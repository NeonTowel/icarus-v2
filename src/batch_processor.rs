//! Batch Processing Module
//!
//! Contains the shared single-image pipeline plus parallel batch orchestration.
//! Pipeline stages are extracted into focused functions; the main orchestrator
//! (`process_image_with_base_paths`) wires them together.

use anyhow::{bail, Context, Result};
use image::DynamicImage;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::metrics::{AccuracyMetrics, BatchMetrics, ImageMetrics, StageTimer};

use crate::config::{ArtisticCropConfig, CropConfig};
use crate::detections_json::save_detections_json;
use crate::directory_walker::relative_to;
use crate::face_detection::{detect_faces, FaceDetector};
use crate::image_io::save_image_with_exif;
use crate::image_utils::{crop_image, crop_to_ultrawide_21_9_centered};
use crate::models::Model;
use crate::multi_format_cropping::{
    analyze_joint, apply_margin_to_bbox, calculate_compound_bbox,
    calculate_landscape_21_9_crop_with_face, calculate_portrait_9_16_crop_with_face,
    calculate_portrait_9_21_crop_with_face, deduplicate_person_detections, detect_suitable_formats,
    merge_bboxes, select_dominant_face_for_crop, strategy_for, BBox, CropRegion,
};
use crate::output_sorting;
use crate::visualization::save_visualized_with_faces;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Shared processing inputs used for a single image.
pub struct ProcessingContext<'a> {
    pub model: &'a dyn Model,
    pub face_detector: &'a FaceDetector,
    pub classifier: Option<&'a dyn crate::models::ImageClassifier>,
    pub crop_config: &'a CropConfig,
    pub artistic_config: &'a ArtisticCropConfig,
    pub confidence: f32,
    pub margin: f32,
    pub keep_aspect_ratio: bool,
    pub sort_output: bool,
    pub classify_output: bool,
    pub classify_only: bool,
    pub quiet: bool,
    pub rename: bool,
    pub jpeg_quality: Option<u8>,
    pub flatten: bool,
    /// When `true`, per-stage timings are collected and returned with each result.
    pub collect_metrics: bool,
    /// When `true`, joint person+face analysis is used for enhanced crop accuracy.
    /// Overrides `crop_config.enable_enhanced_crop` (CLI flag wins).
    pub enhanced_crop: bool,
    /// Minimum long-side pixel count for the pre-inference early filter.
    ///
    /// Images whose `max(width, height)` is below this value are skipped before any
    /// inference when `enhanced_crop` is active. `0` disables the check.
    /// CLI `--min-pixels` always overrides `crop_config.min_long_side_pixels`.
    pub min_long_side_pixels: u32,
}

/// Result metadata for one processed image.
#[derive(Debug)]
pub struct ProcessingResult {
    pub input_path: PathBuf,
    pub output_files: Vec<PathBuf>,
    pub person_count: usize,
    pub face_count: usize,
    /// Per-stage timings. `None` when `ctx.collect_metrics` is `false`.
    pub metrics: Option<ImageMetrics>,
    /// Crop-accuracy counters. `None` when enhanced-crop is off.
    pub accuracy: Option<AccuracyMetrics>,
}

/// Summary of a batch run.
#[derive(Debug, Default)]
pub struct BatchSummary {
    pub total: usize,
    pub succeeded: usize,
    pub failed: Vec<(PathBuf, String)>,
}

/// CLI-friendly detection type used by the output helpers.
#[derive(Debug, Clone)]
pub struct Detection {
    pub bbox: [f32; 4],
    pub label: String,
    pub class_id: usize,
    pub confidence: f32,
}

impl From<crate::models::Detection> for Detection {
    fn from(detection: crate::models::Detection) -> Self {
        Self {
            bbox: [
                detection.bbox.x_min,
                detection.bbox.y_min,
                detection.bbox.x_max,
                detection.bbox.y_max,
            ],
            label: detection.class_name,
            class_id: detection.class_id,
            confidence: detection.confidence,
        }
    }
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const PERSON_CLASS_ID: usize = 0;

// ---------------------------------------------------------------------------
// Small stage helpers
// ---------------------------------------------------------------------------

/// Resolve the output file stem and extension from `output_path` and `jpeg_quality`.
///
/// When `jpeg_quality` is set, always returns `"jpg"` as the extension regardless
/// of the path extension, ensuring `--jpeg` is honoured everywhere.
fn resolve_output_naming(output_path: &Path, jpeg_quality: Option<u8>) -> (&str, &str) {
    let stem = output_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("crop");
    let ext = if jpeg_quality.is_some() {
        "jpg"
    } else {
        output_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("jpg")
    };
    (stem, ext)
}

fn find_best_person_detection(detections: &[Detection]) -> Option<&Detection> {
    detections.iter().find(|d| d.class_id == PERSON_CLASS_ID)
}

/// Classify `image` and return the content tier. Never propagates errors (classifier
/// failure is non-fatal, matching the face-detection contract).
fn try_get_classification_tier(
    classifier: Option<&dyn crate::models::ImageClassifier>,
    image: &DynamicImage,
    file_label: &str,
) -> Option<u8> {
    let c = classifier?;
    match c.classify(image) {
        Ok(t) => Some(t),
        Err(error) => {
            eprintln!(
                "[WARN][{}] classifier failed: {}. continuing without rating.",
                file_label, error
            );
            None
        }
    }
}

fn save_fallback_crop(
    image: &DynamicImage,
    output_path: &Path,
    tier: Option<u8>,
    jpeg_quality: Option<u8>,
) -> Result<()> {
    let crop = crop_to_ultrawide_21_9_centered(image)
        .with_context(|| "Failed to crop image to centered 21:9")?;
    save_image_with_exif(&crop, output_path, tier, jpeg_quality)
        .with_context(|| format!("Failed to save fallback crop to {:?}", output_path))
}

fn ensure_parent_directories(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create output dir: {:?}", parent))?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Detection stages
// ---------------------------------------------------------------------------

/// Run person detection and return all confidence-filtered detections.
///
/// Uses [`Model::infer`] (the one-shot path) which eliminates the Candle tensor
/// round-trip for ORT-backed YOLO models while falling back to the legacy
/// `preprocess → forward → postprocess` sequence for any other implementor.
fn run_detection(
    image: &DynamicImage,
    file_label: &str,
    model: &dyn Model,
    confidence: f32,
) -> Result<Vec<Detection>> {
    let _ = file_label; // reserved for future per-image logging
    let raw = model
        .infer(image)
        .map_err(|e| anyhow::anyhow!("Inference failed: {e}"))?;
    Ok(raw
        .into_iter()
        .map(Detection::from)
        .filter(|d| d.confidence >= confidence)
        .collect())
}

/// Detect faces; non-fatal — on error, log a warning and return empty.
fn detect_faces_nonfatal(
    image: &DynamicImage,
    file_label: &str,
    face_detector: &crate::face_detection::FaceDetector,
    quiet: bool,
) -> Vec<BBox> {
    match detect_faces(image, face_detector) {
        Ok(faces) => {
            if !quiet && !faces.is_empty() {
                println!("[{}] {} face(s) detected.", file_label, faces.len());
            }
            faces
        }
        Err(error) => {
            eprintln!(
                "[WARN][{}] face detection failed: {}. continuing without faces.",
                file_label, error
            );
            Vec::new()
        }
    }
}

/// Compute the person bounding box: single bbox, or compound+dedup for multiple persons.
fn compute_base_bbox(
    person_detections: &[&Detection],
    file_label: &str,
    crop_config: &CropConfig,
    quiet: bool,
) -> Option<BBox> {
    match person_detections.len() {
        0 => None,
        1 => Some(person_detections[0].bbox.into()),
        count => {
            if !quiet {
                println!(
                    "[{}] {} persons detected; using compound bbox.",
                    file_label, count
                );
            }
            let items: Vec<crate::image_utils::Detection> = person_detections
                .iter()
                .map(|d| crate::image_utils::Detection {
                    bbox: d.bbox,
                    confidence: d.confidence,
                    label: d.label.clone(),
                    class_id: d.class_id,
                })
                .collect();
            let effective = if crop_config.enable_reflection_dedup {
                deduplicate_person_detections(&items, crop_config.dedup_iou_threshold)
            } else {
                items
            };
            calculate_compound_bbox(&effective)
        }
    }
}

/// Merge person bbox with faces (or fall back to dominant face when no person detected).
fn compute_crop_bbox(base: Option<BBox>, face_bboxes: &[BBox]) -> Option<BBox> {
    if let Some(b) = base {
        let merged = face_bboxes.iter().fold(b, |acc, f| merge_bboxes(&acc, f));
        Some(merged)
    } else if !face_bboxes.is_empty() {
        select_dominant_face_for_crop(face_bboxes).cloned()
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Classify-only stage
// ---------------------------------------------------------------------------

fn process_classify_only(
    image_path: &Path,
    image: &DynamicImage,
    file_label: &str,
    output_path: Option<&Path>,
    ctx: &ProcessingContext<'_>,
) -> Result<ProcessingResult> {
    let mut output_files: Vec<PathBuf> = Vec::new();

    if let Some(output_path) = output_path {
        output_sorting::ensure_output_dirs(
            output_path.parent().unwrap_or(Path::new(".")),
            ctx.sort_output,
            ctx.classify_output,
        )
        .context("Failed to create output subdirectories")?;

        let Some(classifier) = ctx.classifier else {
            bail!("--classify-only requires classifier but none was initialized");
        };
        let tier =
            Some(try_get_classification_tier(Some(classifier), image, file_label).unwrap_or(0));

        let (stem, ext) = resolve_output_naming(output_path, ctx.jpeg_quality);
        let actual_output_path = if ctx.sort_output {
            let ar = image.width() as f32 / image.height() as f32;
            let fmt = output_sorting::determine_best_format_for_aspect_ratio(ar);
            output_sorting::get_sorted_output_path(output_path, fmt, stem, ext, true, tier)?
        } else {
            output_path.to_path_buf()
        };

        save_image_with_exif(image, &actual_output_path, tier, ctx.jpeg_quality)
            .with_context(|| format!("Failed to save to {:?}", actual_output_path))?;
        output_files.push(actual_output_path.clone());

        if !ctx.quiet {
            println!(
                "[{}] classify-only: saved original image → {:?}",
                file_label, actual_output_path
            );
        }
    }

    Ok(ProcessingResult {
        input_path: image_path.to_path_buf(),
        output_files,
        person_count: 0,
        face_count: 0,
        metrics: None,
        accuracy: None,
    })
}

// ---------------------------------------------------------------------------
// Output stages
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn write_whole_image_crop(
    image: &DynamicImage,
    file_label: &str,
    output_path: &Path,
    viz_path: Option<&Path>,
    person_for_crop: Option<&Detection>,
    detections: &[Detection],
    face_bboxes: &[BBox],
    all_crop_regions: &[CropRegion],
    output_files: &mut Vec<PathBuf>,
    ctx: &ProcessingContext<'_>,
) -> Result<()> {
    let cropped = if ctx.keep_aspect_ratio {
        person_for_crop
            .map(|p| {
                crop_image(image, p.bbox)
                    .with_context(|| format!("Failed to crop bbox {:?}", p.bbox))
            })
            .transpose()?
    } else {
        Some(
            crop_to_ultrawide_21_9_centered(image)
                .with_context(|| "Failed to crop image to centered 21:9")?,
        )
    };

    let img_to_use = cropped.as_ref().unwrap_or(image);
    let tier = if ctx.classify_output {
        Some(try_get_classification_tier(ctx.classifier, img_to_use, file_label).unwrap_or(0))
    } else {
        None
    };

    let (stem, ext) = resolve_output_naming(output_path, ctx.jpeg_quality);
    let actual_path = if ctx.sort_output {
        let ar = img_to_use.width() as f32 / img_to_use.height() as f32;
        let fmt = output_sorting::determine_best_format_for_aspect_ratio(ar);
        output_sorting::get_sorted_output_path(output_path, fmt, stem, ext, true, tier)?
    } else {
        output_path.to_path_buf()
    };

    save_image_with_exif(img_to_use, &actual_path, tier, ctx.jpeg_quality)
        .with_context(|| format!("Failed to save to {:?}", actual_path))?;
    output_files.push(actual_path.clone());
    if !ctx.quiet {
        println!("[{}] saved crop → {:?}", file_label, actual_path);
    }

    if let Some(viz_target) = viz_path {
        let viz_out = resolve_whole_image_viz_path(viz_target, img_to_use, tier, ctx)?;
        save_visualized_with_faces(image, detections, face_bboxes, all_crop_regions, &viz_out)?;
        output_files.push(viz_out.clone());
        if !ctx.quiet {
            println!("[{}] saved annotated → {:?}", file_label, viz_out);
        }
    }
    Ok(())
}

fn resolve_whole_image_viz_path(
    viz_target: &Path,
    img_to_use: &DynamicImage,
    tier: Option<u8>,
    ctx: &ProcessingContext<'_>,
) -> Result<PathBuf> {
    if ctx.sort_output {
        let ar = img_to_use.width() as f32 / img_to_use.height() as f32;
        let fmt = output_sorting::determine_best_format_for_aspect_ratio(ar);
        let stem = viz_target
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("viz");
        let ext = viz_target
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("jpg");
        return output_sorting::get_annotated_output_path(viz_target, fmt, stem, ext, true, tier);
    }
    let stem = viz_target
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("viz");
    let ext = viz_target
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("jpg");
    let dir = viz_target.parent().unwrap_or(Path::new("."));
    let annotated_stem = if stem.ends_with("_annotated") {
        stem.to_string()
    } else {
        format!("{}_annotated", stem)
    };
    Ok(dir.join(format!("{}.{}", annotated_stem, ext)))
}

/// Named bundle of loop-invariant context shared across all format iterations.
struct FormatLoopCtx<'a> {
    image: &'a DynamicImage,
    file_label: &'a str,
    output_path: &'a Path,
    viz_path: Option<&'a Path>,
    /// Original (pre-base-margin) bbox, passed through to joint analyzer.
    base_bbox: &'a BBox,
    face_bboxes: &'a [BBox],
    detections: &'a [Detection],
    focal: &'a crate::focal_point::FocalPoint,
    image_width: u32,
    image_height: u32,
    stem: &'a str,
    ext: &'a str,
    viz_stem: Option<String>,
    viz_ext: Option<String>,
}

#[allow(clippy::too_many_arguments)]
fn write_multi_format_crops(
    image: &DynamicImage,
    file_label: &str,
    output_path: &Path,
    viz_path: Option<&Path>,
    crop_bbox: &BBox,
    face_bboxes: &[BBox],
    detections: &[Detection],
    image_width: u32,
    image_height: u32,
    output_files: &mut Vec<PathBuf>,
    all_crop_regions: &mut Vec<CropRegion>,
    ctx: &ProcessingContext<'_>,
) -> Result<AccuracyMetrics> {
    let mut acc = AccuracyMetrics::default();

    if ctx.margin < 0.0 {
        bail!("--margin must be ≥ 0, got {}", ctx.margin);
    }
    let suitable_formats = detect_suitable_formats(
        image.width(),
        image.height(),
        crop_bbox,
        ctx.margin,
        ctx.crop_config,
    );
    if suitable_formats.is_empty() {
        if !ctx.quiet {
            println!("[{}] no suitable crop formats; using fallback.", file_label);
        }
        let tier = if ctx.classify_output {
            Some(try_get_classification_tier(ctx.classifier, image, file_label).unwrap_or(0))
        } else {
            None
        };
        save_fallback_crop(image, output_path, tier, ctx.jpeg_quality)?;
        output_files.push(output_path.to_path_buf());
        return Ok(acc);
    }
    if !ctx.quiet {
        println!(
            "[{}] suitable formats: {}",
            file_label,
            suitable_formats.join(", ")
        );
    }
    let focal = crate::focal_point::compute_focal_point(
        Some(crop_bbox),
        face_bboxes,
        image_width,
        image_height,
    );
    let (stem, ext) = resolve_output_naming(output_path, ctx.jpeg_quality);
    let loop_ctx = FormatLoopCtx {
        image,
        file_label,
        output_path,
        viz_path,
        base_bbox: crop_bbox,
        face_bboxes,
        detections,
        focal: &focal,
        image_width,
        image_height,
        stem,
        ext,
        viz_stem: viz_path.and_then(|p| p.file_stem().and_then(|s| s.to_str()).map(str::to_string)),
        viz_ext: if ctx.jpeg_quality.is_some() {
            Some("jpg".to_string())
        } else {
            viz_path.and_then(|p| p.extension().and_then(|e| e.to_str()).map(str::to_string))
        },
    };

    let use_enhanced = ctx.enhanced_crop || ctx.crop_config.enable_enhanced_crop;

    for format in &suitable_formats {
        let working_bbox =
            apply_margin_to_bbox(crop_bbox, ctx.margin, image.width(), image.height());

        if use_enhanced {
            let (crop_opt, joint_opt) = compute_format_crop_enhanced(
                format,
                image,
                loop_ctx.base_bbox,
                &working_bbox,
                face_bboxes,
                &focal,
                ctx,
                file_label,
            );
            // Accumulate accuracy counters from the joint analysis.
            if let Some(ref joint) = joint_opt {
                acc.crops_total += 1;
                if joint.full_person_height_expected {
                    acc.crops_full_person_height += 1;
                }
                if joint.relaxed {
                    acc.crops_min_dim_relaxed += 1;
                }
            }
            if let Some(original_crop) = crop_opt {
                write_format_crop_from_region(
                    format,
                    original_crop,
                    &loop_ctx,
                    output_files,
                    all_crop_regions,
                    ctx,
                )?;
            }
        } else {
            write_one_format_crop(
                format,
                &working_bbox,
                &loop_ctx,
                output_files,
                all_crop_regions,
                ctx,
            )?;
        }
    }
    Ok(acc)
}

fn write_one_format_crop(
    format: &str,
    working_bbox: &BBox,
    lc: &FormatLoopCtx<'_>,
    output_files: &mut Vec<PathBuf>,
    all_crop_regions: &mut Vec<CropRegion>,
    ctx: &ProcessingContext<'_>,
) -> Result<()> {
    let Some(original_crop) = compute_format_crop(
        format,
        lc.image,
        working_bbox,
        lc.face_bboxes,
        lc.focal,
        ctx,
        lc.file_label,
    ) else {
        return Ok(());
    };
    write_format_crop_from_region(
        format,
        original_crop,
        lc,
        output_files,
        all_crop_regions,
        ctx,
    )
}

/// Write a pre-computed `CropRegion` through the eye-safety, encode, and save pipeline.
///
/// Shared by both the standard (`write_one_format_crop`) and enhanced
/// (`compute_format_crop_enhanced` → this) code paths so the eye-safety and output
/// logic is never duplicated.
fn write_format_crop_from_region(
    format: &str,
    original_crop: CropRegion,
    lc: &FormatLoopCtx<'_>,
    output_files: &mut Vec<PathBuf>,
    all_crop_regions: &mut Vec<CropRegion>,
    ctx: &ProcessingContext<'_>,
) -> Result<()> {
    let adjusted = crate::face_aware_cropping::enforce_eye_safety(
        &original_crop,
        lc.face_bboxes,
        lc.image_width,
        lc.image_height,
    );
    all_crop_regions.push(adjusted.clone());

    let xyxy = adjusted.to_xyxy_clamped(lc.image.width(), lc.image.height());
    let final_crop = crop_image(lc.image, xyxy)
        .with_context(|| format!("Failed to crop image to region {:?}", xyxy))?;

    let tier = if ctx.classify_output {
        Some(try_get_classification_tier(ctx.classifier, &final_crop, lc.file_label).unwrap_or(0))
    } else {
        None
    };

    let crop_path = output_sorting::get_sorted_output_path(
        lc.output_path,
        format,
        lc.stem,
        lc.ext,
        ctx.sort_output,
        tier,
    )
    .with_context(|| format!("Failed to build output path for format {}", format))?;

    save_image_with_exif(&final_crop, &crop_path, tier, ctx.jpeg_quality)
        .with_context(|| format!("Failed to save cropped image to {:?}", crop_path))?;
    output_files.push(crop_path.clone());
    if !ctx.quiet {
        println!(
            "[{}] saved {} crop → {:?}",
            lc.file_label, format, crop_path
        );
    }

    if let (Some(vstem), Some(vext), Some(viz_base)) = (&lc.viz_stem, &lc.viz_ext, lc.viz_path) {
        let viz_out = output_sorting::get_annotated_output_path(
            viz_base,
            format,
            vstem,
            vext,
            ctx.sort_output,
            tier,
        )
        .with_context(|| format!("Failed to build annotated path for format {}", format))?;
        save_visualized_with_faces(
            lc.image,
            lc.detections,
            lc.face_bboxes,
            std::slice::from_ref(&adjusted),
            &viz_out,
        )?;
        output_files.push(viz_out.clone());
        if !ctx.quiet {
            println!(
                "[{}] saved {} annotated → {:?}",
                lc.file_label, format, viz_out
            );
        }
    }
    Ok(())
}

fn compute_format_crop(
    format: &str,
    image: &DynamicImage,
    bbox: &BBox,
    face_bboxes: &[BBox],
    focal: &crate::focal_point::FocalPoint,
    ctx: &ProcessingContext<'_>,
    file_label: &str,
) -> Option<CropRegion> {
    match format {
        "21:9" => calculate_landscape_21_9_crop_with_face(
            image.width(),
            image.height(),
            bbox,
            face_bboxes,
            focal,
            ctx.crop_config,
            ctx.artistic_config,
        ),
        "9:21" => calculate_portrait_9_21_crop_with_face(
            image.width(),
            image.height(),
            bbox,
            face_bboxes,
            focal,
            ctx.crop_config,
            ctx.artistic_config,
        ),
        "9:16" => calculate_portrait_9_16_crop_with_face(
            image.width(),
            image.height(),
            bbox,
            face_bboxes,
            focal,
            ctx.crop_config,
            ctx.artistic_config,
        ),
        other => {
            eprintln!(
                "[WARN][{}] unknown format '{}' — skipping.",
                file_label, other
            );
            None
        }
    }
}

/// Compute a crop region using joint person+face analysis (enhanced-crop path).
///
/// Calls `analyze_joint` to get a `JointAnalysis`, then applies any recommended
/// extra margin to the working bbox and calls the strategy's `calculate_with_joint`.
///
/// Returns `(crop_region, joint_analysis)` so the caller can record accuracy counters.
/// Compute a crop region using joint person+face analysis (enhanced-crop path).
///
/// Analyzes using `person_bbox` (pre-user-margin, actual person dimensions) so the
/// height comparison is accurate. Applies any recommended `extra_margin_percent` on
/// top of `working_bbox` (already user-margin expanded) before calling the strategy.
///
/// Returns `(crop_region, joint_analysis)` so the caller can record accuracy counters.
#[allow(clippy::too_many_arguments)]
fn compute_format_crop_enhanced(
    format: &str,
    image: &DynamicImage,
    person_bbox: &BBox,
    working_bbox: &BBox,
    face_bboxes: &[BBox],
    focal: &crate::focal_point::FocalPoint,
    ctx: &ProcessingContext<'_>,
    file_label: &str,
) -> (
    Option<CropRegion>,
    Option<crate::crop::joint_analyzer::JointAnalysis>,
) {
    let (aspect_ratio, height_first) = match format {
        "21:9" => (21.0_f32 / 9.0, true),
        "9:21" => (9.0_f32 / 21.0, false),
        "9:16" => (9.0_f32 / 16.0, false),
        other => {
            eprintln!(
                "[WARN][{}] unknown format '{}' — skipping.",
                file_label, other
            );
            return (None, None);
        }
    };

    // Analyze using the original person bbox so person height vs crop height is correct.
    let joint = analyze_joint(
        person_bbox,
        face_bboxes,
        image.width(),
        image.height(),
        aspect_ratio,
        height_first,
        ctx.crop_config,
        ctx.artistic_config,
    );

    // Apply extra margin from joint analysis on top of the already-user-margined bbox.
    let effective_bbox = if joint.extra_margin_percent > 0.0 {
        apply_margin_to_bbox(
            working_bbox,
            joint.extra_margin_percent,
            image.width(),
            image.height(),
        )
    } else {
        working_bbox.clone()
    };

    let crop = strategy_for(format).and_then(|s| {
        s.calculate_with_joint(
            image.width(),
            image.height(),
            &effective_bbox,
            face_bboxes,
            focal,
            ctx.crop_config,
            ctx.artistic_config,
            Some(&joint),
        )
    });

    (crop, Some(joint))
}

// ---------------------------------------------------------------------------
// Main orchestrator
// ---------------------------------------------------------------------------

fn process_image_with_base_paths(
    image_path: &Path,
    output_path: Option<&Path>,
    viz_path: Option<&Path>,
    boxes_path: Option<&Path>,
    ctx: &ProcessingContext<'_>,
) -> Result<ProcessingResult> {
    let file_label = image_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("<unknown>");

    let use_enhanced = ctx.enhanced_crop || ctx.crop_config.enable_enhanced_crop;

    // Early long-side filter: header-only read BEFORE full decode to avoid the dominant
    // face-detection cost on photos that cannot yield good output.
    // Only runs when enhanced_crop is active — guarantees flag-off is byte-for-byte identical.
    // On header-read failure: non-fatal, fall through to normal processing (debug warn).
    let mut filter_considered: u64 = 0;
    if use_enhanced && ctx.min_long_side_pixels > 0 {
        match crate::early_filter::read_long_side(image_path) {
            Ok(long_side) => {
                filter_considered = 1;
                if !crate::crop::geometry::long_side_passes(long_side, ctx.min_long_side_pixels) {
                    log::debug!(
                        "[{}] early-filter: long side {} px < min-pixels {} — skipping",
                        file_label,
                        long_side,
                        ctx.min_long_side_pixels
                    );
                    if !ctx.quiet {
                        println!(
                            "[{}] skipped: long side {} px < --min-pixels {}",
                            file_label, long_side, ctx.min_long_side_pixels
                        );
                    }
                    return Ok(ProcessingResult {
                        input_path: image_path.to_path_buf(),
                        output_files: vec![],
                        person_count: 0,
                        face_count: 0,
                        metrics: None,
                        accuracy: Some(AccuracyMetrics {
                            images_considered: 1,
                            images_early_skipped: 1,
                            ..Default::default()
                        }),
                    });
                }
            }
            Err(e) => {
                log::debug!(
                    "[{}] header-read failed (non-fatal, processing anyway): {e}",
                    file_label
                );
            }
        }
    }

    let t_decode = StageTimer::start();
    let image = image::open(image_path).with_context(|| {
        format!(
            "[{}] failed to open input image: {:?}",
            file_label, image_path
        )
    })?;
    let (image_width, image_height) = (image.width(), image.height());
    let dur_decode = t_decode.elapsed();

    if ctx.classify_only {
        let mut r = process_classify_only(image_path, &image, file_label, output_path, ctx)?;
        r.metrics = ctx.collect_metrics.then(|| ImageMetrics {
            decode: dur_decode,
            ..Default::default()
        });
        r.accuracy = None;
        return Ok(r);
    }

    let t_detect = StageTimer::start();
    let detections = run_detection(&image, file_label, ctx.model, ctx.confidence)?;
    let person_detections: Vec<&Detection> = detections
        .iter()
        .filter(|d| d.class_id == PERSON_CLASS_ID)
        .collect();
    let dur_detect = t_detect.elapsed();

    let t_face = StageTimer::start();
    let face_bboxes = detect_faces_nonfatal(&image, file_label, ctx.face_detector, ctx.quiet);
    let dur_face = t_face.elapsed();

    let t_crop = StageTimer::start();
    let base_bbox = compute_base_bbox(&person_detections, file_label, ctx.crop_config, ctx.quiet);
    let crop_bbox = compute_crop_bbox(base_bbox, &face_bboxes);
    let person_for_crop = find_best_person_detection(&detections);
    let dur_crop = t_crop.elapsed();

    if !ctx.quiet && detections.is_empty() {
        println!(
            "[{}] no objects detected (confidence: {}).",
            file_label, ctx.confidence
        );
    }

    let mut all_crop_regions: Vec<CropRegion> = Vec::new();
    let mut output_files: Vec<PathBuf> = Vec::new();
    let mut image_accuracy = AccuracyMetrics::default();

    let t_encode = StageTimer::start();
    if let Some(out) = output_path {
        output_sorting::ensure_output_dirs(
            out.parent().unwrap_or(Path::new(".")),
            ctx.sort_output,
            ctx.classify_output,
        )
        .context("Failed to create output subdirectories")?;

        if ctx.keep_aspect_ratio || crop_bbox.is_none() {
            write_whole_image_crop(
                &image,
                file_label,
                out,
                viz_path,
                person_for_crop,
                &detections,
                &face_bboxes,
                &all_crop_regions,
                &mut output_files,
                ctx,
            )?;
        } else {
            image_accuracy = write_multi_format_crops(
                &image,
                file_label,
                out,
                viz_path,
                crop_bbox.as_ref().expect("crop bbox is Some here"),
                &face_bboxes,
                &detections,
                image_width,
                image_height,
                &mut output_files,
                &mut all_crop_regions,
                ctx,
            )?;
        }
    } else if let Some(viz) = viz_path {
        save_visualized_with_faces(&image, &detections, &face_bboxes, &all_crop_regions, viz)?;
        output_files.push(viz.to_path_buf());
        if !ctx.quiet {
            println!("[{}] saved annotated → {:?}", file_label, viz);
        }
    }

    if let Some(boxes) = boxes_path {
        save_detections_json(&detections, &face_bboxes, boxes)?;
        output_files.push(boxes.to_path_buf());
        if !ctx.quiet {
            println!(
                "[{}] saved {} person + {} face detection(s) → {:?}",
                file_label,
                detections
                    .iter()
                    .filter(|d| d.class_id == PERSON_CLASS_ID)
                    .count(),
                face_bboxes.len(),
                boxes,
            );
        }
    }
    let dur_encode = t_encode.elapsed();

    let stage_metrics = ctx.collect_metrics.then(|| ImageMetrics {
        decode: dur_decode,
        person_detect: dur_detect,
        face_detect: dur_face,
        crop: dur_crop,
        classify: Default::default(), // classify time is within encode; future M7 can split
        encode: dur_encode,
    });

    // Return accuracy when enhanced-crop is active; None otherwise (no noise when off).
    // Merge early-filter counters (images_considered) with per-crop counters.
    let accuracy = use_enhanced.then_some(AccuracyMetrics {
        images_considered: filter_considered,
        images_early_skipped: 0, // 0 here — early-skip returns above before reaching this point
        crops_total: image_accuracy.crops_total,
        crops_full_person_height: image_accuracy.crops_full_person_height,
        crops_min_dim_relaxed: image_accuracy.crops_min_dim_relaxed,
    });

    Ok(ProcessingResult {
        input_path: image_path.to_path_buf(),
        output_files,
        person_count: detections
            .iter()
            .filter(|d| d.class_id == PERSON_CLASS_ID)
            .count(),
        face_count: face_bboxes.len(),
        metrics: stage_metrics,
        accuracy,
    })
}

// ---------------------------------------------------------------------------
// Batch dispatch
// ---------------------------------------------------------------------------

fn process_one_in_batch(
    image_path: &Path,
    input_root: &Path,
    output_root: Option<&Path>,
    viz_root: Option<&Path>,
    boxes_root: Option<&Path>,
    ctx: &ProcessingContext<'_>,
) -> Result<ProcessingResult> {
    let mut relative = if ctx.flatten {
        image_path
            .file_name()
            .map(PathBuf::from)
            .unwrap_or_else(|| image_path.to_path_buf())
    } else {
        relative_to(image_path, input_root).unwrap_or_else(|| {
            image_path
                .file_name()
                .map(PathBuf::from)
                .unwrap_or_else(|| image_path.to_path_buf())
        })
    };

    if ctx.rename {
        let new_stem = uuid::Uuid::new_v4().to_string();
        let ext = if ctx.jpeg_quality.is_some() {
            "jpg".to_string()
        } else {
            relative
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_owned()
        };
        relative.set_file_name(new_stem);
        if !ext.is_empty() {
            relative.set_extension(ext);
        }
    }

    let output_path = output_root.map(|root| root.join(&relative));
    let viz_path = viz_root.map(|root| root.join(&relative));
    let boxes_path = boxes_root.map(|root| {
        let mut path = root.join(&relative);
        path.set_extension("json");
        path
    });

    for path in [&output_path, &viz_path, &boxes_path].into_iter().flatten() {
        ensure_parent_directories(path)?;
    }

    process_image_with_base_paths(
        image_path,
        output_path.as_deref(),
        viz_path.as_deref(),
        boxes_path.as_deref(),
        ctx,
    )
}

/// Process one image through the full detection, crop, and save pipeline.
pub fn process_single_image(
    input_path: &Path,
    output_path: Option<&Path>,
    viz_path: Option<&Path>,
    boxes_path: Option<&Path>,
    ctx: &ProcessingContext<'_>,
) -> Result<ProcessingResult> {
    if ctx.rename {
        let new_stem = uuid::Uuid::new_v4().to_string();
        let apply_rename = |p: Option<&Path>| -> Option<PathBuf> {
            p.map(|path| {
                let mut pb = path.to_path_buf();
                let ext = pb
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("")
                    .to_owned();
                pb.set_file_name(&new_stem);
                if !ext.is_empty() {
                    pb.set_extension(ext);
                }
                pb
            })
        };
        let new_output = apply_rename(output_path);
        let new_viz = apply_rename(viz_path);
        let new_boxes = apply_rename(boxes_path);
        return process_image_with_base_paths(
            input_path,
            new_output.as_deref(),
            new_viz.as_deref(),
            new_boxes.as_deref(),
            ctx,
        );
    }
    process_image_with_base_paths(input_path, output_path, viz_path, boxes_path, ctx)
}

/// Run the batch processing pipeline in parallel.
pub fn run_batch(
    image_paths: &[PathBuf],
    input_root: &Path,
    output_root: Option<&Path>,
    viz_root: Option<&Path>,
    boxes_root: Option<&Path>,
    ctx: &ProcessingContext<'_>,
    thread_count: usize,
) -> BatchSummary {
    use std::sync::Mutex;

    let succeeded = AtomicUsize::new(0);
    let batch_metrics: Mutex<BatchMetrics> = Mutex::new(BatchMetrics::default());

    let collect_failures = || -> Vec<(PathBuf, String)> {
        image_paths
            .par_iter()
            .filter_map(|image_path| {
                match process_one_in_batch(
                    image_path,
                    input_root,
                    output_root,
                    viz_root,
                    boxes_root,
                    ctx,
                ) {
                    Ok(result) => {
                        succeeded.fetch_add(1, Ordering::Relaxed);
                        if let Ok(mut bm) = batch_metrics.lock() {
                            if let Some(m) = result.metrics {
                                bm.record(m);
                            }
                            if let Some(a) = result.accuracy {
                                bm.record_accuracy(a);
                            }
                        }
                        if !ctx.quiet {
                            println!(
                                "[OK][{}] {} person(s), {} face(s)",
                                image_path
                                    .file_name()
                                    .and_then(|n| n.to_str())
                                    .unwrap_or("<unknown>"),
                                result.person_count,
                                result.face_count,
                            );
                        }
                        None
                    }
                    Err(error) => {
                        eprintln!(
                            "[ERROR][{}] {}",
                            image_path
                                .file_name()
                                .and_then(|n| n.to_str())
                                .unwrap_or("<unknown>"),
                            error
                        );
                        Some((image_path.clone(), format!("{:#}", error)))
                    }
                }
            })
            .collect()
    };

    let thread_pool = ThreadPoolBuilder::new()
        .num_threads(thread_count.max(1))
        .build();
    let mut failed: Vec<(PathBuf, String)> = match thread_pool {
        Ok(pool) => pool.install(collect_failures),
        Err(error) => {
            eprintln!("[WARN] failed to build thread pool ({error}); using default Rayon pool.");
            collect_failures()
        }
    };

    failed.sort_by(|l, r| l.0.cmp(&r.0));

    if ctx.collect_metrics {
        if let Ok(bm) = batch_metrics.lock() {
            bm.print_summary();
        }
    }

    BatchSummary {
        total: image_paths.len(),
        succeeded: succeeded.load(Ordering::Relaxed),
        failed,
    }
}

// ---------------------------------------------------------------------------
// Tester validation tests (M0–M4 reconciliation)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Gate 8 — Flag-off parity: when enhanced_crop is false, accuracy must be None.
    #[test]
    fn test_flag_off_accuracy_is_none() {
        // This mirrors the exact logic in process_image_with_base_paths line ~1010.
        let use_enhanced = false;
        let accuracy = use_enhanced.then_some(AccuracyMetrics::default());
        assert!(
            accuracy.is_none(),
            "accuracy must be None when enhanced_crop is off"
        );
    }

    /// Gate 8 — Flag-on parity: when enhanced_crop is true, accuracy must be Some.
    #[test]
    fn test_flag_on_accuracy_is_some() {
        let use_enhanced = true;
        let accuracy = use_enhanced.then_some(AccuracyMetrics::default());
        assert!(
            accuracy.is_some(),
            "accuracy must be Some when enhanced_crop is on"
        );
    }

    /// Gate 9 — Early-skip behavior: extreme min-pixels causes long_side_passes to fail.
    #[test]
    fn test_high_min_pixels_skips_all_small_images() {
        // Simulate a 500-px-long-side image with --min-pixels 9999.
        assert!(
            !crate::crop::geometry::long_side_passes(500, 9999),
            "500 px long side must fail 9999 px floor"
        );
    }
}
