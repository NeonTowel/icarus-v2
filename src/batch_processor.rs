//! Batch Processing Module
//!
//! Contains the shared single-image pipeline plus parallel batch orchestration.

use anyhow::{bail, Context, Result};
use image::DynamicImage;
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::config::{ArtisticCropConfig, CropConfig};
use crate::directory_walker::relative_to;
use crate::face_detection::{detect_faces, FaceDetector};
use crate::image_utils::{crop_image, crop_to_ultrawide_21_9_centered};
use crate::models::Model;
use crate::multi_format_cropping::{
    apply_margin_to_bbox, calculate_compound_bbox, calculate_landscape_21_9_crop_with_face,
    calculate_portrait_9_16_crop_with_face, calculate_portrait_9_21_crop_with_face,
    deduplicate_person_detections, detect_suitable_formats, BBox, CropRegion,
};
use crate::output_sorting;

/// Shared processing inputs used for a single image.
pub struct ProcessingContext<'a> {
    pub model: &'a dyn Model,
    pub face_detector: &'a FaceDetector,
    pub crop_config: &'a CropConfig,
    pub artistic_config: &'a ArtisticCropConfig,
    pub confidence: f32,
    pub margin: f32,
    pub keep_aspect_ratio: bool,
    pub sort_output: bool,
    pub quiet: bool,
}

/// Result metadata for one processed image.
#[derive(Debug)]
pub struct ProcessingResult {
    pub input_path: PathBuf,
    pub output_files: Vec<PathBuf>,
    pub person_count: usize,
    pub face_count: usize,
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

const PERSON_CLASS_ID: usize = 0;

fn find_best_person_detection(detections: &[Detection]) -> Option<&Detection> {
    detections
        .iter()
        .find(|detection| detection.class_id == PERSON_CLASS_ID)
}

fn save_fallback_crop(image: &DynamicImage, output_path: &Path) -> Result<()> {
    let crop = crop_to_ultrawide_21_9_centered(image)
        .with_context(|| "Failed to crop image to centered 21:9")?;
    crop.save(output_path)
        .with_context(|| format!("Failed to save cropped image to {:?}", output_path))?;
    Ok(())
}

fn save_crop_region(image: &DynamicImage, crop: &CropRegion, output_path: &Path) -> Result<()> {
    let xyxy = crop.to_xyxy_clamped(image.width(), image.height());
    let cropped = crop_image(image, xyxy)
        .with_context(|| format!("Failed to crop image to region {:?}", xyxy))?;
    cropped
        .save(output_path)
        .with_context(|| format!("Failed to save cropped image to {:?}", output_path))?;
    Ok(())
}

fn save_detections_json(
    detections: &[Detection],
    face_bboxes: &[BBox],
    output_path: &Path,
) -> Result<()> {
    use serde_json::json;

    let person_records: Vec<serde_json::Value> = detections
        .iter()
        .enumerate()
        .map(|(index, detection)| {
            json!({
                "person_id": index,
                "label": detection.label,
                "class_id": detection.class_id,
                "confidence": detection.confidence,
                "bbox": detection.bbox,
            })
        })
        .collect();

    let face_records: Vec<serde_json::Value> = face_bboxes
        .iter()
        .enumerate()
        .map(|(index, bbox)| {
            json!({
                "face_id": index,
                "bbox": [bbox.x1, bbox.y1, bbox.x2, bbox.y2],
            })
        })
        .collect();

    let output = json!({
        "persons": person_records,
        "faces": face_records,
    });

    let serialised = serde_json::to_string_pretty(&output)
        .context("Failed to serialise detections+faces to JSON")?;

    std::fs::write(output_path, serialised)
        .with_context(|| format!("Failed to write JSON to {:?}", output_path))?;
    Ok(())
}

fn save_visualized_with_faces(
    image: &DynamicImage,
    detections: &[Detection],
    face_bboxes: &[BBox],
    crop_regions: &[CropRegion],
    output_path: &Path,
) -> Result<()> {
    use image::{Rgba, RgbaImage};

    let mut canvas: RgbaImage = image.to_rgba8();
    let width = canvas.width();
    let height = canvas.height();

    const PERSON_COLOR: [u8; 4] = [0, 255, 0, 220];
    const OTHER_COLOR: [u8; 4] = [128, 0, 0, 220];
    const FACE_COLOR: [u8; 4] = [0, 255, 255, 220];
    const CROP_COLOR: [u8; 4] = [255, 0, 0, 220];

    fn draw_rect(canvas: &mut RgbaImage, x1: u32, y1: u32, x2: u32, y2: u32, colour: Rgba<u8>) {
        let width = canvas.width();
        let height = canvas.height();
        for thickness in 0..2u32 {
            for x in x1..=x2 {
                for dy in 0..=thickness {
                    if y1 + dy < height {
                        canvas.put_pixel(x, y1 + dy, colour);
                    }
                    if y2 >= dy && y2 - dy < height {
                        canvas.put_pixel(x, y2 - dy, colour);
                    }
                }
            }
            for y in y1..=y2 {
                for dx in 0..=thickness {
                    if x1 + dx < width {
                        canvas.put_pixel(x1 + dx, y, colour);
                    }
                    if x2 >= dx && x2 - dx < width {
                        canvas.put_pixel(x2 - dx, y, colour);
                    }
                }
            }
        }
    }

    for detection in detections {
        let colour = if detection.class_id == PERSON_CLASS_ID {
            Rgba(PERSON_COLOR)
        } else {
            Rgba(OTHER_COLOR)
        };
        let [x1, y1, x2, y2] = detection.bbox;
        draw_rect(
            &mut canvas,
            (x1 as u32).min(width.saturating_sub(1)),
            (y1 as u32).min(height.saturating_sub(1)),
            (x2 as u32).min(width.saturating_sub(1)),
            (y2 as u32).min(height.saturating_sub(1)),
            colour,
        );
    }

    for face in face_bboxes {
        draw_rect(
            &mut canvas,
            (face.x1 as u32).min(width.saturating_sub(1)),
            (face.y1 as u32).min(height.saturating_sub(1)),
            (face.x2 as u32).min(width.saturating_sub(1)),
            (face.y2 as u32).min(height.saturating_sub(1)),
            Rgba(FACE_COLOR),
        );
    }

    for crop in crop_regions {
        let x2 = crop.x + crop.width;
        let y2 = crop.y + crop.height;
        draw_rect(
            &mut canvas,
            (crop.x as u32).min(width.saturating_sub(1)),
            (crop.y as u32).min(height.saturating_sub(1)),
            (x2 as u32).min(width.saturating_sub(1)),
            (y2 as u32).min(height.saturating_sub(1)),
            Rgba(CROP_COLOR),
        );
    }

    DynamicImage::ImageRgba8(canvas)
        .save(output_path)
        .with_context(|| format!("Failed to save visualized image to {:?}", output_path))?;

    Ok(())
}

fn ensure_parent_directories(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create output dir: {:?}", parent))?;
    }

    Ok(())
}

fn process_image_with_base_paths(
    image_path: &Path,
    output_path: Option<&Path>,
    viz_path: Option<&Path>,
    boxes_path: Option<&Path>,
    ctx: &ProcessingContext<'_>,
) -> Result<ProcessingResult> {
    let image = image::open(image_path)
        .with_context(|| format!("Failed to open input image: {:?}", image_path))?;

    let image_width = image.width();
    let image_height = image.height();

    let input_tensor = ctx
        .model
        .preprocess(std::slice::from_ref(&image))
        .map_err(|error| anyhow::anyhow!("Preprocessing failed: {error}"))?;

    let (logits, boxes) = ctx
        .model
        .forward(&input_tensor)
        .map_err(|error| anyhow::anyhow!("Inference failed: {error}"))?;

    let raw_detections = ctx
        .model
        .postprocess(logits, boxes)
        .map_err(|error| anyhow::anyhow!("Postprocessing failed: {error}"))?;

    let detections: Vec<Detection> = raw_detections
        .into_iter()
        .map(Detection::from)
        .filter(|detection| detection.confidence >= ctx.confidence)
        .collect();

    let person_detections: Vec<&Detection> = detections
        .iter()
        .filter(|detection| detection.class_id == PERSON_CLASS_ID)
        .collect();

    let crop_bbox: Option<BBox> = match person_detections.len() {
        0 => None,
        1 => Some(person_detections[0].bbox.into()),
        count => {
            if !ctx.quiet {
                println!(
                    "  Detected {} persons — using compound bbox for cropping.",
                    count
                );
            }

            let detections_for_bbox: Vec<crate::image_utils::Detection> = person_detections
                .iter()
                .map(|detection| crate::image_utils::Detection {
                    bbox: detection.bbox,
                    confidence: detection.confidence,
                    label: detection.label.clone(),
                    class_id: detection.class_id,
                })
                .collect();

            let effective_detections = if ctx.crop_config.enable_reflection_dedup {
                deduplicate_person_detections(
                    &detections_for_bbox,
                    ctx.crop_config.dedup_iou_threshold,
                )
            } else {
                detections_for_bbox
            };
            calculate_compound_bbox(&effective_detections)
        }
    };

    let person_for_crop = find_best_person_detection(&detections);

    let face_bboxes = match detect_faces(&image, ctx.face_detector) {
        Ok(faces) => {
            if !ctx.quiet && !faces.is_empty() {
                println!("  Face detections: {}", faces.len());
            }
            faces
        }
        Err(error) => {
            eprintln!(
                "Warning: face detection failed ({}). Continuing without faces.",
                error
            );
            Vec::new()
        }
    };

    if !ctx.quiet {
        if detections.is_empty() {
            println!(
                "No objects detected (confidence threshold: {}).",
                ctx.confidence
            );
        } else {
            println!("Found {} object(s):", detections.len());
        }
    }

    let mut all_crop_regions: Vec<CropRegion> = Vec::new();
    let mut output_files: Vec<PathBuf> = Vec::new();

    if let Some(output_path) = output_path {
        output_sorting::ensure_output_dirs(
            output_path.parent().unwrap_or(Path::new(".")),
            ctx.sort_output,
        )
        .context("Failed to create output subdirectories")?;

        if ctx.keep_aspect_ratio || crop_bbox.is_none() {
            let cropped_image = if ctx.keep_aspect_ratio {
                if let Some(person) = person_for_crop {
                    Some(
                        crop_image(&image, person.bbox)
                            .with_context(|| format!("Failed to crop bbox {:?}", person.bbox))?,
                    )
                } else {
                    None
                }
            } else {
                Some(
                    crop_to_ultrawide_21_9_centered(&image)
                        .with_context(|| "Failed to crop image to centered 21:9")?,
                )
            };

            let actual_output_path = if ctx.sort_output {
                let (crop_width, crop_height) = match &cropped_image {
                    Some(cropped) => (cropped.width(), cropped.height()),
                    None => (image.width(), image.height()),
                };
                let aspect_ratio = crop_width as f32 / crop_height as f32;
                let format = output_sorting::determine_best_format_for_aspect_ratio(aspect_ratio);
                let stem = output_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("crop");
                let ext = output_path
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("jpg");
                output_sorting::get_sorted_output_path(output_path, format, stem, ext, true)?
            } else {
                output_path.to_path_buf()
            };

            match &cropped_image {
                Some(cropped) => cropped
                    .save(&actual_output_path)
                    .with_context(|| format!("Failed to save to {:?}", actual_output_path))?,
                None => image
                    .save(&actual_output_path)
                    .with_context(|| format!("Failed to save to {:?}", actual_output_path))?,
            }

            output_files.push(actual_output_path.clone());

            if !ctx.quiet {
                println!("Saved cropped image to {:?}", actual_output_path);
            }

            if let Some(viz_target) = viz_path {
                let actual_viz_path = if ctx.sort_output {
                    let (crop_width, crop_height) = match &cropped_image {
                        Some(cropped) => (cropped.width(), cropped.height()),
                        None => (image.width(), image.height()),
                    };
                    let aspect_ratio = crop_width as f32 / crop_height as f32;
                    let format =
                        output_sorting::determine_best_format_for_aspect_ratio(aspect_ratio);
                    let stem = viz_target
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("viz");
                    let ext = viz_target
                        .extension()
                        .and_then(|e| e.to_str())
                        .unwrap_or("jpg");
                    output_sorting::get_annotated_output_path(viz_target, format, stem, ext, true)?
                } else {
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
                    dir.join(format!("{}.{}", annotated_stem, ext))
                };

                save_visualized_with_faces(
                    &image,
                    &detections,
                    &face_bboxes,
                    &all_crop_regions,
                    &actual_viz_path,
                )?;
                output_files.push(actual_viz_path.clone());

                if !ctx.quiet {
                    println!("Saved visualized image to {:?}", actual_viz_path);
                }
            }
        } else {
            if ctx.margin < 0.0 {
                bail!("--margin must be ≥ 0, got {}", ctx.margin);
            }

            let raw_bbox: BBox = crop_bbox.clone().expect("crop bbox should exist");
            let person_bbox_for_adjustment = crop_bbox;
            let suitable_formats = detect_suitable_formats(
                image.width(),
                image.height(),
                &raw_bbox,
                ctx.margin,
                ctx.crop_config,
            );

            if suitable_formats.is_empty() {
                if !ctx.quiet {
                    println!("  No suitable crop formats found for this photo.");
                }
                save_fallback_crop(&image, output_path)?;
                output_files.push(output_path.to_path_buf());
            } else {
                if !ctx.quiet {
                    println!("  Suitable formats: {}", suitable_formats.join(", "));
                }

                let stem = output_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("crop");
                let ext = output_path
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("jpg");
                let viz_stem = viz_path.and_then(|path| {
                    path.file_stem()
                        .and_then(|s| s.to_str())
                        .map(|s| s.to_string())
                });
                let viz_ext = viz_path.and_then(|path| {
                    path.extension()
                        .and_then(|e| e.to_str())
                        .map(|s| s.to_string())
                });

                let focal = crate::focal_point::compute_focal_point(
                    person_bbox_for_adjustment.as_ref(),
                    &face_bboxes,
                    image_width,
                    image_height,
                );

                for format in &suitable_formats {
                    let working_bbox =
                        apply_margin_to_bbox(&raw_bbox, ctx.margin, image.width(), image.height());
                    let original_crop = match format.as_str() {
                        "21:9" => calculate_landscape_21_9_crop_with_face(
                            image.width(),
                            image.height(),
                            &working_bbox,
                            &face_bboxes,
                            &focal,
                            ctx.crop_config,
                            ctx.artistic_config,
                        ),
                        "9:21" => calculate_portrait_9_21_crop_with_face(
                            image.width(),
                            image.height(),
                            &working_bbox,
                            &face_bboxes,
                            &focal,
                            ctx.crop_config,
                            ctx.artistic_config,
                        ),
                        "9:16" => calculate_portrait_9_16_crop_with_face(
                            image.width(),
                            image.height(),
                            &working_bbox,
                            &face_bboxes,
                            &focal,
                            ctx.crop_config,
                            ctx.artistic_config,
                        ),
                        other => {
                            eprintln!("Warning: unknown format '{}' — skipping.", other);
                            None
                        }
                    };

                    let original_crop = match original_crop {
                        Some(crop) => crop,
                        None => continue,
                    };

                    let adjusted_crop = crate::face_aware_cropping::enforce_eye_safety(
                        &original_crop,
                        &face_bboxes,
                        image_width,
                        image_height,
                    );

                    all_crop_regions.push(adjusted_crop.clone());

                    let crop_path = output_sorting::get_sorted_output_path(
                        output_path,
                        format,
                        stem,
                        ext,
                        ctx.sort_output,
                    )
                    .with_context(|| {
                        format!("Failed to build output path for format {}", format)
                    })?;

                    save_crop_region(&image, &adjusted_crop, &crop_path)?;
                    output_files.push(crop_path.clone());

                    if !ctx.quiet {
                        println!("  Saved {} crop → {:?}", format, crop_path);
                    }

                    if let (Some(vstem), Some(vext), Some(viz_target)) =
                        (&viz_stem, &viz_ext, viz_path)
                    {
                        let viz_base = viz_target;
                        let viz_output_path = output_sorting::get_annotated_output_path(
                            viz_base,
                            format,
                            vstem,
                            vext,
                            ctx.sort_output,
                        )
                        .with_context(|| {
                            format!("Failed to build annotated path for format {}", format)
                        })?;

                        save_visualized_with_faces(
                            &image,
                            &detections,
                            &face_bboxes,
                            std::slice::from_ref(&adjusted_crop),
                            &viz_output_path,
                        )?;
                        output_files.push(viz_output_path.clone());

                        if !ctx.quiet {
                            println!("  Saved {} annotated → {:?}", format, viz_output_path);
                        }
                    }
                }
            }
        }
    } else if let Some(viz_path) = viz_path {
        save_visualized_with_faces(
            &image,
            &detections,
            &face_bboxes,
            &all_crop_regions,
            viz_path,
        )?;
        output_files.push(viz_path.to_path_buf());

        if !ctx.quiet {
            println!("Saved visualized image to {:?}", viz_path);
        }
    }

    if let Some(boxes_path) = boxes_path {
        save_detections_json(&detections, &face_bboxes, boxes_path)?;
        output_files.push(boxes_path.to_path_buf());

        if !ctx.quiet {
            println!(
                "Saved {} person + {} face detection(s) to {:?}",
                detections
                    .iter()
                    .filter(|detection| detection.class_id == PERSON_CLASS_ID)
                    .count(),
                face_bboxes.len(),
                boxes_path,
            );
        }
    }

    Ok(ProcessingResult {
        input_path: image_path.to_path_buf(),
        output_files,
        person_count: detections
            .iter()
            .filter(|detection| detection.class_id == PERSON_CLASS_ID)
            .count(),
        face_count: face_bboxes.len(),
    })
}

fn process_one_in_batch(
    image_path: &Path,
    input_root: &Path,
    output_root: Option<&Path>,
    viz_root: Option<&Path>,
    boxes_root: Option<&Path>,
    ctx: &ProcessingContext<'_>,
) -> Result<ProcessingResult> {
    let relative = relative_to(image_path, input_root).unwrap_or_else(|| {
        image_path
            .file_name()
            .map(PathBuf::from)
            .unwrap_or_else(|| image_path.to_path_buf())
    });

    let output_path = output_root.map(|root| root.join(&relative));
    let viz_path = viz_root.map(|root| root.join(&relative));
    let boxes_path = boxes_root.map(|root| {
        let mut path = root.join(&relative);
        path.set_extension("json");
        path
    });

    if let Some(ref path) = output_path {
        ensure_parent_directories(path)?;
    }
    if let Some(ref path) = viz_path {
        ensure_parent_directories(path)?;
    }
    if let Some(ref path) = boxes_path {
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
) -> BatchSummary {
    let succeeded = AtomicUsize::new(0);

    let mut failed: Vec<(PathBuf, String)> = image_paths
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
                    if !ctx.quiet {
                        println!(
                            "  [OK] {:?} — {} person(s), {} face(s)",
                            image_path.file_name().unwrap_or_default(),
                            result.person_count,
                            result.face_count,
                        );
                    }
                    None
                }
                Err(error) => {
                    eprintln!(
                        "  [FAIL] {:?} — {}",
                        image_path.file_name().unwrap_or_default(),
                        error
                    );
                    Some((image_path.clone(), format!("{:#}", error)))
                }
            }
        })
        .collect();

    failed.sort_by(|left, right| left.0.cmp(&right.0));

    BatchSummary {
        total: image_paths.len(),
        succeeded: succeeded.load(Ordering::Relaxed),
        failed,
    }
}
