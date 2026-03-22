/// Icarus-v2 CLI Entry Point
///
/// AI image cropping system. Runs YOLOv10 object detection on an input image via
/// ONNX Runtime, then optionally saves cropped regions, annotated images, or raw
/// detection JSON to disk.
///
/// The YOLOv10n ONNX model is downloaded from HuggingFace Hub on first use and
/// cached locally in `~/.cache/huggingface/hub/`.
///
/// # Example
/// ```
/// icarus-v2 --input photo.jpg --output crop.jpg --model yolov10 --confidence 0.3
/// ```
use anyhow::{bail, Context, Result};
use candle_core::Device;
use clap::Parser;
use icarus_v2::image_utils::{crop_image, crop_to_ultrawide_21_9_centered};
use icarus_v2::models::load_candle_model;
use icarus_v2::multi_format_cropping::{
    BBox, CropRegion, calculate_landscape_21_9_crop, calculate_portrait_9_16_crop,
    calculate_portrait_9_21_crop, detect_suitable_formats,
};
use image::DynamicImage;
use serde::Serialize;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Argument definitions
// ---------------------------------------------------------------------------

/// Known model names supported by the detection backend.
const VALID_MODELS: &[&str] = &[
    "yolov10", // YOLOv10n via ONNX Runtime
];

#[derive(Parser, Debug)]
#[command(
    name = "icarus-v2",
    about = "AI Image Cropping System",
    long_about = "Detect objects in images using YOLOv10 via ONNX Runtime. \
                  Supports saving cropped regions, annotated images, and raw detection JSON. \
                  The YOLOv10n ONNX model (~9 MB) is downloaded from HuggingFace Hub on first use.",
    version
)]
struct Args {
    /// Input image path (JPEG, PNG, BMP, TIFF, GIF supported)
    #[arg(short, long, value_name = "FILE")]
    input: PathBuf,

    /// Output image path for cropped detection (skipped if omitted)
    #[arg(short, long, value_name = "FILE")]
    output: Option<PathBuf>,

    /// Model to use for detection
    #[arg(
        long,
        default_value = "yolov10",
        value_name = "MODEL",
        help = "Currently supported: yolov10 (YOLOv10n via ONNX Runtime)"
    )]
    model: String,

    /// (Ignored — retained for backward compatibility; weights are loaded from HuggingFace Hub)
    #[arg(long, value_name = "FILE")]
    model_path: Option<PathBuf>,

    /// Minimum confidence threshold for detections (0.0–1.0)
    #[arg(long, default_value = "0.5", value_name = "FLOAT")]
    confidence: f32,

    /// Save detection bounding boxes as JSON to this path
    #[arg(long, value_name = "FILE")]
    output_boxes: Option<PathBuf>,

    /// Draw detection boxes on image and save to this path (alias: --annotate)
    #[arg(long, alias = "annotate", value_name = "FILE")]
    visualize: Option<PathBuf>,

    /// Suppress all informational output (errors are still shown)
    #[arg(long)]
    quiet: bool,

    /// Disable ultrawide 21:9 cropping and use the original detected bbox instead.
    ///
    /// By default, Icarus-v2 crops to a 21:9 (2.33:1) ultrawide aspect ratio,
    /// centering the detected person horizontally and positioning them in the
    /// upper-third of the frame — ideal for desktop wallpapers.
    ///
    /// Pass this flag to revert to the raw bounding-box crop with no aspect-ratio
    /// adjustment.
    #[arg(long, default_value_t = false)]
    keep_aspect_ratio: bool,

    /// Add symmetric padding around the detected person bounding box before cropping.
    ///
    /// The value is a percentage of the bbox dimensions:
    /// - Horizontal margin = bbox_width  × (margin / 100)
    /// - Vertical margin   = bbox_height × (margin / 100)
    ///
    /// For example, `--margin 10` adds 10% padding on each side.
    /// Margins are clamped to photo bounds, so they will never go negative or exceed the photo.
    #[arg(long, default_value_t = 0.0, value_name = "PERCENT")]
    margin: f32,
}

// ---------------------------------------------------------------------------
// Unified Detection type for CLI output
// ---------------------------------------------------------------------------

/// A detection result in a form convenient for the CLI output helpers.
///
/// Converted from `candle_backend::Detection` after inference.
#[derive(Debug, Clone)]
struct Detection {
    /// Bounding box in XYXY pixel coordinates: [x1, y1, x2, y2].
    bbox: [f32; 4],
    /// Class label string (e.g. "person").
    label: String,
    /// Zero-based COCO class index.
    class_id: usize,
    /// Detection confidence in (0.0, 1.0].
    confidence: f32,
}

impl From<icarus_v2::models::Detection> for Detection {
    fn from(d: icarus_v2::models::Detection) -> Self {
        Self {
            bbox: [d.bbox.x_min, d.bbox.y_min, d.bbox.x_max, d.bbox.y_max],
            label: d.class_name,
            class_id: d.class_id,
            confidence: d.confidence,
        }
    }
}

// ---------------------------------------------------------------------------
// Serialisable detection output
// ---------------------------------------------------------------------------

/// JSON-serialisable form of a single detection.
#[derive(Serialize)]
struct DetectionRecord {
    label: String,
    class_id: usize,
    confidence: f32,
    /// [x1, y1, x2, y2] in pixel coordinates
    bbox: [f32; 4],
}

impl From<&Detection> for DetectionRecord {
    fn from(d: &Detection) -> Self {
        Self {
            label: d.label.clone(),
            class_id: d.class_id,
            confidence: d.confidence,
            bbox: d.bbox,
        }
    }
}

// ---------------------------------------------------------------------------
// Person detection helpers
// ---------------------------------------------------------------------------

/// COCO class index for the "person" category.
const PERSON_CLASS_ID: usize = 0;

/// Return the highest-confidence person detection from a slice of detections.
///
/// Detections are expected to be sorted by confidence descending (as produced
/// by the ONNX postprocessor). The first match is therefore the best person.
///
/// # Returns
/// `Some(&Detection)` if any detection has `class_id == PERSON_CLASS_ID`,
/// `None` otherwise.
fn find_best_person_detection(detections: &[Detection]) -> Option<&Detection> {
    detections
        .iter()
        .find(|d| d.class_id == PERSON_CLASS_ID)
}

// ---------------------------------------------------------------------------
// Output helpers
// ---------------------------------------------------------------------------

/// Save a raw bounding-box crop (fallback when `--keep-aspect-ratio` is specified or no
/// multi-format output path is configured).
///
/// When `keep_aspect_ratio` is `false` (the default), falls back to a centered 21:9 crop
/// of the full image. Pass `true` to return the full image unchanged (no person found).
fn save_fallback_crop(
    image: &DynamicImage,
    output_path: &Path,
) -> Result<()> {
    let crop = crop_to_ultrawide_21_9_centered(image)
        .with_context(|| "Failed to crop image to centered 21:9")?;
    crop.save(output_path)
        .with_context(|| format!("Failed to save cropped image to {:?}", output_path))?;
    Ok(())
}

/// Crop the image to the given `CropRegion` and save to `output_path`.
///
/// The crop coordinates are clamped to photo bounds before saving.
fn save_crop_region(
    image: &DynamicImage,
    crop: &CropRegion,
    output_path: &Path,
) -> Result<()> {
    let xyxy = crop.to_xyxy_clamped(image.width(), image.height());
    let cropped = crop_image(image, xyxy)
        .with_context(|| format!("Failed to crop image to region {:?}", xyxy))?;
    cropped
        .save(output_path)
        .with_context(|| format!("Failed to save cropped image to {:?}", output_path))?;
    Ok(())
}

/// Save detection bounding boxes as a JSON array to `output_path`.
fn save_detections_json(detections: &[Detection], output_path: &Path) -> Result<()> {
    let records: Vec<DetectionRecord> = detections.iter().map(DetectionRecord::from).collect();
    let json = serde_json::to_string_pretty(&records)
        .context("Failed to serialise detections to JSON")?;

    std::fs::write(output_path, json)
        .with_context(|| format!("Failed to write detections JSON to {:?}", output_path))?;

    Ok(())
}

/// Draw bounding boxes on a copy of `image` and save to `output_path`.
///
/// Each box is drawn as a 2-pixel-wide coloured rectangle with the label printed
/// above the top-left corner using a simple ASCII rendering approach (no font
/// dependency). The colour cycles through a small palette to distinguish classes.
///
/// # Limitations
/// Text rendering is rasterised as white pixels only (no font support without
/// external dependencies). Labels are best inspected in the JSON output.
// TODO: Replace with a proper font-rendering solution (e.g., `imageproc` + `rusttype`)
//       once the dependency budget allows it.
fn save_visualized(
    image: &DynamicImage,
    detections: &[Detection],
    output_path: &Path,
) -> Result<()> {
    use image::{Rgba, RgbaImage};

    let mut canvas: RgbaImage = image.to_rgba8();
    let w = canvas.width();
    let h = canvas.height();

    // Class-aware colours: persons (class 0) in bright green, everything else in dark red.
    // This makes it immediately obvious which detections drove cropping decisions.
    const VIZ_PERSON_COLOR: [u8; 4] = [0, 255, 0, 220];     // bright green (#00FF00)
    const VIZ_NON_PERSON_COLOR: [u8; 4] = [128, 0, 0, 220]; // dark red (#800000)

    for det in detections {
        let colour = if det.class_id == PERSON_CLASS_ID {
            VIZ_PERSON_COLOR
        } else {
            VIZ_NON_PERSON_COLOR
        };
        let colour_px = Rgba(colour);

        let [x1, y1, x2, y2] = det.bbox;
        let x1u = (x1 as u32).min(w.saturating_sub(1));
        let y1u = (y1 as u32).min(h.saturating_sub(1));
        let x2u = (x2 as u32).min(w.saturating_sub(1));
        let y2u = (y2 as u32).min(h.saturating_sub(1));

        // Draw 2-pixel-wide rectangle.
        for thickness in 0..2u32 {
            // Top & bottom edges
            for x in x1u..=x2u {
                for dy in 0..=thickness {
                    if y1u + dy < h {
                        canvas.put_pixel(x, y1u + dy, colour_px);
                    }
                    if y2u >= dy && y2u - dy < h {
                        canvas.put_pixel(x, y2u - dy, colour_px);
                    }
                }
            }
            // Left & right edges
            for y in y1u..=y2u {
                for dx in 0..=thickness {
                    if x1u + dx < w {
                        canvas.put_pixel(x1u + dx, y, colour_px);
                    }
                    if x2u >= dx && x2u - dx < w {
                        canvas.put_pixel(x2u - dx, y, colour_px);
                    }
                }
            }
        }
    }

    DynamicImage::ImageRgba8(canvas)
        .save(output_path)
        .with_context(|| format!("Failed to save visualized image to {:?}", output_path))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();

    // ── Validate confidence range ──────────────────────────────────────────
    if args.confidence < 0.0 || args.confidence > 1.0 {
        bail!(
            "--confidence must be between 0.0 and 1.0, got {}",
            args.confidence
        );
    }

    // ── Validate model name early ──────────────────────────────────────────
    if !VALID_MODELS.contains(&args.model.as_str()) {
        bail!(
            "Unknown model '{}'. Valid options are:\n  {}",
            args.model,
            VALID_MODELS.join("\n  ")
        );
    }

    // ── Warn if --model-path was supplied (no longer used) ─────────────────
    if let Some(ref p) = args.model_path {
        if !args.quiet {
            eprintln!(
                "Warning: --model-path {:?} is ignored; weights are loaded from \
                 HuggingFace Hub automatically.",
                p
            );
        }
    }

    // ── Load input image ───────────────────────────────────────────────────
    if !args.input.exists() {
        bail!(
            "Input file not found: {:?}\nPlease check the path and try again.",
            args.input
        );
    }

    if !args.quiet {
        println!("Icarus-v2: loading image from {:?}", args.input);
    }

    let image = image::open(&args.input)
        .with_context(|| format!("Failed to open input image: {:?}", args.input))?;

    let image_width = image.width();
    let image_height = image.height();

    if !args.quiet {
        println!(
            "  Image dimensions: {}×{} pixels",
            image_width, image_height
        );
    }

    // ── Select compute device ──────────────────────────────────────────────
    // Always CPU for now; GPU support (Device::Cuda(0)) can be added later.
    // TODO: Add --device flag to expose GPU selection once GPU testing is available.
    let device = Device::Cpu;

    if !args.quiet {
        println!(
            "  Model: {} (loading weights from HuggingFace Hub…)",
            args.model
        );
        println!("  Confidence threshold: {}", args.confidence);
    }

    // ── Load model ─────────────────────────────────────────────────────────
    let model = load_candle_model(&args.model, &device)
        .await
        .with_context(|| format!("Failed to load model '{}'", args.model))?;

    if !args.quiet {
        println!("  Running inference…");
    }

    // ── Run inference ──────────────────────────────────────────────────────
    let input_tensor = model
        .preprocess(&[image.clone()])
        .map_err(|e| anyhow::anyhow!("Preprocessing failed for '{}': {e}", args.model))?;

    let (logits, boxes) = model
        .forward(&input_tensor)
        .map_err(|e| anyhow::anyhow!("Inference failed for '{}': {e}", args.model))?;

    let raw_detections = model
        .postprocess(logits, boxes)
        .map_err(|e| anyhow::anyhow!("Postprocessing failed for '{}': {e}", args.model))?;

    // Convert from candle_backend::Detection to the CLI's Detection type, then
    // apply the user-supplied confidence threshold (the model already applies its
    // own internal threshold at inference time; this allows users to tighten it
    // further via --confidence without recompiling).
    let detections: Vec<Detection> = raw_detections
        .into_iter()
        .map(Detection::from)
        .filter(|d| d.confidence >= args.confidence)
        .collect();

    // Select the best person detection for cropping (may be None if no person found).
    let person_for_crop = find_best_person_detection(&detections);

    // ── Report results ─────────────────────────────────────────────────────
    if detections.is_empty() {
        if !args.quiet {
            println!(
                "No objects detected (confidence threshold: {}).",
                args.confidence
            );
            println!(
                "Tip: try lowering --confidence below {} to see more results.",
                args.confidence
            );
        }
    } else if !args.quiet {
        println!("Found {} object(s):", detections.len());
        for (i, det) in detections.iter().enumerate() {
            println!(
                "  [{:2}] {:<20} conf={:.3}  bbox=[{:.0},{:.0},{:.0},{:.0}]",
                i + 1,
                det.label,
                det.confidence,
                det.bbox[0],
                det.bbox[1],
                det.bbox[2],
                det.bbox[3]
            );
        }
    }

    // ── Multi-format cropping ──────────────────────────────────────────────
    //
    // When --output is supplied and a person was detected, we run the multi-format
    // intelligent cropping pipeline:
    //
    //   1. Detect suitable formats (21:9, 9:21, 9:16) based on bbox orientation
    //      and visibility checks.
    //   2. For each suitable format, calculate the crop region and save a
    //      format-suffixed file (e.g. photo_21_9.jpg).
    //   3. If --visualize is also supplied, save a per-format annotated image
    //      (e.g. photo_21_9_annotated.jpg).
    //
    // Fallback: when no person is detected OR --keep-aspect-ratio is set, we
    // fall back to the old behaviour (centered 21:9 of the whole image).

    if let Some(ref output_path) = args.output {
        if args.keep_aspect_ratio || person_for_crop.is_none() {
            // ── Legacy / keep-aspect-ratio path ───────────────────────────
            if person_for_crop.is_none() && !args.quiet {
                println!("  No person detected — falling back to centered 21:9 crop.");
            }
            if args.keep_aspect_ratio {
                // Raw bbox crop (no aspect-ratio expansion)
                if let Some(person) = person_for_crop {
                    let cropped = crop_image(&image, person.bbox)
                        .with_context(|| format!("Failed to crop bbox {:?}", person.bbox))?;
                    cropped.save(output_path)
                        .with_context(|| format!("Failed to save to {:?}", output_path))?;
                } else {
                    image.save(output_path)
                        .with_context(|| format!("Failed to save to {:?}", output_path))?;
                }
            } else {
                save_fallback_crop(&image, output_path)
                    .with_context(|| format!("Failed to save fallback crop to {:?}", output_path))?;
            }
            if !args.quiet {
                println!("Saved cropped image to {:?}", output_path);
            }
            // Single-format visualisation
            if let Some(ref viz_path) = args.visualize {
                save_visualized(&image, &detections, viz_path)
                    .with_context(|| format!("Failed to save visualization to {:?}", viz_path))?;
                if !args.quiet {
                    println!("Saved visualized image to {:?}", viz_path);
                }
            }
        } else {
            // ── Multi-format intelligent cropping path ─────────────────────
            let person = person_for_crop.unwrap(); // safe: checked above
            let raw_bbox: BBox = person.bbox.into();

            if args.margin < 0.0 {
                bail!("--margin must be ≥ 0, got {}", args.margin);
            }

            let suitable_formats =
                detect_suitable_formats(image.width(), image.height(), &raw_bbox, args.margin);

            if suitable_formats.is_empty() {
                if !args.quiet {
                    println!(
                        "  No suitable crop formats found for this photo \
                         (person visibility < 50% in all formats). \
                         Try lowering --margin or adjusting --confidence."
                    );
                }
                // Still write the fallback so callers get *some* output
                save_fallback_crop(&image, output_path)
                    .with_context(|| "Failed to save fallback crop")?;
                if !args.quiet {
                    println!("Saved fallback centered 21:9 crop to {:?}", output_path);
                }
            } else {
                if !args.quiet {
                    println!("  Suitable formats: {}", suitable_formats.join(", "));
                }

                // Build the output stem + extension so we can construct per-format paths.
                // e.g. "output/photo.jpg" → stem="photo", ext="jpg"
                let stem = output_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("crop");
                let ext = output_path
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("jpg");
                let dir = output_path.parent().unwrap_or(Path::new("."));

                // Visualise path stem for annotated outputs
                let viz_stem = args.visualize.as_ref().and_then(|p| {
                    p.file_stem().and_then(|s| s.to_str()).map(|s| s.to_string())
                });
                let viz_ext = args.visualize.as_ref().and_then(|p| {
                    p.extension().and_then(|e| e.to_str()).map(|s| s.to_string())
                });
                let viz_dir = args
                    .visualize
                    .as_ref()
                    .and_then(|p| p.parent())
                    .unwrap_or(Path::new("."))
                    .to_path_buf();

                for format in &suitable_formats {
                    // Derive format-safe filename suffix: "21:9" → "21_9"
                    let suffix = format.replace(':', "_");

                    // Calculate crop region for this format
                    let maybe_crop: Option<CropRegion> = {
                        // Apply margin to get the working bbox (same as detect_suitable_formats)
                        use icarus_v2::multi_format_cropping::apply_margin_to_bbox;
                        let working_bbox =
                            apply_margin_to_bbox(&raw_bbox, args.margin, image.width(), image.height());
                        match format.as_str() {
                            "21:9" => calculate_landscape_21_9_crop(
                                image.width(),
                                image.height(),
                                &working_bbox,
                            ),
                            "9:21" => calculate_portrait_9_21_crop(
                                image.width(),
                                image.height(),
                                &working_bbox,
                            ),
                            "9:16" => calculate_portrait_9_16_crop(
                                image.width(),
                                image.height(),
                                &working_bbox,
                            ),
                            other => {
                                eprintln!("Warning: unknown format '{}' — skipping.", other);
                                None
                            }
                        }
                    };

                    let crop = match maybe_crop {
                        Some(c) => c,
                        None => {
                            if !args.quiet {
                                eprintln!(
                                    "Warning: format {} became unavailable during crop calculation — skipping.",
                                    format
                                );
                            }
                            continue;
                        }
                    };

                    // e.g. output/photo_21_9.jpg
                    let crop_path = dir.join(format!("{}_{}.{}", stem, suffix, ext));
                    save_crop_region(&image, &crop, &crop_path)
                        .with_context(|| format!("Failed to save {} crop to {:?}", format, crop_path))?;
                    if !args.quiet {
                        println!("  Saved {} crop → {:?}", format, crop_path);
                    }

                    // Annotated visualisation for this format
                    if let (Some(ref vstem), Some(ref vext)) = (&viz_stem, &viz_ext) {
                        // e.g. output/photo_annotated_21_9.jpg
                        let viz_path = viz_dir.join(format!("{}_{}.{}", vstem, suffix, vext));
                        save_visualized(&image, &detections, &viz_path)
                            .with_context(|| {
                                format!("Failed to save {} visualization to {:?}", format, viz_path)
                            })?;
                        if !args.quiet {
                            println!("  Saved {} annotated → {:?}", format, viz_path);
                        }
                    }
                }
            }
        }
    } else {
        // No --output supplied; still emit the visualisation if requested.
        if let Some(ref viz_path) = args.visualize {
            save_visualized(&image, &detections, viz_path)
                .with_context(|| format!("Failed to save visualization to {:?}", viz_path))?;
            if !args.quiet {
                println!("Saved visualized image to {:?}", viz_path);
            }
        }
    }

    // ── Save detection JSON ────────────────────────────────────────────────
    if let Some(ref boxes_path) = args.output_boxes {
        save_detections_json(&detections, boxes_path)
            .with_context(|| format!("Failed to save detections JSON to {:?}", boxes_path))?;
        if !args.quiet {
            println!(
                "Saved {} detection(s) to {:?}",
                detections.len(),
                boxes_path
            );
        }
    }

    // ── Final summary line ─────────────────────────────────────────────────
    if !args.quiet {
        println!(
            "Done. Found {} object(s).",
            detections.len(),
        );
    }

    Ok(())
}

#[cfg(feature = "debug_tensors")]
fn debug_tensor_inspection() -> anyhow::Result<()> {
    use std::path::Path;
    let weights_path = Path::new("/home/developer/.cache/huggingface/hub/models--facebook--detr-resnet-50/snapshots/1d5f47bd3bdd2c4bbfa585418ffe6da5028b4c0b/model.safetensors");
    
    let data = std::fs::read(weights_path)?;
    let tensors = safetensors::SafeTensors::deserialize(&data)?;
    
    let mut keys: Vec<_> = tensors.names().collect();
    keys.sort();
    
    println!("Total tensors: {}\n", keys.len());
    
    println!("=== ALL KEYS (first 100) ===");
    for (i, key) in keys.iter().take(100).enumerate() {
        println!("{:3}: {}", i, key);
    }
    
    println!("\n\n=== BACKBONE KEYS ===");
    let backbone_keys: Vec<_> = keys.iter()
        .filter(|k| k.contains("backbone"))
        .collect();
    
    println!("Total backbone keys: {}", backbone_keys.len());
    for key in backbone_keys.iter().take(60) {
        println!("  {}", key);
    }
    
    Ok(())
}
