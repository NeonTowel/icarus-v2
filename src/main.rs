/// Icarus-v2 CLI Entry Point
///
/// AI image cropping system. Runs YOLOv10 object detection on an input image via
/// ONNX Runtime for person detection, then runs YOLOv11x-Face via ONNX Runtime for
/// face detection, and applies face-aware crop adjustment to all output formats.
///
/// Both models are downloaded from HuggingFace Hub on first use and cached locally
/// in `~/.cache/huggingface/hub/`.
///
/// # Example
/// ```
/// icarus-v2 --input photo.jpg --output crop.jpg --model yolov10 --confidence 0.3
/// icarus-v2 --input photo.jpg --output crop.jpg --artistic-mode balanced --visualize viz.jpg
/// ```
use anyhow::{bail, Context, Result};
use candle_core::Device;
use clap::Parser;
use icarus_v2::config::{load_crop_config, ArtisticCropConfig, ArtisticMode, CropConfig};
use icarus_v2::face_aware_cropping::apply_face_aware_adjustment;
use icarus_v2::face_detection::{detect_faces, load_face_detector};
use icarus_v2::image_utils::{crop_image, crop_to_ultrawide_21_9_centered};
use icarus_v2::models::load_candle_model;
use icarus_v2::multi_format_cropping::{
    apply_margin_to_bbox, calculate_compound_bbox, calculate_landscape_21_9_crop,
    calculate_portrait_9_16_crop, calculate_portrait_9_21_crop, detect_suitable_formats, BBox,
    CropRegion,
};
use icarus_v2::output_sorting;
use image::DynamicImage;
use serde::Serialize;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Argument definitions
// ---------------------------------------------------------------------------

/// Known person-detection model names supported by the detection backend.
const VALID_MODELS: &[&str] = &[
    "yolov10", // YOLOv10n via ONNX Runtime (stage-1 person detection)
];

#[derive(Parser, Debug)]
#[command(
    name = "icarus-v2",
    about = "AI Image Cropping System",
    long_about = "Detect objects in images using YOLOv10 via ONNX Runtime. \
                  Supports saving cropped regions, annotated images, and raw detection JSON. \
                  The YOLOv10n ONNX model (~9 MB) and YOLOv11x-Face model (~60 MB) are \
                  downloaded from HuggingFace Hub on first use.",
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

    /// Draw detection boxes on image and save to this path (alias: --annotate).
    ///
    /// When set, draws green person boxes, cyan face boxes, and red crop regions.
    /// Face detection is always shown when this flag is set.
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

    /// Organize output images into aspect-ratio subfolders.
    ///
    /// When enabled, output images are sorted into:
    /// - `landscape/`  (21:9 format — ultrawide)
    /// - `portrait/`   (9:16 format — vertical)
    /// - `mobile/`     (9:21 format — tall mobile)
    ///
    /// Annotated images (from `--visualize`) automatically receive an `_annotated`
    /// suffix before the format suffix (e.g. `viz_annotated_21_9.jpg`).
    ///
    /// Without this flag, output is placed flat in the specified directory, but
    /// `_annotated` suffix is still automatically applied to visualized images.
    #[arg(long, default_value_t = false)]
    sort_output: bool,

    /// Vertical headroom: fraction of crop height placed above the bbox center (0.0–1.0).
    ///
    /// `0.40` means the person's bbox center sits 40% from the top of the crop, with
    /// 60% footroom below — ideal for fashion/entertainment on ultrawide displays.
    /// Overrides the value from `--crop-config` if both are provided.
    #[arg(long, value_name = "FLOAT")]
    headroom_ratio: Option<f32>,

    /// Minimum person visibility for a crop format to be accepted (0.0–100.0, as percent).
    ///
    /// If the person would be less than this percentage visible in the proposed crop,
    /// that format is skipped. Overrides the value from `--crop-config` if both are provided.
    /// Stored internally as a ratio; pass `50.0` for the default 50% threshold.
    #[arg(long, value_name = "FLOAT")]
    visibility_threshold: Option<f32>,

    /// Path to a YAML file containing crop configuration.
    ///
    /// Supported fields:
    ///   headroom_ratio: 0.40        # fraction in [0.0, 1.0]
    ///   visibility_threshold: 50.0  # percentage in [0.0, 100.0]
    ///
    /// CLI flags (`--headroom-ratio`, `--visibility-threshold`) take precedence over
    /// any values loaded from this file.
    #[arg(long, value_name = "FILE")]
    crop_config: Option<PathBuf>,

    // ── Face detection and artistic options ────────────────────────────────
    /// Artistic composition mode controlling face margin and shift budget.
    ///
    /// Face detection is always on. This mode controls how aggressively face
    /// visibility is prioritised when adjusting crop placement:
    ///   conservative — Larger safety margin (20px); minimal crop adjustment.
    ///   balanced     — Medium safety margin (15px); default for most photos.
    ///   aggressive   — Smaller margin (10px); tighter framing around the face.
    #[arg(long, default_value = "balanced", value_name = "MODE")]
    artistic_mode: String,
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
#[allow(dead_code)]
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
    detections.iter().find(|d| d.class_id == PERSON_CLASS_ID)
}

// ---------------------------------------------------------------------------
// Output helpers
// ---------------------------------------------------------------------------

/// Save a centered 21:9 fallback crop when no person was detected.
fn save_fallback_crop(image: &DynamicImage, output_path: &Path) -> Result<()> {
    let crop = crop_to_ultrawide_21_9_centered(image)
        .with_context(|| "Failed to crop image to centered 21:9")?;
    crop.save(output_path)
        .with_context(|| format!("Failed to save cropped image to {:?}", output_path))?;
    Ok(())
}

/// Crop the image to the given `CropRegion` and save to `output_path`.
///
/// The crop coordinates are clamped to photo bounds before saving.
fn save_crop_region(image: &DynamicImage, crop: &CropRegion, output_path: &Path) -> Result<()> {
    let xyxy = crop.to_xyxy_clamped(image.width(), image.height());
    let cropped = crop_image(image, xyxy)
        .with_context(|| format!("Failed to crop image to region {:?}", xyxy))?;
    cropped
        .save(output_path)
        .with_context(|| format!("Failed to save cropped image to {:?}", output_path))?;
    Ok(())
}

/// Save detection bounding boxes as a JSON array to `output_path`.
fn save_detections_json(
    detections: &[Detection],
    face_bboxes: &[BBox],
    output_path: &Path,
) -> Result<()> {
    use serde_json::json;

    let person_records: Vec<serde_json::Value> = detections
        .iter()
        .enumerate()
        .map(|(i, d)| {
            json!({
                "person_id": i,
                "label": d.label,
                "class_id": d.class_id,
                "confidence": d.confidence,
                "bbox": d.bbox,
            })
        })
        .collect();

    let face_records: Vec<serde_json::Value> = face_bboxes
        .iter()
        .enumerate()
        .map(|(i, f)| {
            json!({
                "face_id": i,
                "bbox": [f.x1, f.y1, f.x2, f.y2],
            })
        })
        .collect();

    let output = json!({
        "persons": person_records,
        "faces": face_records,
    });

    let json = serde_json::to_string_pretty(&output)
        .context("Failed to serialise detections+faces to JSON")?;

    std::fs::write(output_path, json)
        .with_context(|| format!("Failed to write JSON to {:?}", output_path))?;

    Ok(())
}

/// Draw bounding boxes on a copy of `image` and save to `output_path`.
///
/// Colour coding:
/// - Persons (class 0): bright green (#00FF00)
/// - Other classes: dark red (#800000)
/// - Faces: cyan (#00FFFF)
///
/// # Limitations
/// Text rendering is rasterised as white pixels only (no font support without
/// external dependencies). Labels are best inspected in the JSON output.
// TODO: Replace with a proper font-rendering solution (e.g., `imageproc` + `rusttype`)
//       once the dependency budget allows it.
fn save_visualized_with_faces(
    image: &DynamicImage,
    detections: &[Detection],
    face_bboxes: &[BBox],
    crop_regions: &[CropRegion],
    output_path: &Path,
) -> Result<()> {
    use image::{Rgba, RgbaImage};

    let mut canvas: RgbaImage = image.to_rgba8();
    let w = canvas.width();
    let h = canvas.height();

    // Colour palette.
    const VIZ_PERSON_COLOR: [u8; 4] = [0, 255, 0, 220]; // bright green (#00FF00)
    const VIZ_NON_PERSON_COLOR: [u8; 4] = [128, 0, 0, 220]; // dark red (#800000)
    const VIZ_FACE_COLOR: [u8; 4] = [0, 255, 255, 220]; // cyan (#00FFFF)
    const VIZ_CROP_COLOR: [u8; 4] = [255, 0, 0, 220]; // red (#FF0000)

    /// Draw a 2-pixel-wide axis-aligned rectangle.
    fn draw_rect(canvas: &mut RgbaImage, x1u: u32, y1u: u32, x2u: u32, y2u: u32, colour: Rgba<u8>) {
        let w = canvas.width();
        let h = canvas.height();
        for thickness in 0..2u32 {
            for x in x1u..=x2u {
                for dy in 0..=thickness {
                    if y1u + dy < h {
                        canvas.put_pixel(x, y1u + dy, colour);
                    }
                    if y2u >= dy && y2u - dy < h {
                        canvas.put_pixel(x, y2u - dy, colour);
                    }
                }
            }
            for y in y1u..=y2u {
                for dx in 0..=thickness {
                    if x1u + dx < w {
                        canvas.put_pixel(x1u + dx, y, colour);
                    }
                    if x2u >= dx && x2u - dx < w {
                        canvas.put_pixel(x2u - dx, y, colour);
                    }
                }
            }
        }
    }

    // Draw person / object detections.
    for det in detections {
        let colour = if det.class_id == PERSON_CLASS_ID {
            Rgba(VIZ_PERSON_COLOR)
        } else {
            Rgba(VIZ_NON_PERSON_COLOR)
        };
        let [x1, y1, x2, y2] = det.bbox;
        draw_rect(
            &mut canvas,
            (x1 as u32).min(w.saturating_sub(1)),
            (y1 as u32).min(h.saturating_sub(1)),
            (x2 as u32).min(w.saturating_sub(1)),
            (y2 as u32).min(h.saturating_sub(1)),
            colour,
        );
    }

    // Draw face detections on top (cyan).
    for face in face_bboxes {
        draw_rect(
            &mut canvas,
            (face.x1 as u32).min(w.saturating_sub(1)),
            (face.y1 as u32).min(h.saturating_sub(1)),
            (face.x2 as u32).min(w.saturating_sub(1)),
            (face.y2 as u32).min(h.saturating_sub(1)),
            Rgba(VIZ_FACE_COLOR),
        );
    }

    // Draw crop regions (red).
    for crop in crop_regions {
        let x2 = crop.x + crop.width;
        let y2 = crop.y + crop.height;
        draw_rect(
            &mut canvas,
            (crop.x as u32).min(w.saturating_sub(1)),
            (crop.y as u32).min(h.saturating_sub(1)),
            (x2 as u32).min(w.saturating_sub(1)),
            (y2 as u32).min(h.saturating_sub(1)),
            Rgba(VIZ_CROP_COLOR),
        );
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

    // ── Build CropConfig (YAML file → CLI overrides) ───────────────────────
    //
    // Priority order (highest → lowest):
    //   1. CLI flags: --headroom-ratio, --visibility-threshold
    //   2. YAML file: --crop-config <path>
    //   3. Built-in defaults: headroom_ratio=0.40, visibility_threshold=0.50
    let mut crop_config: CropConfig = if let Some(ref config_path) = args.crop_config {
        if !config_path.exists() {
            bail!(
                "Crop config file not found: {:?}\nPlease check the path and try again.",
                config_path
            );
        }
        load_crop_config(config_path)
            .with_context(|| format!("Failed to load crop config from {:?}", config_path))?
    } else {
        CropConfig::default()
    };

    // CLI flags override YAML values (or defaults).
    if let Some(ratio) = args.headroom_ratio {
        if !(0.0..=1.0).contains(&ratio) {
            bail!(
                "--headroom-ratio must be between 0.0 and 1.0, got {}",
                ratio
            );
        }
        crop_config.headroom_ratio = ratio;
    }
    if let Some(pct) = args.visibility_threshold {
        if !(0.0..=100.0).contains(&pct) {
            bail!(
                "--visibility-threshold must be between 0.0 and 100.0, got {}",
                pct
            );
        }
        // Convert from percentage (CLI) to ratio (internal representation).
        crop_config.visibility_threshold = pct / 100.0;
    }

    if !args.quiet {
        println!(
            "  Crop config: headroom_ratio={:.2}, visibility_threshold={:.0}%",
            crop_config.headroom_ratio,
            crop_config.visibility_threshold * 100.0,
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

    // ── Build ArtisticCropConfig from CLI flags ────────────────────────────
    let artistic_mode = args
        .artistic_mode
        .parse::<ArtisticMode>()
        .with_context(|| {
            format!(
                "Invalid --artistic-mode '{}'. Options: conservative, balanced, aggressive",
                args.artistic_mode
            )
        })?;

    let artistic_config = ArtisticCropConfig::from_mode(artistic_mode);

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

    // ── Load ALL models at startup ─────────────────────────────────────────
    // Both models stay in memory for the entire run. Fatal if either fails.
    let model = load_candle_model(&args.model, &device)
        .await
        .with_context(|| format!("Failed to load model '{}'", args.model))?;

    if !args.quiet {
        println!("  Loading face detection model (YOLOv11x-Face)…");
    }

    // Load face detector in a blocking task (HF Hub download is synchronous).
    let face_detector = tokio::task::spawn_blocking(load_face_detector)
        .await
        .map_err(|e| anyhow::anyhow!("face model load task panicked: {e}"))??;

    if !args.quiet {
        println!("  Running stage-1 inference (person detection)…");
    }

    // ── Stage 1: Person detection via YOLOv10 ─────────────────────────────
    let input_tensor = model
        .preprocess(std::slice::from_ref(&image))
        .map_err(|e| anyhow::anyhow!("Preprocessing failed for '{}': {e}", args.model))?;

    let (logits, boxes) = model
        .forward(&input_tensor)
        .map_err(|e| anyhow::anyhow!("Inference failed for '{}': {e}", args.model))?;

    let raw_detections = model
        .postprocess(logits, boxes)
        .map_err(|e| anyhow::anyhow!("Postprocessing failed for '{}': {e}", args.model))?;

    // Convert from candle_backend::Detection to the CLI's Detection type, then
    // apply the user-supplied confidence threshold.
    let detections: Vec<Detection> = raw_detections
        .into_iter()
        .map(Detection::from)
        .filter(|d| d.confidence >= args.confidence)
        .collect();

    // ── Resolve bbox for cropping (single or compound) ────────────────────
    //
    // Collect all person detections that passed the confidence filter, then:
    //   - 0 persons  → no crop bbox (fallback path below)
    //   - 1 person   → use that detection's bbox directly (backward compatible)
    //   - 2+ persons → compound bbox encompassing all detected persons
    let person_detections: Vec<&Detection> = detections
        .iter()
        .filter(|d| d.class_id == PERSON_CLASS_ID)
        .collect();

    let crop_bbox: Option<BBox> = match person_detections.len() {
        0 => None,
        1 => {
            // Single person: use their bbox directly (no compound calculation needed).
            Some(person_detections[0].bbox.into())
        }
        n => {
            // Multiple persons: build a compound bbox via image_utils::Detection.
            if !args.quiet {
                println!(
                    "  Detected {} persons — using compound bbox for cropping.",
                    n
                );
            }
            let img_utils_detections: Vec<icarus_v2::image_utils::Detection> = person_detections
                .iter()
                .map(|d| icarus_v2::image_utils::Detection {
                    bbox: d.bbox,
                    confidence: d.confidence,
                    label: d.label.clone(),
                    class_id: d.class_id,
                })
                .collect();
            calculate_compound_bbox(&img_utils_detections)
        }
    };

    // Keep the legacy helper available for the keep-aspect-ratio / no-person path.
    let person_for_crop = find_best_person_detection(&detections);

    // ── Stage 2: Face detection via YOLOv11x-Face (always runs) ──────────
    //
    // Face detection is default behavior. If inference fails, we log a warning
    // and continue with an empty face list (graceful degradation).
    if !args.quiet {
        println!("  Running stage-2 inference (face detection via YOLOv11x-Face)…");
    }

    let face_bboxes: Vec<BBox> = match detect_faces(&image, &face_detector) {
        Ok(faces) => {
            if !args.quiet && !faces.is_empty() {
                println!("  Face detections: {}", faces.len());
                for (i, f) in faces.iter().enumerate() {
                    println!(
                        "    [{:2}] face  bbox=[{:.0},{:.0},{:.0},{:.0}]",
                        i + 1,
                        f.x1,
                        f.y1,
                        f.x2,
                        f.y2
                    );
                }
            }
            faces
        }
        Err(e) => {
            eprintln!(
                "Warning: face detection failed ({}). Continuing without faces.",
                e
            );
            vec![]
        }
    };

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
    //   2. For each suitable format:
    //      A. Original algorithm calculates initial crop (person-bbox driven).
    //      B. Face-aware adjustment nudges the crop (within shift budget) so the
    //         dominant face is not clipped.
    //   3. Save each format-suffixed output file.
    //   4. If --visualize is also supplied, save an annotated image per format.
    //
    // Fallback: when no person is detected OR --keep-aspect-ratio is set, we
    // fall back to the old behaviour (centered 21:9 of the whole image).

    // Track all crop regions produced for visualization.
    let mut all_crop_regions: Vec<CropRegion> = Vec::new();

    if let Some(ref output_path) = args.output {
        // Ensure output subdirectories exist when --sort-output is enabled.
        // This is idempotent: running multiple times is safe.
        let output_base_dir = output_path.parent().unwrap_or(Path::new("."));
        output_sorting::ensure_output_dirs(output_base_dir, args.sort_output)
            .context("Failed to create output subdirectories")?;

        if args.keep_aspect_ratio || crop_bbox.is_none() {
            // ── Legacy / keep-aspect-ratio path ───────────────────────────
            //
            // When --keep-aspect-ratio is set or no person is detected, we produce
            // a single output image. With --sort-output, we calculate the actual
            // aspect ratio of the cropped image and place it in the correct subfolder.
            if crop_bbox.is_none() && !args.quiet {
                println!("  No person detected — falling back to centered 21:9 crop.");
            }

            let cropped_image = if args.keep_aspect_ratio {
                // Raw bbox crop (no aspect-ratio expansion)
                if let Some(person) = person_for_crop {
                    let cropped = crop_image(&image, person.bbox)
                        .with_context(|| format!("Failed to crop bbox {:?}", person.bbox))?;
                    Some(cropped)
                } else {
                    // No person and --keep-aspect-ratio: save the original image
                    None
                }
            } else {
                // Centered 21:9 fallback crop
                let cropped = crop_to_ultrawide_21_9_centered(&image)
                    .with_context(|| "Failed to crop image to centered 21:9")?;
                Some(cropped)
            };

            // Determine actual output path (with optional subfolder based on aspect ratio)
            let actual_output_path = if args.sort_output {
                let (crop_w, crop_h) = match &cropped_image {
                    Some(c) => (c.width(), c.height()),
                    None => (image.width(), image.height()),
                };
                let aspect_ratio = crop_w as f32 / crop_h as f32;
                let format = output_sorting::determine_best_format_for_aspect_ratio(aspect_ratio);
                let stem = output_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("crop");
                let ext = output_path
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("jpg");
                output_sorting::get_sorted_output_path(output_path, format, stem, ext, true)
                    .with_context(|| "Failed to build sorted output path")?
            } else {
                output_path.clone()
            };

            // Save the cropped (or original) image
            match &cropped_image {
                Some(c) => c
                    .save(&actual_output_path)
                    .with_context(|| format!("Failed to save to {:?}", actual_output_path))?,
                None => image
                    .save(&actual_output_path)
                    .with_context(|| format!("Failed to save to {:?}", actual_output_path))?,
            }

            if !args.quiet {
                println!("Saved cropped image to {:?}", actual_output_path);
            }

            // Single-format visualisation (always inject _annotated suffix)
            if let Some(ref viz_path) = args.visualize {
                let actual_viz_path = if args.sort_output {
                    // Reuse the format determined above for the sorted crop
                    let (crop_w, crop_h) = match &cropped_image {
                        Some(c) => (c.width(), c.height()),
                        None => (image.width(), image.height()),
                    };
                    let aspect_ratio = crop_w as f32 / crop_h as f32;
                    let format =
                        output_sorting::determine_best_format_for_aspect_ratio(aspect_ratio);
                    let vstem = viz_path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("viz");
                    let vext = viz_path
                        .extension()
                        .and_then(|e| e.to_str())
                        .unwrap_or("jpg");
                    output_sorting::get_annotated_output_path(viz_path, format, vstem, vext, true)
                        .with_context(|| "Failed to build sorted annotated path")?
                } else {
                    // Flat output: inject _annotated suffix into filename only
                    let vstem = viz_path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("viz");
                    let vext = viz_path
                        .extension()
                        .and_then(|e| e.to_str())
                        .unwrap_or("jpg");
                    let viz_dir = viz_path.parent().unwrap_or(Path::new("."));
                    let annotated_stem = if vstem.ends_with("_annotated") {
                        vstem.to_string()
                    } else {
                        format!("{}_annotated", vstem)
                    };
                    viz_dir.join(format!("{}.{}", annotated_stem, vext))
                };

                save_visualized_with_faces(
                    &image,
                    &detections,
                    &face_bboxes,
                    &all_crop_regions,
                    &actual_viz_path,
                )
                .with_context(|| {
                    format!("Failed to save visualization to {:?}", actual_viz_path)
                })?;
                if !args.quiet {
                    println!("Saved visualized image to {:?}", actual_viz_path);
                }
            }
        } else {
            // ── Multi-format intelligent cropping path ─────────────────────
            // crop_bbox is Some(...) here — guaranteed by the branch condition above.
            let raw_bbox: BBox = crop_bbox.clone().unwrap(); // safe: checked by branch condition

            if args.margin < 0.0 {
                bail!("--margin must be ≥ 0, got {}", args.margin);
            }

            // Get the person bbox (for passing to face-aware adjustment).
            let person_bbox_for_adjustment: Option<BBox> = crop_bbox;

            // Detect suitable formats using the standard person-bbox algorithm.
            let suitable_formats = detect_suitable_formats(
                image.width(),
                image.height(),
                &raw_bbox,
                args.margin,
                &crop_config,
            );

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
                    if !face_bboxes.is_empty() {
                        println!(
                            "  Applying face-aware adjustment ({} face(s), mode: {:?}).",
                            face_bboxes.len(),
                            artistic_config.artistic_mode,
                        );
                    }
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

                // Visualise path stem for annotated outputs
                let viz_stem = args.visualize.as_ref().and_then(|p| {
                    p.file_stem()
                        .and_then(|s| s.to_str())
                        .map(|s| s.to_string())
                });
                let viz_ext = args.visualize.as_ref().and_then(|p| {
                    p.extension()
                        .and_then(|e| e.to_str())
                        .map(|s| s.to_string())
                });

                for format in &suitable_formats {
                    // Step A: Original algorithm calculates initial crop (person-bbox driven).
                    let working_bbox =
                        apply_margin_to_bbox(&raw_bbox, args.margin, image.width(), image.height());
                    let maybe_original_crop: Option<CropRegion> = match format.as_str() {
                        "21:9" => calculate_landscape_21_9_crop(
                            image.width(),
                            image.height(),
                            &working_bbox,
                            &crop_config,
                        ),
                        "9:21" => calculate_portrait_9_21_crop(
                            image.width(),
                            image.height(),
                            &working_bbox,
                            &crop_config,
                        ),
                        "9:16" => calculate_portrait_9_16_crop(
                            image.width(),
                            image.height(),
                            &working_bbox,
                            &crop_config,
                        ),
                        other => {
                            eprintln!("Warning: unknown format '{}' — skipping.", other);
                            None
                        }
                    };

                    let original_crop = match maybe_original_crop {
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

                    // Step B: Face-aware adjustment (secondary, additive).
                    // Nudges the crop minimally to ensure the dominant face is not clipped.
                    // Returns the original crop if adjustment is impossible or unnecessary.
                    let adjusted_crop = apply_face_aware_adjustment(
                        &original_crop,
                        person_bbox_for_adjustment.as_ref(),
                        &face_bboxes,
                        &artistic_config,
                        image_width,
                        image_height,
                    );

                    // Track the final crop for visualization.
                    all_crop_regions.push(adjusted_crop.clone());

                    // Step C: Save the (adjusted) crop.
                    let crop_path = output_sorting::get_sorted_output_path(
                        output_path,
                        format,
                        stem,
                        ext,
                        args.sort_output,
                    )
                    .with_context(|| {
                        format!("Failed to build output path for format {}", format)
                    })?;
                    save_crop_region(&image, &adjusted_crop, &crop_path).with_context(|| {
                        format!("Failed to save {} crop to {:?}", format, crop_path)
                    })?;
                    if !args.quiet {
                        println!("  Saved {} crop → {:?}", format, crop_path);
                    }

                    // Annotated visualisation for this format (always injects _annotated suffix)
                    if let (Some(ref vstem), Some(ref vext)) = (&viz_stem, &viz_ext) {
                        // e.g. output/landscape/viz_annotated_21_9.jpg
                        let viz_base = args.visualize.as_deref().unwrap_or(Path::new("viz.jpg"));
                        let viz_path = output_sorting::get_annotated_output_path(
                            viz_base,
                            format,
                            vstem,
                            vext,
                            args.sort_output,
                        )
                        .with_context(|| {
                            format!("Failed to build annotated path for format {}", format)
                        })?;
                        save_visualized_with_faces(
                            &image,
                            &detections,
                            &face_bboxes,
                            std::slice::from_ref(&adjusted_crop),
                            &viz_path,
                        )
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
            save_visualized_with_faces(
                &image,
                &detections,
                &face_bboxes,
                &all_crop_regions,
                viz_path,
            )
            .with_context(|| format!("Failed to save visualization to {:?}", viz_path))?;
            if !args.quiet {
                println!("Saved visualized image to {:?}", viz_path);
            }
        }
    }

    // ── Save detection JSON ────────────────────────────────────────────────
    if let Some(ref boxes_path) = args.output_boxes {
        save_detections_json(&detections, &face_bboxes, boxes_path)
            .with_context(|| format!("Failed to save detections JSON to {:?}", boxes_path))?;
        if !args.quiet {
            println!(
                "Saved {} person + {} face detection(s) to {:?}",
                detections
                    .iter()
                    .filter(|d| d.class_id == PERSON_CLASS_ID)
                    .count(),
                face_bboxes.len(),
                boxes_path
            );
        }
    }

    // ── Final summary line ─────────────────────────────────────────────────
    if !args.quiet {
        println!(
            "Done. Found {} person(s), {} face(s).",
            detections
                .iter()
                .filter(|d| d.class_id == PERSON_CLASS_ID)
                .count(),
            face_bboxes.len(),
        );
    }

    Ok(())
}
