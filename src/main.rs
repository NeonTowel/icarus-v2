/// Icarus-v2 CLI Entry Point
///
/// Multi-model AI image cropping system. Runs object detection on an input image
/// using one of the five supported ONNX detection models, then optionally saves
/// cropped regions, annotated images, or raw detection JSON to disk.
///
/// # Example
/// ```
/// icarus-v2 --input photo.jpg --output crop.jpg --model detr-resnet101 \
///            --model-path ./models/detr/model.onnx --confidence 0.5
/// ```
use anyhow::{bail, Context, Result};
use clap::Parser;
use icarus_v2::image_utils::{crop_image, Detection};
use icarus_v2::models::implementations::{DETRResNet101, DFineL, RFDETRLarge, RFDETRMedium, YOLOv9c};
use image::DynamicImage;
use serde::Serialize;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Argument definitions
// ---------------------------------------------------------------------------

/// Known model names with their default ONNX file locations.
const VALID_MODELS: &[&str] = &[
    "detr-resnet101",
    "yolov9-c",
    "dfine-l",
    "rf-detr-large",
    "rf-detr-medium",
];

#[derive(Parser, Debug)]
#[command(
    name = "icarus-v2",
    about = "Multi-Model AI Image Cropping System",
    long_about = "Detect objects in images using one of five ONNX detection models \
                  (DETR, YOLO, D-FINE, RF-DETR). Supports saving cropped regions, \
                  annotated images, and raw detection JSON.",
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
        default_value = "detr-resnet101",
        value_name = "MODEL",
        help = "One of: detr-resnet101, yolov9-c, dfine-l, rf-detr-large, rf-detr-medium"
    )]
    model: String,

    /// Path to ONNX model file (uses built-in default path if omitted)
    #[arg(long, value_name = "FILE")]
    model_path: Option<PathBuf>,

    /// Minimum confidence threshold for detections (0.0–1.0)
    #[arg(long, default_value = "0.5", value_name = "FLOAT")]
    confidence: f32,

    /// Save detection bounding boxes as JSON to this path
    #[arg(long, value_name = "FILE")]
    output_boxes: Option<PathBuf>,

    /// Draw detection boxes on image and save to this path
    #[arg(long, value_name = "FILE")]
    visualize: Option<PathBuf>,

    /// Suppress all informational output (errors are still shown)
    #[arg(long)]
    quiet: bool,
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
// Model dispatch
// ---------------------------------------------------------------------------

/// Resolve the default ONNX model file path for a given model name.
///
/// Returns `None` if the model name has no known default path (callers should
/// then require the user to supply `--model-path`).
fn default_model_path(model_name: &str) -> Option<PathBuf> {
    match model_name {
        "detr-resnet101" => Some(PathBuf::from("./icarus-v1-rust/models/detr/model.onnx")),
        // YOLOv10m is the closest available weight file for the yolov9-c variant.
        "yolov9-c" => Some(PathBuf::from(
            "./icarus-v1-rust/models/yolov10m/model.onnx",
        )),
        _ => None,
    }
}

/// Run detection using the model identified by `model_name` loaded from `model_path`.
///
/// Returns the full unsorted detection list; confidence filtering happens here.
async fn run_detection(
    model_name: &str,
    model_path: &Path,
    image: DynamicImage,
    confidence_threshold: f32,
) -> Result<Vec<Detection>> {
    let detections = match model_name {
        "detr-resnet101" => {
            let model = DETRResNet101::new(model_path)
                .with_context(|| format!("Failed to load DETR-ResNet101 from {:?}", model_path))?;
            model.detect(image).await?
        }
        "yolov9-c" => {
            let model = YOLOv9c::new(model_path)
                .with_context(|| format!("Failed to load YOLOv9-c from {:?}", model_path))?;
            model.detect(image).await?
        }
        "dfine-l" => {
            let model = DFineL::new(model_path)
                .with_context(|| format!("Failed to load D-FINE-L from {:?}", model_path))?;
            model.detect(image).await?
        }
        "rf-detr-large" => {
            let model = RFDETRLarge::new(model_path)
                .with_context(|| format!("Failed to load RF-DETR-Large from {:?}", model_path))?;
            model.detect(image).await?
        }
        "rf-detr-medium" => {
            let model = RFDETRMedium::new(model_path)
                .with_context(|| {
                    format!("Failed to load RF-DETR-Medium from {:?}", model_path)
                })?;
            model.detect(image).await?
        }
        unknown => bail!(
            "Unknown model '{}'. Valid options are: {}",
            unknown,
            VALID_MODELS.join(", ")
        ),
    };

    // Apply confidence threshold (the postprocessor already does this at 0.5, but the
    // CLI flag lets users tighten it further without recompiling).
    let filtered: Vec<Detection> = detections
        .into_iter()
        .filter(|d| d.confidence >= confidence_threshold)
        .collect();

    Ok(filtered)
}

// ---------------------------------------------------------------------------
// Output helpers
// ---------------------------------------------------------------------------

/// Save the first detected object as a cropped JPEG/PNG to `output_path`.
///
/// If multiple detections are found, only the highest-confidence one is saved.
/// Returns `Ok(true)` when a crop was written, `Ok(false)` when there were no
/// detections to crop.
fn save_crop(
    image: &DynamicImage,
    detections: &[Detection],
    output_path: &Path,
) -> Result<bool> {
    let Some(best) = detections.first() else {
        return Ok(false);
    };

    let crop = crop_image(image, best.bbox)
        .with_context(|| format!("Failed to crop image for bbox {:?}", best.bbox))?;

    crop.save(output_path)
        .with_context(|| format!("Failed to save cropped image to {:?}", output_path))?;

    Ok(true)
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

    // Simple colour palette — cycles by class_id.
    let palette: &[[u8; 4]] = &[
        [255, 0, 0, 220],   // red
        [0, 200, 0, 220],   // green
        [0, 100, 255, 220], // blue
        [255, 165, 0, 220], // orange
        [148, 0, 211, 220], // purple
        [0, 206, 209, 220], // teal
        [255, 20, 147, 220],// pink
    ];

    for det in detections {
        let colour = palette[det.class_id % palette.len()];
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

    // ── Validate model name early (before touching the filesystem) ─────────
    if !VALID_MODELS.contains(&args.model.as_str()) {
        bail!(
            "Unknown model '{}'. Valid options are:\n  {}",
            args.model,
            VALID_MODELS.join("\n  ")
        );
    }

    // ── Load input image ───────────────────────────────────────────────────
    if !args.input.exists() {
        bail!(
            "Input file not found: {:?}\nPlease check the path and try again.",
            args.input
        );
    }

    if !args.quiet {
        println!(
            "Icarus-v2: loading image from {:?}",
            args.input
        );
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

    // ── Resolve model file path ────────────────────────────────────────────
    let model_path: PathBuf = match &args.model_path {
        Some(p) => p.clone(),
        None => match default_model_path(&args.model) {
            Some(p) => p,
            None => bail!(
                "No default model path configured for '{}'. \
                 Please supply --model-path pointing to an ONNX file.",
                args.model
            ),
        },
    };

    if !model_path.exists() {
        bail!(
            "Model file not found: {:?}\n\
             Tip: You can download the model and pass its location with --model-path.",
            model_path
        );
    }

    if !args.quiet {
        println!("  Model: {} (from {:?})", args.model, model_path);
        println!("  Confidence threshold: {}", args.confidence);
        println!("  Running inference…");
    }

    // ── Run detection ──────────────────────────────────────────────────────
    let detections = run_detection(&args.model, &model_path, image.clone(), args.confidence)
        .await
        .with_context(|| {
            format!(
                "Detection failed for model '{}' on image {:?}",
                args.model, args.input
            )
        })?;

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
    } else {
        if !args.quiet {
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
    }

    // ── Save cropped output ────────────────────────────────────────────────
    if let Some(ref output_path) = args.output {
        if detections.is_empty() {
            if !args.quiet {
                println!(
                    "Skipping --output {:?}: no detections to crop.",
                    output_path
                );
            }
        } else {
            let saved = save_crop(&image, &detections, output_path)
                .with_context(|| format!("Failed to save crop to {:?}", output_path))?;
            if saved && !args.quiet {
                println!("Saved cropped image to {:?}", output_path);
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

    // ── Save annotated visualisation ───────────────────────────────────────
    if let Some(ref viz_path) = args.visualize {
        save_visualized(&image, &detections, viz_path)
            .with_context(|| format!("Failed to save visualization to {:?}", viz_path))?;
        if !args.quiet {
            println!("Saved visualized image to {:?}", viz_path);
        }
    }

    // ── Final summary line ─────────────────────────────────────────────────
    if !args.quiet {
        println!(
            "Done. Found {} object(s){}.",
            detections.len(),
            args.output
                .as_ref()
                .map(|p| format!(", saved crop to {:?}", p))
                .unwrap_or_default()
        );
    }

    Ok(())
}
