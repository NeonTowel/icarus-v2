/// Face Detection Module
///
/// Encapsulates all face detection inference. Provides an opaque handle to a loaded
/// face detection model ([`FaceDetector`]) and two public functions:
/// [`load_face_detector`] to initialise the model, and [`detect_faces`] to run inference.
///
/// This module has zero knowledge of cropping, persons, or artistic modes.
/// It converts model-domain [`crate::models::candle_backend::BBox`] coordinates into
/// crop-domain [`crate::multi_format_cropping::BBox`] coordinates — this is the single
/// type-conversion boundary in the entire pipeline.
///
/// # Lifecycle
/// Load once at startup, reuse for all images:
///
/// ```rust,ignore
/// let detector = load_face_detector()?;
/// for image in images {
///     let faces = detect_faces(&image, &detector)?;
/// }
/// ```
///
/// # Model Source
/// Downloads `model.onnx` from `AdamCodd/YOLOv11x-face-detection` on HuggingFace Hub
/// on first use (~60 MB). Subsequent runs load from disk cache with no network I/O.
use crate::models::YoloV11xFaceOrt;
use crate::multi_format_cropping::BBox;
use anyhow::Result;
use candle_core::Device;
use image::DynamicImage;

// ---------------------------------------------------------------------------
// FaceDetector — opaque handle to the loaded face model
// ---------------------------------------------------------------------------

/// Opaque handle to a loaded YOLOv11x-Face detection model.
///
/// This struct wraps [`YoloV11xFaceOrt`] but does not expose it publicly.
/// Callers interact only through [`load_face_detector`] and [`detect_faces`].
///
/// `FaceDetector` is `Send + Sync` (the underlying ORT session is wrapped in a `Mutex`).
///
/// # Example
/// ```rust,ignore
/// let detector = load_face_detector()?;
/// let faces = detect_faces(&image, &detector)?;
/// ```
pub struct FaceDetector {
    /// Underlying YOLOv11x-Face ONNX Runtime model.
    model: YoloV11xFaceOrt,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Load the YOLOv11x-Face detector from HuggingFace Hub cache.
///
/// Downloads the model on first use (~60 MB). Subsequent calls use the local
/// disk cache at `~/.cache/huggingface/hub/` with no network I/O.
///
/// This is a **blocking** operation. In async code, call from a blocking context:
/// ```rust,ignore
/// let detector = tokio::task::spawn_blocking(load_face_detector).await??;
/// ```
///
/// # Errors
/// Returns `Err` if:
/// - The model download fails (network error, disk full, etc.)
/// - The ONNX model file cannot be parsed by ONNX Runtime
///
/// These errors are **fatal** — the caller should exit with a clear message
/// rather than silently falling back to no face detection.
///
/// # Example
/// ```rust,ignore
/// let detector = load_face_detector().expect("face model should load");
/// ```
pub fn load_face_detector() -> Result<FaceDetector> {
    let device = Device::Cpu;
    let model = YoloV11xFaceOrt::from_hub(&device)?;
    Ok(FaceDetector { model })
}

/// Run face detection on the given image.
///
/// Returns zero or more face bounding boxes in image pixel coordinates (`[x1, y1, x2, y2]`),
/// sorted by confidence descending (highest confidence first).
///
/// Converts from model-domain [`crate::models::candle_backend::BBox`]
/// (`x_min`, `y_min`, `x_max`, `y_max`) to crop-domain [`BBox`] (`x1`, `y1`, `x2`, `y2`).
/// No other module performs this conversion.
///
/// # Returns
/// - `Ok(vec![])` if no faces are detected (not an error).
/// - `Err(...)` only on model inference failure (ORT crash, malformed tensor).
///
/// # Arguments
/// * `image`    — Source image to run inference on.
/// * `detector` — Loaded face model handle from [`load_face_detector`].
///
/// # Example
/// ```rust,ignore
/// let faces = detect_faces(&image, &detector)?;
/// if faces.is_empty() {
///     println!("No faces detected");
/// }
/// ```
pub fn detect_faces(image: &DynamicImage, detector: &FaceDetector) -> Result<Vec<BBox>> {
    use crate::models::Model;

    let tensor = detector
        .model
        .preprocess(std::slice::from_ref(image))
        .map_err(|e| anyhow::anyhow!("Face detection preprocess failed: {e}"))?;

    let (logits, boxes) = detector
        .model
        .forward(&tensor)
        .map_err(|e| anyhow::anyhow!("Face detection forward failed: {e}"))?;

    let detections = detector
        .model
        .postprocess(logits, boxes)
        .map_err(|e| anyhow::anyhow!("Face detection postprocess failed: {e}"))?;

    // Convert from model-domain BBox (x_min, y_min, x_max, y_max) to
    // crop-domain BBox (x1, y1, x2, y2). This is the single conversion boundary.
    let face_bboxes: Vec<BBox> = detections
        .into_iter()
        .map(|d| BBox {
            x1: d.bbox.x_min,
            y1: d.bbox.y_min,
            x2: d.bbox.x_max,
            y2: d.bbox.y_max,
        })
        .collect();

    Ok(face_bboxes)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: the face detector model can be loaded from HF Hub cache.
    ///
    /// This test requires the model to be present in the HuggingFace Hub cache
    /// (`~/.cache/huggingface/hub/`). On CI, the model should be pre-cached.
    /// If the cache is empty, the test will attempt to download (~60 MB).
    #[test]
    fn test_load_face_detector_succeeds() {
        let result = load_face_detector();
        assert!(
            result.is_ok(),
            "face detector should load: {:?}",
            result.err()
        );
    }

    /// Blank images should produce zero detections (not an error).
    #[test]
    fn test_detect_faces_on_blank_image_returns_empty() {
        let detector = match load_face_detector() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Skipping test (model not available): {e}");
                return;
            }
        };
        let blank = DynamicImage::new_rgb8(640, 640);
        let faces =
            detect_faces(&blank, &detector).expect("inference on blank image should succeed");
        assert!(
            faces.is_empty(),
            "blank image should yield no faces, got {}",
            faces.len()
        );
    }
}
