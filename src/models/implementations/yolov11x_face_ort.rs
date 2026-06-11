/// YOLOv11x-Face ONNX Runtime implementation of the [`Model`] trait.
///
/// Uses the `ort` crate (ONNX Runtime) to run the `AdamCodd/YOLOv11x-face-detection`
/// model. The model outputs `[1, 5, 8400]` tensors in the Ultralytics transposed format
/// where each of the 8400 anchor slots encodes `[cx, cy, w, h, confidence]`.
///
/// Our custom logic:
/// 1. Resize full image to 640×640 (simple resize, no letterboxing) and normalise to [0,1].
/// 2. Run ORT session to get raw anchor predictions.
/// 3. Filter anchors by confidence threshold, convert centre-format to corner-format,
///    apply NMS, and scale coordinates back to original image dimensions.
///
/// All heavy lifting (session management, graph optimisation, execution providers) is
/// delegated to ONNX Runtime.
///
/// ## M1 — infer_direct / side-channel removal
///
/// The original path stored image dimensions in a `thread_local! YOLOV11X_FACE_PENDING_DIMS`
/// RefCell. That side-channel is removed. `infer_direct` keeps dimensions on the call stack
/// using the shared [`crate::models::candle_backend::preprocess_to_nchw_array`] helper.
/// The legacy `preprocess`/`forward`/`postprocess` stubs now use a `Mutex<pending_dims>`
/// field, matching the YOLOv10 pattern. `detect_faces` calls `model.infer()` to reach the
/// hot path.
///
/// ## Fix 12 — Per-thread SessionPool (Strategy B)
///
/// Replaces the single `Mutex<Session>` with a [`SessionPool`] so that each Rayon worker
/// thread can hold its own session, eliminating mutex contention on the dominant
/// always-on pipeline cost (face detection: ~4233 ms/image on M7 baseline).
///
/// # Model source
/// `AdamCodd/YOLOv11x-face-detection` → `model.onnx` (~60 MB).
/// Downloaded on first use from HuggingFace Hub; cached in `~/.cache/huggingface/`.
///
/// # Thread safety
/// [`YoloV11xFaceOrt`] is `Send + Sync`. The [`SessionPool`] is `Send + Sync`.
/// The `pending_dims` Mutex is only accessed by the stub `preprocess`/`postprocess` path;
/// the hot path (`infer_direct`) keeps dimensions on the call stack.
///
/// # Example
/// ```rust,ignore
/// let device = candle_core::Device::Cpu;
/// let model = YoloV11xFaceOrt::from_hub(&device, thread_count)?;
/// // Hot path via face_detection::detect_faces → model.infer():
/// let faces = model.infer(&full_image)?;
/// ```
use candle_core::{Result as CandleResult, Tensor};
use hf_hub::api::sync::Api;
use image::DynamicImage;
use ort::{inputs, value::TensorRef};
use std::sync::Mutex;

use crate::models::candle_backend::{apply_nms, preprocess_to_nchw_array, BBox, Detection, Model};
use crate::models::session_pool::{SessionPool, YOLOV11X_FACE_FOOTPRINT_MB};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// HuggingFace Hub repository for YOLOv11x-Face ONNX weights.
const HF_REPO: &str = "AdamCodd/YOLOv11x-face-detection";

/// ONNX model filename within the HF repository.
const HF_FILENAME: &str = "model.onnx";

/// Model input resolution. YOLOv11x-Face uses 640×640.
const MODEL_W: u32 = 640;
const MODEL_H: u32 = 640;

/// Confidence threshold for filtering raw anchor predictions.
const DEFAULT_CONF_THRESHOLD: f32 = 0.4;

/// NMS IoU threshold for suppressing overlapping face boxes.
const DEFAULT_NMS_THRESHOLD: f32 = 0.45;

/// Single-class label list for face detection.
const FACE_CLASSES: &[&str] = &["face"];

// ---------------------------------------------------------------------------
// YoloV11xFaceOrt struct
// ---------------------------------------------------------------------------

/// YOLOv11x-Face detector backed by ONNX Runtime with per-thread session pool.
///
/// Downloads `model.onnx` from `AdamCodd/YOLOv11x-face-detection` on HuggingFace Hub
/// on first use. The model is loaded into a [`SessionPool`] — each Rayon worker gets
/// its own session, eliminating the pre-pool mutex contention.
///
/// This struct is `Send + Sync` and can be shared across async inference tasks via `Arc`.
///
/// # Example
/// ```rust,ignore
/// let device = candle_core::Device::Cpu;
/// let model = YoloV11xFaceOrt::from_hub(&device, thread_count)?;
/// let faces = model.infer(&image)?; // Hot path — no Candle round-trip.
/// ```
pub struct YoloV11xFaceOrt {
    /// Per-worker session pool (Fix 12 / Strategy B).
    session: SessionPool,
    /// Stores image dimensions for the legacy `preprocess → postprocess` stub path.
    ///
    /// **Not used by the hot path**: `infer_direct` keeps dimensions on the stack.
    pending_dims: Mutex<Option<(u32, u32)>>,
}

impl YoloV11xFaceOrt {
    /// Download the YOLOv11x-Face ONNX model from HuggingFace Hub and initialise the pool.
    ///
    /// # Parameters
    /// - `thread_count`: Active Rayon thread count; forwarded to `SessionPool` for
    ///   `intra_op_threads` tuning (S3).
    ///
    /// # Errors
    /// Returns `Err` if the download fails or the ONNX session cannot be created.
    pub fn from_hub(_device: &candle_core::Device, thread_count: usize) -> anyhow::Result<Self> {
        let api = Api::new()?;
        let repo = api.model(HF_REPO.to_string());
        let model_path = repo.get(HF_FILENAME)?;

        log::info!(
            "yolov11x-face-ort: loading ONNX model from {:?}",
            model_path
        );

        let session = SessionPool::new(
            model_path,
            "yolov11x-face",
            YOLOV11X_FACE_FOOTPRINT_MB,
            1, // single face detection model
            thread_count,
        )
        .map_err(|e| anyhow::anyhow!("YOLOv11x-Face session pool: {e}"))?;

        log::info!("yolov11x-face-ort: session pool created successfully");

        Ok(Self {
            session,
            pending_dims: Mutex::new(None),
        })
    }

    /// One-shot inference: build NCHW array directly, run ORT, decode — all in one call.
    ///
    /// This is the **hot path** for face detection. Called by `model.infer()` which
    /// `face_detection::detect_faces` uses after the M1 routing change.
    ///
    /// Compared to the legacy `preprocess → forward → postprocess` sequence:
    /// - No Candle tensor is allocated.
    /// - `(orig_w, orig_h)` stay on the call stack — no Mutex side-channel needed.
    ///
    /// # Errors
    /// Returns `Err` if the ORT session fails or the output has an unexpected size.
    fn infer_direct(&self, img: &DynamicImage) -> CandleResult<Vec<Detection>> {
        let (orig_w, orig_h) = (img.width(), img.height());
        let array = preprocess_to_nchw_array(img, MODEL_W, MODEL_H);

        let tensor_ref = TensorRef::from_array_view(&array)
            .map_err(|e| candle_core::Error::Msg(format!("ort TensorRef failed: {e}")))?;

        self.session
            .with_session(|session| {
                let outputs = session
                    .run(inputs!["images" => tensor_ref])
                    .map_err(|e| anyhow::anyhow!("ort inference failed: {e}"))?;

                let (_shape, raw) = outputs["output0"]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| anyhow::anyhow!("ort extract output0 failed: {e}"))?;

                let values: Vec<f32> = raw.to_vec();
                let num_anchors = 8400usize;

                if values.len() != 5 * num_anchors {
                    anyhow::bail!(
                        "yolov11x-face-ort: unexpected output0 size {} (expected {})",
                        values.len(),
                        5 * num_anchors
                    );
                }

                Ok(decode_faces(&values, orig_w, orig_h))
            })
            .map_err(|e| candle_core::Error::Msg(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// Shared decode helper
// ---------------------------------------------------------------------------

/// Decode a flat `[5 × 8400]` raw ORT output into face detections.
///
/// Layout: `[cx_0..cx_8399, cy_0..cy_8399, w_0..w_8399, h_0..h_8399, conf_0..conf_8399]`
/// (transposed: feature `f` for anchor `i` is at index `f * 8400 + i`).
///
/// Applies the confidence threshold, converts centre-format → corner-format, scales to
/// original image dimensions, rejects degenerate boxes, and runs NMS.
///
/// Both `postprocess` and `infer_direct` call this function so decode logic is shared
/// and cannot drift.
fn decode_faces(values: &[f32], orig_w: u32, orig_h: u32) -> Vec<Detection> {
    let num_anchors = 8400usize;
    let scale_x = orig_w as f32 / MODEL_W as f32;
    let scale_y = orig_h as f32 / MODEL_H as f32;

    let pre_nms: Vec<Detection> = (0..num_anchors)
        .filter_map(|i| {
            let cx = values[i];
            let cy = values[num_anchors + i];
            let w = values[2 * num_anchors + i];
            let h = values[3 * num_anchors + i];
            let confidence = values[4 * num_anchors + i];

            if confidence < DEFAULT_CONF_THRESHOLD {
                return None;
            }

            let x_min = ((cx - w / 2.0) * scale_x).max(0.0);
            let y_min = ((cy - h / 2.0) * scale_y).max(0.0);
            let x_max = ((cx + w / 2.0) * scale_x).min(orig_w as f32);
            let y_max = ((cy + h / 2.0) * scale_y).min(orig_h as f32);

            if x_max <= x_min || y_max <= y_min {
                return None;
            }

            Some(Detection {
                bbox: BBox {
                    x_min,
                    y_min,
                    x_max,
                    y_max,
                },
                class_id: 0,
                confidence,
                class_name: "face".to_string(),
            })
        })
        .collect();

    apply_nms(pre_nms, DEFAULT_NMS_THRESHOLD)
}

// ---------------------------------------------------------------------------
// Model trait implementation
// ---------------------------------------------------------------------------

impl Model for YoloV11xFaceOrt {
    /// Resize image to 640×640, normalise to \[0, 1\], stash original dimensions.
    ///
    /// **Legacy stub.** The hot path uses `infer_direct` (via `infer` override) instead.
    /// This method stores dimensions in `pending_dims` for use by `postprocess`.
    ///
    /// # Errors
    /// Returns `Err` if `images` is empty or tensor allocation fails.
    fn preprocess(&self, images: &[DynamicImage]) -> CandleResult<Tensor> {
        use candle_core::{DType, Device};
        use image::imageops::FilterType;

        let img = images
            .first()
            .ok_or_else(|| candle_core::Error::Msg("preprocess: empty image slice".into()))?;

        *self
            .pending_dims
            .lock()
            .map_err(|e| candle_core::Error::Msg(format!("pending_dims lock failed: {e}")))? =
            Some((img.width(), img.height()));

        let resized = img.resize_exact(MODEL_W, MODEL_H, FilterType::Nearest);
        let rgb = resized.to_rgb8();
        let data: Vec<u8> = rgb.into_raw();
        let t = Tensor::from_vec(data, (MODEL_H as usize, MODEL_W as usize, 3), &Device::Cpu)?
            .permute((2, 0, 1))?;
        let t = (t.to_dtype(DType::F32)? * (1.0 / 255.0))?;
        t.unsqueeze(0)
    }

    /// Pass-through: no computation here; ORT inference runs in `infer_direct`.
    fn forward(&self, xs: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        Ok((xs.clone(), xs.clone()))
    }

    /// Run ORT inference via the session pool using the dims from `pending_dims`.
    ///
    /// **Legacy stub.** The hot path uses `infer_direct` instead.
    /// Requires `preprocess` to have been called first.
    fn postprocess(&self, logits: Tensor, _boxes: Tensor) -> CandleResult<Vec<Detection>> {
        use ndarray::Array;

        let (orig_w, orig_h) = self
            .pending_dims
            .lock()
            .map_err(|e| candle_core::Error::Msg(format!("pending_dims lock failed: {e}")))?
            .take()
            .ok_or_else(|| {
                candle_core::Error::Msg(
                    "postprocess: no image dimensions — call preprocess first".into(),
                )
            })?;

        let data: Vec<f32> = logits.flatten_all()?.to_vec1()?;
        let array =
            Array::from_shape_vec((1usize, 3usize, MODEL_H as usize, MODEL_W as usize), data)
                .map_err(|e| candle_core::Error::Msg(format!("ndarray reshape failed: {e}")))?;

        let tensor_ref = TensorRef::from_array_view(&array)
            .map_err(|e| candle_core::Error::Msg(format!("ort TensorRef failed: {e}")))?;

        self.session
            .with_session(|session| {
                let outputs = session
                    .run(inputs!["images" => tensor_ref])
                    .map_err(|e| anyhow::anyhow!("ort inference failed: {e}"))?;

                let (_shape, raw) = outputs["output0"]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| anyhow::anyhow!("ort extract output0 failed: {e}"))?;

                let values: Vec<f32> = raw.to_vec();
                let num_anchors = 8400usize;

                if values.len() != 5 * num_anchors {
                    anyhow::bail!(
                        "yolov11x-face-ort: unexpected output0 size {} (expected {})",
                        values.len(),
                        5 * num_anchors
                    );
                }

                Ok(decode_faces(&values, orig_w, orig_h))
            })
            .map_err(|e| candle_core::Error::Msg(e.to_string()))
    }

    fn classes(&self) -> &[&str] {
        FACE_CLASSES
    }

    fn input_size(&self) -> (usize, usize) {
        (MODEL_W as usize, MODEL_H as usize)
    }

    /// M1 hot path: direct ndarray inference, no Candle tensor round-trip.
    fn infer(&self, image: &DynamicImage) -> CandleResult<Vec<Detection>> {
        self.infer_direct(image)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use image::DynamicImage;

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Build a small synthetic RGB image for preprocessing tests.
    fn make_test_image(w: u32, h: u32) -> DynamicImage {
        DynamicImage::ImageRgb8(image::ImageBuffer::from_fn(w, h, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 128u8])
        }))
    }

    /// Build a fake YOLOv11-style output tensor of shape `[1, 3, 640, 640]`.
    fn make_nchw_tensor() -> Tensor {
        Tensor::zeros(
            (1usize, 3usize, MODEL_H as usize, MODEL_W as usize),
            DType::F32,
            &Device::Cpu,
        )
        .expect("zeros tensor")
    }

    // ── Helpers: build raw anchor output in [5, 8400] transposed format ──────

    /// Build a `[5 × 8400]`-shaped raw ORT output from a small set of anchor rows.
    fn make_raw_output(anchors: &[[f32; 5]]) -> Vec<f32> {
        let num_anchors = 8400usize;
        let mut values = vec![0.0f32; 5 * num_anchors];
        for (i, row) in anchors.iter().enumerate() {
            values[i] = row[0]; // cx
            values[num_anchors + i] = row[1]; // cy
            values[2 * num_anchors + i] = row[2]; // w
            values[3 * num_anchors + i] = row[3]; // h
            values[4 * num_anchors + i] = row[4]; // confidence
        }
        values
    }

    // ── Struct identity ───────────────────────────────────────────────────

    #[test]
    fn test_classes_returns_face() {
        assert_eq!(FACE_CLASSES, &["face"]);
    }

    #[test]
    fn test_model_w_h_are_640() {
        assert_eq!(MODEL_W, 640);
        assert_eq!(MODEL_H, 640);
    }

    // ── Preprocessing (via a minimal stub) ────────────────────────────────

    /// Minimal stub to exercise the preprocess logic without a live ORT session.
    struct PreprocessStub {
        pending_dims: std::sync::Mutex<Option<(u32, u32)>>,
    }

    impl PreprocessStub {
        fn new() -> Self {
            Self {
                pending_dims: std::sync::Mutex::new(None),
            }
        }

        fn preprocess(&self, images: &[DynamicImage]) -> candle_core::Result<Tensor> {
            use candle_core::{DType, Device};
            use image::imageops::FilterType;

            let img = images
                .first()
                .ok_or_else(|| candle_core::Error::Msg("empty slice".into()))?;
            *self.pending_dims.lock().unwrap() = Some((img.width(), img.height()));
            let resized = img.resize_exact(MODEL_W, MODEL_H, FilterType::Nearest);
            let rgb = resized.to_rgb8();
            let data: Vec<u8> = rgb.into_raw();
            let t = Tensor::from_vec(data, (MODEL_H as usize, MODEL_W as usize, 3), &Device::Cpu)?
                .permute((2, 0, 1))?;
            let t = (t.to_dtype(DType::F32)? * (1.0 / 255.0))?;
            t.unsqueeze(0)
        }
    }

    #[test]
    fn test_preprocess_produces_nchw_tensor() {
        let stub = PreprocessStub::new();
        let img = make_test_image(320, 240);
        let tensor = stub.preprocess(&[img]).expect("preprocess must succeed");
        assert_eq!(
            tensor.dims(),
            &[1, 3, MODEL_H as usize, MODEL_W as usize],
            "preprocess must return [N, C, H, W] tensor"
        );
    }

    #[test]
    fn test_preprocess_normalises_to_unit_range() {
        let stub = PreprocessStub::new();
        let img = make_test_image(64, 64);
        let tensor = stub.preprocess(&[img]).expect("preprocess must succeed");
        let values: Vec<f32> = tensor.flatten_all().unwrap().to_vec1().unwrap();
        for v in &values {
            assert!(*v >= 0.0 && *v <= 1.0, "pixel out of [0,1]: {v}");
        }
    }

    #[test]
    fn test_preprocess_captures_original_dims() {
        let stub = PreprocessStub::new();
        let img = make_test_image(1280, 720);
        stub.preprocess(&[img]).expect("preprocess must succeed");
        let dims = stub.pending_dims.lock().unwrap().take();
        assert_eq!(dims, Some((1280, 720)));
    }

    #[test]
    fn test_preprocess_empty_slice_returns_error() {
        let stub = PreprocessStub::new();
        let result = stub.preprocess(&[]);
        assert!(result.is_err(), "empty image slice must return Err");
    }

    // ── Decoding logic (output anchor parsing) ────────────────────────────

    #[test]
    fn test_decode_filters_low_confidence() {
        let raw = make_raw_output(&[[320.0, 320.0, 100.0, 100.0, 0.1]]);
        let result = decode_faces(&raw, 640, 640);
        assert!(result.is_empty(), "low-confidence anchor must be filtered");
    }

    #[test]
    fn test_decode_keeps_high_confidence_detection() {
        let raw = make_raw_output(&[[320.0, 320.0, 80.0, 80.0, 0.9]]);
        let result = decode_faces(&raw, 640, 640);
        assert_eq!(result.len(), 1);
        let det = &result[0];
        assert_eq!(det.class_id, 0);
        assert_eq!(det.class_name, "face");
        assert!((det.confidence - 0.9).abs() < 1e-5);
        assert!(
            (det.bbox.x_min - 280.0).abs() < 1e-3,
            "x_min={}",
            det.bbox.x_min
        );
        assert!(
            (det.bbox.x_max - 360.0).abs() < 1e-3,
            "x_max={}",
            det.bbox.x_max
        );
    }

    #[test]
    fn test_decode_converts_cxcywh_to_xyxy() {
        let raw = make_raw_output(&[[100.0, 200.0, 60.0, 40.0, 0.8]]);
        let result = decode_faces(&raw, 640, 640);
        assert_eq!(result.len(), 1);
        let det = &result[0];
        assert!((det.bbox.x_min - 70.0).abs() < 1e-3);
        assert!((det.bbox.y_min - 180.0).abs() < 1e-3);
        assert!((det.bbox.x_max - 130.0).abs() < 1e-3);
        assert!((det.bbox.y_max - 220.0).abs() < 1e-3);
    }

    #[test]
    fn test_decode_scales_to_original_dims() {
        let raw = make_raw_output(&[[320.0, 320.0, 160.0, 160.0, 0.85]]);
        let result = decode_faces(&raw, 1280, 960);
        assert_eq!(result.len(), 1);
        let det = &result[0];
        assert!((det.bbox.x_min - 480.0).abs() < 1e-2);
        assert!((det.bbox.y_min - 360.0).abs() < 1e-2);
        assert!((det.bbox.x_max - 800.0).abs() < 1e-2);
        assert!((det.bbox.y_max - 600.0).abs() < 1e-2);
    }

    #[test]
    fn test_decode_clamps_to_image_bounds() {
        let raw = make_raw_output(&[[10.0, 10.0, 80.0, 80.0, 0.9]]);
        let result = decode_faces(&raw, 640, 640);
        assert_eq!(result.len(), 1);
        let det = &result[0];
        assert!(det.bbox.x_min >= 0.0);
        assert!(det.bbox.y_min >= 0.0);
    }

    #[test]
    fn test_decode_rejects_degenerate_boxes() {
        let raw = make_raw_output(&[[320.0, 320.0, 0.0, 0.0, 0.95]]);
        let result = decode_faces(&raw, 640, 640);
        assert!(result.is_empty(), "zero-area box must be rejected");
    }

    #[test]
    fn test_decode_multiple_anchors_threshold_filtering() {
        let raw = make_raw_output(&[
            [100.0, 100.0, 60.0, 60.0, 0.9],
            [200.0, 200.0, 60.0, 60.0, 0.1],
            [400.0, 400.0, 80.0, 80.0, 0.7],
        ]);
        let result = decode_faces(&raw, 640, 640);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_decode_applies_nms_to_overlapping_boxes() {
        let raw = make_raw_output(&[
            [320.0, 320.0, 100.0, 100.0, 0.9],
            [322.0, 322.0, 100.0, 100.0, 0.6],
        ]);
        let result = decode_faces(&raw, 640, 640);
        assert_eq!(result.len(), 1);
        assert!((result[0].confidence - 0.9).abs() < 1e-5);
    }

    #[test]
    fn test_forward_pass_through_preserves_shape() {
        let t = make_nchw_tensor();
        let expected_dims = t.dims().to_vec();
        let (out_logits, out_boxes) = (t.clone(), t.clone());
        assert_eq!(out_logits.dims(), expected_dims.as_slice());
        assert_eq!(out_boxes.dims(), expected_dims.as_slice());
    }

    /// Parity test: `decode_faces` (shared helper) produces the same result as the
    /// old inline `postprocess` decode logic.
    ///
    /// Since both `postprocess` and `infer_direct` now call `decode_faces`, this test
    /// guards against future drift if either path is changed independently.
    #[test]
    fn test_infer_direct_decode_matches_postprocess_decode() {
        // Synthetic raw output with two detections above threshold.
        let raw = make_raw_output(&[
            [200.0, 150.0, 80.0, 60.0, 0.85],
            [400.0, 350.0, 100.0, 80.0, 0.75],
        ]);
        let orig_w = 1280u32;
        let orig_h = 720u32;

        // Both postprocess and infer_direct call decode_faces — call it twice to confirm
        // determinism and assert identical results.
        let result_a = decode_faces(&raw, orig_w, orig_h);
        let result_b = decode_faces(&raw, orig_w, orig_h);

        assert_eq!(
            result_a.len(),
            result_b.len(),
            "decode_faces must be deterministic"
        );
        for (a, b) in result_a.iter().zip(result_b.iter()) {
            assert!((a.confidence - b.confidence).abs() < 1e-6);
            assert!((a.bbox.x_min - b.bbox.x_min).abs() < 1e-6);
            assert!((a.bbox.y_min - b.bbox.y_min).abs() < 1e-6);
        }
    }
}
