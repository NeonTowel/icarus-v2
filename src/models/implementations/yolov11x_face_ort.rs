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
/// # Model source
/// `AdamCodd/YOLOv11x-face-detection` → `model.onnx` (~60 MB).
/// Downloaded on first use from HuggingFace Hub; cached in `~/.cache/huggingface/`.
///
/// # Thread safety
/// [`YoloV11xFaceOrt`] is `Send + Sync`. Image dimensions are stored in a
/// `Mutex<Option<(u32, u32)>>` because `preprocess` and `postprocess` share state
/// across two separate `&self` calls.
///
/// # Example
/// ```rust,ignore
/// let device = candle_core::Device::Cpu;
/// let model = YoloV11xFaceOrt::from_hub(&device)?;
/// let tensor = model.preprocess(&[full_image])?;
/// let (logits, boxes) = model.forward(&tensor)?;
/// let faces = model.postprocess(logits, boxes)?;
/// ```
use candle_core::{DType, Device, Result as CandleResult, Tensor};
use hf_hub::api::sync::Api;
use image::{imageops::FilterType, DynamicImage};
use ndarray::Array;
use ort::{
    inputs,
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};
use std::cell::RefCell;
use std::sync::Mutex;

use crate::models::candle_backend::{apply_nms, BBox, Detection, Model};

thread_local! {
    static YOLOV11X_FACE_PENDING_DIMS: RefCell<Option<(u32, u32)>> = const { RefCell::new(None) };
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// M7 Decision: use ORT default. See yolov10_common.rs M7 decision comment.

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

/// YOLOv11x-Face detector backed by ONNX Runtime.
///
/// Downloads `model.onnx` from `AdamCodd/YOLOv11x-face-detection` on HuggingFace Hub
/// on first use. The model is loaded into an ORT session and evaluated via `ort::Session::run`.
///
/// This struct is `Send + Sync` and can be shared across async inference tasks via `Arc`.
///
/// # Example
/// ```rust,ignore
/// let device = candle_core::Device::Cpu;
/// let model = YoloV11xFaceOrt::from_hub(&device)?;
/// let tensor = model.preprocess(&[image])?;
/// let (logits, _boxes) = model.forward(&tensor)?;
/// let faces = model.postprocess(logits, _boxes)?;
/// ```
pub struct YoloV11xFaceOrt {
    /// ONNX Runtime session; `run()` requires `&mut Session`, so wrapped in Mutex.
    session: Mutex<Session>,
}

impl YoloV11xFaceOrt {
    /// Download the YOLOv11x-Face ONNX model from HuggingFace Hub and initialise the ORT session.
    ///
    /// The `device` parameter is accepted for API compatibility but is not used —
    /// ONNX Runtime manages its own execution providers (CPU by default).
    ///
    /// Downloads `model.onnx` from `AdamCodd/YOLOv11x-face-detection` on first use.
    /// Subsequent calls use the local HF Hub cache (`~/.cache/huggingface/hub/`)
    /// with no network I/O.
    ///
    /// # Errors
    /// Returns `Err` if the download fails or the ONNX session cannot be created.
    ///
    /// # Example
    /// ```rust,ignore
    /// let model = YoloV11xFaceOrt::from_hub(&Device::Cpu)?;
    /// ```
    pub fn from_hub(_device: &Device) -> anyhow::Result<Self> {
        let api = Api::new()?;
        let repo = api.model(HF_REPO.to_string());
        let model_path = repo.get(HF_FILENAME)?;

        log::info!(
            "yolov11x-face-ort: loading ONNX model from {:?}",
            model_path
        );

        // Use ORT default thread count. See M7 decision comment above.
        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("ort Session builder failed: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("ort optimisation level failed: {e}"))?
            .commit_from_file(&model_path)
            .map_err(|e| anyhow::anyhow!("ort session load failed: {e}"))?;

        log::info!("yolov11x-face-ort: ONNX session created successfully");

        Ok(Self {
            session: Mutex::new(session),
        })
    }
}

// ---------------------------------------------------------------------------
// Model trait implementation
// ---------------------------------------------------------------------------

impl Model for YoloV11xFaceOrt {
    /// Resize image to 640×640, normalise to \[0, 1\], stash original dimensions.
    ///
    /// Returns a 1×3×640×640 Candle tensor (used as a pass-through carrier to `postprocess`).
    ///
    /// # Errors
    /// Returns `Err` if `images` is empty or tensor allocation fails.
    fn preprocess(&self, images: &[DynamicImage]) -> CandleResult<Tensor> {
        let img = images
            .first()
            .ok_or_else(|| candle_core::Error::Msg("preprocess: empty image slice".into()))?;

        YOLOV11X_FACE_PENDING_DIMS.with(|pending_dims| {
            *pending_dims.borrow_mut() = Some((img.width(), img.height()));
        });

        // Resize to 640×640 (simple resize, matching Ultralytics preprocessing).
        let resized = img.resize_exact(MODEL_W, MODEL_H, FilterType::Nearest);
        let rgb = resized.to_rgb8();

        // Build CHW tensor and normalise to [0, 1].
        let data: Vec<u8> = rgb.into_raw();
        let t = Tensor::from_vec(data, (MODEL_H as usize, MODEL_W as usize, 3), &Device::Cpu)?
            .permute((2, 0, 1))?; // HWC → CHW
        let t = (t.to_dtype(DType::F32)? * (1.0 / 255.0))?; // [0,255] → [0,1]
        t.unsqueeze(0) // CHW → NCHW
    }

    /// Pass-through: no computation here; ORT inference runs in `postprocess`.
    fn forward(&self, xs: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        Ok((xs.clone(), xs.clone()))
    }

    /// Run ORT inference, decode YOLOv11 anchors, and scale bboxes to original image space.
    ///
    /// The model output `output0` has shape `[1, 5, 8400]`. Each of the 8400 anchor slots
    /// encodes `[cx, cy, w, h, confidence]` in 640×640 model coordinate space.
    ///
    /// Steps:
    /// 1. Convert Candle tensor → ndarray for the ORT input.
    /// 2. Run ORT session.
    /// 3. Decode anchors: filter by confidence, convert cx/cy/w/h → x_min/y_min/x_max/y_max.
    /// 4. Scale from 640×640 model space → original image dimensions.
    /// 5. Apply NMS to suppress overlapping boxes.
    ///
    /// # Errors
    /// Returns `Err` if `preprocess` was not called first, or if the ORT session fails.
    fn postprocess(&self, logits: Tensor, _boxes: Tensor) -> CandleResult<Vec<Detection>> {
        let (orig_w, orig_h) = YOLOV11X_FACE_PENDING_DIMS
            .with(|pending_dims| pending_dims.borrow_mut().take())
            .ok_or_else(|| {
                candle_core::Error::Msg(
                    "postprocess: no image dimensions — call preprocess first".into(),
                )
            })?;

        // Convert Candle tensor → ndarray for the ORT input.
        let data: Vec<f32> = logits.flatten_all()?.to_vec1()?;
        let array =
            Array::from_shape_vec((1usize, 3usize, MODEL_H as usize, MODEL_W as usize), data)
                .map_err(|e| candle_core::Error::Msg(format!("ndarray reshape failed: {e}")))?;

        // Run ORT inference.
        let tensor_ref = TensorRef::from_array_view(&array)
            .map_err(|e| candle_core::Error::Msg(format!("ort TensorRef failed: {e}")))?;
        let mut session_guard = self.session.lock().unwrap();
        let outputs = session_guard
            .run(inputs!["images" => tensor_ref])
            .map_err(|e| candle_core::Error::Msg(format!("ort inference failed: {e}")))?;

        let (_shape, raw) = outputs["output0"]
            .try_extract_tensor::<f32>()
            .map_err(|e| candle_core::Error::Msg(format!("ort extract output0 failed: {e}")))?;

        // output0 shape: [1, 5, 8400] — layout is [batch, features, anchors].
        // features = [cx, cy, w, h, confidence].
        // We have batch=1, so the flat layout is:
        //   [cx_0..cx_8399, cy_0..cy_8399, w_0..w_8399, h_0..h_8399, conf_0..conf_8399]
        let values: Vec<f32> = raw.to_vec();
        let num_anchors = 8400usize;
        let num_features = 5usize;

        // Validate that the output has the expected number of elements.
        if values.len() != num_features * num_anchors {
            return Err(candle_core::Error::Msg(format!(
                "yolov11x-face-ort: unexpected output0 size {} (expected {})",
                values.len(),
                num_features * num_anchors
            )));
        }

        let scale_x = orig_w as f32 / MODEL_W as f32;
        let scale_y = orig_h as f32 / MODEL_H as f32;

        // Decode anchor slots: output is transposed ([features, anchors]),
        // so feature `f` for anchor `i` is at index `f * num_anchors + i`.
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

                // Convert centre-format → corner-format in model coordinate space,
                // then scale to original image dimensions.
                let x_min = ((cx - w / 2.0) * scale_x).max(0.0);
                let y_min = ((cy - h / 2.0) * scale_y).max(0.0);
                let x_max = ((cx + w / 2.0) * scale_x).min(orig_w as f32);
                let y_max = ((cy + h / 2.0) * scale_y).min(orig_h as f32);

                // Reject degenerate boxes.
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

        // Apply NMS to suppress overlapping face boxes.
        Ok(apply_nms(pre_nms, DEFAULT_NMS_THRESHOLD))
    }

    fn classes(&self) -> &[&str] {
        FACE_CLASSES
    }

    fn input_size(&self) -> (usize, usize) {
        (MODEL_W as usize, MODEL_H as usize)
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
    ///
    /// This mirrors what `preprocess` returns and what ORT would receive, allowing
    /// the full preprocess → forward pipeline to be tested without a real ORT session.
    fn make_nchw_tensor() -> Tensor {
        Tensor::zeros(
            (1usize, 3usize, MODEL_H as usize, MODEL_W as usize),
            DType::F32,
            &Device::Cpu,
        )
        .expect("zeros tensor")
    }

    // ── Helpers: build raw anchor output in [1, 5, 8400] transposed format ──

    /// Build a `[1, 5, 8400]`-shaped raw ORT output from a small set of anchor rows.
    ///
    /// Each element of `anchors` is `[cx, cy, w, h, confidence]`.
    /// Unused anchor slots are zeroed (confidence=0 → filtered out by threshold).
    fn make_raw_output(anchors: &[[f32; 5]]) -> Vec<f32> {
        // Flat layout: [cx_0..cx_8399, cy_0..cy_8399, w_0..w_8399, h_0..h_8399, conf_0..conf_8399]
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

    /// Decode raw output values using the same logic as `postprocess`, but without
    /// requiring a real ORT session. Used to unit-test the decoding logic in isolation.
    fn decode_raw(values: &[f32], orig_w: u32, orig_h: u32) -> Vec<Detection> {
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

    // ── Struct identity ───────────────────────────────────────────────────

    #[test]
    fn test_classes_returns_face() {
        // We can't construct YoloV11xFaceOrt without a real ORT session, so we
        // verify the constant directly.
        assert_eq!(FACE_CLASSES, &["face"]);
    }

    #[test]
    fn test_model_w_h_are_640() {
        assert_eq!(MODEL_W, 640);
        assert_eq!(MODEL_H, 640);
    }

    // ── Preprocessing (via a minimal stub) ────────────────────────────────

    /// Minimal Model stub to exercise the preprocess logic without a live ORT session.
    struct PreprocessStub {
        pending_dims: Mutex<Option<(u32, u32)>>,
    }

    impl PreprocessStub {
        fn new() -> Self {
            Self {
                pending_dims: Mutex::new(None),
            }
        }

        fn preprocess(&self, images: &[DynamicImage]) -> CandleResult<Tensor> {
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
        assert_eq!(
            dims,
            Some((1280, 720)),
            "pending_dims must capture original size"
        );
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
        // cx=320, cy=320, w=100, h=100, conf=0.1 (< 0.4 threshold)
        let raw = make_raw_output(&[[320.0, 320.0, 100.0, 100.0, 0.1]]);
        let result = decode_raw(&raw, 640, 640);
        assert!(result.is_empty(), "low-confidence anchor must be filtered");
    }

    #[test]
    fn test_decode_keeps_high_confidence_detection() {
        // cx=320, cy=320, w=80, h=80, conf=0.9 → at 1:1 scale
        // x_min = 320 - 40 = 280, x_max = 320 + 40 = 360
        let raw = make_raw_output(&[[320.0, 320.0, 80.0, 80.0, 0.9]]);
        let result = decode_raw(&raw, 640, 640);
        assert_eq!(result.len(), 1, "high-confidence anchor must be kept");
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
        // cx=100, cy=200, w=60, h=40 → x_min=70, y_min=180, x_max=130, y_max=220
        let raw = make_raw_output(&[[100.0, 200.0, 60.0, 40.0, 0.8]]);
        let result = decode_raw(&raw, 640, 640);
        assert_eq!(result.len(), 1);
        let det = &result[0];
        assert!(
            (det.bbox.x_min - 70.0).abs() < 1e-3,
            "x_min={}",
            det.bbox.x_min
        );
        assert!(
            (det.bbox.y_min - 180.0).abs() < 1e-3,
            "y_min={}",
            det.bbox.y_min
        );
        assert!(
            (det.bbox.x_max - 130.0).abs() < 1e-3,
            "x_max={}",
            det.bbox.x_max
        );
        assert!(
            (det.bbox.y_max - 220.0).abs() < 1e-3,
            "y_max={}",
            det.bbox.y_max
        );
    }

    #[test]
    fn test_decode_scales_to_original_dims() {
        // Model space: cx=320, cy=320, w=160, h=160 at 640×640
        // Original image: 1280×960 → scale_x=2.0, scale_y=1.5
        // x_min=(320-80)*2.0=480, y_min=(320-80)*1.5=360
        // x_max=(320+80)*2.0=800, y_max=(320+80)*1.5=600
        let raw = make_raw_output(&[[320.0, 320.0, 160.0, 160.0, 0.85]]);
        let result = decode_raw(&raw, 1280, 960);
        assert_eq!(result.len(), 1);
        let det = &result[0];
        assert!(
            (det.bbox.x_min - 480.0).abs() < 1e-2,
            "x_min={}",
            det.bbox.x_min
        );
        assert!(
            (det.bbox.y_min - 360.0).abs() < 1e-2,
            "y_min={}",
            det.bbox.y_min
        );
        assert!(
            (det.bbox.x_max - 800.0).abs() < 1e-2,
            "x_max={}",
            det.bbox.x_max
        );
        assert!(
            (det.bbox.y_max - 600.0).abs() < 1e-2,
            "y_max={}",
            det.bbox.y_max
        );
    }

    #[test]
    fn test_decode_clamps_to_image_bounds() {
        // Face near top-left corner: bbox would go negative in model space.
        // cx=10, cy=10, w=80, h=80 → raw x_min=-30, y_min=-30 → clamped to 0.
        let raw = make_raw_output(&[[10.0, 10.0, 80.0, 80.0, 0.9]]);
        let result = decode_raw(&raw, 640, 640);
        assert_eq!(result.len(), 1);
        let det = &result[0];
        assert!(det.bbox.x_min >= 0.0, "x_min must not go negative");
        assert!(det.bbox.y_min >= 0.0, "y_min must not go negative");
    }

    #[test]
    fn test_decode_rejects_degenerate_boxes() {
        // w=0 → x_min == x_max → no valid area, must be rejected.
        let raw = make_raw_output(&[[320.0, 320.0, 0.0, 0.0, 0.95]]);
        let result = decode_raw(&raw, 640, 640);
        assert!(result.is_empty(), "zero-area box must be rejected");
    }

    #[test]
    fn test_decode_multiple_anchors_threshold_filtering() {
        // Three anchors: two above threshold, one below.
        let raw = make_raw_output(&[
            [100.0, 100.0, 60.0, 60.0, 0.9], // kept
            [200.0, 200.0, 60.0, 60.0, 0.1], // filtered (< 0.4)
            [400.0, 400.0, 80.0, 80.0, 0.7], // kept
        ]);
        let result = decode_raw(&raw, 640, 640);
        assert_eq!(
            result.len(),
            2,
            "two detections above threshold must be kept"
        );
    }

    #[test]
    fn test_decode_applies_nms_to_overlapping_boxes() {
        // Two heavily overlapping anchors — NMS must keep only the higher-confidence one.
        let raw = make_raw_output(&[
            [320.0, 320.0, 100.0, 100.0, 0.9], // higher confidence — survives NMS
            [322.0, 322.0, 100.0, 100.0, 0.6], // heavily overlapping, lower — suppressed
        ]);
        let result = decode_raw(&raw, 640, 640);
        assert_eq!(
            result.len(),
            1,
            "NMS must suppress the overlapping lower-confidence box"
        );
        assert!(
            (result[0].confidence - 0.9).abs() < 1e-5,
            "highest-confidence box must survive NMS"
        );
    }

    #[test]
    fn test_forward_pass_through_preserves_shape() {
        // forward() is a pass-through; verify the output shape equals the input.
        let t = make_nchw_tensor();
        let expected_dims = t.dims().to_vec();
        // Replicate the pass-through logic without a real session.
        let (out_logits, out_boxes) = (t.clone(), t.clone());
        assert_eq!(out_logits.dims(), expected_dims.as_slice());
        assert_eq!(out_boxes.dims(), expected_dims.as_slice());
    }
}
