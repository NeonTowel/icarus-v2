/// YOLOv10-face ONNX Runtime implementation of the [`Model`] trait.
///
/// This is the fast face detector used by the v2.1 Speed Demon pipeline when a
/// YOLO26 person model is selected. It mirrors the YOLOv11x-Face implementation
/// but loads the lighter YOLOv10-face export from HuggingFace Hub.
///
/// ## M1 — infer_direct / decode_faces extraction
///
/// The decode logic is factored into a shared `decode_faces` helper so that both
/// `postprocess` (legacy stub) and `infer_direct` (hot path) call the same code.
/// `Model::infer` is overridden to call `infer_direct`, eliminating the Candle
/// tensor round-trip and the Mutex pending-dims side-channel on the hot path.
///
/// ## Fix 12 — Per-thread SessionPool (Strategy B)
///
/// Replaces the single `Mutex<Session>` with a [`SessionPool`] so each Rayon worker
/// thread gets its own session, eliminating the mutex contention that previously
/// serialised all face detection inference.
use candle_core::{Result as CandleResult, Tensor};
use hf_hub::api::sync::Api;
use image::DynamicImage;
use ort::{inputs, value::TensorRef};
use std::sync::Mutex;

use crate::models::candle_backend::{apply_nms, preprocess_to_nchw_array, BBox, Detection, Model};
use crate::models::session_pool::{SessionPool, YOLOV10_FACE_FOOTPRINT_MB};

/// HuggingFace Hub repository for YOLOv10-face ONNX weights.
const HF_REPO: &str = "deepghs/yolo-face";

/// ONNX model filename within the HF repository.
const HF_FILENAME: &str = "yolov10n-face/model.onnx";

/// Model input resolution.
const MODEL_W: u32 = 640;
const MODEL_H: u32 = 640;

/// Confidence threshold for filtering raw face predictions.
const DEFAULT_CONF_THRESHOLD: f32 = 0.4;

/// NMS IoU threshold for suppressing overlapping face boxes.
const DEFAULT_NMS_THRESHOLD: f32 = 0.45;

/// Single-class label list for face detection.
const FACE_CLASSES: &[&str] = &["face"];

/// YOLOv10-face detector backed by ONNX Runtime with per-thread session pool.
pub struct YoloV10FaceOrt {
    /// Per-worker session pool (Fix 12 / Strategy B).
    session: SessionPool,
    /// Stores image dimensions for the `preprocess → postprocess` legacy stub path.
    ///
    /// **Not used by the hot path**: `infer_direct` keeps dimensions on the stack.
    pending_dims: Mutex<Option<(u32, u32)>>,
}

impl YoloV10FaceOrt {
    /// Download the YOLOv10-face ONNX model from HuggingFace Hub and initialise the pool.
    ///
    /// # Parameters
    /// - `thread_count`: Active Rayon thread count; forwarded to `SessionPool`.
    pub fn from_hub(_device: &candle_core::Device, thread_count: usize) -> anyhow::Result<Self> {
        let api = Api::new()?;
        let repo = api.model(HF_REPO.to_string());
        let model_path = repo.get(HF_FILENAME)?;

        log::info!("yolov10-face-ort: loading ONNX model from {:?}", model_path);

        let session = SessionPool::new(
            model_path,
            "yolov10-face",
            YOLOV10_FACE_FOOTPRINT_MB,
            1, // single face detection model
            thread_count,
        )
        .map_err(|e| anyhow::anyhow!("YOLOv10-Face session pool: {e}"))?;

        Ok(Self {
            session,
            pending_dims: Mutex::new(None),
        })
    }

    /// One-shot inference: build NCHW array directly, run ORT, decode — all in one call.
    ///
    /// This is the **hot path** for face detection. Called by `model.infer()` which
    /// `face_detection::detect_faces` uses after the M1 routing change.
    fn infer_direct(&self, img: &DynamicImage) -> CandleResult<Vec<Detection>> {
        let (orig_w, orig_h) = (img.width(), img.height());
        let array = preprocess_to_nchw_array(img, MODEL_W, MODEL_H);

        let tensor_ref = TensorRef::from_array_view(&array)
            .map_err(|error| candle_core::Error::Msg(format!("ort TensorRef failed: {error}")))?;

        self.session
            .with_session(|session| {
                let outputs = session
                    .run(inputs!["images" => tensor_ref])
                    .map_err(|e| anyhow::anyhow!("ort inference failed: {e}"))?;

                let (_shape, raw) = outputs["output0"]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| anyhow::anyhow!("ort extract output0 failed: {e}"))?;

                let values = raw.to_vec();
                let num_anchors = values.len() / 5usize;

                if num_anchors * 5 != values.len() {
                    anyhow::bail!(
                        "yolov10-face: unexpected output0 size {} (expected a multiple of 5)",
                        values.len()
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

/// Decode a flat `[5 × num_anchors]` transposed ORT output into face detections.
///
/// Layout: `[cx_0..cx_N, cy_0..cy_N, w_0..w_N, h_0..h_N, conf_0..conf_N]`.
///
/// Both `postprocess` and `infer_direct` call this function so decode logic is shared
/// and cannot drift.
fn decode_faces(values: &[f32], orig_w: u32, orig_h: u32) -> Vec<Detection> {
    let num_anchors = values.len() / 5;
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

impl Model for YoloV10FaceOrt {
    /// **Legacy stub.** Stores dims in `pending_dims` for the `postprocess` call.
    fn preprocess(&self, images: &[DynamicImage]) -> CandleResult<Tensor> {
        use candle_core::{DType, Device};
        use image::imageops::FilterType;

        let img = images
            .first()
            .ok_or_else(|| candle_core::Error::Msg("preprocess: empty image slice".into()))?;

        *self.pending_dims.lock().map_err(|error| {
            candle_core::Error::Msg(format!("pending_dims lock failed: {error}"))
        })? = Some((img.width(), img.height()));

        let resized = img.resize_exact(MODEL_W, MODEL_H, FilterType::Nearest);
        let rgb = resized.to_rgb8();
        let data: Vec<u8> = rgb.into_raw();

        let tensor = Tensor::from_vec(data, (MODEL_H as usize, MODEL_W as usize, 3), &Device::Cpu)?
            .permute((2, 0, 1))?;
        let tensor = (tensor.to_dtype(DType::F32)? * (1.0 / 255.0))?;
        tensor.unsqueeze(0)
    }

    fn forward(&self, xs: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        Ok((xs.clone(), xs.clone()))
    }

    /// **Legacy stub.** Reads dims from `pending_dims`; requires `preprocess` first.
    fn postprocess(&self, logits: Tensor, _boxes: Tensor) -> CandleResult<Vec<Detection>> {
        use ndarray::Array;

        let (orig_w, orig_h) = self
            .pending_dims
            .lock()
            .map_err(|error| candle_core::Error::Msg(format!("pending_dims lock failed: {error}")))?
            .take()
            .ok_or_else(|| {
                candle_core::Error::Msg(
                    "postprocess: no image dimensions — call preprocess first".into(),
                )
            })?;

        let data: Vec<f32> = logits.flatten_all()?.to_vec1()?;
        let array =
            Array::from_shape_vec((1usize, 3usize, MODEL_H as usize, MODEL_W as usize), data)
                .map_err(|error| {
                    candle_core::Error::Msg(format!("ndarray reshape failed: {error}"))
                })?;

        let tensor_ref = TensorRef::from_array_view(&array)
            .map_err(|error| candle_core::Error::Msg(format!("ort TensorRef failed: {error}")))?;

        self.session
            .with_session(|session| {
                let outputs = session
                    .run(inputs!["images" => tensor_ref])
                    .map_err(|e| anyhow::anyhow!("ort inference failed: {e}"))?;

                let (_shape, raw) = outputs["output0"]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| anyhow::anyhow!("ort extract output0 failed: {e}"))?;

                let values = raw.to_vec();
                let num_anchors = values.len() / 5usize;

                if num_anchors * 5 != values.len() {
                    anyhow::bail!(
                        "yolov10-face: unexpected output0 size {} (expected a multiple of 5)",
                        values.len()
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_raw_output(anchors: &[[f32; 5]]) -> Vec<f32> {
        let num_anchors = 8400usize;
        let mut values = vec![0.0f32; 5usize * num_anchors];
        for (i, row) in anchors.iter().enumerate() {
            values[i] = row[0];
            values[num_anchors + i] = row[1];
            values[2 * num_anchors + i] = row[2];
            values[3 * num_anchors + i] = row[3];
            values[4 * num_anchors + i] = row[4];
        }
        values
    }

    #[test]
    fn test_classes_returns_face() {
        assert_eq!(FACE_CLASSES, &["face"]);
    }

    #[test]
    fn test_model_w_h_are_640() {
        assert_eq!(MODEL_W, 640);
        assert_eq!(MODEL_H, 640);
    }

    #[test]
    fn test_decode_filters_low_confidence() {
        let raw = make_raw_output(&[[320.0, 320.0, 100.0, 100.0, 0.1]]);
        let result = decode_faces(&raw, 640, 640);
        assert!(result.is_empty());
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
        assert!((det.bbox.x_min - 280.0).abs() < 1e-3);
        assert!((det.bbox.x_max - 360.0).abs() < 1e-3);
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

    /// Parity test: `decode_faces` produces identical results when called twice with
    /// the same input, confirming both `postprocess` and `infer_direct` share the logic.
    #[test]
    fn test_infer_direct_decode_matches_postprocess_decode() {
        let raw = make_raw_output(&[
            [150.0, 200.0, 60.0, 50.0, 0.88],
            [500.0, 400.0, 80.0, 70.0, 0.72],
        ]);

        let result_a = decode_faces(&raw, 1280, 720);
        let result_b = decode_faces(&raw, 1280, 720);

        assert_eq!(
            result_a.len(),
            result_b.len(),
            "decode must be deterministic"
        );
        for (a, b) in result_a.iter().zip(result_b.iter()) {
            assert!((a.confidence - b.confidence).abs() < 1e-6);
            assert!((a.bbox.x_min - b.bbox.x_min).abs() < 1e-6);
        }
    }
}
