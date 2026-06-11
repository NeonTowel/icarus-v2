//! Shared helpers for YOLO26 ONNX Runtime model variants.
//!
//! YOLO26 uses the same end-to-end `[1, 300, 6]` detection format as YOLOv10,
//! so preprocessing and decoding mirror the YOLOv10 implementation while keeping
//! the YOLO26-specific HuggingFace repository and variant metadata isolated here.
//!
//! ## M1 — infer_direct / side-channel removal
//!
//! The original path used a `thread_local! YOLO26_PENDING_DIMS` to pass image
//! dimensions from `preprocess` to `postprocess`. That side-channel is removed.
//! [`YOLOv26OrtInner::infer_direct`] collapses all three trait steps into one
//! function: dimensions stay on the call stack, preprocessing uses the shared
//! [`crate::models::candle_backend::preprocess_to_nchw_array`] helper. The legacy
//! `preprocess`/`forward`/`postprocess` stubs now store dims in a
//! `Mutex<Option<(u32, u32)>>` field, matching the YOLOv10 pattern.
//!
//! ## Fix 12 — Per-thread SessionPool (Strategy B)
//!
//! Replaces the single `Mutex<Session>` with a [`SessionPool`] so each Rayon worker
//! thread gets its own ORT session, eliminating mutex contention on the person
//! detection stage.
use candle_core::{Device, Result as CandleResult, Tensor};
use hf_hub::api::sync::Api;
use image::DynamicImage;
use ndarray::Array;
use ort::{inputs, value::TensorRef};
use std::sync::Mutex;

use crate::models::candle_backend::{preprocess_to_nchw_array, BBox, Detection, COCO_CLASSES};
use crate::models::session_pool::{SessionPool, YOLO26_FOOTPRINT_MB};

/// Model input width and height in pixels.
pub(super) const MODEL_W: u32 = 640;
pub(super) const MODEL_H: u32 = 640;

/// Configuration for a specific YOLO26 model variant.
pub(super) struct YOLOv26VariantConfig {
    /// HuggingFace Hub repository ID, always `zwh20081/yolo26-onnx`.
    pub(super) hf_repo: &'static str,
    /// Filename of the ONNX model inside the repository.
    pub(super) hf_filename: &'static str,
    /// Human-readable variant name for logs and debugging.
    pub(super) display_name: &'static str,
}

/// Shared inference state for any YOLO26 ONNX Runtime variant.
pub(super) struct YOLOv26OrtInner {
    /// Per-worker session pool (Fix 12 / Strategy B).
    session: SessionPool,
    /// Stores image dimensions for the legacy `preprocess → postprocess` stub path.
    ///
    /// **Not used by the hot path**: [`infer_direct`] keeps dimensions on the stack.
    pending_dims: Mutex<Option<(u32, u32)>>,
}

impl YOLOv26OrtInner {
    /// Download the ONNX model from HuggingFace Hub and create a session pool.
    ///
    /// # Parameters
    /// - `thread_count`: Active Rayon thread count; used to tune `intra_op_threads`.
    pub fn from_hub(
        config: &YOLOv26VariantConfig,
        _device: &Device,
        thread_count: usize,
    ) -> anyhow::Result<Self> {
        let api = Api::new()?;
        let repo = api.model(config.hf_repo.to_string());
        let model_path = repo.get(config.hf_filename)?;

        log::info!(
            "{}: loading ONNX model from {:?}",
            config.display_name,
            model_path
        );

        let session = SessionPool::new(
            model_path,
            config.display_name,
            YOLO26_FOOTPRINT_MB,
            1, // single detection model
            thread_count,
        )
        .map_err(|e| anyhow::anyhow!("YOLO26 session pool: {e}"))?;

        Ok(Self {
            session,
            pending_dims: Mutex::new(None),
        })
    }

    /// Resize an image to the model input size and normalise pixels to `[0, 1]`.
    ///
    /// **Legacy stub.** The hot path uses [`infer_direct`] instead. This method stores
    /// original dimensions in `pending_dims` for a subsequent [`postprocess`] call.
    pub fn preprocess(&self, images: &[DynamicImage]) -> CandleResult<Tensor> {
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

        let tensor = Tensor::from_vec(data, (MODEL_H as usize, MODEL_W as usize, 3), &Device::Cpu)?
            .permute((2, 0, 1))?;
        let tensor = (tensor.to_dtype(DType::F32)? * (1.0 / 255.0))?;
        tensor.unsqueeze(0)
    }

    /// Pass the preprocessed tensor through unchanged.
    pub fn forward(&self, xs: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        Ok((xs.clone(), xs.clone()))
    }

    /// Run ONNX Runtime inference and decode detections.
    ///
    /// **Legacy stub.** The hot path uses [`infer_direct`] instead. This method reads
    /// dimensions from `pending_dims` (set by [`preprocess`]).
    pub fn postprocess(&self, logits: Tensor, _boxes: Tensor) -> CandleResult<Vec<Detection>> {
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

                decode_output0(raw.to_vec(), orig_w, orig_h)
                    .map_err(|e| anyhow::anyhow!("yolo26 decode failed: {e}"))
            })
            .map_err(|e| candle_core::Error::Msg(e.to_string()))
    }

    /// One-shot inference: build NCHW array directly, run ORT, decode — all in one call.
    ///
    /// This is the **hot path** used by YOLO26n/s/m through the
    /// [`crate::models::candle_backend::Model::infer`] override.
    ///
    /// Compared to the legacy `preprocess → forward → postprocess` sequence:
    /// - No Candle tensor is allocated (the ndarray is built via the shared helper).
    /// - Original `(width, height)` stays on the call stack — no Mutex side-channel needed.
    /// - The session pool is consulted once; the calling thread's session is reused.
    ///
    /// # Errors
    /// Returns `Err` if the ORT session fails or the model output has an unexpected shape.
    pub(super) fn infer_direct(&self, img: &DynamicImage) -> CandleResult<Vec<Detection>> {
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

                decode_output0(raw.to_vec(), orig_w, orig_h)
                    .map_err(|e| anyhow::anyhow!("yolo26 decode failed: {e}"))
            })
            .map_err(|e| candle_core::Error::Msg(e.to_string()))
    }

    /// Return the COCO class names used by all YOLO26 variants.
    pub fn classes(&self) -> &[&str] {
        &COCO_CLASSES
    }

    /// Return the model input size in pixels.
    pub fn input_size(&self) -> (usize, usize) {
        (MODEL_W as usize, MODEL_H as usize)
    }
}

fn decode_output0(values: Vec<f32>, orig_w: u32, orig_h: u32) -> CandleResult<Vec<Detection>> {
    let expected_len = 300usize * 6usize;
    if values.len() != expected_len {
        return Err(candle_core::Error::Msg(format!(
            "yolo26: unexpected output0 size {} (expected {})",
            values.len(),
            expected_len
        )));
    }

    let scale_x = orig_w as f32 / MODEL_W as f32;
    let scale_y = orig_h as f32 / MODEL_H as f32;

    let detections = values
        .chunks_exact(6)
        .filter_map(|row| {
            let (x1, y1, x2, y2, score, class_raw) =
                (row[0], row[1], row[2], row[3], row[4], row[5]);

            if score <= 0.0 || !(0.0..=1.0).contains(&score) {
                return None;
            }

            let class_id = class_raw as usize;
            if class_id >= COCO_CLASSES.len() {
                return None;
            }

            Some(Detection {
                bbox: BBox {
                    x_min: (x1 * scale_x).max(0.0),
                    y_min: (y1 * scale_y).max(0.0),
                    x_max: (x2 * scale_x).min(orig_w as f32),
                    y_max: (y2 * scale_y).min(orig_h as f32),
                },
                class_id,
                confidence: score,
                class_name: COCO_CLASSES[class_id].to_string(),
            })
        })
        .collect();

    Ok(detections)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use image::imageops::FilterType;

    #[test]
    fn test_model_input_size_is_640x640() {
        assert_eq!(MODEL_W, 640);
        assert_eq!(MODEL_H, 640);
    }

    #[test]
    fn test_variant_config_yolo26n() {
        let config = YOLOv26VariantConfig {
            hf_repo: "zwh20081/yolo26-onnx",
            hf_filename: "yolo26n.onnx",
            display_name: "YOLO26n",
        };

        assert_eq!(config.hf_repo, "zwh20081/yolo26-onnx");
        assert_eq!(config.hf_filename, "yolo26n.onnx");
        assert_eq!(config.display_name, "YOLO26n");
    }

    #[test]
    fn test_variant_config_yolo26s() {
        let config = YOLOv26VariantConfig {
            hf_repo: "zwh20081/yolo26-onnx",
            hf_filename: "yolo26s.onnx",
            display_name: "YOLO26s",
        };

        assert_eq!(config.hf_filename, "yolo26s.onnx");
    }

    #[test]
    fn test_variant_config_yolo26m() {
        let config = YOLOv26VariantConfig {
            hf_repo: "zwh20081/yolo26-onnx",
            hf_filename: "yolo26m.onnx",
            display_name: "YOLO26m",
        };

        assert_eq!(config.hf_filename, "yolo26m.onnx");
    }

    #[test]
    fn test_decode_output0_filters_padding_rows() {
        let mut values = vec![0.0f32; 300usize * 6usize];
        values[0] = 10.0;
        values[1] = 20.0;
        values[2] = 30.0;
        values[3] = 40.0;
        values[4] = 0.75;
        values[5] = 0.0;

        let detections = decode_output0(values, 1280, 720).expect("decoding should succeed");

        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].confidence, 0.75);
        assert_eq!(detections[0].class_id, 0);
        assert_eq!(detections[0].class_name, "person");
    }

    #[test]
    fn test_decode_output0_scales_to_original_dims() {
        let mut values = vec![0.0f32; 300usize * 6usize];
        values[0] = 320.0;
        values[1] = 160.0;
        values[2] = 960.0;
        values[3] = 480.0;
        values[4] = 0.9;
        values[5] = 0.0;

        let detections = decode_output0(values, 1280, 720).expect("decoding should succeed");
        let detection = &detections[0];

        assert!((detection.bbox.x_min - 640.0).abs() < 1e-3);
        assert!((detection.bbox.y_min - 180.0).abs() < 1e-3);
        assert!((detection.bbox.x_max - 1280.0).abs() < 1e-3);
        assert!((detection.bbox.y_max - 540.0).abs() < 1e-3);
    }

    /// Parity test: `preprocess_to_nchw_array` (shared helper) produces byte-for-byte
    /// identical values to the Candle round-trip for YOLO26 input dimensions.
    ///
    /// This guards M1: if this passes, YOLO26's `infer_direct` is numerically equivalent
    /// to the old `preprocess → forward → postprocess` path.
    #[test]
    fn test_direct_preprocess_matches_candle_preprocess() {
        let img = image::DynamicImage::ImageRgb8(image::ImageBuffer::from_fn(100, 75, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 200u8])
        }));

        // --- Candle round-trip (legacy path) ---
        let resized = img.resize_exact(MODEL_W, MODEL_H, FilterType::Nearest);
        let rgb = resized.to_rgb8();
        let raw_u8: Vec<u8> = rgb.into_raw();
        let t = Tensor::from_vec(
            raw_u8,
            (MODEL_H as usize, MODEL_W as usize, 3),
            &Device::Cpu,
        )
        .unwrap()
        .permute((2, 0, 1))
        .unwrap();
        let t = (t.to_dtype(DType::F32).unwrap() * (1.0f64 / 255.0)).unwrap();
        let t = t.unsqueeze(0).unwrap();
        let candle_data: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();

        // --- Shared helper (M1 hot path) ---
        let direct_arr = preprocess_to_nchw_array(&img, MODEL_W, MODEL_H);
        let direct_data: Vec<f32> = direct_arr.iter().copied().collect();

        assert_eq!(candle_data.len(), direct_data.len(), "lengths must match");

        let max_diff = candle_data
            .iter()
            .zip(direct_data.iter())
            .map(|(c, d)| (c - d).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-6,
            "max diff {max_diff}: YOLO26 shared preprocess must equal Candle round-trip"
        );
    }
}
