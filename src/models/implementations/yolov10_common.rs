//! Shared helpers for YOLOv10 ONNX Runtime model variants.
//!
//! All YOLOv10 variants in this crate use the same preprocessing, ONNX Runtime
//! session setup, and output decoding. The only difference is the HuggingFace Hub
//! repository that provides `onnx/model.onnx`.
//!
//! ## M6 — Candle round-trip removal
//!
//! The original path did `DynamicImage → candle Tensor → Vec<f32> → ndarray → ORT`.
//! [`preprocess_to_nchw_array`] builds the ndarray directly, skipping candle entirely
//! on the hot path. [`YOLOv10OrtInner::infer_direct`] collapses all three trait steps
//! (`preprocess → forward → postprocess`) into one function so that original image
//! dimensions never leave the call stack, eliminating the old `thread_local!` side-channel.
//!
//! The legacy `preprocess` / `forward` / `postprocess` methods remain as stubs that
//! delegate via a `Mutex<Option<(u32, u32)>>` struct field. They are not used by the
//! normal batch pipeline (which calls [`Model::infer`] → [`infer_direct`] instead) but
//! are retained so the trait contract is fulfillable by any code that calls them directly.
//!
//! ## Fix 12 — Per-thread SessionPool (Strategy B)
//!
//! Replaces the single `Mutex<Session>` with a [`SessionPool`] so each Rayon worker
//! thread can hold its own ORT session, eliminating serialised inference. The pool
//! falls back to a shared locked session when the RAM-derived cap is reached, so
//! degradation is graceful. `intra_op_threads` is sized to `cores / min(threads, cap)`
//! to prevent CPU oversubscription.
use candle_core::{DType, Device, Result as CandleResult, Tensor};
use hf_hub::api::sync::Api;
use image::{imageops::FilterType, DynamicImage};
use ndarray::Array;
use ort::{inputs, value::TensorRef};
use std::sync::Mutex;

use crate::models::candle_backend::{preprocess_to_nchw_array, BBox, Detection, COCO_CLASSES};
use crate::models::session_pool::{SessionPool, YOLOV10_FOOTPRINT_MB};

/// Model input width and height in pixels.
pub(super) const MODEL_W: u32 = 640;
pub(super) const MODEL_H: u32 = 640;

/// Configuration for a specific YOLOv10 model variant.
pub(super) struct YOLOv10VariantConfig {
    /// HuggingFace Hub repository ID, such as `onnx-community/yolov10n`.
    pub(super) hf_repo: &'static str,
    /// Filename of the ONNX model inside the repository.
    pub(super) hf_filename: &'static str,
    /// Human-readable variant name for logs and debugging.
    pub(super) display_name: &'static str,
}

/// Shared inference state for any YOLOv10 ONNX Runtime variant.
///
/// Holds a [`SessionPool`] (one ORT session per Rayon worker, bounded by RAM)
/// and a legacy `pending_dims` mutex for the `preprocess → postprocess` stub path.
pub(super) struct YOLOv10OrtInner {
    /// Per-worker session pool (Fix 12 / Strategy B).
    session: SessionPool,
    /// Stores image dimensions for the legacy `preprocess → postprocess` stub path.
    ///
    /// **Not used by the hot path**: [`infer_direct`] keeps dimensions on the stack.
    /// This field exists solely so that callers that invoke `preprocess` + `postprocess`
    /// directly (bypassing [`Model::infer`]) still get correct output.
    pending_dims: Mutex<Option<(u32, u32)>>,
}

impl YOLOv10OrtInner {
    /// Download the ONNX model from HuggingFace Hub and create a session pool.
    ///
    /// # Parameters
    /// - `thread_count`: Active Rayon thread count; used to tune `intra_op_threads`.
    pub fn from_hub(
        config: &YOLOv10VariantConfig,
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
            YOLOV10_FOOTPRINT_MB,
            1, // single detection model
            thread_count,
        )
        .map_err(|e| anyhow::anyhow!("YOLOv10 session pool: {e}"))?;

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
                    .map_err(|e| anyhow::anyhow!("yolov10 decode failed: {e}"))
            })
            .map_err(|e| candle_core::Error::Msg(e.to_string()))
    }

    /// One-shot inference: build NCHW array directly, run ORT, decode — all in one call.
    ///
    /// This is the **hot path** used by [`YOLOv10Ort`] / variants through the
    /// [`crate::models::candle_backend::Model::infer`] override.
    ///
    /// Compared to the legacy `preprocess → forward → postprocess` sequence:
    /// - No Candle tensor is allocated (the ndarray is built directly from pixel bytes).
    /// - Original `(width, height)` stays on the call stack — no `thread_local!` or
    ///   shared-mutable side-channel is needed.
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
                    .map_err(|e| anyhow::anyhow!("yolov10 decode failed: {e}"))
            })
            .map_err(|e| candle_core::Error::Msg(e.to_string()))
    }

    /// Return the COCO class names used by all YOLOv10 variants.
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
            "yolov10: unexpected output0 size {} (expected {})",
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

    #[test]
    fn test_model_input_size_is_640x640() {
        assert_eq!(MODEL_W, 640);
        assert_eq!(MODEL_H, 640);
    }

    #[test]
    fn test_variant_config_nano() {
        let config = YOLOv10VariantConfig {
            hf_repo: "onnx-community/yolov10n",
            hf_filename: "onnx/model.onnx",
            display_name: "yolov10n",
        };

        assert_eq!(config.hf_repo, "onnx-community/yolov10n");
        assert_eq!(config.hf_filename, "onnx/model.onnx");
        assert_eq!(config.display_name, "yolov10n");
    }

    #[test]
    fn test_variant_config_small() {
        let config = YOLOv10VariantConfig {
            hf_repo: "onnx-community/yolov10s",
            hf_filename: "onnx/model.onnx",
            display_name: "yolov10s",
        };

        assert_eq!(config.hf_repo, "onnx-community/yolov10s");
    }

    #[test]
    fn test_variant_config_medium() {
        let config = YOLOv10VariantConfig {
            hf_repo: "onnx-community/yolov10m",
            hf_filename: "onnx/model.onnx",
            display_name: "yolov10m",
        };

        assert_eq!(config.hf_repo, "onnx-community/yolov10m");
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
        values[6] = 1.0;
        values[7] = 2.0;
        values[8] = 3.0;
        values[9] = 4.0;
        values[10] = 0.0;
        values[11] = 0.0;

        let detections = decode_output0(values, 1280, 720).expect("decoding should succeed");

        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].confidence, 0.75);
        assert_eq!(detections[0].class_id, 0);
        assert_eq!(detections[0].class_name, "person");
    }

    /// Verify that the shared `preprocess_to_nchw_array` (now in `candle_backend`) produces
    /// byte-for-byte identical values to the legacy Candle round-trip.
    ///
    /// This is the primary parity gate for M1: if this test passes, the direct-ndarray
    /// hot path is numerically equivalent to the original code and detection results
    /// will be identical.
    #[test]
    fn test_direct_preprocess_matches_candle_preprocess() {
        let img = image::DynamicImage::ImageRgb8(image::ImageBuffer::from_fn(100, 75, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 128u8])
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

        // --- Shared helper from candle_backend (M1 hot path) ---
        let direct_array = preprocess_to_nchw_array(&img, MODEL_W, MODEL_H);
        let direct_data: Vec<f32> = direct_array.iter().copied().collect();

        assert_eq!(
            candle_data.len(),
            direct_data.len(),
            "output lengths must match"
        );

        let max_diff = candle_data
            .iter()
            .zip(direct_data.iter())
            .map(|(c, d)| (c - d).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-6,
            "max pixel-value difference {max_diff} exceeds tolerance: \
             shared preprocess_to_nchw_array must be numerically equivalent to candle round-trip"
        );
    }

    /// Verify that `preprocess_to_nchw_array` always produces the expected NCHW shape.
    #[test]
    fn test_preprocess_to_nchw_array_shape() {
        let img = image::DynamicImage::new_rgb8(320, 240);
        let arr = preprocess_to_nchw_array(&img, MODEL_W, MODEL_H);
        assert_eq!(
            arr.shape(),
            &[1, 3, MODEL_H as usize, MODEL_W as usize],
            "preprocess_to_nchw_array must return [N=1, C=3, H=640, W=640]"
        );
    }

    /// Verify that all normalised pixel values are in `[0.0, 1.0]`.
    #[test]
    fn test_preprocess_to_nchw_array_values_in_unit_range() {
        let img = image::DynamicImage::ImageRgb8(image::ImageBuffer::from_fn(64, 64, |_, _| {
            image::Rgb([0u8, 128u8, 255u8])
        }));
        let arr = preprocess_to_nchw_array(&img, MODEL_W, MODEL_H);
        for &v in arr.iter() {
            assert!(
                (0.0..=1.0).contains(&v),
                "pixel value {v} is outside [0.0, 1.0]"
            );
        }
    }
}
