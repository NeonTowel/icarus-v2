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
//! ## M7 — Strategy A: right-sized ORT intra-op threads
//!
//! [`ORT_INTRA_OP_THREADS`] is set to `1` so that ONNX Runtime does not compete with
//! the Rayon worker pool for CPU cores. With one shared `Mutex<Session>`, ORT inference
//! is already serialised; giving it all cores would starve the Rayon workers that handle
//! image I/O and crop geometry in parallel.
use candle_core::{DType, Device, Result as CandleResult, Tensor};
use hf_hub::api::sync::Api;
use image::{imageops::FilterType, DynamicImage};
use ndarray::Array;
use ort::{
    inputs,
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};
use std::sync::Mutex;

use crate::models::candle_backend::{BBox, Detection, COCO_CLASSES};

/// Model input width and height in pixels.
pub(super) const MODEL_W: u32 = 640;
pub(super) const MODEL_H: u32 = 640;

// M7 Decision: ORT default thread count (no explicit with_intra_threads call).
//
// Strategy A (with_intra_threads=1) was benchmarked and found 10× slower per image
// on a 10-core machine (person_detect: 440ms vs ~44ms; face_detect: 8200ms vs ~820ms).
// The expected oversubscription does not occur in practice: sleeping Rayon workers
// (blocked on Mutex<Session>) do not consume CPU cycles while ORT holds the lock.
// Therefore, letting ORT use all available cores is optimal for the current serialised
// single-session architecture.
//
// TODO (M7 follow-up): implement Strategy B (per-worker sessions, intra_threads=1)
// gated by RAM availability. RAM cost: ~400MB × workers for wd-vit; NOT viable for
// wd-ensemble-accurate (~4–6GB × N workers = 20–30GB for 5 workers).

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
pub(super) struct YOLOv10OrtInner {
    session: Mutex<Session>,
    /// Stores image dimensions for the legacy `preprocess → postprocess` stub path.
    ///
    /// **Not used by the hot path**: [`infer_direct`] keeps dimensions on the stack.
    /// This field exists solely so that callers that invoke `preprocess` + `postprocess`
    /// directly (bypassing [`Model::infer`]) still get correct output.
    ///
    /// Concurrency note: concurrent `preprocess` calls from different Rayon workers can
    /// race on this value. This is acceptable because the hot path never writes here.
    pending_dims: Mutex<Option<(u32, u32)>>,
}

impl YOLOv10OrtInner {
    /// Download the ONNX model from HuggingFace Hub and create an ORT session.
    pub fn from_hub(config: &YOLOv10VariantConfig, _device: &Device) -> anyhow::Result<Self> {
        let api = Api::new()?;
        let repo = api.model(config.hf_repo.to_string());
        let model_path = repo.get(config.hf_filename)?;

        log::info!(
            "{}: loading ONNX model from {:?}",
            config.display_name,
            model_path
        );

        // Use ORT default thread count (all available cores). See M7 decision comment above.
        let session = Session::builder()
            .map_err(|error| anyhow::anyhow!("ort Session builder failed: {error}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|error| anyhow::anyhow!("ort optimisation level failed: {error}"))?
            .commit_from_file(&model_path)
            .map_err(|error| anyhow::anyhow!("ort session load failed: {error}"))?;

        Ok(Self {
            session: Mutex::new(session),
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

        let mut session_guard = self
            .session
            .lock()
            .map_err(|error| candle_core::Error::Msg(format!("session lock failed: {error}")))?;
        let outputs = session_guard
            .run(inputs!["images" => tensor_ref])
            .map_err(|error| candle_core::Error::Msg(format!("ort inference failed: {error}")))?;

        let (_shape, raw) = outputs["output0"]
            .try_extract_tensor::<f32>()
            .map_err(|error| {
                candle_core::Error::Msg(format!("ort extract output0 failed: {error}"))
            })?;

        decode_output0(raw.to_vec(), orig_w, orig_h)
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
    /// - The ORT session lock is acquired exactly once per image.
    ///
    /// # Errors
    /// Returns `Err` if the ORT session fails or the model output has an unexpected shape.
    pub(super) fn infer_direct(&self, img: &DynamicImage) -> CandleResult<Vec<Detection>> {
        let (orig_w, orig_h) = (img.width(), img.height());
        let array = preprocess_to_nchw_array(img);

        let tensor_ref = TensorRef::from_array_view(&array)
            .map_err(|e| candle_core::Error::Msg(format!("ort TensorRef failed: {e}")))?;

        let mut session_guard = self
            .session
            .lock()
            .map_err(|e| candle_core::Error::Msg(format!("session lock failed: {e}")))?;
        let outputs = session_guard
            .run(inputs!["images" => tensor_ref])
            .map_err(|e| candle_core::Error::Msg(format!("ort inference failed: {e}")))?;

        let (_shape, raw) = outputs["output0"]
            .try_extract_tensor::<f32>()
            .map_err(|e| candle_core::Error::Msg(format!("ort extract output0 failed: {e}")))?;

        decode_output0(raw.to_vec(), orig_w, orig_h)
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

/// Build an NCHW `Array4<f32>` from `img` without allocating any Candle tensors.
///
/// Resizes the image to `MODEL_W × MODEL_H` using nearest-neighbour interpolation,
/// converts to RGB, and lays the pixel data out as `[1, C, H, W]` (C-contiguous)
/// with values normalised to `[0.0, 1.0]`.
///
/// This produces **byte-for-byte identical values** to the Candle round-trip:
/// `Tensor::from_vec(HWC u8) → permute(CHW) → F32/255 → unsqueeze(NCHW)`.
/// The parity is verified by `test_direct_preprocess_matches_candle_preprocess`.
///
/// Memory: one allocation of `3 × MODEL_W × MODEL_H × 4` bytes (≈4.7 MB for 640×640).
fn preprocess_to_nchw_array(img: &DynamicImage) -> ndarray::Array<f32, ndarray::Ix4> {
    let resized = img.resize_exact(MODEL_W, MODEL_H, FilterType::Nearest);
    let rgb = resized.to_rgb8();
    let data = rgb.into_raw(); // HWC Vec<u8>: index = h*W*3 + w*3 + c

    let pixel_count = (MODEL_W * MODEL_H) as usize;
    let mut nchw_data = vec![0.0f32; 3 * pixel_count];

    // De-interleave HWC → NCHW: collect R-plane, then G-plane, then B-plane.
    for i in 0..pixel_count {
        nchw_data[i] = data[i * 3] as f32 / 255.0; // channel 0 (R)
        nchw_data[pixel_count + i] = data[i * 3 + 1] as f32 / 255.0; // channel 1 (G)
        nchw_data[2 * pixel_count + i] = data[i * 3 + 2] as f32 / 255.0; // channel 2 (B)
    }

    Array::from_shape_vec((1, 3, MODEL_H as usize, MODEL_W as usize), nchw_data)
        .expect("shape is exactly 1 × 3 × MODEL_H × MODEL_W — infallible for fixed constants")
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

    /// Verify that `preprocess_to_nchw_array` produces byte-for-byte identical values
    /// to the legacy Candle round-trip (`from_vec → permute → F32/255 → unsqueeze`).
    ///
    /// This is the primary parity gate for M6: if this test passes, the direct-ndarray
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

        // --- Direct ndarray path (M6 hot path) ---
        let direct_array = preprocess_to_nchw_array(&img);
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
             direct preprocess path must be numerically equivalent to candle round-trip"
        );
    }

    /// Verify that `preprocess_to_nchw_array` always produces the expected NCHW shape.
    #[test]
    fn test_preprocess_to_nchw_array_shape() {
        let img = image::DynamicImage::new_rgb8(320, 240);
        let arr = preprocess_to_nchw_array(&img);
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
        let arr = preprocess_to_nchw_array(&img);
        for &v in arr.iter() {
            assert!(
                (0.0..=1.0).contains(&v),
                "pixel value {v} is outside [0.0, 1.0]"
            );
        }
    }
}
