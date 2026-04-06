//! Shared helpers for YOLOv10 ONNX Runtime model variants.
//!
//! All YOLOv10 variants in this crate use the same preprocessing, ONNX Runtime
//! session setup, and output decoding. The only difference is the HuggingFace Hub
//! repository that provides `onnx/model.onnx`.
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

thread_local! {
    static YOLOV10_PENDING_DIMS: RefCell<Option<(u32, u32)>> = const { RefCell::new(None) };
}

use crate::models::candle_backend::{BBox, Detection, COCO_CLASSES};

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
pub(super) struct YOLOv10OrtInner {
    session: Mutex<Session>,
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

        let session = Session::builder()
            .map_err(|error| anyhow::anyhow!("ort Session builder failed: {error}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|error| anyhow::anyhow!("ort optimisation level failed: {error}"))?
            .commit_from_file(&model_path)
            .map_err(|error| anyhow::anyhow!("ort session load failed: {error}"))?;

        Ok(Self {
            session: Mutex::new(session),
        })
    }

    /// Resize an image to the model input size and normalise pixels to `[0, 1]`.
    pub fn preprocess(&self, images: &[DynamicImage]) -> CandleResult<Tensor> {
        let img = images
            .first()
            .ok_or_else(|| candle_core::Error::Msg("preprocess: empty image slice".into()))?;

        YOLOV10_PENDING_DIMS.with(|pending_dims| {
            *pending_dims.borrow_mut() = Some((img.width(), img.height()));
        });

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
    pub fn postprocess(&self, logits: Tensor, _boxes: Tensor) -> CandleResult<Vec<Detection>> {
        let (orig_w, orig_h) = YOLOV10_PENDING_DIMS
            .with(|pending_dims| pending_dims.borrow_mut().take())
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
}
