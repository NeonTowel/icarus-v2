//! YOLOv10 ONNX Runtime implementation of the [`Model`] trait.
//!
//! This is the default YOLOv10n (nano) detector. The shared preprocessing, session
//! setup, and postprocessing live in `yolov10_common.rs`; this file is a thin wrapper.
use candle_core::{Device, Result as CandleResult, Tensor};
use image::DynamicImage;

use crate::models::candle_backend::{Detection, Model};

use super::yolov10_common::{YOLOv10OrtInner, YOLOv10VariantConfig};

/// Nano-variant configuration.
const VARIANT_CONFIG: YOLOv10VariantConfig = YOLOv10VariantConfig {
    hf_repo: "onnx-community/yolov10n",
    hf_filename: "onnx/model.onnx",
    display_name: "yolov10n",
};

/// YOLOv10n detector backed by ONNX Runtime.
///
/// # Example
/// ```rust,ignore
/// let device = candle_core::Device::Cpu;
/// let model = YOLOv10Ort::from_hub(&device)?;
/// // Hot path: one-shot inference (no Candle tensor round-trip)
/// let detections = model.infer(&img)?;
/// ```
pub struct YOLOv10Ort {
    inner: YOLOv10OrtInner,
}

impl YOLOv10Ort {
    /// Download the YOLOv10n ONNX model from HuggingFace Hub and initialise the session pool.
    ///
    /// # Parameters
    /// - `thread_count`: Active Rayon thread count; forwarded to `SessionPool` for
    ///   intra-op thread tuning (S3).
    pub fn from_hub(device: &Device, thread_count: usize) -> anyhow::Result<Self> {
        Ok(Self {
            inner: YOLOv10OrtInner::from_hub(&VARIANT_CONFIG, device, thread_count)?,
        })
    }
}

impl Model for YOLOv10Ort {
    fn preprocess(&self, images: &[DynamicImage]) -> CandleResult<Tensor> {
        self.inner.preprocess(images)
    }

    fn forward(&self, xs: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        self.inner.forward(xs)
    }

    fn postprocess(&self, logits: Tensor, boxes: Tensor) -> CandleResult<Vec<Detection>> {
        self.inner.postprocess(logits, boxes)
    }

    fn classes(&self) -> &[&str] {
        self.inner.classes()
    }

    fn input_size(&self) -> (usize, usize) {
        self.inner.input_size()
    }

    /// M6 hot path: direct ndarray inference, no Candle tensor round-trip.
    fn infer(&self, image: &DynamicImage) -> CandleResult<Vec<Detection>> {
        self.inner.infer_direct(image)
    }
}
