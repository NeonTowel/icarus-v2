//! YOLO26m ONNX Runtime implementation of the [`Model`] trait.
use candle_core::{Device, Result as CandleResult, Tensor};
use image::DynamicImage;

use crate::models::candle_backend::{Detection, Model};

use super::yolo26_common::{YOLOv26OrtInner, YOLOv26VariantConfig};

const VARIANT_CONFIG: YOLOv26VariantConfig = YOLOv26VariantConfig {
    hf_repo: "zwh20081/yolo26-onnx",
    hf_filename: "yolo26m.onnx",
    display_name: "YOLO26m",
};

/// YOLO26m detector backed by ONNX Runtime.
pub struct YOLOv26mOrt {
    inner: YOLOv26OrtInner,
}

impl YOLOv26mOrt {
    /// Download the YOLO26m ONNX model from HuggingFace Hub and initialise the session pool.
    pub fn from_hub(device: &Device, thread_count: usize) -> anyhow::Result<Self> {
        Ok(Self {
            inner: YOLOv26OrtInner::from_hub(&VARIANT_CONFIG, device, thread_count)?,
        })
    }
}

impl Model for YOLOv26mOrt {
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

    /// M1 hot path: direct ndarray inference, no Candle tensor round-trip.
    fn infer(&self, image: &DynamicImage) -> CandleResult<Vec<Detection>> {
        self.inner.infer_direct(image)
    }
}
