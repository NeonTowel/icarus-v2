//! YOLOv10m ONNX Runtime implementation of the [`Model`] trait.
use candle_core::{Device, Result as CandleResult, Tensor};
use image::DynamicImage;

use crate::models::candle_backend::{Detection, Model};

use super::yolov10_common::{YOLOv10OrtInner, YOLOv10VariantConfig};

const VARIANT_CONFIG: YOLOv10VariantConfig = YOLOv10VariantConfig {
    hf_repo: "onnx-community/yolov10m",
    hf_filename: "onnx/model.onnx",
    display_name: "yolov10m",
};

/// YOLOv10m detector backed by ONNX Runtime.
pub struct YOLOv10mOrt {
    inner: YOLOv10OrtInner,
}

impl YOLOv10mOrt {
    /// Download the YOLOv10m ONNX model from HuggingFace Hub and initialise the session.
    pub fn from_hub(device: &Device) -> anyhow::Result<Self> {
        Ok(Self {
            inner: YOLOv10OrtInner::from_hub(&VARIANT_CONFIG, device)?,
        })
    }
}

impl Model for YOLOv10mOrt {
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
