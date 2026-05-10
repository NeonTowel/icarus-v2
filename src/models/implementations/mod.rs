//! Model implementations.
//!
//! Active models:
//! - [`YOLOv10Ort`] — YOLOv10n via ONNX Runtime (`onnx-community/yolov10n` on HF Hub)
//! - [`YOLOv10sOrt`] — YOLOv10s via ONNX Runtime (`onnx-community/yolov10s` on HF Hub)
//! - [`YOLOv10mOrt`] — YOLOv10m via ONNX Runtime (`onnx-community/yolov10m` on HF Hub)
//! - [`YOLOv26nOrt`] — YOLO26n via ONNX Runtime (`zwh20081/yolo26-onnx` on HF Hub)
//! - [`YOLOv26sOrt`] — YOLO26s via ONNX Runtime (`zwh20081/yolo26-onnx` on HF Hub)
//! - [`YOLOv26mOrt`] — YOLO26m via ONNX Runtime (`zwh20081/yolo26-onnx` on HF Hub)
//! - [`YoloV10FaceOrt`] — YOLOv10-face via ONNX Runtime (`deepghs/yolo-face` on HF Hub)
//! - [`YoloV11xFaceOrt`] — YOLOv11x-Face via ONNX Runtime (`AdamCodd/YOLOv11x-face-detection` on HF Hub)
pub mod eva2_classifier;
pub mod yolo26_common;
pub mod yolo26m_ort;
pub mod yolo26n_ort;
pub mod yolo26s_ort;
pub mod yolov10_common;
pub mod yolov10_face_ort;
pub mod yolov10_ort;
pub mod yolov10m_ort;
pub mod yolov10s_ort;
pub mod yolov11x_face_ort;

pub use yolo26m_ort::YOLOv26mOrt;
pub use yolo26n_ort::YOLOv26nOrt;
pub use yolo26s_ort::YOLOv26sOrt;
pub use yolov10_face_ort::YoloV10FaceOrt;
pub use yolov10_ort::YOLOv10Ort;
pub use yolov10m_ort::YOLOv10mOrt;
pub use yolov10s_ort::YOLOv10sOrt;
pub use yolov11x_face_ort::YoloV11xFaceOrt;

use candle_core::Device;
use image::DynamicImage;

use crate::models::Model;

macro_rules! define_candle_wrapper {
    ($name:ident) => {
        pub struct $name {
            inner: YOLOv10Ort,
        }

        impl $name {
            pub fn from_file(path: &str, device: &Device) -> anyhow::Result<Self> {
                if !std::path::Path::new(path).exists() {
                    anyhow::bail!("model file not found: {}", path);
                }

                Self::from_hub(device)
            }

            pub fn from_hub(device: &Device) -> anyhow::Result<Self> {
                Ok(Self {
                    inner: YOLOv10Ort::from_hub(device)?,
                })
            }

            pub fn detect_image(
                &self,
                image: &DynamicImage,
            ) -> anyhow::Result<Vec<crate::models::Detection>> {
                let tensor = self.inner.preprocess(std::slice::from_ref(image))?;
                let (logits, boxes) = self.inner.forward(&tensor)?;
                Ok(self.inner.postprocess(logits, boxes)?)
            }
        }
    };
}

define_candle_wrapper!(DetrCandle);
define_candle_wrapper!(RtDetrCandle);
define_candle_wrapper!(YOLOv9cCandle);

macro_rules! define_deferred_model {
    ($name:ident) => {
        pub struct $name;

        impl $name {
            pub fn from_hub(_device: &Device) -> anyhow::Result<Self> {
                anyhow::bail!(
                    "{} is deferred to v3.0 (deformable attention not yet implemented)",
                    stringify!($name)
                );
            }
        }
    };
}

define_deferred_model!(RfDetrMedium);
define_deferred_model!(RfDetrLarge);
