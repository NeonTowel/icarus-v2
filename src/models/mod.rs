//! Models module: inference backend and model implementations.
//!
//! Active models:
//! - `yolov10` — YOLOv10n via ONNX Runtime (default person detection)
//! - `yolov10s` — YOLOv10s via ONNX Runtime (higher-accuracy person detection)
//! - `yolov10m` — YOLOv10m via ONNX Runtime (highest-accuracy person detection)
//! - `yolo26` — YOLO26n via ONNX Runtime (fastest person detection)
//! - `yolo26s` — YOLO26s via ONNX Runtime (balanced person detection)
//! - `yolo26m` — YOLO26m via ONNX Runtime (highest-accuracy YOLO26 variant)
//! - `yolov11x-face` — YOLOv11x-Face via ONNX Runtime (stage-2 face detection)
///
/// All models implement the [`Model`] trait (preprocess → forward → postprocess).
pub mod backbones;
pub mod candle_backend;
pub mod implementations;

pub use candle_backend::{BBox, Detection, ImageClassifier, Model};
pub use implementations::{
    eva2_classifier::FreepikEva02, YOLOv10Ort, YOLOv10mOrt, YOLOv10sOrt, YOLOv26mOrt, YOLOv26nOrt,
    YOLOv26sOrt, YoloV10FaceOrt, YoloV11xFaceOrt,
};

use candle_core::Device;

/// Load a detection model by CLI name.
///
/// # Supported names
/// - `"yolov10"` — YOLOv10n via ONNX Runtime (`onnx-community/yolov10n` on HF Hub)
/// - `"yolov10s"` — YOLOv10s via ONNX Runtime (`onnx-community/yolov10s` on HF Hub)
/// - `"yolov10m"` — YOLOv10m via ONNX Runtime (`onnx-community/yolov10m` on HF Hub)
/// - `"yolo26"` — YOLO26n via ONNX Runtime (`zwh20081/yolo26-onnx` on HF Hub)
/// - `"yolo26s"` — YOLO26s via ONNX Runtime (`zwh20081/yolo26-onnx` on HF Hub)
/// - `"yolo26m"` — YOLO26m via ONNX Runtime (`zwh20081/yolo26-onnx` on HF Hub)
/// - `"yolov11x-face"` — YOLOv11x-Face via ONNX Runtime (`AdamCodd/YOLOv11x-face-detection` on HF Hub)
///
/// # Errors
/// Returns `Err` if the model name is unknown or weight download/parsing fails.
pub async fn load_candle_model(
    model_name: &str,
    device: &Device,
) -> anyhow::Result<Box<dyn Model>> {
    let device = device.clone();
    let name = model_name.to_string();
    tokio::task::spawn_blocking(move || -> anyhow::Result<Box<dyn Model>> {
        match name.as_str() {
            "yolov10" => Ok(Box::new(YOLOv10Ort::from_hub(&device)?)),
            "yolov10s" => Ok(Box::new(YOLOv10sOrt::from_hub(&device)?)),
            "yolov10m" => Ok(Box::new(YOLOv10mOrt::from_hub(&device)?)),
            "yolo26" => Ok(Box::new(YOLOv26nOrt::from_hub(&device)?)),
            "yolo26s" => Ok(Box::new(YOLOv26sOrt::from_hub(&device)?)),
            "yolo26m" => Ok(Box::new(YOLOv26mOrt::from_hub(&device)?)),
            "yolov11x-face" => Ok(Box::new(YoloV11xFaceOrt::from_hub(&device)?)),
            unknown => Err(anyhow::anyhow!(
                "Unknown model '{}'. Supported models: yolov10, yolov10s, yolov10m, yolo26, yolo26s, yolo26m, yolov11x-face",
                unknown
            )),
        }
    })
    .await
    .map_err(|e| anyhow::anyhow!("model load task panicked: {e}"))?
}

/// Load the Freepik NSFW image classifier.
pub async fn load_classifier(device: &Device) -> anyhow::Result<Box<dyn ImageClassifier>> {
    let device = device.clone();
    tokio::task::spawn_blocking(move || -> anyhow::Result<Box<dyn ImageClassifier>> {
        Ok(Box::new(FreepikEva02::from_hub(&device)?))
    })
    .await
    .map_err(|e| anyhow::anyhow!("classifier load task panicked: {e}"))?
}
