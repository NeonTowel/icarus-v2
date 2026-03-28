/// Models module: inference backend and model implementations.
///
/// Active models:
/// - `yolov10` — YOLOv10n via ONNX Runtime (stage-1 person detection)
/// - `yolov11x-face` — YOLOv11x-Face via ONNX Runtime (stage-2 face detection)
///
/// All models implement the [`Model`] trait (preprocess → forward → postprocess).
pub mod backbones;
pub mod candle_backend;
pub mod implementations;

pub use candle_backend::{BBox, Detection, Model};
pub use implementations::{YOLOv10Ort, YoloV11xFaceOrt};

use candle_core::Device;

/// Load a detection model by CLI name.
///
/// # Supported names
/// - `"yolov10"` — YOLOv10n via ONNX Runtime (`onnx-community/yolov10n` on HF Hub)
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
            "yolov11x-face" => Ok(Box::new(YoloV11xFaceOrt::from_hub(&device)?)),
            unknown => Err(anyhow::anyhow!(
                "Unknown model '{}'. Supported models: yolov10, yolov11x-face",
                unknown
            )),
        }
    })
    .await
    .map_err(|e| anyhow::anyhow!("model load task panicked: {e}"))?
}
