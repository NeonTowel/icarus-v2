/// Models module: inference backend and model implementations.
///
/// Active models: `yolov10` (YOLOv10n via ONNX Runtime).
///
/// All models implement the [`Model`] trait (preprocess → forward → postprocess).
pub mod backbones;
pub mod candle_backend;
pub mod implementations;

pub use candle_backend::{BBox, Detection, Model};
pub use implementations::YOLOv10Ort;

use candle_core::Device;

/// Load a detection model by CLI name.
///
/// # Supported names
/// - `"yolov10"` — YOLOv10n via ONNX Runtime (`Kalray/yolov10` on HF Hub)
///
/// # Errors
/// Returns `Err` if the model name is unknown or weight download fails.
pub async fn load_candle_model(
    model_name: &str,
    device: &Device,
) -> anyhow::Result<Box<dyn Model>> {
    let device = device.clone();
    let name = model_name.to_string();
    tokio::task::spawn_blocking(move || -> anyhow::Result<Box<dyn Model>> {
        match name.as_str() {
            "yolov10" => Ok(Box::new(YOLOv10Ort::from_hub(&device)?)),
            unknown => Err(anyhow::anyhow!(
                "Unknown model '{}'. Supported models: yolov10",
                unknown
            )),
        }
    })
    .await
    .map_err(|e| anyhow::anyhow!("model load task panicked: {e}"))?
}
