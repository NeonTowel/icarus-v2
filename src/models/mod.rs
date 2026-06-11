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
pub mod ensemble;
pub mod implementations;
pub mod session_pool;

pub use candle_backend::{BBox, Detection, ImageClassifier, Model};
pub use ensemble::{ClassifierKind, SingleWdClassifier, WdEnsembleClassifier};
pub use implementations::{
    eva2_classifier::FreepikEva02, YOLOv10Ort, YOLOv10mOrt, YOLOv10sOrt, YOLOv26mOrt, YOLOv26nOrt,
    YOLOv26sOrt, YoloV10FaceOrt, YoloV11xFaceOrt,
};
pub use session_pool::SessionPool;

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
/// # Parameters
/// - `thread_count`: Active Rayon thread count; used by `SessionPool` to tune
///   `intra_op_threads` so parallel sessions do not oversubscribe the CPU.
///
/// # Errors
/// Returns `Err` if the model name is unknown or weight download/parsing fails.
pub async fn load_candle_model(
    model_name: &str,
    device: &Device,
    thread_count: usize,
) -> anyhow::Result<Box<dyn Model>> {
    let device = device.clone();
    let name = model_name.to_string();
    tokio::task::spawn_blocking(move || -> anyhow::Result<Box<dyn Model>> {
        match name.as_str() {
            "yolov10" => Ok(Box::new(YOLOv10Ort::from_hub(&device, thread_count)?)),
            "yolov10s" => Ok(Box::new(YOLOv10sOrt::from_hub(&device, thread_count)?)),
            "yolov10m" => Ok(Box::new(YOLOv10mOrt::from_hub(&device, thread_count)?)),
            "yolo26" => Ok(Box::new(YOLOv26nOrt::from_hub(&device, thread_count)?)),
            "yolo26s" => Ok(Box::new(YOLOv26sOrt::from_hub(&device, thread_count)?)),
            "yolo26m" => Ok(Box::new(YOLOv26mOrt::from_hub(&device, thread_count)?)),
            "yolov11x-face" => Ok(Box::new(YoloV11xFaceOrt::from_hub(&device, thread_count)?)),
            unknown => Err(anyhow::anyhow!(
                "Unknown model '{}'. Supported models: yolov10, yolov10s, yolov10m, yolo26, yolo26s, yolo26m, yolov11x-face",
                unknown
            )),
        }
    })
    .await
    .map_err(|e| anyhow::anyhow!("model load task panicked: {e}"))?
}

/// Load an image classifier by runtime kind.
///
/// # Parameters
/// - `thread_count`: Active Rayon thread count; passed to `SessionPool` for
///   intra-op thread tuning (S3).
pub async fn load_classifier(
    kind: ClassifierKind,
    device: &Device,
    thread_count: usize,
) -> anyhow::Result<Box<dyn ImageClassifier>> {
    let device = device.clone();
    tokio::task::spawn_blocking(move || -> anyhow::Result<Box<dyn ImageClassifier>> {
        match kind {
            ClassifierKind::Freepik => Ok(Box::new(FreepikEva02::from_hub(&device)?)),
            ClassifierKind::WdEva02 => Ok(Box::new(SingleWdClassifier::wd_eva02(thread_count)?)),
            ClassifierKind::Idolsankaku => {
                Ok(Box::new(SingleWdClassifier::idolsankaku(thread_count)?))
            }
            ClassifierKind::WdSwinv2 => Ok(Box::new(SingleWdClassifier::wd_swinv2(thread_count)?)),
            ClassifierKind::IdolsankakuSwinv2 => Ok(Box::new(
                SingleWdClassifier::idolsankaku_swinv2(thread_count)?,
            )),
            ClassifierKind::WdVit => Ok(Box::new(SingleWdClassifier::wd_vit(thread_count)?)),
            ClassifierKind::WdEnsembleFast => {
                Ok(Box::new(WdEnsembleClassifier::fast(thread_count)?))
            }
            ClassifierKind::WdEnsembleAccurate => {
                Ok(Box::new(WdEnsembleClassifier::accurate(thread_count)?))
            }
        }
    })
    .await
    .map_err(|e| anyhow::anyhow!("classifier load task panicked: {e}"))?
}
