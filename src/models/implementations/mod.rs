/// Model implementations.
///
/// Active models:
/// - [`YOLOv10Ort`] — YOLOv10n via ONNX Runtime (`Kalray/yolov10` on HF Hub)
pub mod yolov10_ort;

pub use yolov10_ort::YOLOv10Ort;
