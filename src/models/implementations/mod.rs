/// Model implementations.
///
/// Active models:
/// - [`YOLOv10Ort`] — YOLOv10n via ONNX Runtime (`onnx-community/yolov10n` on HF Hub)
/// - [`YoloV11xFaceOrt`] — YOLOv11x-Face via ONNX Runtime (`AdamCodd/YOLOv11x-face-detection` on HF Hub)
pub mod yolov10_ort;
pub mod yolov11x_face_ort;

pub use yolov10_ort::YOLOv10Ort;
pub use yolov11x_face_ort::YoloV11xFaceOrt;
