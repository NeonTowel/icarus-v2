/// Model-specific preprocessor implementations
pub mod detr;
pub mod yolo;

pub use detr::DetrPreprocessor;
pub use yolo::YoloPreprocessor;
