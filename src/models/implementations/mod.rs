/// Model implementations: Individual model wrappers
pub mod detr_resnet101;
pub mod dfine_l;
pub mod rf_detr_large;
pub mod rf_detr_medium;
pub mod yolov9_c;

pub use detr_resnet101::DETRResNet101;
pub use dfine_l::DFineL;
pub use rf_detr_large::RFDETRLarge;
pub use rf_detr_medium::RFDETRMedium;
pub use yolov9_c::YOLOv9c;
