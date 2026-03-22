/// Backbone neural network architectures for Candle-based object detectors.
///
/// # Active
/// ResNet architectures are now delegated entirely to `candle_transformers::models::resnet`.
/// DETR uses `resnet50_no_final_layer` and RT-DETR uses `resnet34_no_final_layer`.
///
/// # Deferred (v3.0)
/// [`dinov2::DINOv2Medium`] and [`dinov2::DINOv2Large`] are stubs pending
/// deformable-attention support in Candle for RF-DETR.
pub mod dinov2;

pub use dinov2::{DINOv2Large, DINOv2Medium};
