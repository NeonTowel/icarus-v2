//! DINOv2 Backbone (Deferred to v3.0)
//!
//! RF-DETR requires DINOv2 which in turn needs deformable attention — a primitive
//! not yet available in `candle-transformers` 0.9.  Deferring to v3.0 when either:
//! 1. Candle adds deformable attention primitives, OR
//! 2. An alternative CPU-optimised RF-DETR variant becomes available.
//!
//! At that point replace this stub with:
//!   `use candle_transformers::models::dinov2::*;`
//!
//! See: https://github.com/roboflow/rf-detr

/// DINOv2-medium placeholder (ViT-S/14 derivative, 256-dim).
pub struct DINOv2Medium;

/// DINOv2-large placeholder (ViT-L/14 derivative, 768-dim).
pub struct DINOv2Large;

impl DINOv2Medium {
    pub fn _placeholder() {
        // Reserved for v3.0 implementation
    }
}

impl DINOv2Large {
    pub fn _placeholder() {
        // Reserved for v3.0 implementation
    }
}
