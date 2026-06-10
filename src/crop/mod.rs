//! Multi-format intelligent cropping.
//!
//! This module is split into focused submodules:
//! - `geometry` — bounding boxes, crop regions, placement math
//! - `pose` — pose type/direction detection and headroom selection
//! - `dedup` — multi-person deduplication and compound bbox
//! - `strategy` — `CropStrategy` trait and per-format implementations

pub mod dedup;
pub mod geometry;
pub mod pose;
pub mod strategy;

// Re-export the full public surface so callers can use `crate::crop::*`
// or `crate::multi_format_cropping::*` (via the re-export shim).
pub use dedup::*;
pub use geometry::*;
pub use pose::*;
pub use strategy::*;
