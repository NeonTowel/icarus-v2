//! Multi-Format Intelligent Cropping Module — thin re-export shim.
//!
//! The implementation now lives in [`crate::crop`]. This shim preserves all
//! existing import paths (`use crate::multi_format_cropping::*`) without change.
pub use crate::crop::*;
