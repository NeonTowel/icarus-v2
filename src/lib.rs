/// Icarus-v2: Multi-Model AI Image Cropping System
///
/// A production-ready Rust system for detecting and cropping objects in images
/// using multiple AI models (DETR, YOLO, DFINE, RF-DETR).
pub mod batch_processor;
pub mod config;
pub mod crop;
pub mod detections_json;
pub mod directory_walker;
pub mod early_filter;
pub mod error;
pub mod face_aware_cropping;
pub mod face_detection;
pub mod focal_point;
pub mod image_io;
pub mod image_utils;
pub mod metrics;
pub mod models;
pub mod multi_format_cropping;
pub mod output_sorting;
pub mod review;
pub mod visualization;

pub use error::{Error, Result};
