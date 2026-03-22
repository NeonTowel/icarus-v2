/// Icarus-v2: Multi-Model AI Image Cropping System
/// 
/// A production-ready Rust system for detecting and cropping objects in images
/// using multiple AI models (DETR, YOLO, DFINE, RF-DETR).

pub mod config;
pub mod error;
pub mod image_utils;
pub mod models;

pub use error::{Error, Result};
