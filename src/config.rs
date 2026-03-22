/// Configuration types for Icarus-v2
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Main configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Path to ONNX model files
    pub model_path: PathBuf,

    /// Default model to use
    pub default_model: String,

    /// Image processing parameters
    pub image_config: ImageConfig,

    /// Detection parameters
    pub detection_config: DetectionConfig,
}

/// Image processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageConfig {
    /// Maximum image dimension
    pub max_dimension: u32,

    /// Minimum image dimension
    pub min_dimension: u32,

    /// JPEG quality for output
    pub jpeg_quality: u8,
}

/// Detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionConfig {
    /// Confidence threshold for detections
    pub confidence_threshold: f32,

    /// NMS IoU threshold
    pub nms_threshold: f32,

    /// Maximum detections per image
    pub max_detections: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("./models"),
            default_model: "detr-resnet101".to_string(),
            image_config: ImageConfig::default(),
            detection_config: DetectionConfig::default(),
        }
    }
}

impl Default for ImageConfig {
    fn default() -> Self {
        Self {
            max_dimension: 2048,
            min_dimension: 32,
            jpeg_quality: 95,
        }
    }
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.5,
            nms_threshold: 0.5,
            max_detections: 100,
        }
    }
}
