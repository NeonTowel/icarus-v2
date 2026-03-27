/// Configuration types for Icarus-v2
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

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

// ---------------------------------------------------------------------------
// CropConfig — intelligent cropping parameters with YAML deserialization
// ---------------------------------------------------------------------------

fn default_headroom_ratio() -> f32 {
    0.40
}

fn default_visibility_threshold() -> f32 {
    0.50
}

/// Runtime-configurable parameters that govern the intelligent multi-format cropping algorithm.
///
/// Instances can be deserialized from a YAML file via [`load_crop_config`], or constructed
/// programmatically with `CropConfig::default()` for sensible defaults.
///
/// CLI flags (`--headroom-ratio`, `--visibility-threshold`) take precedence over YAML values.
///
/// # Fields
/// - `headroom_ratio`: Fraction of crop height above the bbox center. Default `0.40` (40%).
/// - `visibility_threshold`: Minimum visible fraction required for a format to be valid.
///   Stored internally as a ratio in `[0.0, 1.0]`. Default `0.50` (50%).
///
/// # YAML example
/// ```yaml
/// headroom_ratio: 0.40
/// visibility_threshold: 0.50
/// ```
///
/// # Future fields (Phase 2 — not yet implemented)
/// - `horizontal_offset_percent`: Shift crop center horizontally by a fixed percentage.
/// - `crop_scale_factor`: Scale the crop region before clamping to photo bounds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CropConfig {
    /// Fraction of the crop height allocated above the bbox center (headroom).
    ///
    /// `0.40` → bbox center sits at 40% from the top; 60% footroom below.
    /// Valid range: `[0.0, 1.0]`. Default: `0.40`.
    #[serde(default = "default_headroom_ratio")]
    pub headroom_ratio: f32,

    /// Minimum fraction of the person's bbox area that must fall inside the proposed crop.
    ///
    /// Stored as a ratio `[0.0, 1.0]`. The CLI accepts this as a percentage (0–100)
    /// and converts before storing here. Default: `0.50` (50%).
    #[serde(default = "default_visibility_threshold")]
    pub visibility_threshold: f32,
    // TODO(Phase 2): horizontal_offset_percent: f32  — shift crop center horizontally
    // TODO(Phase 2): crop_scale_factor: f32           — scale crop before clamping
}

impl Default for CropConfig {
    fn default() -> Self {
        Self {
            headroom_ratio: default_headroom_ratio(),
            visibility_threshold: default_visibility_threshold(),
        }
    }
}

/// Load a [`CropConfig`] from a YAML file on disk.
///
/// Missing fields fall back to their serde defaults, so a partial YAML file is valid.
/// CLI flags should be applied **after** this call to override any loaded values.
///
/// # Parameters
/// - `path`: Path to the YAML configuration file.
///
/// # Errors
/// Returns an error if the file cannot be read or if the YAML is malformed.
///
/// # Example
/// ```rust,ignore
/// let config = load_crop_config(Path::new("crop_config.yaml"))?;
/// ```
pub fn load_crop_config(path: &Path) -> Result<CropConfig> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read crop config file: {:?}", path))?;
    let config: CropConfig = serde_yaml::from_str(&contents)
        .with_context(|| format!("Failed to parse crop config YAML from {:?}", path))?;
    Ok(config)
}
