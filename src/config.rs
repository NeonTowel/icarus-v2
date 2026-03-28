/// Configuration types for Icarus-v2
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::str::FromStr;

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

// ---------------------------------------------------------------------------
// Face-aware artistic crop configuration
// ---------------------------------------------------------------------------

/// Strategy for selecting the dominant face when multiple faces are detected.
///
/// # Deprecation Notice
/// This enum is deprecated. Face selection is now always `MostCentral` (closest
/// face centroid to image center). Use [`ArtisticMode`] to control aggressiveness.
///
/// # Example
/// ```rust,ignore
/// let strategy = FaceSelectionStrategy::MostCentral;
/// ```
#[deprecated(
    since = "2.1.0",
    note = "Face selection is now always MostCentral. Use ArtisticMode to control aggressiveness."
)]
#[allow(deprecated)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum FaceSelectionStrategy {
    /// Select the face with the largest bounding box area.
    Largest,
    /// Select the face with the highest detection confidence.
    HighestConfidence,
    /// Select the face closest to the image center (Euclidean distance).
    MostCentral,
    /// Composite score: 50% bbox area + 30% confidence + 20% centrality (default).
    #[default]
    WeightedScore,
}

#[allow(deprecated)]
impl FromStr for FaceSelectionStrategy {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "largest" => Ok(Self::Largest),
            "highest_confidence" => Ok(Self::HighestConfidence),
            "most_central" => Ok(Self::MostCentral),
            "weighted_score" => Ok(Self::WeightedScore),
            other => anyhow::bail!(
                "Unknown face selection strategy '{}'. Valid options: \
                 largest, highest_confidence, most_central, weighted_score",
                other
            ),
        }
    }
}

/// Controls how aggressively face-aware crop adjustment is applied.
///
/// - **Conservative**: Larger safety margin (20px) around face; minimal adjustment.
///   Best for editorial content where environmental context matters.
/// - **Balanced**: Medium safety margin (15px). Default for fashion and entertainment.
/// - **Aggressive**: Smaller safety margin (10px); more shift budget used for face
///   visibility. Ideal for headshots and profile images.
///
/// The mode is the only user-facing parameter — all internal tuning values
/// are derived from it via [`ArtisticCropConfig::from_mode`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ArtisticMode {
    /// Larger safety margin, less aggressive face adjustment.
    Conservative,
    /// Balanced face framing and body visibility (default).
    #[default]
    Balanced,
    /// Smaller margin, tighter framing around the dominant face.
    Aggressive,
}

impl FromStr for ArtisticMode {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "conservative" => Ok(Self::Conservative),
            "balanced" => Ok(Self::Balanced),
            "aggressive" => Ok(Self::Aggressive),
            other => anyhow::bail!(
                "Unknown artistic mode '{}'. Valid options: conservative, balanced, aggressive",
                other
            ),
        }
    }
}

/// Derived margins and bias values for each [`ArtisticMode`].
///
/// # Deprecation Notice
/// This struct is deprecated. Parameters are now embedded in [`ArtisticCropConfig`]
/// via [`ArtisticCropConfig::from_mode`].
#[deprecated(
    since = "2.1.0",
    note = "Use ArtisticCropConfig::from_mode() to get the new config with embedded parameters."
)]
#[derive(Debug, Clone)]
pub struct ArtisticModeParams {
    /// Bias of the crop center toward the face centroid (0.0 = no bias, 1.0 = full).
    pub face_centroid_bias: f32,
    /// Multiplier applied to `face_margin_px` (1.0 = base, >1 = more margin).
    pub margin_multiplier: f32,
    /// Minimum fraction of the person body that must remain visible (0.0–1.0).
    pub min_body_visibility: f32,
}

/// Runtime configuration for the face-aware crop adjustment algorithm.
///
/// Controls the artistic composition mode, face safety margin, and shift budget.
/// Face detection is always enabled. The user-facing API has exactly two knobs:
/// the [`ArtisticMode`] (via CLI `--artistic-mode`) and the `--visualize` flag
/// (handled in `main.rs`, not stored here).
///
/// # Construction
/// Prefer [`ArtisticCropConfig::from_mode`] over manual construction:
///
/// ```rust,ignore
/// let config = ArtisticCropConfig::from_mode(ArtisticMode::Balanced);
/// assert_eq!(config.face_safety_margin_px, 15);
/// ```
///
/// # Defaults
/// `ArtisticCropConfig::default()` is equivalent to `from_mode(ArtisticMode::Balanced)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArtisticCropConfig {
    /// Artistic composition mode controlling face margin and shift budget.
    #[serde(default)]
    pub artistic_mode: ArtisticMode,

    /// Maximum fraction of crop dimension allowed for face adjustment shift.
    ///
    /// Shift budget in X = `crop_width  * max_shift_fraction`.
    /// Shift budget in Y = `crop_height * max_shift_fraction`.
    /// Default: 0.10 (10%).
    #[serde(default = "default_max_shift_fraction")]
    pub max_shift_fraction: f32,

    /// Pixel safety margin around the detected face bbox that must be inside the crop.
    ///
    /// Derived from `artistic_mode` by default:
    /// - Conservative → 20 px
    /// - Balanced     → 15 px
    /// - Aggressive   → 10 px
    #[serde(default = "default_face_safety_margin_px")]
    pub face_safety_margin_px: u32,
}

fn default_max_shift_fraction() -> f32 {
    0.10
}

fn default_face_safety_margin_px() -> u32 {
    15 // Balanced default
}

impl Default for ArtisticCropConfig {
    fn default() -> Self {
        Self::from_mode(ArtisticMode::Balanced)
    }
}

impl ArtisticCropConfig {
    /// Create a config from an [`ArtisticMode`] with default shift fraction (10%).
    ///
    /// | Mode         | face_safety_margin_px | max_shift_fraction |
    /// |--------------|----------------------|-------------------|
    /// | Conservative | 20 px                | 10%               |
    /// | Balanced     | 15 px                | 10%               |
    /// | Aggressive   | 10 px                | 10%               |
    ///
    /// # Example
    /// ```rust,ignore
    /// let config = ArtisticCropConfig::from_mode(ArtisticMode::Conservative);
    /// assert_eq!(config.face_safety_margin_px, 20);
    /// assert!((config.max_shift_fraction - 0.10).abs() < 0.01);
    /// ```
    pub fn from_mode(mode: ArtisticMode) -> Self {
        let margin = match mode {
            ArtisticMode::Conservative => 20,
            ArtisticMode::Balanced => 15,
            ArtisticMode::Aggressive => 10,
        };
        Self {
            artistic_mode: mode,
            max_shift_fraction: 0.10,
            face_safety_margin_px: margin,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_mode_conservative() {
        let config = ArtisticCropConfig::from_mode(ArtisticMode::Conservative);
        assert_eq!(config.face_safety_margin_px, 20);
        assert!((config.max_shift_fraction - 0.10).abs() < 0.001);
        assert_eq!(config.artistic_mode, ArtisticMode::Conservative);
    }

    #[test]
    fn test_from_mode_balanced() {
        let config = ArtisticCropConfig::from_mode(ArtisticMode::Balanced);
        assert_eq!(config.face_safety_margin_px, 15);
        assert!((config.max_shift_fraction - 0.10).abs() < 0.001);
    }

    #[test]
    fn test_from_mode_aggressive() {
        let config = ArtisticCropConfig::from_mode(ArtisticMode::Aggressive);
        assert_eq!(config.face_safety_margin_px, 10);
        assert!((config.max_shift_fraction - 0.10).abs() < 0.001);
    }

    #[test]
    fn test_default_is_balanced() {
        let default = ArtisticCropConfig::default();
        let balanced = ArtisticCropConfig::from_mode(ArtisticMode::Balanced);
        assert_eq!(
            default.face_safety_margin_px,
            balanced.face_safety_margin_px
        );
        assert_eq!(default.artistic_mode, ArtisticMode::Balanced);
    }

    #[test]
    fn test_artistic_mode_fromstr() {
        assert_eq!(
            "conservative".parse::<ArtisticMode>().unwrap(),
            ArtisticMode::Conservative
        );
        assert_eq!(
            "balanced".parse::<ArtisticMode>().unwrap(),
            ArtisticMode::Balanced
        );
        assert_eq!(
            "aggressive".parse::<ArtisticMode>().unwrap(),
            ArtisticMode::Aggressive
        );
        assert!("unknown".parse::<ArtisticMode>().is_err());
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
