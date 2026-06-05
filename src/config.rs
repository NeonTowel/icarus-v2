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

fn default_target_y_landscape() -> f32 {
    0.50
}
fn default_target_y_portrait() -> f32 {
    0.50
}
fn default_target_y_mobile() -> f32 {
    0.50
}
fn default_visibility_threshold() -> f32 {
    0.50
}
fn default_true() -> bool {
    true
}
fn default_dedup_iou_threshold() -> f32 {
    0.50
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CropConfig {
    #[serde(default = "default_target_y_landscape", alias = "headroom_ratio")]
    pub target_y_frac_landscape: f32, // 0.50 — places crop center at bbox vertical center

    #[serde(default = "default_target_y_portrait")]
    pub target_y_frac_portrait: f32, // 0.50 — places crop center at bbox vertical center

    #[serde(default = "default_target_y_mobile")]
    pub target_y_frac_mobile: f32, // 0.50 — places crop center at bbox vertical center

    #[serde(default = "default_visibility_threshold")]
    pub visibility_threshold: f32, // 0.50

    #[serde(default = "default_true")]
    pub enable_reflection_dedup: bool,

    #[serde(default = "default_dedup_iou_threshold")]
    pub dedup_iou_threshold: f32, // 0.50

    #[serde(default = "default_true")]
    pub enable_directional_thirds: bool,
}

impl Default for CropConfig {
    fn default() -> Self {
        Self {
            target_y_frac_landscape: default_target_y_landscape(),
            target_y_frac_portrait: default_target_y_portrait(),
            target_y_frac_mobile: default_target_y_mobile(),
            visibility_threshold: default_visibility_threshold(),
            enable_reflection_dedup: default_true(),
            dedup_iou_threshold: default_dedup_iou_threshold(),
            enable_directional_thirds: default_true(),
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
    #[serde(default)]
    pub artistic_mode: ArtisticMode,
}

impl Default for ArtisticCropConfig {
    fn default() -> Self {
        Self::from_mode(ArtisticMode::Balanced)
    }
}

impl ArtisticCropConfig {
    pub fn from_mode(mode: ArtisticMode) -> Self {
        Self {
            artistic_mode: mode,
        }
    }

    pub fn target_y_offset(&self) -> f32 {
        match self.artistic_mode {
            ArtisticMode::Aggressive => -0.05,
            ArtisticMode::Balanced => 0.00,
            ArtisticMode::Conservative => 0.05,
        }
    }
}

/// Resolve the number of Rayon worker threads for batch processing.
///
/// Policy:
/// - `requested == None`  → 50% of `available` cores, floored, minimum 1.
/// - `requested == Some(n)` where `n > available` → capped at `available`.
/// - `requested == Some(n)` where `1 <= n <= available` → `n` unchanged.
///
/// # Parameters
/// - `requested`: Optional CLI override from `--threads`.
/// - `available`: Logical core count visible to this process.
///
/// # Example
/// ```rust
/// use icarus_v2::config::resolve_thread_count;
///
/// assert_eq!(resolve_thread_count(None, 8), 4);
/// assert_eq!(resolve_thread_count(Some(999), 8), 8);
/// ```
pub fn resolve_thread_count(requested: Option<usize>, available: usize) -> usize {
    let available = available.max(1);
    match requested {
        None => (available / 2).max(1),
        Some(count) => count.clamp(1, available),
    }
}

/// Number of logical cores available to this process, never less than 1.
///
/// Uses [`std::thread::available_parallelism`], which respects affinity and
/// cgroup limits when supported by the platform.
///
/// # Example
/// ```rust
/// use icarus_v2::config::available_core_count;
///
/// assert!(available_core_count() >= 1);
/// ```
pub fn available_core_count() -> usize {
    std::thread::available_parallelism()
        .map(|core_count| core_count.get())
        .unwrap_or(1)
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;

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

    #[test]
    fn resolve_thread_count_defaults_to_half() {
        assert_eq!(resolve_thread_count(None, 8), 4);
        assert_eq!(resolve_thread_count(None, 16), 8);
    }

    #[test]
    fn resolve_thread_count_floors_to_at_least_one() {
        assert_eq!(resolve_thread_count(None, 1), 1);
        assert_eq!(resolve_thread_count(None, 0), 1);
    }

    #[test]
    fn resolve_thread_count_caps_at_available() {
        assert_eq!(resolve_thread_count(Some(9999), 8), 8);
        assert_eq!(resolve_thread_count(Some(8), 8), 8);
    }

    #[test]
    fn resolve_thread_count_passes_through_valid_request() {
        assert_eq!(resolve_thread_count(Some(3), 8), 3);
        assert_eq!(resolve_thread_count(Some(1), 8), 1);
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
