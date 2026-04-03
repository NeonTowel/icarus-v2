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

    // ===== PHASE 3: Aspect-ratio-specific breathing room =====
    /// Breathing room above forehead for mobile (9:21) crops.
    ///
    /// Expressed as a percentage of crop height. A value of 12.5 means
    /// 12.5% of the crop height is reserved as empty space above the
    /// forehead. Tight by design: mobile frames are tall, so headroom
    /// competes with body visibility.
    ///
    /// Valid range: 10.0..=15.0. Default: 12.5.
    #[serde(default = "default_breathing_room_mobile")]
    pub breathing_room_percent_mobile: f32,

    /// Breathing room above forehead for portrait (9:16) crops.
    ///
    /// Balanced between head comfort and body visibility.
    ///
    /// Valid range: 15.0..=25.0. Default: 20.0.
    #[serde(default = "default_breathing_room_portrait")]
    pub breathing_room_percent_portrait: f32,

    /// Breathing room above forehead for landscape (21:9) crops.
    ///
    /// Generous: landscape frames are wide and short, so head breathing
    /// room improves the visual weight above the subject.
    ///
    /// Valid range: 20.0..=30.0. Default: 25.0.
    #[serde(default = "default_breathing_room_landscape")]
    pub breathing_room_percent_landscape: f32,

    // ===== PHASE 3: Aspect-ratio-specific face bbox penetration =====
    /// Max percentage of face bbox height the crop edge may penetrate (mobile).
    ///
    /// "Penetration" is how far the crop top edge is allowed to cut into
    /// the face bounding box from above (forehead zone only). The protected
    /// eye zone (30–65% of face height) is always preserved regardless of
    /// this value.
    ///
    /// Mobile gets the highest penetration to show maximum body below the
    /// face in the tall 9:21 frame.
    ///
    /// Valid range: 15.0..=20.0. Default: 18.0.
    #[serde(default = "default_penetration_mobile")]
    pub max_face_bbox_penetration_percent_mobile: f32,

    /// Max face bbox penetration for portrait (9:16) crops.
    ///
    /// Moderate: balanced body visibility vs. comfortable head framing.
    ///
    /// Valid range: 12.0..=15.0. Default: 14.0.
    #[serde(default = "default_penetration_portrait")]
    pub max_face_bbox_penetration_percent_portrait: f32,

    /// Max face bbox penetration for landscape (21:9) crops.
    ///
    /// Conservative: landscape frames show torso + arms by horizontal
    /// extension; minimal vertical penetration is appropriate.
    ///
    /// Valid range: 10.0..=12.0. Default: 11.0.
    #[serde(default = "default_penetration_landscape")]
    pub max_face_bbox_penetration_percent_landscape: f32,
}

fn default_max_shift_fraction() -> f32 {
    0.10
}

fn default_face_safety_margin_px() -> u32 {
    15 // Balanced default
}

fn default_breathing_room_mobile() -> f32 {
    12.5
}

fn default_breathing_room_portrait() -> f32 {
    20.0
}

fn default_breathing_room_landscape() -> f32 {
    25.0
}

fn default_penetration_mobile() -> f32 {
    18.0
}

fn default_penetration_portrait() -> f32 {
    14.0
}

fn default_penetration_landscape() -> f32 {
    11.0
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
            // Phase 3 defaults are mode-independent: per-aspect-ratio
            // differentiation replaces mode-based differentiation here.
            breathing_room_percent_mobile: 12.5,
            breathing_room_percent_portrait: 20.0,
            breathing_room_percent_landscape: 25.0,
            max_face_bbox_penetration_percent_mobile: 18.0,
            max_face_bbox_penetration_percent_portrait: 14.0,
            max_face_bbox_penetration_percent_landscape: 11.0,
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

    #[test]
    fn test_phase3_defaults_present() {
        // Phase 3 fields must be present with the approved default values
        // regardless of whether the config is constructed via default() or
        // from_mode().
        let config = ArtisticCropConfig::default();
        assert!(
            (config.breathing_room_percent_mobile - 12.5).abs() < 0.01,
            "mobile breathing room default should be 12.5, got {}",
            config.breathing_room_percent_mobile
        );
        assert!(
            (config.breathing_room_percent_portrait - 20.0).abs() < 0.01,
            "portrait breathing room default should be 20.0, got {}",
            config.breathing_room_percent_portrait
        );
        assert!(
            (config.breathing_room_percent_landscape - 25.0).abs() < 0.01,
            "landscape breathing room default should be 25.0, got {}",
            config.breathing_room_percent_landscape
        );
        assert!(
            (config.max_face_bbox_penetration_percent_mobile - 18.0).abs() < 0.01,
            "mobile penetration default should be 18.0, got {}",
            config.max_face_bbox_penetration_percent_mobile
        );
        assert!(
            (config.max_face_bbox_penetration_percent_portrait - 14.0).abs() < 0.01,
            "portrait penetration default should be 14.0, got {}",
            config.max_face_bbox_penetration_percent_portrait
        );
        assert!(
            (config.max_face_bbox_penetration_percent_landscape - 11.0).abs() < 0.01,
            "landscape penetration default should be 11.0, got {}",
            config.max_face_bbox_penetration_percent_landscape
        );
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
