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
/// Used by [`ArtisticCropConfig::face_selection_strategy`] to control how the
/// primary face centroid is determined from a multi-face scene.
///
/// # Example
/// ```rust,ignore
/// let strategy = FaceSelectionStrategy::WeightedScore;
/// ```
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

/// Controls how aggressively the crop is composed around the face centroid.
///
/// - **Conservative**: Minimal face bias; generous body/context composition. Safe for
///   editorial content where environmental context matters.
/// - **Balanced**: Equal weight on face framing and body visibility. Default for
///   fashion and entertainment.
/// - **Aggressive**: Maximum face prominence; tighter crop with face near center.
///   Ideal for headshots and profile images.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ArtisticMode {
    /// Minimal face bias; body and context preserved.
    Conservative,
    /// Balanced face framing and body visibility (default).
    #[default]
    Balanced,
    /// Maximum face prominence; tight crop centred on face.
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
/// These are the internal values computed from the mode and used by the cropping
/// algorithm. Callers should use [`ArtisticCropConfig::effective_params`] rather
/// than constructing this directly.
#[derive(Debug, Clone)]
pub struct ArtisticModeParams {
    /// Bias of the crop center toward the face centroid (0.0 = no bias, 1.0 = full).
    pub face_centroid_bias: f32,
    /// Multiplier applied to `face_margin_px` (1.0 = base, >1 = more margin).
    pub margin_multiplier: f32,
    /// Minimum fraction of the person body that must remain visible (0.0–1.0).
    pub min_body_visibility: f32,
}

/// Runtime configuration for the artistic face-centric cropping algorithm.
///
/// Controls face detection usage, face selection strategy, and artistic composition
/// parameters. Instances can be constructed programmatically or supplied via CLI flags.
///
/// # Defaults
/// All fields have sensible defaults matching the "balanced" artistic mode:
/// - Face detection enabled.
/// - 20 px face margin.
/// - 0.5 head-to-body ratio.
/// - Weighted-score face selection.
/// - Balanced artistic mode.
///
/// # Example
/// ```rust,ignore
/// let config = ArtisticCropConfig::default();
/// let params = config.effective_params();
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArtisticCropConfig {
    /// Enable face detection in the two-stage pipeline (default: true).
    #[serde(default = "default_use_face_detection")]
    pub use_face_detection: bool,

    /// Pixel margin to add around detected face bounding boxes before composing the crop.
    ///
    /// Valid range: 10–30 px. Default: 20 px.
    #[serde(default = "default_face_margin_px")]
    pub face_margin_px: u32,

    /// Fraction of the crop height dedicated to the head region.
    ///
    /// `0.5` means the head occupies 50% of the crop height, leaving 50% for body/context.
    /// Valid range: 0.3–0.7. Default: 0.5.
    #[serde(default = "default_head_to_body_ratio")]
    pub head_to_body_ratio: f32,

    /// Face selection strategy when multiple faces are detected (default: weighted_score).
    #[serde(default)]
    pub face_selection_strategy: FaceSelectionStrategy,

    /// Artistic composition mode (default: balanced).
    #[serde(default)]
    pub artistic_mode: ArtisticMode,
}

fn default_use_face_detection() -> bool {
    true
}

fn default_face_margin_px() -> u32 {
    20
}

fn default_head_to_body_ratio() -> f32 {
    0.5
}

impl Default for ArtisticCropConfig {
    fn default() -> Self {
        Self {
            use_face_detection: default_use_face_detection(),
            face_margin_px: default_face_margin_px(),
            head_to_body_ratio: default_head_to_body_ratio(),
            face_selection_strategy: FaceSelectionStrategy::default(),
            artistic_mode: ArtisticMode::default(),
        }
    }
}

impl ArtisticCropConfig {
    /// Derive the effective algorithmic parameters for the current [`ArtisticMode`].
    ///
    /// The returned [`ArtisticModeParams`] encapsulates the numeric values that the
    /// cropping algorithm uses, derived from the high-level mode selection.
    ///
    /// | Mode         | face_centroid_bias | margin_multiplier | min_body_visibility |
    /// |------------- |--------------------|-------------------|---------------------|
    /// | Conservative | 0.3                | 1.5               | 0.6                 |
    /// | Balanced     | 0.6                | 1.0               | 0.4                 |
    /// | Aggressive   | 0.9                | 0.7               | 0.2                 |
    ///
    /// # Example
    /// ```rust,ignore
    /// let config = ArtisticCropConfig::default();
    /// let params = config.effective_params();
    /// assert!((params.face_centroid_bias - 0.6).abs() < 0.01);
    /// ```
    pub fn effective_params(&self) -> ArtisticModeParams {
        match self.artistic_mode {
            ArtisticMode::Conservative => ArtisticModeParams {
                face_centroid_bias: 0.3,
                margin_multiplier: 1.5,
                min_body_visibility: 0.6,
            },
            ArtisticMode::Balanced => ArtisticModeParams {
                face_centroid_bias: 0.6,
                margin_multiplier: 1.0,
                min_body_visibility: 0.4,
            },
            ArtisticMode::Aggressive => ArtisticModeParams {
                face_centroid_bias: 0.9,
                margin_multiplier: 0.7,
                min_body_visibility: 0.2,
            },
        }
    }

    /// Validated face margin in pixels, clamped to the allowed range [10, 30].
    pub fn clamped_face_margin_px(&self) -> u32 {
        self.face_margin_px.clamp(10, 30)
    }

    /// Validated head-to-body ratio, clamped to the allowed range [0.3, 0.7].
    pub fn clamped_head_to_body_ratio(&self) -> f32 {
        self.head_to_body_ratio.clamp(0.3, 0.7)
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
