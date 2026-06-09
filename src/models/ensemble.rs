use std::str::FromStr;

use anyhow::Result;
use image::DynamicImage;

use crate::models::candle_backend::ImageClassifier;
use crate::models::implementations::wd_tagger_ort::{
    WdTaggerOnnx, CONFIG_IDOLSANKAKU, CONFIG_WD_EVA02,
};

/// Identifies which image classifier to load and use at runtime.
///
/// Selected via the `--classifier <NAME>` CLI flag. `Freepik` is the
/// pre-existing default and is preserved bit-for-bit by this ExecPlan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassifierKind {
    /// Existing 4-tier classifier (Freepik/nsfw_image_detector via Candle).
    Freepik,
    /// SmilingWolf/wd-eva02-large-tagger-v3, rating head only, 5 tiers.
    WdEva02,
    /// deepghs/idolsankaku-eva02-large-tagger-v1, rating head only, 5 tiers.
    Idolsankaku,
    /// Both above models combined via confidence-weighted severity ensemble, 5 tiers.
    WdEnsemble,
}

impl FromStr for ClassifierKind {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "freepik" => Ok(Self::Freepik),
            "wd-eva02" => Ok(Self::WdEva02),
            "idolsankaku" => Ok(Self::Idolsankaku),
            "wd-ensemble" => Ok(Self::WdEnsemble),
            other => anyhow::bail!(
                "Unknown classifier '{}'. Valid options: freepik, wd-eva02, idolsankaku, wd-ensemble",
                other
            ),
        }
    }
}

pub const ENSEMBLE_EPS_MIN: f32 = 0.01;
pub const ENSEMBLE_SHARPNESS: f32 = 1.5;

/// Single-model wrapper over one WD tagger instance.
pub struct SingleWdClassifier {
    inner: WdTaggerOnnx,
}

impl SingleWdClassifier {
    pub fn wd_eva02() -> Result<Self> {
        Ok(Self {
            inner: WdTaggerOnnx::from_hub(CONFIG_WD_EVA02)?,
        })
    }

    pub fn idolsankaku() -> Result<Self> {
        Ok(Self {
            inner: WdTaggerOnnx::from_hub(CONFIG_IDOLSANKAKU)?,
        })
    }
}

impl ImageClassifier for SingleWdClassifier {
    fn classify(&self, image: &DynamicImage) -> Result<u8> {
        let raw = self.inner.predict_ratings(image)?;
        let normalized = normalize_ratings(raw);
        Ok(severity_to_tier(severity_from_distribution(normalized)))
    }

    fn name(&self) -> &str {
        self.inner.display_name()
    }
}

pub struct WdEnsembleClassifier {
    anime: WdTaggerOnnx,
    real: WdTaggerOnnx,
}

impl WdEnsembleClassifier {
    pub fn from_hub() -> Result<Self> {
        let anime = WdTaggerOnnx::from_hub(CONFIG_WD_EVA02)?;
        let real = WdTaggerOnnx::from_hub(CONFIG_IDOLSANKAKU)?;
        Ok(Self { anime, real })
    }
}

/// Normalize independent sigmoid ratings into a distribution for severity scoring.
pub fn normalize_ratings(raw: [f32; 4]) -> [f32; 4] {
    let sum = raw[0] + raw[1] + raw[2] + raw[3];
    if sum < 1e-6 {
        return [1.0, 0.0, 0.0, 0.0];
    }

    [raw[0] / sum, raw[1] / sum, raw[2] / sum, raw[3] / sum]
}

pub fn severity_from_distribution(distribution: [f32; 4]) -> f32 {
    0.0 * distribution[0] + 1.0 * distribution[1] + 2.0 * distribution[2] + 3.0 * distribution[3]
}

pub fn severity_to_tier(severity: f32) -> u8 {
    if severity < 0.30 {
        1
    } else if severity < 1.00 {
        2
    } else if severity < 1.80 {
        3
    } else if severity < 2.60 {
        4
    } else {
        5
    }
}

fn confidence(raw: [f32; 4]) -> f32 {
    let mut sorted = raw;
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    sorted[0] - sorted[1]
}

fn argmax4(values: [f32; 4]) -> usize {
    let mut best_index = 0usize;
    let mut best_value = f32::NEG_INFINITY;

    for (index, value) in values.iter().enumerate() {
        if *value > best_value {
            best_index = index;
            best_value = *value;
        }
    }

    best_index
}

pub fn ensemble_tier(raw_anime: [f32; 4], raw_real: [f32; 4]) -> u8 {
    let weight_anime = confidence(raw_anime)
        .max(ENSEMBLE_EPS_MIN)
        .powf(ENSEMBLE_SHARPNESS);
    let weight_real = confidence(raw_real)
        .max(ENSEMBLE_EPS_MIN)
        .powf(ENSEMBLE_SHARPNESS);
    let total_weight = weight_anime + weight_real;

    let combined_raw = [
        (weight_anime * raw_anime[0] + weight_real * raw_real[0]) / total_weight,
        (weight_anime * raw_anime[1] + weight_real * raw_real[1]) / total_weight,
        (weight_anime * raw_anime[2] + weight_real * raw_real[2]) / total_weight,
        (weight_anime * raw_anime[3] + weight_real * raw_real[3]) / total_weight,
    ];

    let normalized = normalize_ratings(combined_raw);
    let raw_tier = severity_to_tier(severity_from_distribution(normalized));

    let top_anime = argmax4(raw_anime) as i32;
    let top_real = argmax4(raw_real) as i32;
    let has_major_disagreement = (top_anime - top_real).abs() >= 2;

    if has_major_disagreement && raw_tier < 5 {
        raw_tier + 1
    } else {
        raw_tier
    }
}

impl ImageClassifier for WdEnsembleClassifier {
    fn classify(&self, image: &DynamicImage) -> Result<u8> {
        let raw_anime = self.anime.predict_ratings(image)?;
        let raw_real = self.real.predict_ratings(image)?;
        Ok(ensemble_tier(raw_anime, raw_real))
    }

    fn name(&self) -> &str {
        "wd-ensemble (eva02 + idolsankaku)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_freepik() {
        assert_eq!(
            "freepik".parse::<ClassifierKind>().unwrap(),
            ClassifierKind::Freepik
        );
    }

    #[test]
    fn parses_wd_ensemble() {
        assert_eq!(
            "wd-ensemble".parse::<ClassifierKind>().unwrap(),
            ClassifierKind::WdEnsemble
        );
    }

    #[test]
    fn rejects_garbage() {
        assert!("nonsense".parse::<ClassifierKind>().is_err());
    }

    #[test]
    fn pure_general_normalized_is_tier_1() {
        let raw = [0.9, 0.05, 0.03, 0.02];
        let normalized = normalize_ratings(raw);
        assert_eq!(severity_to_tier(severity_from_distribution(normalized)), 1);
    }

    #[test]
    fn pure_explicit_normalized_is_tier_5() {
        let raw = [0.01, 0.01, 0.03, 0.95];
        let normalized = normalize_ratings(raw);
        assert_eq!(severity_to_tier(severity_from_distribution(normalized)), 5);
    }

    #[test]
    fn confident_anime_dominates_uncertain_real() {
        let raw_anime = [0.9, 0.05, 0.03, 0.02];
        let raw_real = [0.25, 0.25, 0.25, 0.25];
        assert_eq!(ensemble_tier(raw_anime, raw_real), 1);
    }

    #[test]
    fn both_models_confident_explicit_is_tier_5() {
        let raw = [0.01, 0.01, 0.03, 0.95];
        assert_eq!(ensemble_tier(raw, raw), 5);
    }

    #[test]
    fn major_disagreement_bumps_tier_up() {
        let raw_anime = [0.95, 0.03, 0.01, 0.01];
        let raw_real = [0.01, 0.01, 0.03, 0.95];
        let tier = ensemble_tier(raw_anime, raw_real);
        assert!(
            tier >= 4,
            "expected >=4 after disagreement bump, got {tier}"
        );
    }

    #[test]
    fn minor_disagreement_does_not_bump() {
        let raw_anime = [0.6, 0.3, 0.05, 0.05];
        let raw_real = [0.3, 0.6, 0.05, 0.05];
        assert!(ensemble_tier(raw_anime, raw_real) <= 2);
    }

    #[test]
    fn pathological_zero_sigmoids_default_to_general() {
        let raw = [0.0, 0.0, 0.0, 0.0];
        assert_eq!(ensemble_tier(raw, raw), 1);
    }
}
