//! Crop strategy trait and per-format implementations.
//!
//! Each output format (21:9, 9:21, 9:16) is represented as a zero-sized unit struct
//! that implements `CropStrategy`. The shared calculation logic lives in the trait's
//! default `calculate()` method — the three implementations differ only in their
//! aspect ratio, orientation (height-first vs width-first), and which config field
//! supplies the base vertical bias.
//!
//! # Example
//! ```rust,ignore
//! let crop = strategy_for("21:9")
//!     .and_then(|s| s.calculate(w, h, &bbox, &faces, &focal, &config, &artistic));
//! ```

use super::geometry::{
    person_is_reasonably_visible_threshold, place_focal_point, BBox, CropRegion,
};
use super::pose::{detect_pose_direction, Direction};
use crate::config::{ArtisticCropConfig, CropConfig};
use crate::focal_point::FocalPoint;

// ---------------------------------------------------------------------------
// CropParams and Orientation
// ---------------------------------------------------------------------------

/// Whether to derive crop dimensions height-first (landscape) or width-first (portrait).
pub enum Orientation {
    /// `crop_height = photo_height`, then cap width to photo_width.
    HeightFirst,
    /// `crop_width = photo_width`, then cap height to photo_height.
    WidthFirst,
}

/// Parameters that vary between crop strategies, derived from the format and config.
pub struct CropParams {
    pub aspect_ratio: f32,
    pub target_y_frac_base: f32,
    pub orientation: Orientation,
}

// ---------------------------------------------------------------------------
// CropStrategy trait
// ---------------------------------------------------------------------------

/// A crop algorithm for one output format (21:9, 9:21, or 9:16).
///
/// Implementors supply `params()` and `format_name()`; the shared `calculate()`
/// default method handles all geometry so there is no duplicated logic.
pub trait CropStrategy {
    /// Returns aspect ratio, vertical bias, and dimension ordering for this format.
    fn params(&self, config: &CropConfig) -> CropParams;

    /// Human-readable format name, e.g. `"21:9"`.
    fn format_name(&self) -> &'static str;

    /// Compute the crop region for this format. Returns `None` when the person
    /// would not be sufficiently visible at the computed position.
    #[allow(clippy::too_many_arguments)]
    fn calculate(
        &self,
        photo_w: u32,
        photo_h: u32,
        bbox: &BBox,
        faces: &[BBox],
        focal: &FocalPoint,
        config: &CropConfig,
        artistic: &ArtisticCropConfig,
    ) -> Option<CropRegion> {
        let CropParams {
            aspect_ratio,
            target_y_frac_base,
            orientation,
        } = self.params(config);
        let pw = photo_w as f32;
        let ph = photo_h as f32;

        let (crop_width, crop_height) = match orientation {
            Orientation::HeightFirst => {
                let h = ph;
                let w = h * aspect_ratio;
                if w > pw {
                    let w2 = pw;
                    (w2, w2 / aspect_ratio)
                } else {
                    (w, h)
                }
            }
            Orientation::WidthFirst => {
                let w = pw;
                let h = w / aspect_ratio;
                if h > ph {
                    let h2 = ph;
                    (h2 * aspect_ratio, h2)
                } else {
                    (w, h)
                }
            }
        };

        let mut target_x_frac = 0.50;
        if config.enable_directional_thirds && !faces.is_empty() {
            target_x_frac = match detect_pose_direction(bbox, &faces[0]) {
                Direction::FacingRight => 0.33,
                Direction::FacingLeft => 0.67,
                Direction::Frontal => 0.50,
            };
        }

        let target_y_frac = (target_y_frac_base + artistic.target_y_offset()).clamp(0.10, 0.50);
        let crop = place_focal_point(
            photo_w,
            photo_h,
            crop_width,
            crop_height,
            focal,
            target_x_frac,
            target_y_frac,
        );

        if person_is_reasonably_visible_threshold(
            bbox,
            &crop,
            config.visibility_threshold,
            photo_w,
            photo_h,
        ) {
            Some(crop)
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Unit-struct implementations
// ---------------------------------------------------------------------------

/// Landscape 21:9 crop strategy.
pub struct Landscape21x9;

impl CropStrategy for Landscape21x9 {
    fn params(&self, config: &CropConfig) -> CropParams {
        CropParams {
            aspect_ratio: 21.0 / 9.0,
            target_y_frac_base: config.target_y_frac_landscape,
            orientation: Orientation::HeightFirst,
        }
    }
    fn format_name(&self) -> &'static str {
        "21:9"
    }
}

/// Portrait 9:21 (ultrawide mobile) crop strategy.
pub struct Portrait9x21;

impl CropStrategy for Portrait9x21 {
    fn params(&self, config: &CropConfig) -> CropParams {
        CropParams {
            aspect_ratio: 9.0 / 21.0,
            target_y_frac_base: config.target_y_frac_mobile,
            orientation: Orientation::WidthFirst,
        }
    }
    fn format_name(&self) -> &'static str {
        "9:21"
    }
}

/// Portrait 9:16 (standard portrait) crop strategy.
pub struct Portrait9x16;

impl CropStrategy for Portrait9x16 {
    fn params(&self, config: &CropConfig) -> CropParams {
        CropParams {
            aspect_ratio: 9.0 / 16.0,
            target_y_frac_base: config.target_y_frac_portrait,
            orientation: Orientation::WidthFirst,
        }
    }
    fn format_name(&self) -> &'static str {
        "9:16"
    }
}

// ---------------------------------------------------------------------------
// Dispatch helper
// ---------------------------------------------------------------------------

/// Return a reference to the strategy for `format`, or `None` for unknown formats.
///
/// # Example
/// ```rust,ignore
/// if let Some(s) = strategy_for("21:9") {
///     let crop = s.calculate(w, h, &bbox, &faces, &focal, &config, &artistic);
/// }
/// ```
pub fn strategy_for(format: &str) -> Option<&'static dyn CropStrategy> {
    match format {
        "21:9" => Some(&Landscape21x9),
        "9:21" => Some(&Portrait9x21),
        "9:16" => Some(&Portrait9x16),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Legacy free-function wrappers (thin delegates)
// ---------------------------------------------------------------------------
// These keep callers that import the named functions compiling without changes.

#[allow(clippy::too_many_arguments)]
pub fn calculate_landscape_21_9_crop_with_face(
    photo_w: u32,
    photo_h: u32,
    bbox: &BBox,
    faces: &[BBox],
    focal: &FocalPoint,
    config: &CropConfig,
    artistic: &ArtisticCropConfig,
) -> Option<CropRegion> {
    Landscape21x9.calculate(photo_w, photo_h, bbox, faces, focal, config, artistic)
}

#[allow(clippy::too_many_arguments)]
pub fn calculate_portrait_9_21_crop_with_face(
    photo_w: u32,
    photo_h: u32,
    bbox: &BBox,
    faces: &[BBox],
    focal: &FocalPoint,
    config: &CropConfig,
    artistic: &ArtisticCropConfig,
) -> Option<CropRegion> {
    Portrait9x21.calculate(photo_w, photo_h, bbox, faces, focal, config, artistic)
}

#[allow(clippy::too_many_arguments)]
pub fn calculate_portrait_9_16_crop_with_face(
    photo_w: u32,
    photo_h: u32,
    bbox: &BBox,
    faces: &[BBox],
    focal: &FocalPoint,
    config: &CropConfig,
    artistic: &ArtisticCropConfig,
) -> Option<CropRegion> {
    Portrait9x16.calculate(photo_w, photo_h, bbox, faces, focal, config, artistic)
}

// ---------------------------------------------------------------------------
// Format suitability detection (probes with empty face slice for geometric gate)
// ---------------------------------------------------------------------------

/// Detect which output formats are suitable for this photo and person bounding box.
///
/// **Orientation logic:**
/// - Wide bbox (width > height): landscape 21:9 only.
/// - Tall/square bbox: landscape 21:9 always + portrait 9:21 and 9:16 if visible.
///
/// Probes each format with an empty face slice — suitability is a person-visibility gate,
/// not a face-placement decision. The caller re-computes crops with real faces later.
pub fn detect_suitable_formats(
    photo_width: u32,
    photo_height: u32,
    bbox: &BBox,
    margin_percent: f32,
    config: &CropConfig,
) -> Vec<String> {
    use super::geometry::apply_margin_to_bbox;

    let working = apply_margin_to_bbox(bbox, margin_percent, photo_width, photo_height);
    let bbox_is_wide = working.width() > working.height();

    let focal =
        crate::focal_point::compute_focal_point(Some(&working), &[], photo_width, photo_height);
    let artistic = ArtisticCropConfig::default();
    let mut suitable = Vec::new();

    if Landscape21x9
        .calculate(
            photo_width,
            photo_height,
            &working,
            &[],
            &focal,
            config,
            &artistic,
        )
        .is_some()
    {
        suitable.push("21:9".to_string());
    }
    if !bbox_is_wide {
        if Portrait9x21
            .calculate(
                photo_width,
                photo_height,
                &working,
                &[],
                &focal,
                config,
                &artistic,
            )
            .is_some()
        {
            suitable.push("9:21".to_string());
        }
        if Portrait9x16
            .calculate(
                photo_width,
                photo_height,
                &working,
                &[],
                &focal,
                config,
                &artistic,
            )
            .is_some()
        {
            suitable.push("9:16".to_string());
        }
    }
    suitable
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn tall_person_bbox() -> BBox {
        BBox {
            x1: 800.0,
            y1: 200.0,
            x2: 2200.0,
            y2: 3800.0,
        }
    }
    fn wide_person_bbox() -> BBox {
        BBox {
            x1: 200.0,
            y1: 500.0,
            x2: 1800.0,
            y2: 900.0,
        }
    }

    #[test]
    fn test_wide_bbox_returns_only_landscape() {
        let config = CropConfig::default();
        let formats = detect_suitable_formats(1920, 1080, &wide_person_bbox(), 0.0, &config);
        assert!(!formats.contains(&"9:21".to_string()));
        assert!(!formats.contains(&"9:16".to_string()));
    }

    #[test]
    fn test_tall_bbox_can_return_portrait_formats() {
        let config = CropConfig::default();
        let formats = detect_suitable_formats(3024, 4032, &tall_person_bbox(), 0.0, &config);
        assert!(
            formats.contains(&"9:21".to_string()) || formats.contains(&"9:16".to_string()),
            "Expected at least one portrait format: {:?}",
            formats
        );
    }

    #[test]
    fn test_detect_formats_with_margin() {
        let config = CropConfig::default();
        let formats = detect_suitable_formats(3024, 4032, &tall_person_bbox(), 10.0, &config);
        assert!(!formats.is_empty());
        assert!(
            formats.contains(&"9:21".to_string()) || formats.contains(&"9:16".to_string()),
            "Expected portrait formats with 10% margin: {:?}",
            formats
        );
    }

    #[test]
    fn test_landscape_suitable_for_landscape_photo_with_person() {
        let bbox = BBox {
            x1: 760.0,
            y1: 240.0,
            x2: 1160.0,
            y2: 840.0,
        };
        let config = CropConfig::default();
        let formats = detect_suitable_formats(1920, 1080, &bbox, 0.0, &config);
        assert!(
            formats.contains(&"21:9".to_string()),
            "Expected 21:9: {:?}",
            formats
        );
    }

    #[test]
    fn test_landscape_strategy_matches_legacy_function() {
        let bbox = BBox {
            x1: 200.0,
            y1: 50.0,
            x2: 500.0,
            y2: 900.0,
        };
        let config = CropConfig::default();
        let artistic = ArtisticCropConfig::default();
        let focal = crate::focal_point::compute_focal_point(Some(&bbox), &[], 3024, 4032);
        let via_trait = Landscape21x9.calculate(3024, 4032, &bbox, &[], &focal, &config, &artistic);
        let via_fn = calculate_landscape_21_9_crop_with_face(
            3024,
            4032,
            &bbox,
            &[],
            &focal,
            &config,
            &artistic,
        );
        assert_eq!(
            via_trait, via_fn,
            "Landscape21x9 trait result must match legacy function"
        );
    }

    #[test]
    fn test_portrait_9x21_strategy_matches_legacy_function() {
        let bbox = BBox {
            x1: 800.0,
            y1: 200.0,
            x2: 2200.0,
            y2: 3800.0,
        };
        let config = CropConfig::default();
        let artistic = ArtisticCropConfig::default();
        let focal = crate::focal_point::compute_focal_point(Some(&bbox), &[], 3024, 4032);
        let via_trait = Portrait9x21.calculate(3024, 4032, &bbox, &[], &focal, &config, &artistic);
        let via_fn = calculate_portrait_9_21_crop_with_face(
            3024,
            4032,
            &bbox,
            &[],
            &focal,
            &config,
            &artistic,
        );
        assert_eq!(
            via_trait, via_fn,
            "Portrait9x21 trait result must match legacy function"
        );
    }

    #[test]
    fn test_portrait_9x16_strategy_matches_legacy_function() {
        let bbox = BBox {
            x1: 800.0,
            y1: 200.0,
            x2: 2200.0,
            y2: 3800.0,
        };
        let config = CropConfig::default();
        let artistic = ArtisticCropConfig::default();
        let focal = crate::focal_point::compute_focal_point(Some(&bbox), &[], 3024, 4032);
        let via_trait = Portrait9x16.calculate(3024, 4032, &bbox, &[], &focal, &config, &artistic);
        let via_fn = calculate_portrait_9_16_crop_with_face(
            3024,
            4032,
            &bbox,
            &[],
            &focal,
            &config,
            &artistic,
        );
        assert_eq!(
            via_trait, via_fn,
            "Portrait9x16 trait result must match legacy function"
        );
    }

    #[test]
    fn test_strategy_for_dispatch() {
        assert!(strategy_for("21:9").is_some());
        assert!(strategy_for("9:21").is_some());
        assert!(strategy_for("9:16").is_some());
        assert!(strategy_for("unknown").is_none());
        assert_eq!(strategy_for("21:9").unwrap().format_name(), "21:9");
    }
}
