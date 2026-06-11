//! Joint person + face crop analysis.
//!
//! This module looks at the person bounding box and face bounding boxes *together*
//! to recommend:
//!
//! 1. A vertical placement bias (`target_y_frac_override`) that keeps the full person
//!    height in frame whenever the crop is large enough, with the face positioned toward
//!    the upper portion of the crop.
//!
//! 2. An extra margin expansion (`extra_margin_percent`) when the proposed crop would
//!    fall below a configured minimum output dimension.
//!
//! ## Design invariants
//!
//! - **Pure data.** This module performs no image I/O and allocates no pixels.
//! - **Advisory only.** The returned [`JointAnalysis`] is a recommendation; the strategy
//!   still passes the crop through `enforce_eye_safety`, `person_is_reasonably_visible`,
//!   and `to_xyxy_clamped` before any write.
//! - **Person/face separation preserved.** `focal_point.rs` is NOT modified; the joint
//!   analyzer introduces a *separate, opt-in* vertical bias consumed by the strategy.
//! - **Relaxation only expands margin.** `extra_margin_percent` is always ≥ 0; it is
//!   bounded by what the photo can satisfy without exceeding its own bounds.

use crate::config::{ArtisticCropConfig, CropConfig};
use crate::crop::geometry::{crop_dimensions, long_side_passes, BBox};

// ---------------------------------------------------------------------------
// JointAnalysis result type
// ---------------------------------------------------------------------------

/// Recommendation produced by analyzing the person box and face boxes together.
///
/// All fields are **advisory**. The strategy applies them after its own computation;
/// the final crop still passes through visibility gating, eye-safety, and coordinate
/// clamping.
///
/// # Example
/// ```rust,ignore
/// let analysis = analyze_joint(
///     &person_bbox, &face_bboxes, 3024, 4032, 9.0/16.0, false,
///     &crop_config, &artistic_config,
/// );
/// if analysis.relaxed {
///     println!("min-dimension relaxation fired: extra margin {:.1}%", analysis.extra_margin_percent);
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct JointAnalysis {
    /// Optional override for `target_y_frac` (vertical focal placement within the crop).
    ///
    /// When `Some(v)`, `v` is already clamped to `[0.10, 0.50]` so it composes
    /// safely with `artistic.target_y_offset()` and the 6% eye-safety budget.
    /// `None` means "use the strategy's normal value".
    pub target_y_frac_override: Option<f32>,

    /// Recommended extra margin percentage to expand the working bbox.
    ///
    /// This is *additional* percentage on top of the user-supplied `--margin`.
    /// Applied via `apply_margin_to_bbox` before calling the strategy.
    /// `0.0` means no relaxation.
    pub extra_margin_percent: f32,

    /// `true` when this analysis triggered min-dimension relaxation.
    pub relaxed: bool,

    /// `true` when the recommended placement is expected to keep both `person.y1`
    /// and `person.y2` inside the crop (full person height preserved).
    pub full_person_height_expected: bool,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Analyze person + faces jointly for one target format.
///
/// Returns a [`JointAnalysis`] with:
/// - A `target_y_frac_override` biased to keep the full person in frame (when the
///   crop is large enough) with the face well-positioned toward the top.
/// - An `extra_margin_percent` recommendation when the proposed crop would fall below
///   the configured minimum dimension.
///
/// # Arguments
/// * `person`       — Person bounding box (working bbox after base margin).
/// * `faces`        — Face bounding boxes in the same coordinate space (may be empty).
/// * `photo_w/h`    — Source photo dimensions.
/// * `aspect_ratio` — Target format aspect ratio (e.g. `9.0/16.0` for portrait 9:16).
/// * `height_first` — `true` for HeightFirst (landscape 21:9), `false` for WidthFirst.
/// * `config`       — Crop configuration (min-dimension thresholds, visibility threshold).
/// * `artistic`     — Artistic mode configuration (vertical offset).
///
/// # Design notes
/// The function is intentionally `#[allow(clippy::too_many_arguments)]` — the arguments
/// mirror the strategy's `calculate` signature to make call-site symmetry obvious.
#[allow(clippy::too_many_arguments)]
pub fn analyze_joint(
    person: &BBox,
    faces: &[BBox],
    photo_w: u32,
    photo_h: u32,
    aspect_ratio: f32,
    height_first: bool,
    config: &CropConfig,
    artistic: &ArtisticCropConfig,
) -> JointAnalysis {
    let (crop_w, crop_h) = crop_dimensions(photo_w, photo_h, aspect_ratio, height_first);
    let person_height = person.height();
    let person_center_y = person.center_y();

    // -------------------------------------------------------------------------
    // 1. Vertical bias for full-person-height preservation
    // -------------------------------------------------------------------------
    //
    // The focal point is at `person_center_y`. The strategy places the crop so that:
    //   `crop_y = (person_center_y - target_y_frac * crop_h).clamp(0, photo_h - crop_h)`
    //
    // For the person to be fully inside:
    //   crop_y ≤ person.y1  →  target_y_frac ≥ (center_y - person.y1) / crop_h
    //                                          = person_height / (2 * crop_h)
    //   crop_y + crop_h ≥ person.y2  →  target_y_frac ≤ 1 - person_height / (2 * crop_h)
    //
    // When faces are present, we further bias toward keeping the face in the upper
    // portion of the crop (face at ~25% from top), which corresponds to a higher
    // target_y_frac (focal point placed lower in the crop window).

    let target_y_frac_override =
        compute_vertical_bias(faces, crop_h, person_height, person_center_y, artistic);

    // Evaluate whether the bias achieves full-person-height coverage.
    let full_person_height_expected = if crop_h >= person_height {
        // Person fits; check if the recommended placement keeps both ends in frame.
        let effective_frac = target_y_frac_override.unwrap_or_else(|| {
            (config.target_y_frac_portrait + artistic.target_y_offset()).clamp(0.10, 0.50)
        });
        let crop_y = (person_center_y - effective_frac * crop_h).max(0.0);
        let crop_y2 = crop_y + crop_h;
        crop_y <= person.y1 && crop_y2 >= person.y2
    } else {
        false
    };

    // -------------------------------------------------------------------------
    // 2. Min-dimension relaxation (long-side semantics)
    // -------------------------------------------------------------------------
    //
    // Use max(crop_w, crop_h) — the long side — as the single dimension to check against
    // config.min_long_side_pixels. This is orientation-correct by construction:
    // - Landscape (21:9): crop_w > crop_h → long side is width.
    // - Portrait  (9:16) and Mobile (9:21): crop_h > crop_w → long side is height.
    // No new aspect ratios; maps only to the three fixed output_sorting.rs categories.
    let long_side = crop_w.max(crop_h);
    let long_side_u32 = long_side as u32;

    let (extra_margin_percent, relaxed) =
        if !long_side_passes(long_side_u32, config.min_long_side_pixels) {
            // Deficit: how much the long side falls short of the minimum.
            let deficit = config.min_long_side_pixels as f32 - long_side;
            // Photo's long side — the upper bound on how much we can grow.
            let photo_long_side = (photo_w as f32).max(photo_h as f32);
            // Person's dominant dimension along the long side (used to express margin as %).
            let person_dominant = if crop_w >= crop_h {
                person.width().max(1.0)
            } else {
                person.height().max(1.0)
            };
            let extra = if photo_long_side > long_side {
                let max_growth = photo_long_side - long_side;
                let target_growth = deficit.min(max_growth);
                // Express as percent of person dimension (same convention as --margin).
                (target_growth / person_dominant * 100.0).min(50.0)
            } else {
                // Photo's own long side is smaller than the floor — cannot grow further.
                // Mark as relaxed but add no extra margin (photo is already at its max).
                0.0
            };
            (extra, true)
        } else {
            (0.0, false)
        };

    JointAnalysis {
        target_y_frac_override,
        extra_margin_percent,
        relaxed,
        full_person_height_expected,
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Compute the `target_y_frac` override for full-person-height bias.
///
/// Returns `None` when the default strategy value is already sufficient (person fits
/// centered, no faces to consider).
fn compute_vertical_bias(
    faces: &[BBox],
    crop_h: f32,
    person_height: f32,
    person_center_y: f32,
    artistic: &ArtisticCropConfig,
) -> Option<f32> {
    if crop_h <= 0.0 {
        return None;
    }

    let artistic_offset = artistic.target_y_offset();

    // Minimum target_y_frac to include the person's top edge.
    let min_frac = (person_height / (2.0 * crop_h)).clamp(0.10, 0.50);
    // Maximum target_y_frac to include the person's bottom edge.
    let max_frac = (1.0 - person_height / (2.0 * crop_h)).clamp(0.10, 0.50);

    if faces.is_empty() {
        // No faces: if the person fits, use the centered value (0.5 + artistic offset),
        // clamped into the valid range. Only override if the default would clip the person.
        if crop_h >= person_height {
            let default_frac = (0.50 + artistic_offset).clamp(0.10, 0.50);
            if default_frac >= min_frac && default_frac <= max_frac {
                // Default already works; no override needed.
                return None;
            }
            // Clamp to ensure the person fits.
            return Some(default_frac.clamp(min_frac, max_frac));
        }
        // Person doesn't fit; can't do better than the default.
        return None;
    }

    // With faces: bias so the dominant face top sits at roughly 20% from crop top.
    // Dominant face = the first entry (highest confidence after NMS, or largest by area).
    let face = &faces[0];
    let face_top = face.y1;

    // We want: crop_y ≈ face_top - 0.10 * crop_h (10% headroom above face)
    //          crop_y = person_center_y - target_y_frac * crop_h
    // Solving: target_y_frac = (person_center_y - face_top + 0.10 * crop_h) / crop_h
    let headroom_frac = 0.10_f32; // 10% of crop height above the face
    let ideal = (person_center_y - face_top + headroom_frac * crop_h) / crop_h;

    // Clamp into [min_frac, max_frac] to ensure person stays in frame (when it fits),
    // then apply the [0.10, 0.50] absolute clamp.
    let clamped = if crop_h >= person_height {
        ideal.clamp(min_frac, max_frac)
    } else {
        // Person is taller than crop; prioritise the face top.
        ideal.clamp(0.10, 0.50)
    };

    // Apply artistic offset, then final safety clamp.
    let with_artistic = (clamped + artistic_offset).clamp(0.10, 0.50);

    Some(with_artistic)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ArtisticCropConfig, ArtisticMode, CropConfig};

    fn default_config() -> CropConfig {
        CropConfig::default()
    }

    fn balanced_artistic() -> ArtisticCropConfig {
        ArtisticCropConfig::from_mode(ArtisticMode::Balanced)
    }

    fn tall_person() -> BBox {
        // Person bbox centred in a 3024×4032 photo.
        BBox {
            x1: 800.0,
            y1: 200.0,
            x2: 2200.0,
            y2: 3800.0,
        }
    }

    fn face_bbox() -> BBox {
        // Face near the top of the person bbox.
        BBox {
            x1: 1000.0,
            y1: 300.0,
            x2: 1900.0,
            y2: 900.0,
        }
    }

    // ── No-op case (enabled_enhanced_crop=false equivalent) ──────────────────

    #[test]
    fn test_analyze_joint_no_faces_no_override_when_fits() {
        let person = BBox {
            x1: 400.0,
            y1: 100.0,
            x2: 800.0,
            y2: 700.0,
        };
        let config = default_config();
        let artistic = balanced_artistic();
        // 9:16 portrait of a 1080×1920 photo — crop fills the photo, person fits.
        let result = analyze_joint(
            &person,
            &[],
            1080,
            1920,
            9.0 / 16.0,
            false,
            &config,
            &artistic,
        );
        // No relaxation needed for a reasonably-sized photo.
        assert!(!result.relaxed, "no relaxation for normal photo");
        assert_eq!(
            result.extra_margin_percent, 0.0,
            "no extra margin for normal photo"
        );
    }

    // ── Full-person-height preservation ──────────────────────────────────────

    #[test]
    fn test_analyze_joint_tall_person_portrait_full_height_expected() {
        // 3024×4032 photo, tall person, 9:16 format.
        // crop_h = photo_h = 4032; person_height = 3600. Person fits.
        let config = default_config();
        let artistic = balanced_artistic();
        let result = analyze_joint(
            &tall_person(),
            &[],
            3024,
            4032,
            9.0 / 16.0,
            false,
            &config,
            &artistic,
        );
        assert!(
            result.full_person_height_expected,
            "tall person should fit in a full-height portrait crop"
        );
    }

    #[test]
    fn test_analyze_joint_with_face_produces_override() {
        let config = default_config();
        let artistic = balanced_artistic();
        let result = analyze_joint(
            &tall_person(),
            &[face_bbox()],
            3024,
            4032,
            9.0 / 16.0,
            false,
            &config,
            &artistic,
        );
        // With a face, we should get a target_y_frac_override.
        assert!(
            result.target_y_frac_override.is_some(),
            "face should trigger vertical bias override"
        );
        let frac = result.target_y_frac_override.unwrap();
        assert!(
            (0.10..=0.50).contains(&frac),
            "override must be in [0.10, 0.50], got {frac}"
        );
    }

    // ── Min-dimension relaxation ─────────────────────────────────────────────

    #[test]
    fn test_analyze_joint_small_photo_triggers_relaxation() {
        // 400×600 photo — portrait 9:16 crop would be 400×711 (capped to 400×600).
        // min_long_side_pixels default = 1200 > 600 → relaxation fires.
        let person = BBox {
            x1: 50.0,
            y1: 50.0,
            x2: 350.0,
            y2: 550.0,
        };
        let config = default_config(); // min_long_side_pixels = 1200
        let artistic = balanced_artistic();
        let result = analyze_joint(
            &person,
            &[],
            400,
            600,
            9.0 / 16.0,
            false,
            &config,
            &artistic,
        );
        assert!(result.relaxed, "small photo must trigger relaxation");
        assert!(
            result.extra_margin_percent >= 0.0,
            "extra margin must be non-negative"
        );
    }

    #[test]
    fn test_analyze_joint_large_photo_no_relaxation() {
        // 3024×4032 photo at 9:16 — crop_w = 3024, crop_h = 5376 capped to 4032.
        // Both sides are well above 1280 defaults.
        let config = default_config();
        let artistic = balanced_artistic();
        let result = analyze_joint(
            &tall_person(),
            &[],
            3024,
            4032,
            9.0 / 16.0,
            false,
            &config,
            &artistic,
        );
        assert!(!result.relaxed, "large photo should not trigger relaxation");
        assert_eq!(result.extra_margin_percent, 0.0);
    }

    /// Landscape (21:9) relaxation fires when the crop width (long side) is below threshold.
    #[test]
    fn test_relaxation_fires_landscape_21x9_small_photo() {
        let person = BBox {
            x1: 10.0,
            y1: 10.0,
            x2: 90.0,
            y2: 90.0,
        };
        let config = default_config(); // min_long_side_pixels = 1200
        let artistic = balanced_artistic();
        // 200×100 photo, 21:9 landscape → crop capped to photo width (200) < 1200.
        let result = analyze_joint(&person, &[], 200, 100, 21.0 / 9.0, true, &config, &artistic);
        assert!(result.relaxed, "small landscape must trigger relaxation");
    }

    /// Mobile (9:21) relaxation fires when the crop height (long side) is below threshold.
    #[test]
    fn test_relaxation_fires_mobile_9x21_small_photo() {
        let person = BBox {
            x1: 10.0,
            y1: 10.0,
            x2: 90.0,
            y2: 190.0,
        };
        let config = default_config(); // min_long_side_pixels = 1200
        let artistic = balanced_artistic();
        // 100×200 photo, 9:21 mobile → crop capped to photo height (200) < 1200.
        let result = analyze_joint(
            &person,
            &[],
            100,
            200,
            9.0 / 21.0,
            false,
            &config,
            &artistic,
        );
        assert!(result.relaxed, "small mobile must trigger relaxation");
    }

    // ── Target_y_frac range safety ────────────────────────────────────────────

    #[test]
    fn test_target_y_frac_override_never_outside_range() {
        let config = default_config();
        let artistic = balanced_artistic();
        // Run with various person/face combinations and assert range is always safe.
        let cases: Vec<(BBox, Vec<BBox>)> = vec![
            (tall_person(), vec![]),
            (tall_person(), vec![face_bbox()]),
            (
                BBox {
                    x1: 0.0,
                    y1: 0.0,
                    x2: 3024.0,
                    y2: 4032.0,
                },
                vec![],
            ),
        ];
        for (person, faces) in cases {
            let result = analyze_joint(
                &person,
                &faces,
                3024,
                4032,
                9.0 / 16.0,
                false,
                &config,
                &artistic,
            );
            if let Some(frac) = result.target_y_frac_override {
                assert!(
                    (0.10..=0.50).contains(&frac),
                    "target_y_frac_override {frac} outside [0.10, 0.50]"
                );
            }
        }
    }

    #[test]
    fn test_extra_margin_never_negative() {
        let config = default_config();
        let artistic = balanced_artistic();
        let result = analyze_joint(
            &tall_person(),
            &[face_bbox()],
            300,
            400,
            9.0 / 16.0,
            false,
            &config,
            &artistic,
        );
        assert!(
            result.extra_margin_percent >= 0.0,
            "extra_margin_percent must never be negative"
        );
    }
}
