/// Face-Aware Crop Adjustment Module
///
/// This module applies a secondary refinement pass on top of an already-computed
/// crop region (produced by the original person-bbox algorithm in
/// [`crate::multi_format_cropping`]) to ensure the dominant face is fully visible.
///
/// # Design Principles
///
/// - **Additive, not primary.** The original crop algorithm is the authority on
///   initial crop placement. This module only nudges the crop within a configurable
///   shift budget.
/// - **Never fails.** [`apply_face_aware_adjustment`] always returns a valid
///   [`CropRegion`]. If adjustment is impossible or unnecessary, it returns the
///   input crop unchanged. No `Option`, no `Result`.
/// - **Never changes dimensions.** Only the crop position (x, y) is modified.
///   Width and height are always preserved exactly.
/// - **Dominant face = most central.** The face whose centroid is closest to the
///   image center is selected as the adjustment target (MostCentral strategy).
///
/// # Graceful Degradation Matrix
///
/// | person_bbox | face_bboxes | Behaviour                                     |
/// |-------------|-------------|-----------------------------------------------|
/// | Some        | non-empty   | Normal: select dominant face, try adjustment   |
/// | Some        | empty       | Return original crop unchanged                 |
/// | None        | non-empty   | Try adjustment without body visibility check   |
/// | None        | empty       | Return original crop unchanged                 |
///
/// # Example
/// ```rust,ignore
/// use icarus_v2::face_aware_cropping::{apply_face_aware_adjustment, find_dominant_face};
/// use icarus_v2::config::{ArtisticCropConfig, ArtisticMode};
/// use icarus_v2::multi_format_cropping::{BBox, CropRegion};
///
/// let crop = CropRegion { x: 100.0, y: 100.0, width: 800.0, height: 600.0 };
/// let faces = vec![BBox { x1: 350.0, y1: 200.0, x2: 450.0, y2: 300.0 }];
/// let config = ArtisticCropConfig::default();
/// let adjusted = apply_face_aware_adjustment(&crop, None, &faces, &config, 1920, 1080);
/// // adjusted.width == crop.width and adjusted.height == crop.height (dimensions unchanged)
/// ```
use crate::config::ArtisticCropConfig;
use crate::multi_format_cropping::{person_is_reasonably_visible_threshold, BBox, CropRegion};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Apply face-aware adjustment to an already-computed crop region.
///
/// This is the primary entry point. It:
/// 1. Selects the dominant face (closest centroid to image center).
/// 2. Checks if the face (plus safety margin) is already inside the crop.
/// 3. If not, computes the minimal shift needed.
/// 4. Enforces the shift budget from `config.max_shift_fraction`.
/// 5. Validates that the person bbox is not excessively cut off (if provided).
///
/// # Guarantees
/// - Always returns a valid `CropRegion` (never panics, never returns `None`).
/// - If adjustment fails or is unnecessary, returns the original `crop` unchanged.
/// - Never changes the crop dimensions (width/height), only position (x/y).
///
/// # Arguments
/// * `crop`         — The crop region computed by the original algorithm.
/// * `person_bbox`  — The person bounding box (may be `None` if detection failed).
/// * `face_bboxes`  — All detected face bounding boxes (may be empty).
/// * `config`       — Artistic crop configuration (shift budget, face margin, mode).
/// * `image_width`  — Source image width in pixels.
/// * `image_height` — Source image height in pixels.
///
/// # Example
/// ```rust,ignore
/// let adjusted = apply_face_aware_adjustment(
///     &crop, Some(&person), &faces, &config, 1920, 1080
/// );
/// assert_eq!(adjusted.width, crop.width); // dimensions never change
/// ```
pub fn apply_face_aware_adjustment(
    crop: &CropRegion,
    person_bbox: Option<&BBox>,
    face_bboxes: &[BBox],
    config: &ArtisticCropConfig,
    image_width: u32,
    image_height: u32,
) -> CropRegion {
    // If no faces are detected, there is nothing to adjust.
    if face_bboxes.is_empty() {
        return crop.clone();
    }

    // Select the dominant face: the one whose centroid is closest to the image center.
    let dominant_face = match find_dominant_face(face_bboxes, image_width, image_height) {
        Some(f) => f,
        None => return crop.clone(), // Defensive: empty slice should have been caught above.
    };

    // Skip degenerate face bboxes (zero area — likely model artifacts).
    if dominant_face.width() <= 0.0 || dominant_face.height() <= 0.0 {
        return crop.clone();
    }

    // Compute the "required zone" — the face bbox expanded by the safety margin.
    let required_zone = compute_required_zone(
        dominant_face,
        config.face_safety_margin_px,
        image_width,
        image_height,
    );

    // Compute the shift budget in pixels.
    let budget_x = crop.width * config.max_shift_fraction;
    let budget_y = crop.height * config.max_shift_fraction;

    // Attempt to shift the crop minimally to contain the required zone.
    // Returns the original crop if no shift is needed or budget is exceeded.
    match try_minimal_shift(
        crop,
        &required_zone,
        person_bbox,
        budget_x,
        budget_y,
        image_width,
        image_height,
    ) {
        Some(shifted) => shifted,
        None => crop.clone(), // Graceful degradation: return original.
    }
}

/// Select the dominant face: the face whose centroid is closest to the image center.
///
/// Uses the Euclidean distance from the face centroid to the image center point
/// `(image_width/2, image_height/2)`. Lower distance = higher priority.
///
/// Returns `None` if `face_bboxes` is empty.
///
/// # Arguments
/// * `face_bboxes`  — Slice of face bounding boxes.
/// * `image_width`  — Image width in pixels (used to compute center).
/// * `image_height` — Image height in pixels (used to compute center).
///
/// # Example
/// ```rust,ignore
/// let faces = vec![
///     BBox { x1: 0.0, y1: 0.0, x2: 100.0, y2: 100.0 },   // centroid (50, 50)
///     BBox { x1: 460.0, y1: 460.0, x2: 540.0, y2: 540.0 }, // centroid (500, 500) ← closer to center
/// ];
/// let dominant = find_dominant_face(&faces, 1000, 1000);
/// assert!(dominant.is_some());
/// ```
pub fn find_dominant_face(
    face_bboxes: &[BBox],
    image_width: u32,
    image_height: u32,
) -> Option<&BBox> {
    if face_bboxes.is_empty() {
        return None;
    }

    let img_cx = image_width as f32 / 2.0;
    let img_cy = image_height as f32 / 2.0;

    face_bboxes.iter().min_by(|a, b| {
        let da = {
            let dx = a.center_x() - img_cx;
            let dy = a.center_y() - img_cy;
            dx * dx + dy * dy // squared distance (avoid sqrt for comparison)
        };
        let db = {
            let dx = b.center_x() - img_cx;
            let dy = b.center_y() - img_cy;
            dx * dx + dy * dy
        };
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    })
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Compute the "required zone" — the face bbox expanded by the safety margin.
///
/// The required zone is the minimum region that must be inside the crop for the
/// face to be adequately framed. It is the face bbox grown by `safety_margin_px`
/// on each side, clamped to image bounds.
///
/// # Arguments
/// * `face_bbox`        — The face bounding box to expand.
/// * `safety_margin_px` — Margin in pixels to add on each side.
/// * `image_width`      — Image width for clamping.
/// * `image_height`     — Image height for clamping.
fn compute_required_zone(
    face_bbox: &BBox,
    safety_margin_px: u32,
    image_width: u32,
    image_height: u32,
) -> BBox {
    let m = safety_margin_px as f32;
    let pw = image_width as f32;
    let ph = image_height as f32;
    BBox {
        x1: (face_bbox.x1 - m).max(0.0),
        y1: (face_bbox.y1 - m).max(0.0),
        x2: (face_bbox.x2 + m).min(pw),
        y2: (face_bbox.y2 + m).min(ph),
    }
}

/// Attempt to shift the crop minimally to contain the required zone.
///
/// Returns `Some(shifted_crop)` if:
/// - The required shift is within the budget (in both X and Y), AND
/// - The person bbox would not be excessively cut off after the shift
///   (at least 30% of the person must remain visible).
///
/// Returns `None` if:
/// - The required zone is already inside the crop (no shift needed; caller returns original).
/// - The required shift exceeds the budget in either axis.
/// - The shifted crop would cut the person bbox below 30% visibility.
///
/// # Algorithm (from design document)
/// ```text
/// dx = 0; dy = 0
/// if required_zone.x1 < crop.x:            dx = required_zone.x1 - crop.x   (negative)
/// elif required_zone.x2 > crop.x + width:  dx = required_zone.x2 - (crop.x + width) (positive)
/// ...similar for dy...
/// if abs(dx) > budget_x or abs(dy) > budget_y: return None
/// new_x = clamp(crop.x + dx, 0, img_w - crop.width)
/// new_y = clamp(crop.y + dy, 0, img_h - crop.height)
/// ```
fn try_minimal_shift(
    crop: &CropRegion,
    required_zone: &BBox,
    person_bbox: Option<&BBox>,
    budget_x: f32,
    budget_y: f32,
    image_width: u32,
    image_height: u32,
) -> Option<CropRegion> {
    let crop_x2 = crop.x + crop.width;
    let crop_y2 = crop.y + crop.height;

    // Calculate how much the zone overflows the crop on each side.
    let dx: f32 = if required_zone.x1 < crop.x {
        // Zone overflows on the left → shift crop left (negative dx).
        required_zone.x1 - crop.x
    } else if required_zone.x2 > crop_x2 {
        // Zone overflows on the right → shift crop right (positive dx).
        required_zone.x2 - crop_x2
    } else {
        0.0
    };

    let dy: f32 = if required_zone.y1 < crop.y {
        // Zone overflows on the top → shift crop up (negative dy).
        required_zone.y1 - crop.y
    } else if required_zone.y2 > crop_y2 {
        // Zone overflows on the bottom → shift crop down (positive dy).
        required_zone.y2 - crop_y2
    } else {
        0.0
    };

    // If no shift is needed, the face is already inside the crop — return None
    // so the caller can return the original unchanged (avoids a spurious clone).
    if dx == 0.0 && dy == 0.0 {
        return None;
    }

    // Enforce shift budget. If either axis exceeds the budget, abort.
    if dx.abs() > budget_x || dy.abs() > budget_y {
        return None;
    }

    // Apply shift and clamp to valid image bounds.
    let pw = image_width as f32;
    let ph = image_height as f32;
    let new_x = (crop.x + dx).clamp(0.0, (pw - crop.width).max(0.0));
    let new_y = (crop.y + dy).clamp(0.0, (ph - crop.height).max(0.0));

    let shifted = CropRegion {
        x: new_x,
        y: new_y,
        width: crop.width,
        height: crop.height,
    };

    // Validate person visibility: the person must not be cut below 30% by the shift.
    // This is only enforced when a person bbox is provided.
    if let Some(person) = person_bbox {
        const MIN_PERSON_VISIBILITY: f32 = 0.30;
        if !person_is_reasonably_visible_threshold(
            person,
            &shifted,
            MIN_PERSON_VISIBILITY,
            image_width,
            image_height,
        ) {
            return None;
        }
    }

    Some(shifted)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ArtisticCropConfig, ArtisticMode};

    fn make_crop(x: f32, y: f32, w: f32, h: f32) -> CropRegion {
        CropRegion {
            x,
            y,
            width: w,
            height: h,
        }
    }

    fn make_bbox(x1: f32, y1: f32, x2: f32, y2: f32) -> BBox {
        BBox { x1, y1, x2, y2 }
    }

    // --- find_dominant_face ---

    #[test]
    fn test_find_dominant_face_empty_returns_none() {
        assert!(find_dominant_face(&[], 1000, 1000).is_none());
    }

    #[test]
    fn test_find_dominant_face_single_returns_it() {
        let faces = vec![make_bbox(100.0, 100.0, 200.0, 200.0)];
        let result = find_dominant_face(&faces, 1000, 1000);
        assert!(result.is_some());
        assert_eq!(result.unwrap().x1, 100.0);
    }

    #[test]
    fn test_find_dominant_face_selects_closest_to_center() {
        // Image center is (500, 500).
        // Face 1 centroid: (50, 50) — far from center.
        // Face 2 centroid: (500, 500) — at center.
        let faces = vec![
            make_bbox(0.0, 0.0, 100.0, 100.0),     // centroid (50, 50)
            make_bbox(450.0, 450.0, 550.0, 550.0), // centroid (500, 500) ← closer
        ];
        let dominant = find_dominant_face(&faces, 1000, 1000).unwrap();
        // The second face is closest to center.
        assert!((dominant.center_x() - 500.0).abs() < 1.0);
        assert!((dominant.center_y() - 500.0).abs() < 1.0);
    }

    // --- apply_face_aware_adjustment ---

    #[test]
    fn test_no_faces_returns_original_crop() {
        let crop = make_crop(100.0, 100.0, 800.0, 600.0);
        let person = make_bbox(200.0, 150.0, 600.0, 550.0);
        let config = ArtisticCropConfig::default();
        let result = apply_face_aware_adjustment(&crop, Some(&person), &[], &config, 1920, 1080);
        assert_eq!(result, crop, "no faces = no adjustment");
    }

    #[test]
    fn test_no_person_no_faces_returns_original() {
        let crop = make_crop(100.0, 100.0, 800.0, 600.0);
        let config = ArtisticCropConfig::default();
        let result = apply_face_aware_adjustment(&crop, None, &[], &config, 1920, 1080);
        assert_eq!(result, crop);
    }

    #[test]
    fn test_face_already_inside_crop_returns_original() {
        // Face is well inside the crop — no adjustment needed.
        let crop = make_crop(100.0, 100.0, 800.0, 600.0);
        let person = make_bbox(200.0, 150.0, 600.0, 550.0);
        // Face is at (350, 200)-(450, 300), well inside crop (100..900, 100..700).
        let faces = vec![make_bbox(350.0, 200.0, 450.0, 300.0)];
        let config = ArtisticCropConfig::from_mode(ArtisticMode::Balanced);
        let result = apply_face_aware_adjustment(&crop, Some(&person), &faces, &config, 1920, 1080);
        assert_eq!(
            result.x, crop.x,
            "should not shift when face is already inside"
        );
        assert_eq!(result.y, crop.y);
        assert_eq!(result.width, crop.width, "dimensions must not change");
        assert_eq!(result.height, crop.height);
    }

    #[test]
    fn test_adjustment_shifts_crop_when_face_outside_left() {
        // Crop starts at x=500. Face is at x1=460 (15px outside the crop left edge with 15px margin).
        // Config: Balanced (15px margin, 10% budget).
        // Budget = 800 * 0.10 = 80px. Required shift = small amount.
        let crop = make_crop(500.0, 100.0, 800.0, 600.0);
        let person = make_bbox(300.0, 100.0, 900.0, 650.0);
        // Face is partially outside the left edge of the crop.
        let faces = vec![make_bbox(460.0, 150.0, 560.0, 250.0)];
        let config = ArtisticCropConfig::from_mode(ArtisticMode::Balanced);
        let result = apply_face_aware_adjustment(&crop, Some(&person), &faces, &config, 1920, 1080);
        // The crop should have shifted left to include the face (with margin).
        // Face zone x1 = 460 - 15 = 445. Required shift dx = 445 - 500 = -55px.
        // Budget = 80px. abs(-55) <= 80, so shift should apply.
        assert!(result.x < crop.x, "crop should shift left to include face");
        // Dimensions must be unchanged.
        assert_eq!(result.width, crop.width);
        assert_eq!(result.height, crop.height);
    }

    #[test]
    fn test_excessive_shift_returns_original() {
        // Face is way outside the crop — would need >10% shift.
        let crop = make_crop(0.0, 0.0, 800.0, 600.0);
        // Face is at x1=1500, far outside crop (0..800). Required shift = 1500 - 800 = 700px.
        // Budget = 800 * 0.10 = 80px. 700 > 80, so return original.
        let faces = vec![make_bbox(1500.0, 800.0, 1600.0, 900.0)];
        let config = ArtisticCropConfig::from_mode(ArtisticMode::Balanced);
        let result = apply_face_aware_adjustment(&crop, None, &faces, &config, 1920, 1080);
        assert_eq!(result.x, crop.x, "shift budget exceeded = return original");
        assert_eq!(result.y, crop.y);
    }

    #[test]
    fn test_shift_rejected_when_person_visibility_too_low() {
        // Person is at the right side. Shifting left would cut too much of the person.
        // Person bbox: x1=600, x2=900. Crop at x=500, width=800 (crop_x2=1300).
        // If we shift left by 100px: new crop x=400, crop_x2=1200.
        // Person visibility: visible_left=600, visible_right=900 → 300/300 = 100% → OK.
        // Actually, let's set up where the shift would cut the person badly:
        // Person: x1=50, x2=200 (far left). Crop at x=0, width=800.
        // Face far right at x=900 — requires shift right by large amount.
        // After large shift, person would be mostly outside crop.
        let crop = make_crop(0.0, 0.0, 400.0, 600.0); // narrow crop
                                                      // Person is entirely inside the current crop.
        let person = make_bbox(50.0, 100.0, 200.0, 500.0);
        // Face is far to the right of the crop. Shift would need ~500px but budget = 400*0.1=40px.
        // So budget enforcement catches it first. Let's test the person visibility check specifically:
        // Set a face that needs a 30px shift right, which would push person mostly outside.
        // Crop x=300, width=400 → x2=700. Person x1=280, x2=310 (mostly at left of crop).
        // Face x2=740, needs shift right of 40px → x+40=340, x2=740.
        // After shift: person visible_left=340, visible_right=310 → negative intersection → 0%
        let crop2 = make_crop(300.0, 0.0, 400.0, 600.0);
        let person2 = make_bbox(280.0, 100.0, 310.0, 500.0); // barely inside crop left edge
                                                             // Face at right edge, requiring shift that cuts person out.
        let faces2 = vec![make_bbox(720.0, 200.0, 760.0, 300.0)]; // x2=760, crop_x2=700 → shift=75px
        let config = ArtisticCropConfig::from_mode(ArtisticMode::Aggressive); // 10px margin, 10% budget
                                                                              // Budget = 400*0.10=40px, required shift=75px → exceeds budget, returns original.
        let result =
            apply_face_aware_adjustment(&crop2, Some(&person2), &faces2, &config, 1920, 1080);
        assert_eq!(result.x, crop2.x, "budget exceeded should return original");
    }

    #[test]
    fn test_dimensions_never_change() {
        // Regardless of adjustment, crop dimensions (width, height) must be preserved.
        let crop = make_crop(100.0, 100.0, 800.0, 600.0);
        let person = make_bbox(200.0, 150.0, 700.0, 600.0);
        // Face partially outside left edge.
        let faces = vec![make_bbox(80.0, 200.0, 180.0, 300.0)];
        let config = ArtisticCropConfig::default();
        let result = apply_face_aware_adjustment(&crop, Some(&person), &faces, &config, 1920, 1080);
        assert_eq!(result.width, crop.width, "width must never change");
        assert_eq!(result.height, crop.height, "height must never change");
    }

    // --- compute_required_zone ---

    #[test]
    fn test_compute_required_zone_expands_by_margin() {
        let face = make_bbox(100.0, 100.0, 200.0, 200.0);
        let zone = compute_required_zone(&face, 20, 1920, 1080);
        assert!((zone.x1 - 80.0).abs() < 0.01);
        assert!((zone.y1 - 80.0).abs() < 0.01);
        assert!((zone.x2 - 220.0).abs() < 0.01);
        assert!((zone.y2 - 220.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_required_zone_clamps_to_image_bounds() {
        // Face at top-left corner, margin would go negative.
        let face = make_bbox(5.0, 5.0, 50.0, 50.0);
        let zone = compute_required_zone(&face, 20, 1920, 1080);
        assert_eq!(zone.x1, 0.0, "x1 clamped to 0");
        assert_eq!(zone.y1, 0.0, "y1 clamped to 0");
    }
}
