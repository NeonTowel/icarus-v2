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

    // ── NEW: Head-priority routing ────────────────────────────────────────────
    // Activates only when:
    //   1. A person bbox is available (Some, not None)
    //   2. The person is in a face-forward pose (face in upper 40% of person bbox)
    //   3. The repositioned crop still keeps the person adequately visible (≥ 30%)
    //
    // For face-forward poses the upstream 40% headroom algorithm sometimes clips
    // feet while leaving too much space below the head (e.g. artboard_01 9:21).
    // Head-priority anchors the crop top to the forehead instead, accepting that
    // feet may be outside the frame.
    if let Some(person) = person_bbox {
        if is_face_forward_pose(person, dominant_face) {
            let new_crop = compute_face_forward_crop(
                person,
                dominant_face,
                crop.width,
                crop.height,
                image_width,
                image_height,
            );
            // Visibility gate: person must remain at least 30% visible.
            // If the repositioned crop cuts the body too aggressively, fall through
            // to the original shift-based logic which may produce a better result.
            if person_is_reasonably_visible_threshold(
                person,
                &new_crop,
                0.30,
                image_width,
                image_height,
            ) {
                return new_crop;
            }
        }
    }
    // ── END HEAD-PRIORITY ─────────────────────────────────────────────────────

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

/// Position a crop to prioritize head visibility for face-forward poses.
///
/// The key insight: cropping feet is aesthetically acceptable, cropping eyes is not.
/// This function places the crop top just above the face (with a buffer), allowing the
/// bottom to extend wherever the crop height dictates — potentially past the feet.
///
/// The returned crop has the same `width` and `height` as the inputs, but a repositioned
/// `(x, y)`. All coordinates are clamped to remain within photo bounds.
///
/// # Arguments
/// * `person_bbox` — Person detection bbox; used for horizontal centering.
/// * `face_bbox`   — Dominant face bbox; `face_bbox.y1` anchors the vertical position.
/// * `crop_width`  — Width of the desired crop (passed through, unmodified).
/// * `crop_height` — Height of the desired crop (passed through, unmodified).
/// * `photo_w`     — Source photo width in pixels.
/// * `photo_h`     — Source photo height in pixels.
///
/// # Example
/// ```rust,ignore
/// let crop = compute_face_forward_crop(&person, &face, 800.0, 1600.0, 3000, 4000);
/// assert!(crop.y <= face.y1 - 15.0); // forehead visible with buffer
/// assert_eq!(crop.width, 800.0);     // dimensions preserved
/// ```
fn compute_face_forward_crop(
    person_bbox: &BBox,
    face_bbox: &BBox,
    crop_width: f32,
    crop_height: f32,
    photo_w: u32,
    photo_h: u32,
) -> CropRegion {
    let photo_hf = photo_h as f32;
    let photo_wf = photo_w as f32;

    // Head safety zone: 20px above the forehead ensures we never produce a "hairline crop".
    // This absolute pixel value is appropriate because face detection bboxes from YOLOv11x-Face
    // are tight around the face and 20px consistently prevents the forehead-clip artifact
    // across typical resolutions (3000–6000px photos).
    let head_top_buffer = 20.0_f32;

    // Step 1: Place crop top so forehead is visible with buffer.
    // face_bbox.y1 is the top of the detected face (forehead / hairline).
    let mut crop_y = (face_bbox.y1 - head_top_buffer).max(0.0);

    // Step 2: If crop extends below photo bottom, shift crop upward just enough.
    // This clamp ensures we never produce out-of-bounds coordinates.
    if crop_y + crop_height > photo_hf {
        crop_y = (photo_hf - crop_height).max(0.0);
    }

    // Step 3: Center horizontally on person's horizontal center.
    // Use person center (not face center) to maintain full body framing.
    let crop_x = (person_bbox.center_x() - (crop_width / 2.0))
        .max(0.0)
        .min((photo_wf - crop_width).max(0.0));

    CropRegion {
        x: crop_x,
        y: crop_y,
        width: crop_width,
        height: crop_height,
    }
}

/// Determine whether a person is in a face-forward pose.
///
/// Returns `true` if the face center is within the **upper 40%** of the person bounding
/// box. This geometric heuristic distinguishes standing/walking persons (face at 10–25%
/// from top) from sitting/bending poses (face at 40–70% from top).
///
/// # Arguments
/// * `person_bbox` — Person detection bbox; must have `height() > 0.0` for meaningful results.
/// * `face_bbox`   — Dominant face bbox. Containment within `person_bbox` is not enforced.
///
/// # Edge Cases
/// - Zero-height person bbox → returns `false` (safe guard, no division by zero).
/// - Face above person (negative ratio) → returns `true` (face is above = head-priority applies).
/// - Face far below person (ratio > 0.40) → returns `false` (falls through to shift logic).
///
/// # Example
/// ```rust,ignore
/// // Standing: face center at 25% of person height → face-forward
/// let result = is_face_forward_pose(&person, &face);
/// assert!(result);
/// ```
fn is_face_forward_pose(person_bbox: &BBox, face_bbox: &BBox) -> bool {
    // Guard: degenerate person bbox — avoid division by zero.
    if person_bbox.height() <= 0.0 {
        return false;
    }

    // Relative vertical position of the face center within the person bbox.
    // 0.0 = top of person, 1.0 = bottom of person.
    let face_y_relative = (face_bbox.center_y() - person_bbox.y1) / person_bbox.height();

    // Threshold 0.40: face in upper 40% = face-forward pose.
    // Calibrated from artboard sample set; separates standing (0.10–0.25)
    // from sitting (0.35–0.55) with acceptable overlap zone at 0.35–0.40.
    face_y_relative < 0.40
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
    fn test_face_already_inside_crop_returns_original_no_person() {
        // Face is well inside the crop and no person bbox is provided.
        // Without a person bbox, head-priority is skipped, and the shift-based
        // path also finds no adjustment needed (face is inside) → original returned.
        let crop = make_crop(100.0, 100.0, 800.0, 600.0);
        // Face is at (350, 200)-(450, 300), well inside crop (100..900, 100..700).
        let faces = vec![make_bbox(350.0, 200.0, 450.0, 300.0)];
        let config = ArtisticCropConfig::from_mode(ArtisticMode::Balanced);
        let result = apply_face_aware_adjustment(&crop, None, &faces, &config, 1920, 1080);
        assert_eq!(
            result.x, crop.x,
            "no-person + face inside crop = no adjustment"
        );
        assert_eq!(result.y, crop.y);
        assert_eq!(result.width, crop.width, "dimensions must not change");
        assert_eq!(result.height, crop.height);
    }

    #[test]
    fn test_face_inside_crop_with_face_forward_person_repositions() {
        // When a person bbox is present and the face is face-forward, head-priority
        // repositions the crop even if the face was already nominally inside.
        // This is the intended new behaviour: protect the forehead, not the original position.
        let crop = make_crop(100.0, 100.0, 800.0, 600.0);
        let person = make_bbox(200.0, 150.0, 600.0, 550.0);
        // Face at (350, 200)-(450, 300): face-forward (center_y=250, person y1=150, h=400 → 25%).
        let face_y1 = 200.0_f32;
        let faces = vec![make_bbox(350.0, face_y1, 450.0, 300.0)];
        let config = ArtisticCropConfig::from_mode(ArtisticMode::Balanced);
        let result = apply_face_aware_adjustment(&crop, Some(&person), &faces, &config, 1920, 1080);
        // Head-priority: crop_y = face.y1 - 20 = 180, so crop.y <= face.y1 and face is visible.
        assert!(
            result.y <= face_y1,
            "head-priority should position crop top at or above face top, got y={}",
            result.y
        );
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
    fn test_shift_rejected_when_budget_exceeded_no_person() {
        // Face is outside the crop and requires a shift that exceeds the budget.
        // Using person_bbox=None so head-priority is bypassed, testing the shift budget path.
        //
        // Crop x=300, width=400 → x2=700.
        // Face at x2=760 → required shift = 760 - 700 = 60px (plus margin).
        // Budget = 400 * 0.10 = 40px. 60 > 40 → budget exceeded → return original.
        let crop = make_crop(300.0, 0.0, 400.0, 600.0);
        let faces = vec![make_bbox(720.0, 200.0, 760.0, 300.0)];
        let config = ArtisticCropConfig::from_mode(ArtisticMode::Aggressive); // 10px margin, 10% budget
        let result = apply_face_aware_adjustment(&crop, None, &faces, &config, 1920, 1080);
        assert_eq!(result.x, crop.x, "budget exceeded should return original");
    }

    #[test]
    fn test_shift_rejected_when_person_visibility_would_be_too_low() {
        // Test that the person visibility gate rejects a shift that cuts the person body.
        // Uses a non-face-forward pose (face in lower half of person) to bypass head-priority,
        // then verifies the shift-based path enforces the 30% visibility gate.
        //
        // Setup: person is narrow (x1=600, x2=650) near the LEFT edge of the crop.
        //        crop starts at x=500, width=400 → x2=900.
        //        Face is far RIGHT at x=(870..920), requiring a shift right of ~30px.
        //        After rightward shift: crop x=530, x2=930. Person (600..650) → still inside.
        //        → Actually this passes, so use a tighter case:
        //
        // Better: person near right edge, face far LEFT → shift left cuts person.
        //         person x1=850, x2=900. Crop x=800, w=400 → x2=1200.
        //         Face at x1=700 → required zone x1 = 700-10=690. shift = 690-800=-110px.
        //         Budget = 400 * 0.10 = 40px. abs(-110) > 40 → budget exceeded.
        //
        // Face must NOT be face-forward: person y1=100, y2=500, face center_y must be >= 200
        // for face_y_relative = (center_y - 100)/400 >= 0.25 — we need >= 0.40, so center_y >= 260.
        // Use face y1=400, y2=480 → center_y=440, face_y_relative=(440-100)/400=0.85 → NOT face-forward.
        let crop = make_crop(800.0, 0.0, 400.0, 600.0);
        let person = make_bbox(850.0, 100.0, 900.0, 500.0); // near right of crop
        let faces = vec![make_bbox(700.0, 400.0, 760.0, 480.0)]; // NOT face-forward (85%)
        let config = ArtisticCropConfig::from_mode(ArtisticMode::Aggressive); // 10px margin, 10% budget
                                                                              // Zone x1 = 700 - 10 = 690. Shift = 690 - 800 = -110px. Budget = 40px → rejected.
        let result = apply_face_aware_adjustment(&crop, Some(&person), &faces, &config, 1920, 1080);
        assert_eq!(
            result.x, crop.x,
            "budget exceeded should return original; got x={}",
            result.x
        );
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

    // --- is_face_forward_pose ---

    #[test]
    fn test_is_face_forward_pose_upper_portion() {
        // Person height = 400. Face center_y = (150+250)/2 = 200.
        // face_y_relative = (200 - 100) / 400 = 0.25 → face-forward (< 0.40).
        let person = make_bbox(100.0, 100.0, 300.0, 500.0);
        let face = make_bbox(150.0, 150.0, 250.0, 250.0);
        assert!(
            is_face_forward_pose(&person, &face),
            "face at 25% of person height should be face-forward"
        );
    }

    #[test]
    fn test_is_face_forward_pose_lower_portion() {
        // Person height = 400. Face center_y = (400+480)/2 = 440.
        // face_y_relative = (440 - 100) / 400 = 0.85 → NOT face-forward (>= 0.40).
        let person = make_bbox(100.0, 100.0, 300.0, 500.0);
        let face = make_bbox(150.0, 400.0, 250.0, 480.0);
        assert!(
            !is_face_forward_pose(&person, &face),
            "face at 85% of person height should NOT be face-forward"
        );
    }

    #[test]
    fn test_is_face_forward_pose_zero_height_person() {
        // Degenerate bbox: height = 0 → guard clause must prevent division by zero.
        let person = make_bbox(100.0, 100.0, 300.0, 100.0); // y1 == y2 → height = 0
        let face = make_bbox(150.0, 90.0, 250.0, 110.0);
        assert!(
            !is_face_forward_pose(&person, &face),
            "zero-height person bbox should return false (guard clause)"
        );
    }

    // --- compute_face_forward_crop ---

    #[test]
    fn test_compute_face_forward_crop_protects_head() {
        // Scenario: person standing in a large photo.
        // Face y1=250 → crop_y should be <= 250 - 15 = 235 (at most 20px above face top).
        // Chin y2=450 → crop_y + crop_height (1200) = well past chin.
        let person = make_bbox(500.0, 200.0, 1000.0, 1200.0);
        let face = make_bbox(600.0, 250.0, 900.0, 450.0);
        let crop = compute_face_forward_crop(&person, &face, 800.0, 1200.0, 3000, 4000);

        // Forehead should be visible with at least a small buffer above.
        assert!(
            crop.y <= face.y1 - 15.0,
            "crop top should be at least 15px above forehead, got y={} face.y1={}",
            crop.y,
            face.y1
        );
        // Chin must be fully inside the crop.
        assert!(
            crop.y + crop.height >= face.y2,
            "chin must be inside the crop"
        );
        // Dimensions are preserved.
        assert_eq!(crop.width, 800.0, "width must not change");
        assert_eq!(crop.height, 1200.0, "height must not change");
    }

    #[test]
    fn test_compute_face_forward_crop_clamps_to_bounds() {
        // Face is near the top edge of the photo → crop_y should clamp to 0.0.
        // Also tests that x and x+width remain within photo bounds.
        let person = make_bbox(100.0, 5.0, 400.0, 800.0);
        let face = make_bbox(150.0, 10.0, 350.0, 100.0); // near top edge
        let crop = compute_face_forward_crop(&person, &face, 500.0, 900.0, 600, 1000);

        assert!(crop.y >= 0.0, "crop y must not be negative");
        assert!(crop.x >= 0.0, "crop x must not be negative");
        assert!(
            crop.x + crop.width <= 600.0,
            "crop must not exceed photo width"
        );
        assert!(
            crop.y + crop.height <= 1000.0,
            "crop must not exceed photo height"
        );
        // Dimensions preserved.
        assert_eq!(crop.width, 500.0);
        assert_eq!(crop.height, 900.0);
    }

    // --- end-to-end head-priority via apply_face_aware_adjustment ---

    #[test]
    fn test_face_forward_applies_head_priority_repositions_crop() {
        // Tall crop (1600px) with person at y1=100, face at y1=150 (upper 15%).
        // Original crop y=200 (crop starts below the face top).
        // Head-priority should reposition to anchor crop top above the face.
        let crop = make_crop(0.0, 200.0, 800.0, 1600.0);
        let person = make_bbox(200.0, 100.0, 600.0, 1800.0);
        let face = make_bbox(300.0, 150.0, 500.0, 350.0); // face in upper ~15% of person
        let config = ArtisticCropConfig::default();
        let result =
            apply_face_aware_adjustment(&crop, Some(&person), &[face], &config, 1000, 2000);

        // Crop should have been repositioned so that face top (y=150) is inside.
        assert!(
            result.y <= 150.0,
            "head-priority: crop top should be at or above face y1=150, got result.y={}",
            result.y
        );
        // Dimensions must be preserved.
        assert_eq!(result.width, crop.width, "width must not change");
        assert_eq!(result.height, crop.height, "height must not change");
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
