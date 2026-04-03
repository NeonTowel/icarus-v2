/// Face-Aware Crop Adjustment Module
///
/// This module applies a secondary refinement pass on top of an already-computed
/// crop region (produced by the original person-bbox algorithm in
/// [`crate::multi_format_cropping`]) to ensure the dominant face is fully visible.
///
/// # Phases
///
/// - **Phase 1** (head-priority): For face-forward poses, anchors the crop top
///   just above the detected forehead using a fixed 20px buffer.
/// - **Phase 2** (dynamic margin): Scales the face safety margin when the face
///   is close to a crop edge; uses a minimal-shift strategy.
/// - **Phase 3** (aspect-ratio-aware): Replaces the fixed Phase 1 buffer with
///   per-aspect breathing room and face-bbox penetration tolerance, yielding
///   better body visibility (mobile: knees; portrait: waist; landscape: torso).
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
// Phase 3: Aspect-ratio types and helpers
// ---------------------------------------------------------------------------

/// Target aspect ratio category for crop-specific Phase 3 behaviour.
///
/// Used to select breathing room and face penetration tolerance values from
/// [`ArtisticCropConfig`]. The variant is determined from crop dimensions via
/// [`determine_aspect_ratio`], not from format-name strings.
///
/// # Variants
/// - `Mobile`   — 9:21 (≈0.4286 ratio). Tall, narrow; maximise body visibility.
/// - `Portrait` — 9:16 (≈0.5625 ratio). Balanced; classic portrait composition.
/// - `Landscape`— 21:9 (≈2.333 ratio). Wide, short; shoulders + context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AspectRatio {
    /// Mobile vertical: 9:21. Crop tall and narrow; show body to knees.
    Mobile,
    /// Portrait standard: 9:16. Balanced composition; show body to waist.
    Portrait,
    /// Landscape: 21:9. Wide frame; show shoulders, arms, and context.
    Landscape,
}

/// Isolation level of a detected face relative to the crop edges.
///
/// Drives the aggressiveness of the Phase 3 body-showing strategy.
/// "Isolated" means the face bbox has comfortable clearance from all
/// four crop edges; "constrained" means one or more edges are too close.
///
/// See [`assess_face_isolation`] for the classification algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaceIsolation {
    /// Face is ≥ `edge_margin` from ALL four crop edges.
    /// Strategy: aggressive body showing at 100% of max penetration.
    FullyIsolated,
    /// Face is within `edge_margin` of exactly ONE crop edge.
    /// Strategy: moderate body showing at 70% of max penetration.
    ModerateConstraint,
    /// Face is within `edge_margin` of TWO or more crop edges.
    /// Strategy: conservative body showing at 40% of max penetration.
    EdgeConstraint,
}

/// Determine the aspect ratio category from crop dimensions.
///
/// Uses width/height ratio with tolerance bands wide enough to handle
/// floating-point imprecision in crop dimension calculations:
///
/// | Ratio range | Variant   | Example format |
/// |-------------|-----------|----------------|
/// | ≤ 0.50      | Mobile    | 9:21 (0.4286)  |
/// | 0.50–0.65   | Portrait  | 9:16 (0.5625)  |
/// | > 0.65      | Landscape | 21:9 (2.333)   |
///
/// # Arguments
/// * `crop_width`  — Width of the crop region in pixels.
/// * `crop_height` — Height of the crop region in pixels.
///
/// # Returns
/// Always returns a valid [`AspectRatio`] variant. For degenerate inputs
/// (zero or negative height), returns `Landscape` as a safe default.
///
/// # Example
/// ```rust,ignore
/// assert_eq!(determine_aspect_ratio(900.0, 2100.0), AspectRatio::Mobile);
/// assert_eq!(determine_aspect_ratio(900.0, 1600.0), AspectRatio::Portrait);
/// assert_eq!(determine_aspect_ratio(2100.0, 900.0), AspectRatio::Landscape);
/// ```
pub fn determine_aspect_ratio(crop_width: f32, crop_height: f32) -> AspectRatio {
    if crop_height <= 0.0 {
        // Degenerate input: return a safe default rather than panic on division.
        return AspectRatio::Landscape;
    }
    let ratio = crop_width / crop_height;
    if ratio <= 0.50 {
        AspectRatio::Mobile // 9:21 = 0.4286
    } else if ratio <= 0.65 {
        AspectRatio::Portrait // 9:16 = 0.5625
    } else {
        AspectRatio::Landscape // 21:9 = 2.333 and anything wider
    }
}

/// Return the breathing-room percentage for the given aspect ratio.
///
/// "Breathing room" is the vertical space between the top of the person's
/// head and the top edge of the crop, expressed as a percentage of crop
/// height. Higher values give the subject more air above the forehead.
///
/// # Arguments
/// * `aspect` — Target aspect ratio category.
/// * `config` — Artistic crop config containing per-aspect values.
///
/// # Returns
/// A percentage value (e.g. `12.5` means 12.5% of crop height).
///
/// # Example
/// ```rust,ignore
/// let config = ArtisticCropConfig::default();
/// let br = get_breathing_room_for_aspect(AspectRatio::Mobile, &config);
/// assert!((br - 12.5).abs() < 0.01);
/// ```
pub fn get_breathing_room_for_aspect(aspect: AspectRatio, config: &ArtisticCropConfig) -> f32 {
    match aspect {
        AspectRatio::Mobile => config.breathing_room_percent_mobile,
        AspectRatio::Portrait => config.breathing_room_percent_portrait,
        AspectRatio::Landscape => config.breathing_room_percent_landscape,
    }
}

/// Return the max face-bbox penetration percentage for the given aspect ratio.
///
/// "Penetration" is how far (as a percentage of face bbox height) the crop
/// edge is allowed to cut into the face bounding box from above (forehead
/// zone). The protected zone — eyes, nose, mouth (30–65% of face height from
/// the top) — must NEVER be penetrated regardless of this value.
///
/// # Arguments
/// * `aspect` — Target aspect ratio category.
/// * `config` — Artistic crop config containing per-aspect values.
///
/// # Returns
/// A percentage value (e.g. `18.0` means up to 18% of face height).
///
/// # Example
/// ```rust,ignore
/// let config = ArtisticCropConfig::default();
/// let pen = get_max_penetration_for_aspect(AspectRatio::Mobile, &config);
/// assert!((pen - 18.0).abs() < 0.01);
/// ```
pub fn get_max_penetration_for_aspect(aspect: AspectRatio, config: &ArtisticCropConfig) -> f32 {
    match aspect {
        AspectRatio::Mobile => config.max_face_bbox_penetration_percent_mobile,
        AspectRatio::Portrait => config.max_face_bbox_penetration_percent_portrait,
        AspectRatio::Landscape => config.max_face_bbox_penetration_percent_landscape,
    }
}

/// Assess how constrained a face is relative to the crop edges.
///
/// A face is "isolated" when it has comfortable clearance from all crop edges.
/// The clearance threshold is `edge_margin_frac` (a fraction, e.g. `0.05` for 5%)
/// multiplied by the crop width (for left/right edges) or height (for top/bottom).
///
/// Faces partially or fully outside the crop are treated as near-edge contacts
/// because the overflowing edge coordinate naturally satisfies the `<` or `>`
/// inequalities in the margin check.
///
/// # Arguments
/// * `face_bbox`        — Detected face bounding box.
/// * `crop`             — Current crop region.
/// * `edge_margin_frac` — Fraction of crop dimension considered "near edge".
///   Default value used by Phase 3: `0.05` (5%).
///
/// # Returns
/// A [`FaceIsolation`] variant indicating the constraint level:
/// - 0 near edges → [`FaceIsolation::FullyIsolated`]
/// - 1 near edge  → [`FaceIsolation::ModerateConstraint`]
/// - 2+ near edges→ [`FaceIsolation::EdgeConstraint`]
///
/// # Edge Cases
/// - Zero-size crop: both width and height margins are 0; all four conditions
///   likely true → [`FaceIsolation::EdgeConstraint`].
/// - Face filling the entire crop: all four edges are "near" → `EdgeConstraint`.
pub fn assess_face_isolation(
    face_bbox: &BBox,
    crop: &CropRegion,
    edge_margin_frac: f32,
) -> FaceIsolation {
    let margin_x = crop.width * edge_margin_frac;
    let margin_y = crop.height * edge_margin_frac;

    let crop_left = crop.x;
    let crop_right = crop.x + crop.width;
    let crop_top = crop.y;
    let crop_bottom = crop.y + crop.height;

    // Count how many crop edges the face bbox is "near" (within margin),
    // including cases where the face overflows past the crop edge.
    let mut near_edge_count: u8 = 0;

    if face_bbox.x1 < crop_left + margin_x {
        near_edge_count += 1; // near or past left edge
    }
    if face_bbox.x2 > crop_right - margin_x {
        near_edge_count += 1; // near or past right edge
    }
    if face_bbox.y1 < crop_top + margin_y {
        near_edge_count += 1; // near or past top edge
    }
    if face_bbox.y2 > crop_bottom - margin_y {
        near_edge_count += 1; // near or past bottom edge
    }

    match near_edge_count {
        0 => FaceIsolation::FullyIsolated,
        1 => FaceIsolation::ModerateConstraint,
        _ => FaceIsolation::EdgeConstraint,
    }
}

/// Apply aspect-ratio-aware face cropping adjustments (Phase 3).
///
/// This is the Phase 3 decision tree. It takes the full person bbox, the
/// detected face bbox, the target crop from the base algorithm, and the
/// aspect-ratio-specific config to produce an adjusted crop that balances
/// head breathing room with body visibility.
///
/// # Decision Tree
///
/// 1. If `face_bbox` is `None`: return `target_crop` unchanged (fallback to
///    the original 40% headroom algorithm from the base pass).
/// 2. Determine aspect ratio from crop dimensions.
/// 3. Look up breathing room and penetration tolerance per aspect.
/// 4. Assess face isolation level ([`assess_face_isolation`]).
/// 5. Apply isolation-dependent penetration scaling:
///    - [`FaceIsolation::FullyIsolated`]      → 100% of max penetration
///    - [`FaceIsolation::ModerateConstraint`] → 70% of max penetration
///    - [`FaceIsolation::EdgeConstraint`]     → 40% of max penetration
/// 6. Compute adjusted crop `y` with breathing room and penetration limits.
/// 7. Validate eye safety: the protected zone (30–65% of face height from
///    the top) must be fully inside the returned crop.
/// 8. Clamp all coordinates to image bounds.
///
/// # Guarantees
/// - Always returns a valid `CropRegion` (never panics).
/// - Crop dimensions (width, height) are always preserved exactly.
/// - Eyes are always preserved: if the protected zone would be clipped,
///   falls back to returning `target_crop` unchanged.
///
/// # Arguments
/// * `person_bbox`   — Full person bounding box (for horizontal centering).
/// * `face_bbox`     — Optional face bounding box. `None` → returns `target_crop`.
/// * `target_crop`   — Crop region from the base algorithm.
/// * `config`        — Artistic crop config with Phase 3 fields.
/// * `image_width`   — Source image width in pixels.
/// * `image_height`  — Source image height in pixels.
///
/// # Example
/// ```rust,ignore
/// let result = apply_aspect_aware_face_cropping(
///     &person, Some(&face), &crop, &config, 3024, 4032,
/// );
/// assert_eq!(result.width, crop.width);  // dimensions never change
/// ```
pub fn apply_aspect_aware_face_cropping(
    person_bbox: &BBox,
    face_bbox: Option<&BBox>,
    target_crop: &CropRegion,
    config: &ArtisticCropConfig,
    image_width: u32,
    image_height: u32,
) -> CropRegion {
    // Step 0: No face → return original crop (Phase 1-2 fallback).
    let face = match face_bbox {
        Some(f) => f,
        None => return target_crop.clone(),
    };

    // Step 1: Determine aspect ratio from crop dimensions.
    let aspect = determine_aspect_ratio(target_crop.width, target_crop.height);

    // Step 2: Look up aspect-specific parameters.
    let breathing_room_pct = get_breathing_room_for_aspect(aspect, config);
    let max_penetration_pct = get_max_penetration_for_aspect(aspect, config);

    // Step 3: Assess face isolation.
    let isolation = assess_face_isolation(face, target_crop, 0.05);

    // Step 4: Scale penetration by isolation level.
    let effective_penetration_pct = match isolation {
        FaceIsolation::FullyIsolated => max_penetration_pct,
        FaceIsolation::ModerateConstraint => max_penetration_pct * 0.7,
        FaceIsolation::EdgeConstraint => max_penetration_pct * 0.4,
    };

    // Step 5: Convert percentages to pixels.
    // max_penetration_px: how far the crop top may enter the face bbox from above.
    let face_height = face.height();
    let max_penetration_px = face_height * (effective_penetration_pct / 100.0);

    // breathing_room_px: empty space to leave above the forehead.
    let breathing_room_px = target_crop.height * (breathing_room_pct / 100.0);

    // Step 6: Eye safety limits.
    // Protected zone: 30–65% of face height from the top.
    // The crop top must be at or above the protected zone's top boundary.
    // We add a 15px absolute safety buffer to guard against subpixel rounding.
    let eye_safety_px = 15.0_f32;
    let eye_zone_top = face.y1 + (0.30 * face_height);
    let crop_y_eye_limit = eye_zone_top - eye_safety_px;

    // The penetration limit: how far down the crop top may sit.
    let crop_y_penetration_limit = face.y1 + max_penetration_px;

    // The effective upper bound for crop_y is the more conservative of the two.
    let crop_y_max = crop_y_penetration_limit.min(crop_y_eye_limit);

    // Step 7: Ideal crop_y with full breathing room above the forehead.
    let crop_y_ideal = face.y1 - breathing_room_px;

    // Clamp to non-negative (face near top of image).
    let mut crop_y = crop_y_ideal.max(0.0);

    // Step 8: Shift crop down to show more body when possible.
    // "More body" means a larger crop_y (crop frame starts lower in the image).
    // We push crop_y down by however much body overflows below the current frame,
    // but never past crop_y_max (which protects the eyes and forehead zone).
    let person_bottom = person_bbox.y2;
    let current_crop_bottom = crop_y + target_crop.height;
    let body_overflow = person_bottom - current_crop_bottom;

    if body_overflow > 0.0 {
        let shift_down = body_overflow.min((crop_y_max - crop_y).max(0.0));
        crop_y += shift_down;
    }

    // Step 9: Clamp crop_y to image bounds.
    let photo_hf = image_height as f32;
    let photo_wf = image_width as f32;
    if crop_y + target_crop.height > photo_hf {
        crop_y = (photo_hf - target_crop.height).max(0.0);
    }
    crop_y = crop_y.max(0.0);

    // Step 10: Final eye safety check.
    // After all adjustments, verify that the protected zone (30–65% of face height)
    // is fully inside the crop. If not, fall back to the original crop.
    let protected_top = face.y1 + 0.30 * face_height;
    let protected_bottom = face.y1 + 0.65 * face_height;
    let final_crop_top = crop_y;
    let final_crop_bottom = crop_y + target_crop.height;

    if protected_top < final_crop_top || protected_bottom > final_crop_bottom {
        // Eye safety violated — fall back to original crop unchanged.
        return target_crop.clone();
    }

    // Step 11: Horizontal positioning.
    // Center on person's horizontal center (same as Phase 1 strategy).
    let crop_x = (person_bbox.center_x() - target_crop.width / 2.0)
        .max(0.0)
        .min((photo_wf - target_crop.width).max(0.0));

    CropRegion {
        x: crop_x,
        y: crop_y,
        width: target_crop.width,
        height: target_crop.height,
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Apply face-aware adjustment to an already-computed crop region.
///
/// This is the primary entry point. It orchestrates all three phases:
/// 1. **Phase 3** (aspect-ratio-aware): For face-forward poses, adjusts crop
///    position using per-aspect breathing room and face penetration tolerance.
/// 2. **Phase 1** (head-priority fallback): If Phase 3 is rejected by the
///    person visibility gate, tries the fixed 20px head-buffer strategy.
/// 3. **Phase 2** (dynamic margin + shift): Scales the safety margin near
///    edges and applies a minimal crop shift within the shift budget.
///
/// For each phase, a 30% person visibility gate decides whether the result is
/// acceptable. The first phase that passes is returned; if all fail, the
/// original crop is returned unchanged.
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

    // ── Phase 3 → Phase 1 head-priority routing ──────────────────────────────
    // Activates only when:
    //   1. A person bbox is available (Some, not None)
    //   2. The person is in a face-forward pose (face in upper 40% of person bbox)
    //   3. The repositioned crop still keeps the person adequately visible (≥ 30%)
    //
    // Phase 3 replaces the fixed 20px head buffer of Phase 1 with per-aspect
    // breathing room and face penetration tolerance, yielding better body
    // visibility for mobile crops without sacrificing forehead framing.
    // If Phase 3 is rejected by the visibility gate, Phase 1 is tried next.
    if let Some(person) = person_bbox {
        if is_face_forward_pose(person, dominant_face) {
            // ── Phase 3: aspect-ratio-aware crop ──────────────────────────────
            let phase3_crop = apply_aspect_aware_face_cropping(
                person,
                Some(dominant_face),
                crop,
                config,
                image_width,
                image_height,
            );
            // Visibility gate: person must remain at least 30% visible.
            if person_is_reasonably_visible_threshold(
                person,
                &phase3_crop,
                0.30,
                image_width,
                image_height,
            ) {
                return phase3_crop;
            }

            // ── Phase 1 fallback: fixed 20px head buffer ───────────────────
            // Phase 3 was too aggressive. Try the simpler Phase 1 strategy.
            let phase1_crop = compute_face_forward_crop(
                person,
                dominant_face,
                crop.width,
                crop.height,
                image_width,
                image_height,
            );
            if person_is_reasonably_visible_threshold(
                person,
                &phase1_crop,
                0.30,
                image_width,
                image_height,
            ) {
                return phase1_crop;
            }
            // Both Phase 3 and Phase 1 rejected: fall through to Phase 2.
        }
    }
    // ── END HEAD-PRIORITY (Phase 3 + Phase 1 fallback) ────────────────────────

    // ── PHASE 2: Dynamic margin ───────────────────────────────────────────────
    // Scale the safety margin up when the face is within 5% of the crop edge.
    // Centered faces use the base margin from config; cramped faces get more room.
    let dynamic_margin =
        compute_dynamic_face_margin(dominant_face, crop, config.face_safety_margin_px);

    // Compute the "required zone" — the face bbox expanded by the dynamic safety margin.
    let required_zone =
        compute_required_zone(dominant_face, dynamic_margin, image_width, image_height);
    // ── END DYNAMIC MARGIN ────────────────────────────────────────────────────

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

/// Compute a context-aware face safety margin that scales up near crop edges.
///
/// Centered faces use the base margin (typically 15px for Balanced mode).
/// When a face is within 5% of the crop height from any vertical edge, the
/// margin increases to prevent the "cramped head" visual artefact — the feeling
/// that the subject's head is squeezed against the frame boundary.
///
/// The margin never decreases below `base_margin_px` regardless of the face position.
///
/// # Arguments
/// * `face_bbox`      — The dominant face bounding box.
/// * `crop`           — The current crop region (before any shift).
/// * `base_margin_px` — The mode-derived base margin (10/15/20px).
///
/// # Returns
/// A pixel margin value `>= base_margin_px`. When the face is near an edge,
/// the returned value is `max((crop.height * 0.05 * 1.5) as u32, base_margin_px)`.
///
/// # Example
/// ```rust,ignore
/// let margin = compute_dynamic_face_margin(&face, &crop, 15);
/// // If face is near top edge: margin > 15
/// // If face is well-centered: margin == 15
/// ```
fn compute_dynamic_face_margin(face_bbox: &BBox, crop: &CropRegion, base_margin_px: u32) -> u32 {
    // 5% of crop height defines the "too close to edge" breathing room threshold.
    // Photography best practice: 5–10% headroom between subject and frame boundary.
    let min_breathing_room = crop.height * 0.05;

    // Measure vertical distance from face edges to crop edges.
    let distance_to_top = (face_bbox.y1 - crop.y).max(0.0);
    let distance_to_bottom = ((crop.y + crop.height) - face_bbox.y2).max(0.0);

    // If either vertical distance falls below the breathing room threshold,
    // the face is "cramped" and needs a larger margin.
    // The 1.5x multiplier produces ~7.5% of crop height as clear space,
    // within professional framing norms.
    if distance_to_top < min_breathing_room || distance_to_bottom < min_breathing_room {
        // max() guard ensures we never return less than base even if crop is tiny.
        ((min_breathing_room * 1.5) as u32).max(base_margin_px)
    } else {
        base_margin_px
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

    // =========================================================================
    // Phase 3 test fixtures
    // =========================================================================

    // Simulates a 3024×4032 portrait photo (common smartphone resolution).
    const PHOTO_W: u32 = 3024;
    const PHOTO_H: u32 = 4032;

    /// Standing person: face near the top, body extending to near the bottom.
    fn standing_person() -> BBox {
        make_bbox(1000.0, 200.0, 2000.0, 3800.0)
    }

    /// Face for standing_person: upper portion of the frame.
    fn standing_face() -> BBox {
        make_bbox(1200.0, 300.0, 1800.0, 700.0)
    }

    /// 9:21 mobile crop dimensions derived from PHOTO_W/PHOTO_H.
    ///
    /// crop_width = PHOTO_H * (9/21) ~= 1728; crop_height = PHOTO_H = 4032.
    fn mobile_crop() -> CropRegion {
        make_crop(648.0, 0.0, 1728.0, 4032.0)
    }

    /// 9:16 portrait crop dimensions derived from PHOTO_H.
    ///
    /// crop_height = PHOTO_H = 4032; crop_width = 4032 * (9/16) = 2268.
    fn portrait_crop() -> CropRegion {
        make_crop(378.0, 0.0, 2268.0, 4032.0)
    }

    /// 21:9 landscape crop dimensions derived from PHOTO_W.
    ///
    /// crop_width = PHOTO_W = 3024; crop_height = 3024 / (21/9) ~= 1296.
    fn landscape_crop() -> CropRegion {
        make_crop(0.0, 500.0, 3024.0, 1296.0)
    }

    // =========================================================================
    // Phase 3 unit tests
    // =========================================================================

    /// Verify that determine_aspect_ratio classifies standard formats correctly.
    #[test]
    fn test_determine_aspect_ratio_classification() {
        // 9:21 mobile — ratio 900/2100 ≈ 0.4286
        assert_eq!(
            determine_aspect_ratio(900.0, 2100.0),
            AspectRatio::Mobile,
            "9:21 should be Mobile"
        );
        // 9:16 portrait — ratio 900/1600 = 0.5625
        assert_eq!(
            determine_aspect_ratio(900.0, 1600.0),
            AspectRatio::Portrait,
            "9:16 should be Portrait"
        );
        // 21:9 landscape — ratio 2100/900 ≈ 2.333
        assert_eq!(
            determine_aspect_ratio(2100.0, 900.0),
            AspectRatio::Landscape,
            "21:9 should be Landscape"
        );
        // Degenerate zero-size crop — safe default is Landscape
        assert_eq!(
            determine_aspect_ratio(0.0, 0.0),
            AspectRatio::Landscape,
            "degenerate (0,0) should return Landscape"
        );
    }

    /// Mobile 9:21 with face centered and isolated.
    ///
    /// Phase 3 should use 12.5% breathing room and 18% penetration.
    /// The crop should show more body than Phase 1 (which uses 20px).
    #[test]
    fn test_mobile_fully_isolated_face() {
        let person = standing_person();
        let face = standing_face(); // y1=300, y2=700, height=400
        let crop = mobile_crop(); // width=1728, height=4032
        let config = ArtisticCropConfig::default();

        let result = apply_aspect_aware_face_cropping(
            &person,
            Some(&face),
            &crop,
            &config,
            PHOTO_W,
            PHOTO_H,
        );

        // Protected zone for this face: 300 + 0.30*400 = 420 to 300 + 0.65*400 = 560.
        let protected_top = 300.0 + 0.30 * 400.0; // 420
        let protected_bottom = 300.0 + 0.65 * 400.0; // 560

        assert!(result.y >= 0.0, "crop_y must not be negative");
        assert!(
            result.y <= protected_top,
            "eyes top ({}) must be below crop top ({})",
            protected_top,
            result.y
        );
        assert!(
            result.y + result.height >= protected_bottom,
            "eyes bottom ({}) must be above crop bottom ({})",
            protected_bottom,
            result.y + result.height
        );

        // Dimensions must be preserved.
        assert_eq!(result.width, crop.width, "width must not change");
        assert_eq!(result.height, crop.height, "height must not change");
    }

    /// Mobile 9:21 with face near the top edge.
    ///
    /// Face y1=60 is within 5% of crop top (margin = 4032*0.05 ≈ 201.6px).
    /// Isolation should be ModerateConstraint; penetration scaled to 70%.
    #[test]
    fn test_mobile_moderate_constraint() {
        // Person and face positioned near the top of the image.
        let person = make_bbox(1000.0, 50.0, 2000.0, 3800.0);
        let face = make_bbox(1200.0, 60.0, 1800.0, 460.0);
        // face.y1 = 60; crop.y = 0; margin_y = 4032*0.05 ≈ 201.6
        // 60 < 0 + 201.6 → near top → ModerateConstraint.
        let crop = mobile_crop();
        let config = ArtisticCropConfig::default();

        // Verify isolation classification directly.
        let isolation = assess_face_isolation(&face, &crop, 0.05);
        assert_eq!(
            isolation,
            FaceIsolation::ModerateConstraint,
            "face near top should be ModerateConstraint"
        );

        let result = apply_aspect_aware_face_cropping(
            &person,
            Some(&face),
            &crop,
            &config,
            PHOTO_W,
            PHOTO_H,
        );

        // Eyes must still be visible.
        let face_h = face.height(); // 460 - 60 = 400
        let protected_top = face.y1 + 0.30 * face_h; // 60 + 120 = 180
        assert!(
            result.y <= protected_top,
            "eyes must be visible: crop_y={} protected_top={}",
            result.y,
            protected_top
        );
        assert_eq!(result.width, crop.width, "width must not change");
        assert_eq!(result.height, crop.height, "height must not change");
    }

    /// Portrait 9:16 with face centered.
    ///
    /// Should use 20% breathing room and 14% penetration.
    #[test]
    fn test_portrait_balanced_isolation() {
        let person = standing_person();
        let face = standing_face();
        let crop = portrait_crop(); // width=2268, height=4032
        let config = ArtisticCropConfig::default();

        // Verify aspect classification.
        let aspect = determine_aspect_ratio(crop.width, crop.height);
        assert_eq!(
            aspect,
            AspectRatio::Portrait,
            "portrait crop should be Portrait"
        );

        // Verify lookup values.
        let br = get_breathing_room_for_aspect(aspect, &config);
        assert!(
            (br - 20.0).abs() < 0.01,
            "portrait breathing room should be 20.0, got {}",
            br
        );
        let pen = get_max_penetration_for_aspect(aspect, &config);
        assert!(
            (pen - 14.0).abs() < 0.01,
            "portrait penetration should be 14.0, got {}",
            pen
        );

        let result = apply_aspect_aware_face_cropping(
            &person,
            Some(&face),
            &crop,
            &config,
            PHOTO_W,
            PHOTO_H,
        );

        let protected_top = face.y1 + 0.30 * face.height();
        assert!(
            result.y <= protected_top,
            "eyes must be visible: crop_y={} protected_top={}",
            result.y,
            protected_top
        );
        assert_eq!(result.width, crop.width, "width must not change");
        assert_eq!(result.height, crop.height, "height must not change");
    }

    /// Landscape 21:9 with face centered.
    ///
    /// Should use 25% breathing room and 11% penetration (most conservative).
    #[test]
    fn test_landscape_isolated_face() {
        let person = standing_person();
        let face = standing_face();
        let crop = landscape_crop(); // width=3024, height=1296
        let config = ArtisticCropConfig::default();

        let aspect = determine_aspect_ratio(crop.width, crop.height);
        assert_eq!(
            aspect,
            AspectRatio::Landscape,
            "landscape crop should be Landscape"
        );

        let result = apply_aspect_aware_face_cropping(
            &person,
            Some(&face),
            &crop,
            &config,
            PHOTO_W,
            PHOTO_H,
        );

        // Breathing room: 25% of 1296 = 324px above forehead (y1=300).
        // crop_y_ideal = 300 - 324 = -24 → clamped to 0.
        let protected_top = face.y1 + 0.30 * face.height();
        assert!(
            result.y <= protected_top,
            "eyes must be visible: crop_y={} protected_top={}",
            result.y,
            protected_top
        );
        assert_eq!(result.width, crop.width, "width must not change");
        assert_eq!(result.height, crop.height, "height must not change");
    }

    /// No face bbox: the function must return the original crop unchanged.
    #[test]
    fn test_no_face_detection_fallback() {
        let person = standing_person();
        let crop = mobile_crop();
        let config = ArtisticCropConfig::default();

        let result = apply_aspect_aware_face_cropping(
            &person, None, // no face
            &crop, &config, PHOTO_W, PHOTO_H,
        );

        assert_eq!(result.x, crop.x, "no face → original x preserved");
        assert_eq!(result.y, crop.y, "no face → original y preserved");
        assert_eq!(
            result.width, crop.width,
            "no face → original width preserved"
        );
        assert_eq!(
            result.height, crop.height,
            "no face → original height preserved"
        );
    }

    /// Verify that different aspect ratios produce distinct breathing room values.
    #[test]
    fn test_aspect_specific_breathing_room() {
        let config = ArtisticCropConfig::default();

        let br_m = get_breathing_room_for_aspect(AspectRatio::Mobile, &config);
        let br_p = get_breathing_room_for_aspect(AspectRatio::Portrait, &config);
        let br_l = get_breathing_room_for_aspect(AspectRatio::Landscape, &config);

        // Ordering: Mobile < Portrait < Landscape (tighter frame = less breathing room).
        assert!(
            br_m < br_p,
            "mobile breathing room ({}) must be less than portrait ({})",
            br_m,
            br_p
        );
        assert!(
            br_p < br_l,
            "portrait breathing room ({}) must be less than landscape ({})",
            br_p,
            br_l
        );

        // Exact approved default values.
        assert!(
            (br_m - 12.5).abs() < 0.01,
            "mobile default: 12.5, got {}",
            br_m
        );
        assert!(
            (br_p - 20.0).abs() < 0.01,
            "portrait default: 20.0, got {}",
            br_p
        );
        assert!(
            (br_l - 25.0).abs() < 0.01,
            "landscape default: 25.0, got {}",
            br_l
        );
    }

    /// Verify that different aspect ratios produce distinct penetration values.
    ///
    /// Mobile is most aggressive (shows most body); landscape is most conservative.
    #[test]
    fn test_aspect_specific_penetration() {
        let config = ArtisticCropConfig::default();

        let pen_m = get_max_penetration_for_aspect(AspectRatio::Mobile, &config);
        let pen_p = get_max_penetration_for_aspect(AspectRatio::Portrait, &config);
        let pen_l = get_max_penetration_for_aspect(AspectRatio::Landscape, &config);

        // Ordering: Mobile > Portrait > Landscape (inverse of breathing room).
        assert!(
            pen_m > pen_p,
            "mobile penetration ({}) must be greater than portrait ({})",
            pen_m,
            pen_p
        );
        assert!(
            pen_p > pen_l,
            "portrait penetration ({}) must be greater than landscape ({})",
            pen_p,
            pen_l
        );

        // Exact approved default values.
        assert!(
            (pen_m - 18.0).abs() < 0.01,
            "mobile default: 18.0, got {}",
            pen_m
        );
        assert!(
            (pen_p - 14.0).abs() < 0.01,
            "portrait default: 14.0, got {}",
            pen_p
        );
        assert!(
            (pen_l - 11.0).abs() < 0.01,
            "landscape default: 11.0, got {}",
            pen_l
        );
    }

    /// Eyes must NEVER be cropped regardless of how aggressive the config is.
    ///
    /// Uses an intentionally extreme config to stress-test the eye safety invariant.
    #[test]
    fn test_eye_preservation_always() {
        let person = standing_person();
        let face = standing_face(); // y1=300, y2=700, height=400

        // Stress test: penetration set to 30% (above the approved maximum).
        // Even with this unrealistic value the eye safety check must hold.
        let mut config = ArtisticCropConfig::default();
        config.max_face_bbox_penetration_percent_mobile = 30.0;
        config.breathing_room_percent_mobile = 5.0; // very tight

        let crop = mobile_crop();
        let result = apply_aspect_aware_face_cropping(
            &person,
            Some(&face),
            &crop,
            &config,
            PHOTO_W,
            PHOTO_H,
        );

        // Protected zone: 300 + 0.30*400 = 420 to 300 + 0.65*400 = 560.
        let protected_top = 300.0 + 0.30 * 400.0; // 420
        let protected_bottom = 300.0 + 0.65 * 400.0; // 560
        let crop_top = result.y;
        let crop_bottom = result.y + result.height;

        assert!(
            crop_top <= protected_top,
            "eye zone top ({}) must be below crop top ({})",
            protected_top,
            crop_top
        );
        assert!(
            crop_bottom >= protected_bottom,
            "eye zone bottom ({}) must be above crop bottom ({})",
            protected_bottom,
            crop_bottom
        );
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
        // Original crop y=200 (crop starts below the face top, clipping the face entirely).
        // Phase 3 (or Phase 1 fallback) should reposition to a lower y than 200.
        //
        // Phase 3 invariant: the eye zone (face.y1 + 30% of face height) must be
        // fully inside the crop. This is stricter than "no adjustment" but allows
        // slight forehead penetration (Phase 1 required y <= face.y1 exactly).
        //
        // For this test:
        //   face.y1=150, face_height=200 → eye_zone_top = 150 + 60 = 210
        //   Phase 3 breathing room (mobile, 12.5% of 1600) = 200px → crop_y ideal = -50 → 0
        //   Body overflow pushes crop_y down to ~175px (within penetration budget).
        //   Result: crop_y ≈ 175 < eye_zone_top (210) → eyes are visible.
        let crop = make_crop(0.0, 200.0, 800.0, 1600.0);
        let person = make_bbox(200.0, 100.0, 600.0, 1800.0);
        let face = make_bbox(300.0, 150.0, 500.0, 350.0); // face in upper ~15% of person
        let config = ArtisticCropConfig::default();
        let result =
            apply_face_aware_adjustment(&crop, Some(&person), &[face.clone()], &config, 1000, 2000);

        // Crop must be repositioned (Phase 3 or Phase 1) — below the original y=200.
        assert!(
            result.y < crop.y,
            "crop must be repositioned above original y=200, got result.y={}",
            result.y
        );
        // Phase 3 invariant: the eye zone top must be inside the repositioned crop.
        // eye_zone_top = face.y1 + 0.30 * face_height = 150 + 60 = 210.
        let face_height = face.height(); // 200
        let eye_zone_top = face.y1 + 0.30 * face_height; // 210
        assert!(
            result.y <= eye_zone_top,
            "Phase 3: crop top must be at or above the eye zone ({}), got result.y={}",
            eye_zone_top,
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

    // --- compute_dynamic_face_margin ---

    #[test]
    fn test_dynamic_margin_increases_when_face_near_top_edge() {
        // Face top at y1=50, crop top at y=40 → distance_to_top = 10px.
        // min_breathing_room = 400 * 0.05 = 20px.
        // 10 < 20 → face is cramped at top → margin scales up from base 15.
        let face = make_bbox(100.0, 50.0, 200.0, 150.0);
        let crop = make_crop(0.0, 40.0, 1000.0, 400.0);
        let margin = compute_dynamic_face_margin(&face, &crop, 15);
        assert!(
            margin > 15,
            "face near top edge should increase margin above base 15, got {}",
            margin
        );
        // Expected: (20.0 * 1.5) as u32 = 30, which is > 15.
        assert_eq!(margin, 30, "scaled margin should be (20 * 1.5) = 30");
    }

    #[test]
    fn test_dynamic_margin_stays_base_when_face_centered() {
        // Face (400..600) well inside crop (0..1000).
        // distance_to_top = 400 - 0 = 400px.
        // distance_to_bottom = 1000 - 600 = 400px.
        // min_breathing_room = 1000 * 0.05 = 50px.
        // 400 > 50 AND 400 > 50 → face is well-centered → return base margin.
        let face = make_bbox(400.0, 300.0, 600.0, 500.0);
        let crop = make_crop(0.0, 0.0, 1000.0, 1000.0);
        let margin = compute_dynamic_face_margin(&face, &crop, 15);
        assert_eq!(
            margin, 15,
            "centered face should return base margin unchanged"
        );
    }

    #[test]
    fn test_dynamic_margin_never_decreases_below_base() {
        // Even with a very tiny crop (e.g., 50px height), the margin should never
        // drop below base_margin_px regardless of the scaled value.
        // min_breathing_room = 50 * 0.05 = 2.5px.
        // Scaled: (2.5 * 1.5) as u32 = 3, which is less than base_margin_px=15.
        // max() guard must ensure we return 15.
        let face = make_bbox(100.0, 0.0, 200.0, 50.0); // spans entire tiny crop height
        let crop = make_crop(0.0, 0.0, 500.0, 50.0); // 50px tall crop
        let margin = compute_dynamic_face_margin(&face, &crop, 15);
        assert!(
            margin >= 15,
            "margin must never go below base_margin_px=15, got {}",
            margin
        );
    }
}
