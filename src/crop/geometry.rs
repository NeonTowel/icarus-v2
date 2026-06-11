//! Geometric primitives: bounding boxes, crop regions, and placement math.
//!
//! All functions are pure and operate only on pixel coordinates.
// Re-export CropConfig so callers can import it from either crop::* or config directly.
pub use crate::config::CropConfig;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Bounding box in `[x1, y1, x2, y2]` pixel coordinates (top-left, bottom-right).
#[derive(Debug, Clone, PartialEq)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl BBox {
    #[inline]
    pub fn width(&self) -> f32 {
        self.x2 - self.x1
    }
    #[inline]
    pub fn height(&self) -> f32 {
        self.y2 - self.y1
    }
    #[inline]
    pub fn center_x(&self) -> f32 {
        (self.x1 + self.x2) / 2.0
    }
    #[inline]
    pub fn center_y(&self) -> f32 {
        (self.y1 + self.y2) / 2.0
    }
}

impl From<[f32; 4]> for BBox {
    fn from(arr: [f32; 4]) -> Self {
        BBox {
            x1: arr[0],
            y1: arr[1],
            x2: arr[2],
            y2: arr[3],
        }
    }
}

/// A rectangular crop region in pixel coordinates (top-left origin, not clamped).
///
/// Use `to_xyxy_clamped` to convert to a `[f32; 4]` array for `crop_image()`.
#[derive(Debug, Clone, PartialEq)]
pub struct CropRegion {
    /// Left edge (pixels from left of photo).
    pub x: f32,
    /// Top edge (pixels from top of photo).
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl CropRegion {
    /// Convert to `[x1, y1, x2, y2]`, clamped to photo bounds.
    pub fn to_xyxy_clamped(&self, photo_width: u32, photo_height: u32) -> [f32; 4] {
        let x1 = self.x.clamp(0.0, photo_width as f32);
        let y1 = self.y.clamp(0.0, photo_height as f32);
        let x2 = (self.x + self.width).clamp(0.0, photo_width as f32);
        let y2 = (self.y + self.height).clamp(0.0, photo_height as f32);
        [x1, y1, x2, y2]
    }
}

// ---------------------------------------------------------------------------
// Bbox helpers
// ---------------------------------------------------------------------------

/// Merge two bboxes into their minimum enclosing rectangle.
pub fn merge_bboxes(a: &BBox, b: &BBox) -> BBox {
    BBox {
        x1: a.x1.min(b.x1),
        y1: a.y1.min(b.y1),
        x2: a.x2.max(b.x2),
        y2: a.y2.max(b.y2),
    }
}

/// Expand a bbox by a fixed pixel margin in all four directions, clamped to photo bounds.
pub fn expand_bbox_px(bbox: &BBox, margin_px: u32, photo_width: u32, photo_height: u32) -> BBox {
    let m = margin_px as f32;
    BBox {
        x1: (bbox.x1 - m).max(0.0),
        y1: (bbox.y1 - m).max(0.0),
        x2: (bbox.x2 + m).min(photo_width as f32),
        y2: (bbox.y2 + m).min(photo_height as f32),
    }
}

/// Expand a bbox by a symmetric percentage margin relative to bbox dimensions.
///
/// `margin_percent = 10.0` adds 10% of bbox width on each horizontal side.
pub fn apply_margin_to_bbox(
    bbox: &BBox,
    margin_percent: f32,
    photo_width: u32,
    photo_height: u32,
) -> BBox {
    if margin_percent <= 0.0 {
        return bbox.clone();
    }
    let mx = bbox.width() * (margin_percent / 100.0);
    let my = bbox.height() * (margin_percent / 100.0);
    BBox {
        x1: (bbox.x1 - mx).max(0.0),
        y1: (bbox.y1 - my).max(0.0),
        x2: (bbox.x2 + mx).min(photo_width as f32),
        y2: (bbox.y2 + my).min(photo_height as f32),
    }
}

// ---------------------------------------------------------------------------
// Long-side quality gate (shared between early_filter and joint_analyzer)
// ---------------------------------------------------------------------------

/// Return `true` when a dimension meets the minimum pixel threshold.
///
/// - `min_px == 0` disables the check (always returns `true`).
/// - Equal-to-minimum counts as passing.
///
/// Pass `max(crop_w, crop_h)` as `long_side_px` so that landscape crops enforce
/// on width and portrait/mobile crops enforce on height — both via this single predicate.
///
/// # Examples
/// ```rust
/// use icarus_v2::crop::geometry::long_side_passes;
///
/// assert!(long_side_passes(1200, 1200)); // equal-to-min passes
/// assert!(long_side_passes(1500, 1200)); // above min passes
/// assert!(!long_side_passes(1199, 1200)); // below min fails
/// assert!(long_side_passes(0, 0));       // 0 min disables
/// assert!(long_side_passes(500, 0));     // 0 min always passes
/// ```
pub fn long_side_passes(long_side_px: u32, min_px: u32) -> bool {
    min_px == 0 || long_side_px >= min_px
}

// ---------------------------------------------------------------------------
// Crop dimension calculation (shared between strategy and joint analyzer)
// ---------------------------------------------------------------------------

/// Compute the pixel dimensions `(crop_width, crop_height)` for a given photo and format.
///
/// Mirrors the sizing logic in [`crate::crop::strategy::CropStrategy::calculate`] so
/// that `joint_analyzer` can reason about the same crop geometry without importing the
/// strategy module (avoiding circular dependencies).
///
/// - `height_first` → `crop_height = photo_height`, width derived (landscape 21:9 style).
/// - `!height_first` → `crop_width = photo_width`, height derived (portrait 9:21 / 9:16 style).
///
/// Both branches cap the unconstrained dimension to the photo bounds.
///
/// # Example
/// ```rust,ignore
/// // 1920×1080 photo at 21:9 (height-first):
/// let (w, h) = crop_dimensions(1920, 1080, 21.0 / 9.0, true);
/// // h == 1080, w == 1080 * 21/9 ≈ 2520 → capped to 1920 → h recalculated
/// ```
pub fn crop_dimensions(
    photo_w: u32,
    photo_h: u32,
    aspect_ratio: f32,
    height_first: bool,
) -> (f32, f32) {
    let pw = photo_w as f32;
    let ph = photo_h as f32;
    if height_first {
        let h = ph;
        let w = h * aspect_ratio;
        if w > pw {
            let w2 = pw;
            (w2, w2 / aspect_ratio)
        } else {
            (w, h)
        }
    } else {
        let w = pw;
        let h = w / aspect_ratio;
        if h > ph {
            let h2 = ph;
            (h2 * aspect_ratio, h2)
        } else {
            (w, h)
        }
    }
}

// ---------------------------------------------------------------------------
// Crop placement
// ---------------------------------------------------------------------------

/// Place the focal point within the photo bounds to produce a `CropRegion`.
///
/// Centers the crop so that `focal.x` falls at `target_x_frac * crop_width` and
/// `focal.y` at `target_y_frac * crop_height`, then clamps to photo bounds.
pub fn place_focal_point(
    photo_width: u32,
    photo_height: u32,
    crop_width: f32,
    crop_height: f32,
    focal: &crate::focal_point::FocalPoint,
    target_x_frac: f32,
    target_y_frac: f32,
) -> CropRegion {
    let x = (focal.x - target_x_frac * crop_width)
        .clamp(0.0, (photo_width as f32 - crop_width).max(0.0));
    let y = (focal.y - target_y_frac * crop_height)
        .clamp(0.0, (photo_height as f32 - crop_height).max(0.0));
    CropRegion {
        x,
        y,
        width: crop_width,
        height: crop_height,
    }
}

// ---------------------------------------------------------------------------
// Visibility helpers
// ---------------------------------------------------------------------------

/// Return `true` if the face zone `zone` is fully contained within `crop`.
pub fn is_region_visible(zone: &BBox, crop: &CropRegion) -> bool {
    let cx2 = crop.x + crop.width;
    let cy2 = crop.y + crop.height;
    zone.x1 >= crop.x && zone.y1 >= crop.y && zone.x2 <= cx2 && zone.y2 <= cy2
}

/// Return `true` if at least 50% of the person (by bbox area) is visible in the crop.
pub fn person_is_reasonably_visible(bbox: &BBox, crop: &CropRegion, _pw: u32, _ph: u32) -> bool {
    person_is_reasonably_visible_threshold(bbox, crop, 0.50, _pw, _ph)
}

/// Return `true` if at least `threshold` fraction of the person is visible in the crop.
pub fn person_is_reasonably_visible_threshold(
    bbox: &BBox,
    crop: &CropRegion,
    threshold: f32,
    _pw: u32,
    _ph: u32,
) -> bool {
    let cx2 = crop.x + crop.width;
    let cy2 = crop.y + crop.height;
    let vl = bbox.x1.max(crop.x);
    let vr = bbox.x2.min(cx2);
    let vt = bbox.y1.max(crop.y);
    let vb = bbox.y2.min(cy2);
    if vl >= vr || vt >= vb {
        return false;
    }
    let visible_area = (vr - vl) * (vb - vt);
    let person_area = bbox.width() * bbox.height();
    if person_area <= 0.0 {
        return false;
    }
    visible_area / person_area >= threshold
}

// ---------------------------------------------------------------------------
// IoU
// ---------------------------------------------------------------------------

/// Intersection-over-Union for two `[x1, y1, x2, y2]` bboxes. Returns 0.0 for non-overlapping.
pub(crate) fn compute_iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let [ax1, ay1, ax2, ay2] = a;
    let [bx1, by1, bx2, by2] = b;
    let iw = (ax2.min(bx2) - ax1.max(bx1)).max(0.0);
    let ih = (ay2.min(by2) - ay1.max(by1)).max(0.0);
    let inter = iw * ih;
    let area_a = (ax2 - ax1).max(0.0) * (ay2 - ay1).max(0.0);
    let area_b = (bx2 - bx1).max(0.0) * (by2 - by1).max(0.0);
    let union = area_a + area_b - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
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

    // --- long_side_passes ---

    #[test]
    fn test_long_side_passes_equal_to_min_passes() {
        assert!(long_side_passes(1200, 1200), "equal-to-min must pass");
    }

    #[test]
    fn test_long_side_passes_above_min_passes() {
        assert!(long_side_passes(1500, 1200), "above min must pass");
    }

    #[test]
    fn test_long_side_passes_below_min_fails() {
        assert!(!long_side_passes(1199, 1200), "one below min must fail");
        assert!(!long_side_passes(0, 1200), "zero must fail when min > 0");
    }

    #[test]
    fn test_long_side_passes_zero_min_disables_check() {
        assert!(long_side_passes(0, 0), "0 min always passes");
        assert!(
            long_side_passes(500, 0),
            "0 min always passes for any value"
        );
        assert!(long_side_passes(1, 0), "0 min always passes for 1");
    }

    #[test]
    fn test_long_side_passes_u32_max_with_zero_min() {
        assert!(
            long_side_passes(u32::MAX, 0),
            "u32::MAX must pass when min is 0 (disabled)"
        );
    }

    /// Confirms landscape (21:9) enforces width and portrait/mobile enforce height,
    /// both via `max(crop_w, crop_h)` — the single long-side predicate.
    #[test]
    fn test_long_side_category_mapping_landscape_portrait_mobile() {
        // Landscape 21:9 on a 4000×2000 photo: crop_w = 4000, crop_h ≈ 1714 → long side = 4000
        let (cw, ch) = crop_dimensions(4000, 2000, 21.0 / 9.0, true);
        let long_side = cw.max(ch) as u32;
        assert!(
            long_side_passes(long_side, 1200),
            "landscape long side ({long_side}) passes 1200"
        );

        // Portrait 9:16 on a 1080×1920 photo: crop_w = 1080, crop_h = 1920 → long side = 1920
        let (cw, ch) = crop_dimensions(1080, 1920, 9.0 / 16.0, false);
        let long_side = cw.max(ch) as u32;
        assert!(
            long_side_passes(long_side, 1200),
            "portrait long side ({long_side}) passes 1200"
        );

        // Mobile 9:21 on a 400×500 photo: crop would be small → long side < 1200
        let (cw, ch) = crop_dimensions(400, 500, 9.0 / 21.0, false);
        let long_side = cw.max(ch) as u32;
        assert!(
            !long_side_passes(long_side, 1200),
            "small mobile long side ({long_side}) fails 1200"
        );
    }

    // --- crop_dimensions ---

    #[test]
    fn test_crop_dimensions_height_first_landscape() {
        // 1920×1080 photo at 21:9 — width would be 2520, capped to 1920.
        let (w, h) = crop_dimensions(1920, 1080, 21.0 / 9.0, true);
        assert!((w - 1920.0).abs() < 0.5, "w={w}");
        assert!((h - 1920.0 / (21.0 / 9.0)).abs() < 0.5, "h={h}");
    }

    #[test]
    fn test_crop_dimensions_height_first_wide_photo() {
        // 4000×2000 photo at 21:9 — width would be ~4666, capped to 4000.
        let (w, h) = crop_dimensions(4000, 2000, 21.0 / 9.0, true);
        assert!((w - 4000.0).abs() < 0.5, "w={w}");
        assert!((h - 4000.0 * 9.0 / 21.0).abs() < 0.5, "h={h}");
    }

    #[test]
    fn test_crop_dimensions_width_first_portrait() {
        // 1080×1920 photo at 9:16 — height would be 1920 exactly.
        let (w, h) = crop_dimensions(1080, 1920, 9.0 / 16.0, false);
        assert!((w - 1080.0).abs() < 0.5, "w={w}");
        assert!((h - 1920.0).abs() < 0.5, "h={h}");
    }

    #[test]
    fn test_crop_dimensions_width_first_short_photo() {
        // 1080×800 photo at 9:16 — height would be 1920, capped to 800.
        let (w, h) = crop_dimensions(1080, 800, 9.0 / 16.0, false);
        assert!((h - 800.0).abs() < 0.5, "h={h}");
        assert!((w - 800.0 * 9.0 / 16.0).abs() < 0.5, "w={w}");
    }

    // --- apply_margin_to_bbox ---

    #[test]
    fn test_apply_margin_zero() {
        let bbox = tall_person_bbox();
        let expanded = apply_margin_to_bbox(&bbox, 0.0, 3024, 4032);
        assert_eq!(expanded, bbox);
    }

    #[test]
    fn test_apply_margin_expands_bbox() {
        let bbox = BBox {
            x1: 100.0,
            y1: 100.0,
            x2: 300.0,
            y2: 600.0,
        };
        let expanded = apply_margin_to_bbox(&bbox, 10.0, 1920, 1080);
        assert!((expanded.x1 - 80.0).abs() < 0.01);
        assert!((expanded.y1 - 50.0).abs() < 0.01);
        assert!((expanded.x2 - 320.0).abs() < 0.01);
        assert!((expanded.y2 - 650.0).abs() < 0.01);
    }

    #[test]
    fn test_apply_margin_clamped_to_photo() {
        let bbox = BBox {
            x1: 0.0,
            y1: 0.0,
            x2: 200.0,
            y2: 500.0,
        };
        let expanded = apply_margin_to_bbox(&bbox, 50.0, 400, 600);
        assert!(expanded.x1 >= 0.0);
        assert!(expanded.y1 >= 0.0);
        assert!(expanded.x2 <= 400.0);
        assert!(expanded.y2 <= 600.0);
    }

    // --- person_is_reasonably_visible ---

    #[test]
    fn test_person_fully_inside_crop_is_visible() {
        let bbox = BBox {
            x1: 100.0,
            y1: 100.0,
            x2: 400.0,
            y2: 600.0,
        };
        let crop = CropRegion {
            x: 0.0,
            y: 0.0,
            width: 1920.0,
            height: 1080.0,
        };
        assert!(person_is_reasonably_visible(&bbox, &crop, 1920, 1080));
    }

    #[test]
    fn test_person_completely_outside_crop_not_visible() {
        let bbox = BBox {
            x1: 1000.0,
            y1: 0.0,
            x2: 1200.0,
            y2: 400.0,
        };
        let crop = CropRegion {
            x: 0.0,
            y: 0.0,
            width: 500.0,
            height: 400.0,
        };
        assert!(!person_is_reasonably_visible(&bbox, &crop, 1920, 1080));
    }

    #[test]
    fn test_person_half_visible_passes_threshold() {
        let bbox = BBox {
            x1: 200.0,
            y1: 0.0,
            x2: 400.0,
            y2: 400.0,
        };
        let crop = CropRegion {
            x: 0.0,
            y: 0.0,
            width: 300.0,
            height: 400.0,
        };
        assert!(person_is_reasonably_visible(&bbox, &crop, 1920, 1080));
    }

    #[test]
    fn test_person_less_than_half_visible_fails() {
        let bbox = BBox {
            x1: 200.0,
            y1: 0.0,
            x2: 400.0,
            y2: 400.0,
        };
        let crop = CropRegion {
            x: 0.0,
            y: 0.0,
            width: 297.9,
            height: 400.0,
        };
        assert!(!person_is_reasonably_visible(&bbox, &crop, 1920, 1080));
    }

    // --- IoU ---

    #[test]
    fn test_compute_iou_full_overlap() {
        let a = [100.0, 200.0, 400.0, 800.0];
        assert!((compute_iou(a, a) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_compute_iou_no_overlap() {
        let result = compute_iou([0.0, 0.0, 100.0, 100.0], [200.0, 200.0, 300.0, 300.0]);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_compute_iou_partial_overlap() {
        let a = [0.0, 0.0, 200.0, 200.0];
        let b = [100.0, 100.0, 300.0, 300.0];
        let expected = 10000.0 / 70000.0;
        assert!((compute_iou(a, b) - expected).abs() < 0.001);
    }

    // --- place_focal_point ---

    #[test]
    fn test_place_focal_point_centered() {
        let f = crate::focal_point::FocalPoint {
            x: 500.0,
            y: 500.0,
            kind: crate::focal_point::FocalKind::ImageCenter,
        };
        let c = place_focal_point(1000, 1000, 400.0, 400.0, &f, 0.5, 0.5);
        assert_eq!(c.x, 300.0);
        assert_eq!(c.y, 300.0);
    }

    #[test]
    fn test_place_focal_point_top_left_clamp() {
        let f = crate::focal_point::FocalPoint {
            x: 100.0,
            y: 100.0,
            kind: crate::focal_point::FocalKind::BboxCenter,
        };
        let c = place_focal_point(1000, 1000, 400.0, 400.0, &f, 0.5, 0.5);
        assert_eq!(c.x, 0.0);
        assert_eq!(c.y, 0.0);
    }

    #[test]
    fn test_place_focal_point_bottom_right_clamp() {
        let f = crate::focal_point::FocalPoint {
            x: 900.0,
            y: 900.0,
            kind: crate::focal_point::FocalKind::BboxCenter,
        };
        let c = place_focal_point(1000, 1000, 400.0, 400.0, &f, 0.5, 0.5);
        assert_eq!(c.x, 600.0);
        assert_eq!(c.y, 600.0);
    }

    #[test]
    fn test_place_focal_point_off_center() {
        let f = crate::focal_point::FocalPoint {
            x: 600.0,
            y: 300.0,
            kind: crate::focal_point::FocalKind::BboxCenter,
        };
        let c = place_focal_point(1000, 1000, 300.0, 400.0, &f, 0.33, 0.30);
        assert_eq!(c.x, 501.0);
        assert_eq!(c.y, 180.0);
    }
}
