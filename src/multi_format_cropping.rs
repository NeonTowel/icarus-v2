/// Multi-Format Intelligent Cropping Module
///
/// Provides automatic detection of suitable wallpaper formats (landscape 21:9,
/// portrait 9:21, portrait 9:16) from a single photo based on the detected
/// person's bounding box and photo dimensions.
///
/// # Algorithm Overview
///
/// 1. **Bbox Orientation**: Wide bbox → landscape only; tall/square bbox → try all formats.
/// 2. **Landscape 21:9 (Face-First)**: Face zone (top 30% of bbox) at upper-third of crop.
/// 3. **Portrait 9:21 & 9:16 (Balanced Upward)**: Person center at 45% from crop top.
/// 4. **Margin**: Optional symmetric padding (percentage of bbox dimensions) before positioning.
/// 5. **Visibility Gate**: Person must be ≥50% visible in the final crop, or format is skipped.
///
/// # Example
/// ```rust,ignore
/// use icarus_v2::multi_format_cropping::{BBox, detect_suitable_formats};
///
/// let bbox = BBox { x1: 100.0, y1: 50.0, x2: 400.0, y2: 900.0 };
/// let formats = detect_suitable_formats(3000, 4000, &bbox, 5.0);
/// // Returns e.g. ["21:9", "9:21", "9:16"] depending on photo/bbox geometry
/// ```

/// A rectangular crop region in pixel coordinates (top-left origin, not clamped).
///
/// Use `to_bbox_clamped` to convert to a `[f32; 4]` array suitable for `crop_image()`.
#[derive(Debug, Clone, PartialEq)]
pub struct CropRegion {
    /// Left edge of the crop (pixels from left of photo).
    pub x: f32,
    /// Top edge of the crop (pixels from top of photo).
    pub y: f32,
    /// Crop width in pixels.
    pub width: f32,
    /// Crop height in pixels.
    pub height: f32,
}

impl CropRegion {
    /// Convert to `[x1, y1, x2, y2]` pixel array, clamped to photo bounds.
    ///
    /// This is the form accepted by `image_utils::crop_image()`.
    pub fn to_xyxy_clamped(&self, photo_width: u32, photo_height: u32) -> [f32; 4] {
        let pw = photo_width as f32;
        let ph = photo_height as f32;
        let x1 = self.x.max(0.0).min(pw);
        let y1 = self.y.max(0.0).min(ph);
        let x2 = (self.x + self.width).max(0.0).min(pw);
        let y2 = (self.y + self.height).max(0.0).min(ph);
        [x1, y1, x2, y2]
    }
}

/// Bounding box in `[x1, y1, x2, y2]` pixel coordinates (top-left, bottom-right).
#[derive(Debug, Clone, PartialEq)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl BBox {
    /// Width of the bounding box in pixels.
    #[inline]
    pub fn width(&self) -> f32 {
        self.x2 - self.x1
    }

    /// Height of the bounding box in pixels.
    #[inline]
    pub fn height(&self) -> f32 {
        self.y2 - self.y1
    }

    /// Horizontal center of the bounding box.
    #[inline]
    pub fn center_x(&self) -> f32 {
        (self.x1 + self.x2) / 2.0
    }

    /// Vertical center of the bounding box.
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

// ---------------------------------------------------------------------------
// Helper: Margin application
// ---------------------------------------------------------------------------

/// Expand a bounding box by a symmetric margin (percentage of bbox dimensions).
///
/// The margin is calculated as `bbox_width * (margin_percent / 100.0)` on the
/// horizontal axis and `bbox_height * (margin_percent / 100.0)` on the vertical
/// axis. The result is clamped to photo bounds so the returned bbox is always valid.
///
/// # Parameters
/// - `bbox`: Source bounding box.
/// - `margin_percent`: Padding percentage relative to bbox dimensions (e.g. `10.0` = 10%).
/// - `photo_width`: Width of the source photo in pixels (clamp limit).
/// - `photo_height`: Height of the source photo in pixels (clamp limit).
///
/// # Example
/// ```rust,ignore
/// let bbox = BBox { x1: 100.0, y1: 100.0, x2: 300.0, y2: 600.0 };
/// let expanded = apply_margin_to_bbox(&bbox, 10.0, 1920, 1080);
/// // Adds 20px horizontally (10% of 200) and 50px vertically (10% of 500) on each side
/// ```
pub fn apply_margin_to_bbox(
    bbox: &BBox,
    margin_percent: f32,
    photo_width: u32,
    photo_height: u32,
) -> BBox {
    if margin_percent <= 0.0 {
        return bbox.clone();
    }

    let margin_x = bbox.width() * (margin_percent / 100.0);
    let margin_y = bbox.height() * (margin_percent / 100.0);

    BBox {
        x1: (bbox.x1 - margin_x).max(0.0),
        y1: (bbox.y1 - margin_y).max(0.0),
        x2: (bbox.x2 + margin_x).min(photo_width as f32),
        y2: (bbox.y2 + margin_y).min(photo_height as f32),
    }
}

// ---------------------------------------------------------------------------
// Helper: Visibility check
// ---------------------------------------------------------------------------

/// Return `true` if at least 50% of the person (by bbox area) is visible inside the crop.
///
/// A format is only considered suitable when the person would be reasonably framed.
/// This prevents generating crops where the person is barely a sliver on one edge.
///
/// # Parameters
/// - `bbox`: Person bounding box (may be margin-expanded).
/// - `crop`: Proposed crop region.
/// - `_photo_width` / `_photo_height`: Unused; reserved for future bounds-checking extension.
pub fn person_is_reasonably_visible(
    bbox: &BBox,
    crop: &CropRegion,
    _photo_width: u32,
    _photo_height: u32,
) -> bool {
    let crop_x2 = crop.x + crop.width;
    let crop_y2 = crop.y + crop.height;

    let visible_left = bbox.x1.max(crop.x);
    let visible_right = bbox.x2.min(crop_x2);
    let visible_top = bbox.y1.max(crop.y);
    let visible_bottom = bbox.y2.min(crop_y2);

    if visible_left >= visible_right || visible_top >= visible_bottom {
        // Person is completely outside the proposed crop.
        return false;
    }

    let visible_area = (visible_right - visible_left) * (visible_bottom - visible_top);
    let person_area = bbox.width() * bbox.height();

    if person_area <= 0.0 {
        return false;
    }

    let visibility_ratio = visible_area / person_area;
    visibility_ratio >= 0.50
}

// ---------------------------------------------------------------------------
// Milestone 2: Landscape 21:9 Crop (Face-First Vertical Positioning)
// ---------------------------------------------------------------------------

/// Calculate a 21:9 landscape crop region with face-first vertical positioning.
///
/// **Algorithm:**
/// 1. Start with `crop_height = photo_height`; derive `crop_width = crop_height × (21/9)`.
/// 2. If `crop_width > photo_width`, reduce to `photo_width` and recalculate height.
/// 3. Face zone = top 30% of bbox. Position the face zone bottom at 33% down from crop top.
/// 4. Center person horizontally.
/// 5. Clamp to photo bounds.
/// 6. Return `None` if person would be less than 50% visible.
///
/// # Parameters
/// - `photo_width` / `photo_height`: Source photo dimensions.
/// - `bbox`: Person bounding box (should already include margin if applicable).
///
/// # Returns
/// `Some(CropRegion)` when the person is ≥50% visible in the computed crop,
/// `None` otherwise.
pub fn calculate_landscape_21_9_crop(
    photo_width: u32,
    photo_height: u32,
    bbox: &BBox,
) -> Option<CropRegion> {
    const ASPECT_RATIO: f32 = 21.0 / 9.0; // ~2.333

    let pw = photo_width as f32;
    let ph = photo_height as f32;

    // Step 1: try using the full photo height.
    let mut crop_height = ph;
    let mut crop_width = crop_height * ASPECT_RATIO;

    // Step 2: if that would exceed photo width, scale down to fit.
    if crop_width > pw {
        crop_width = pw;
        crop_height = crop_width / ASPECT_RATIO;
    }

    // Step 3: Face-first vertical positioning.
    // Face zone = top 30% of bbox. Place the face zone bottom at 33% from crop top.
    let face_zone_bottom = bbox.y1 + bbox.height() * 0.30;
    let face_zone_position_in_crop = crop_height * 0.33;
    let crop_y = face_zone_bottom - face_zone_position_in_crop;

    // Step 4: Center person horizontally.
    let crop_x = bbox.center_x() - (crop_width / 2.0);

    // Step 5: Clamp both axes to valid photo bounds.
    let crop_x = crop_x.max(0.0).min((pw - crop_width).max(0.0));
    let crop_y = crop_y.max(0.0).min((ph - crop_height).max(0.0));

    let crop = CropRegion {
        x: crop_x,
        y: crop_y,
        width: crop_width,
        height: crop_height,
    };

    // Step 6: Visibility gate.
    if person_is_reasonably_visible(bbox, &crop, photo_width, photo_height) {
        Some(crop)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Milestone 3: Portrait Crops (Balanced Upward Bias Positioning)
// ---------------------------------------------------------------------------

/// Internal generic portrait crop calculator used by both 9:21 and 9:16 variants.
///
/// **Algorithm:**
/// 1. Start with `crop_width = photo_width`; derive `crop_height = crop_width / aspect_ratio`.
/// 2. If `crop_height > photo_height`, reduce to `photo_height` and recalculate width.
/// 3. Position person center at 45% from crop top (biased slightly upward).
/// 4. Center person horizontally.
/// 5. Clamp to photo bounds.
/// 6. Return `None` if person visibility < 50%.
fn calculate_portrait_crop(
    photo_width: u32,
    photo_height: u32,
    bbox: &BBox,
    aspect_ratio: f32, // width/height, e.g. 9.0/21.0 for 9:21
) -> Option<CropRegion> {
    let pw = photo_width as f32;
    let ph = photo_height as f32;

    // Step 1: try using the full photo width.
    let mut crop_width = pw;
    let mut crop_height = crop_width / aspect_ratio;

    // Step 2: if height exceeds photo height, scale down to fit.
    if crop_height > ph {
        crop_height = ph;
        crop_width = crop_height * aspect_ratio;
    }

    // Step 3: Balanced upward bias — person center at 45% from crop top.
    let crop_y = bbox.center_y() - (crop_height * 0.45);

    // Step 4: Center person horizontally.
    let crop_x = bbox.center_x() - (crop_width / 2.0);

    // Step 5: Clamp.
    let crop_x = crop_x.max(0.0).min((pw - crop_width).max(0.0));
    let crop_y = crop_y.max(0.0).min((ph - crop_height).max(0.0));

    let crop = CropRegion {
        x: crop_x,
        y: crop_y,
        width: crop_width,
        height: crop_height,
    };

    // Step 6: Visibility gate.
    if person_is_reasonably_visible(bbox, &crop, photo_width, photo_height) {
        Some(crop)
    } else {
        None
    }
}

/// Calculate a 9:21 ultrawide portrait crop with balanced upward bias positioning.
///
/// See [`calculate_portrait_crop`] for the full algorithm description.
///
/// # Returns
/// `Some(CropRegion)` when the person is ≥50% visible, `None` otherwise.
pub fn calculate_portrait_9_21_crop(
    photo_width: u32,
    photo_height: u32,
    bbox: &BBox,
) -> Option<CropRegion> {
    const ASPECT_RATIO: f32 = 9.0 / 21.0; // ~0.4286 (width/height)
    calculate_portrait_crop(photo_width, photo_height, bbox, ASPECT_RATIO)
}

/// Calculate a 9:16 standard portrait crop with balanced upward bias positioning.
///
/// See [`calculate_portrait_crop`] for the full algorithm description.
///
/// # Returns
/// `Some(CropRegion)` when the person is ≥50% visible, `None` otherwise.
pub fn calculate_portrait_9_16_crop(
    photo_width: u32,
    photo_height: u32,
    bbox: &BBox,
) -> Option<CropRegion> {
    const ASPECT_RATIO: f32 = 9.0 / 16.0; // 0.5625 (width/height)
    calculate_portrait_crop(photo_width, photo_height, bbox, ASPECT_RATIO)
}

// ---------------------------------------------------------------------------
// Milestone 4: Format Suitability Detection
// ---------------------------------------------------------------------------

/// Detect which output formats are suitable for this photo and person bounding box.
///
/// **Orientation logic:**
/// - If `bbox_width > bbox_height` (wide/landscape person): landscape 21:9 only.
/// - Otherwise (tall/square person): landscape 21:9 always + portrait 9:21 and 9:16 if suitable.
///
/// Margin is applied symmetrically to the bbox before any calculations.
///
/// # Parameters
/// - `photo_width` / `photo_height`: Source photo dimensions in pixels.
/// - `bbox`: Raw detected bounding box in `[x1, y1, x2, y2]` pixel coordinates.
/// - `margin_percent`: Optional padding around bbox as percentage of bbox size (0 = no margin).
///
/// # Returns
/// A `Vec<String>` of format names (`"21:9"`, `"9:21"`, `"9:16"`) that are viable.
/// Returns an empty vec if no format can accommodate the person.
///
/// # Example
/// ```rust,ignore
/// let bbox = BBox { x1: 200.0, y1: 50.0, x2: 500.0, y2: 1900.0 };
/// let formats = detect_suitable_formats(3024, 4032, &bbox, 5.0);
/// // likely ["21:9", "9:21", "9:16"] for a tall portrait person in a portrait photo
/// ```
pub fn detect_suitable_formats(
    photo_width: u32,
    photo_height: u32,
    bbox: &BBox,
    margin_percent: f32,
) -> Vec<String> {
    let working_bbox = apply_margin_to_bbox(bbox, margin_percent, photo_width, photo_height);

    let bbox_is_wide = working_bbox.width() > working_bbox.height();

    let mut suitable: Vec<String> = Vec::new();

    // Landscape 21:9 is always attempted regardless of bbox orientation.
    if calculate_landscape_21_9_crop(photo_width, photo_height, &working_bbox).is_some() {
        suitable.push("21:9".to_string());
    }

    // Portrait formats are only attempted when the person's bbox is portrait-oriented.
    if !bbox_is_wide {
        if calculate_portrait_9_21_crop(photo_width, photo_height, &working_bbox).is_some() {
            suitable.push("9:21".to_string());
        }
        if calculate_portrait_9_16_crop(photo_width, photo_height, &working_bbox).is_some() {
            suitable.push("9:16".to_string());
        }
    }

    suitable
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn tall_person_bbox() -> BBox {
        // Portrait photo 3024×4032, person roughly centred
        BBox {
            x1: 800.0,
            y1: 200.0,
            x2: 2200.0,
            y2: 3800.0,
        }
    }

    fn wide_person_bbox() -> BBox {
        // Wide bbox — e.g., person lying down
        BBox {
            x1: 200.0,
            y1: 500.0,
            x2: 1800.0,
            y2: 900.0,
        }
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
        // bbox width=200, height=500; 10% margin → x±20, y±50
        let expanded = apply_margin_to_bbox(&bbox, 10.0, 1920, 1080);
        assert!((expanded.x1 - 80.0).abs() < 0.01);
        assert!((expanded.y1 - 50.0).abs() < 0.01);
        assert!((expanded.x2 - 320.0).abs() < 0.01);
        assert!((expanded.y2 - 650.0).abs() < 0.01);
    }

    #[test]
    fn test_apply_margin_clamped_to_photo() {
        // Bbox already touching the edge — margin must not go negative / exceed photo
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
        // Person 200×400 → area 80000; crop captures the left half → visible area 80000*0.5
        let bbox = BBox {
            x1: 200.0,
            y1: 0.0,
            x2: 400.0,
            y2: 400.0,
        };
        let crop = CropRegion {
            x: 0.0,
            y: 0.0,
            width: 300.0, // captures 100 of 200 px width = exactly 50%
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
        // crop captures only 49% of person width
        let crop = CropRegion {
            x: 0.0,
            y: 0.0,
            width: 297.9, // < 50% of 200px width
            height: 400.0,
        };
        assert!(!person_is_reasonably_visible(&bbox, &crop, 1920, 1080));
    }

    // --- calculate_landscape_21_9_crop ---

    #[test]
    fn test_landscape_crop_returns_some_for_visible_person() {
        // 4000×3000 landscape photo, centred tall person
        let bbox = BBox {
            x1: 1600.0,
            y1: 300.0,
            x2: 2400.0,
            y2: 2800.0,
        };
        let result = calculate_landscape_21_9_crop(4000, 3000, &bbox);
        assert!(result.is_some(), "Expected Some for visible person");
        let crop = result.unwrap();
        // Aspect ratio should be close to 21:9
        let ratio = crop.width / crop.height;
        assert!(
            (ratio - 21.0 / 9.0).abs() < 0.05,
            "Expected ~21:9 ratio, got {:.4}",
            ratio
        );
        // Crop stays within photo bounds
        assert!(crop.x >= 0.0 && crop.x + crop.width <= 4000.0);
        assert!(crop.y >= 0.0 && crop.y + crop.height <= 3000.0);
    }

    #[test]
    fn test_landscape_crop_clamps_to_photo_bounds() {
        // Person in corner — crop must not go out of bounds
        let bbox = BBox {
            x1: 0.0,
            y1: 0.0,
            x2: 200.0,
            y2: 800.0,
        };
        let result = calculate_landscape_21_9_crop(1920, 1080, &bbox);
        if let Some(crop) = result {
            assert!(crop.x >= 0.0);
            assert!(crop.y >= 0.0);
            assert!(crop.x + crop.width <= 1920.0 + 0.01); // float tolerance
            assert!(crop.y + crop.height <= 1080.0 + 0.01);
        }
    }

    // --- calculate_portrait_9_21_crop ---

    #[test]
    fn test_portrait_9_21_crop_for_tall_person_in_portrait_photo() {
        let photo_w = 3024u32;
        let photo_h = 4032u32;
        let bbox = tall_person_bbox();
        let result = calculate_portrait_9_21_crop(photo_w, photo_h, &bbox);
        assert!(
            result.is_some(),
            "Expected Some for visible tall person in portrait photo"
        );
        let crop = result.unwrap();
        let ratio = crop.width / crop.height;
        assert!(
            (ratio - 9.0 / 21.0).abs() < 0.05,
            "Expected ~9:21 ratio, got {:.4}",
            ratio
        );
    }

    // --- calculate_portrait_9_16_crop ---

    #[test]
    fn test_portrait_9_16_crop_for_tall_person_in_portrait_photo() {
        let photo_w = 3024u32;
        let photo_h = 4032u32;
        let bbox = tall_person_bbox();
        let result = calculate_portrait_9_16_crop(photo_w, photo_h, &bbox);
        assert!(
            result.is_some(),
            "Expected Some for visible tall person in portrait photo"
        );
        let crop = result.unwrap();
        let ratio = crop.width / crop.height;
        assert!(
            (ratio - 9.0 / 16.0).abs() < 0.05,
            "Expected ~9:16 ratio, got {:.4}",
            ratio
        );
    }

    // --- detect_suitable_formats ---

    #[test]
    fn test_wide_bbox_returns_only_landscape() {
        let bbox = wide_person_bbox();
        let formats = detect_suitable_formats(1920, 1080, &bbox, 0.0);
        // Wide bbox → portrait formats must be skipped
        assert!(!formats.contains(&"9:21".to_string()));
        assert!(!formats.contains(&"9:16".to_string()));
    }

    #[test]
    fn test_tall_bbox_can_return_portrait_formats() {
        let photo_w = 3024u32;
        let photo_h = 4032u32;
        let bbox = tall_person_bbox();
        let formats = detect_suitable_formats(photo_w, photo_h, &bbox, 0.0);
        // A 3600px-tall person in a 4032px photo fills most of the frame.
        // The 21:9 landscape crop is only ~1296px tall, so the person's 3600px height
        // cannot be 50%+ visible → landscape is correctly skipped.
        // Portrait formats cover the person well and should be present.
        assert!(
            formats.contains(&"9:21".to_string()) || formats.contains(&"9:16".to_string()),
            "Expected at least one portrait format for tall bbox in portrait photo: {:?}",
            formats
        );
    }

    #[test]
    fn test_detect_formats_with_margin() {
        let photo_w = 3024u32;
        let photo_h = 4032u32;
        let bbox = tall_person_bbox();
        // With margin, the bbox expands; portrait formats should still be viable
        let formats_with_margin = detect_suitable_formats(photo_w, photo_h, &bbox, 10.0);
        assert!(
            !formats_with_margin.is_empty(),
            "Expected at least one suitable format with 10% margin, got: {:?}",
            formats_with_margin
        );
        // Portrait formats should survive modest margin
        assert!(
            formats_with_margin.contains(&"9:21".to_string())
                || formats_with_margin.contains(&"9:16".to_string()),
            "Expected portrait formats with 10% margin: {:?}",
            formats_with_margin
        );
    }

    #[test]
    fn test_landscape_suitable_for_landscape_photo_with_person() {
        // A landscape photo (wide) with a moderately-tall person should accept 21:9
        // Person 400px wide × 600px tall in a 1920×1080 landscape photo:
        // 21:9 crop_h = 1080, crop_w = 1920*21/9 → capped at 1920, crop_h = 1920/2.333 ≈ 823
        // Person 600px tall / 823px crop_h → ~73% visible → passes
        let bbox = BBox {
            x1: 760.0,
            y1: 240.0,
            x2: 1160.0,
            y2: 840.0,
        };
        let formats = detect_suitable_formats(1920, 1080, &bbox, 0.0);
        assert!(
            formats.contains(&"21:9".to_string()),
            "Landscape format should be viable for this photo/person: {:?}",
            formats
        );
    }
}
