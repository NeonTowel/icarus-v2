/// Multi-Format Intelligent Cropping Module
///
/// Provides automatic detection of suitable wallpaper formats (landscape 21:9,
/// portrait 9:21, portrait 9:16) from a single photo based on the detected
/// person's bounding box and photo dimensions.
///
/// # Algorithm Overview
///
/// 1. **Bbox Orientation**: Wide bbox → landscape only; tall/square bbox → try all formats.
/// 2. **Landscape 21:9 (Bbox-Center Upward Bias)**: Person bbox center at `headroom_ratio`
///    from crop top (default 40%), giving 60% footroom — ideal for fashion/ultrawide.
/// 3. **Portrait 9:21 & 9:16 (Bbox-Center Upward Bias)**: Same configurable ratio.
/// 4. **Margin**: Optional symmetric padding (percentage of bbox dimensions) before positioning.
/// 5. **Visibility Gate**: Person must be ≥`visibility_threshold` visible in the final crop,
///    or format is skipped (configurable, default 50%).
///
/// # Example
/// ```rust,ignore
/// use icarus_v2::multi_format_cropping::{BBox, detect_suitable_formats};
/// use icarus_v2::config::CropConfig;
///
/// let bbox = BBox { x1: 100.0, y1: 50.0, x2: 400.0, y2: 900.0 };
/// let config = CropConfig::default();
/// let formats = detect_suitable_formats(3000, 4000, &bbox, 5.0, &config);
/// // Returns e.g. ["21:9", "9:21", "9:16"] depending on photo/bbox geometry
/// ```
// Re-export CropConfig from config so callers can import it from either path.
// The canonical definition with serde support lives in `crate::config`.
pub use crate::config::CropConfig;

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
/// This is a convenience wrapper around [`person_is_reasonably_visible_threshold`] using
/// the fixed 50% threshold. Prefer [`person_is_reasonably_visible_threshold`] when you
/// need configurable behaviour.
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
    person_is_reasonably_visible_threshold(bbox, crop, 0.50, _photo_width, _photo_height)
}

/// Return `true` if at least `threshold` fraction of the person (by bbox area) is visible
/// inside the crop.
///
/// This is the parameterized version of [`person_is_reasonably_visible`], accepting a
/// configurable visibility threshold instead of the hardcoded 50%.
///
/// # Parameters
/// - `bbox`: Person bounding box (may be margin-expanded).
/// - `crop`: Proposed crop region.
/// - `threshold`: Minimum visible fraction required, in `[0.0, 1.0]`.
///   For example, `0.40` means at least 40% of the person's area must be inside the crop.
/// - `_photo_width` / `_photo_height`: Unused; reserved for future bounds-checking extension.
///
/// # Example
/// ```rust,ignore
/// let visible = person_is_reasonably_visible_threshold(&bbox, &crop, 0.40, 1920, 1080);
/// ```
pub fn person_is_reasonably_visible_threshold(
    bbox: &BBox,
    crop: &CropRegion,
    threshold: f32,
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
    visibility_ratio >= threshold
}

// ---------------------------------------------------------------------------
// Milestone 2: Landscape 21:9 Crop (Bbox-Center Upward Bias Positioning)
// ---------------------------------------------------------------------------

/// Calculate a 21:9 landscape crop region with configurable bbox-center upward bias.
///
/// **Algorithm:**
/// 1. Start with `crop_height = photo_height`; derive `crop_width = crop_height × (21/9)`.
/// 2. If `crop_width > photo_width`, reduce to `photo_width` and recalculate height.
/// 3. Position bbox center at `config.headroom_ratio` from crop top (default 40%).
///    This gives 60% footroom below the center — ideal for fashion/entertainment on ultrawides.
/// 4. Center person horizontally.
/// 5. Clamp to photo bounds.
/// 6. Return `None` if person is less than `config.visibility_threshold` visible.
///
/// # Parameters
/// - `photo_width` / `photo_height`: Source photo dimensions.
/// - `bbox`: Person bounding box (should already include margin if applicable).
/// - `config`: Cropping configuration controlling headroom and visibility threshold.
///
/// # Returns
/// `Some(CropRegion)` when the person is sufficiently visible in the computed crop,
/// `None` otherwise.
///
/// # Example
/// ```rust,ignore
/// let config = CropConfig::default();
/// let crop = calculate_landscape_21_9_crop(4000, 3000, &bbox, &config);
/// ```
pub fn calculate_landscape_21_9_crop(
    photo_width: u32,
    photo_height: u32,
    bbox: &BBox,
    config: &CropConfig,
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

    // Step 3: Bbox-center upward bias — place bbox center at headroom_ratio from crop top.
    // headroom_ratio=0.40 → center is at 40% from top, leaving 60% footroom below.
    let crop_y = bbox.center_y() - (crop_height * config.headroom_ratio);

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

    // Step 6: Visibility gate using configurable threshold.
    if person_is_reasonably_visible_threshold(
        bbox,
        &crop,
        config.visibility_threshold,
        photo_width,
        photo_height,
    ) {
        Some(crop)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// E3: Off-Center Horizontal Softening
// ---------------------------------------------------------------------------

/// Apply a gentle horizontal pull toward the photo center when the
/// subject is in the outer 25% of the frame.
///
/// Reduces the "dead space" artifact in off-center compositions
/// without eliminating the photographer's deliberate negative space.
///
/// # Activation
///
/// Activates when `off_center_fraction > 0.50`, meaning the person
/// center is in the outer 25% on either side (the outermost 25% of
/// the left or right half).
///
/// # Formula
///
/// ```text
/// photo_center = photo_width / 2
/// off_center_frac = |person_center_x − photo_center| / photo_center
///   (0.0 = centered, 1.0 = at edge)
///
/// if off_center_frac ≤ 0.50 → return raw_crop_x unchanged
///
/// activation = (off_center_frac − 0.50) / 0.50
/// blend = activation × config.softening_strength
/// centered_x = photo_center − crop_width / 2
/// softened_x = raw_crop_x × (1 − blend) + centered_x × blend
/// ```
///
/// # Parameters
/// - `raw_crop_x`: Crop X from person-center algorithm (before clamping).
/// - `crop_width`: Width of the crop region in pixels.
/// - `person_center_x`: Horizontal center of the person bbox.
/// - `photo_width`: Photo width in pixels (as f32).
/// - `config`: Must have `softening_strength` in `[0.0, 1.0]`.
///
/// # Returns
/// Adjusted `crop_x` (not clamped — caller clamps to photo bounds).
pub fn apply_horizontal_softening(
    raw_crop_x: f32,
    crop_width: f32,
    person_center_x: f32,
    photo_width: f32,
    config: &CropConfig,
) -> f32 {
    let photo_center = photo_width / 2.0;
    if photo_center <= 0.0 || config.softening_strength <= 0.0 {
        return raw_crop_x;
    }

    let off_center_frac = (person_center_x - photo_center).abs() / photo_center;

    if off_center_frac <= 0.50 {
        return raw_crop_x;
    }

    let activation = (off_center_frac - 0.50) / 0.50;
    let blend = activation * config.softening_strength;
    let centered_x = photo_center - (crop_width / 2.0);

    raw_crop_x * (1.0 - blend) + centered_x * blend
}

// ---------------------------------------------------------------------------
// E2: Context-Aware Landscape Aspect Ratio
// ---------------------------------------------------------------------------

/// Compute a dynamic landscape aspect ratio that widens for narrow
/// subjects to include more environmental context.
///
/// When the person occupies less than 40% of the photo width, the
/// aspect ratio scales linearly from 21:9 (at 40%) toward
/// `config.max_landscape_aspect` (at 0%). Subjects wider than 40%
/// of the photo always get the standard 21:9.
///
/// # Formula
///
/// ```text
/// person_width_ratio = bbox.width() / photo_width
/// if person_width_ratio >= 0.40 → 21/9
/// t = person_width_ratio / 0.40   (0.0 = infinitely narrow, 1.0 = 40% wide)
/// aspect = (21/9) + (1.0 − t) × (max_landscape_aspect − 21/9)
/// ```
///
/// # Parameters
/// - `bbox`: Person bounding box (after margin expansion).
/// - `photo_width`: Photo width in pixels.
/// - `config`: Must have `max_landscape_aspect` set (default 25/9).
///
/// # Returns
/// Aspect ratio as `width / height` in `[21/9, max_landscape_aspect]`.
pub fn calculate_landscape_aspect_ratio(bbox: &BBox, photo_width: u32, config: &CropConfig) -> f32 {
    let pw = photo_width as f32;
    if pw <= 0.0 {
        return 21.0 / 9.0;
    }

    let person_width_ratio = bbox.width() / pw;
    let base_aspect = 21.0_f32 / 9.0;

    if person_width_ratio >= 0.40 {
        return base_aspect;
    }

    let t = person_width_ratio / 0.40;
    let max_aspect = config.max_landscape_aspect;
    let expanded = base_aspect + (1.0 - t) * (max_aspect - base_aspect);

    expanded.clamp(base_aspect, max_aspect)
}

// ---------------------------------------------------------------------------
// Enhanced crop functions (_with_face variants)
// ---------------------------------------------------------------------------

/// Enhanced landscape 21:9 crop with face-aware anchoring, dynamic
/// aspect ratio, and horizontal softening.
///
/// Delegates to the original algorithm when all feature flags are
/// `false` and `face_bbox` is `None`.
///
/// # Enhancements (controlled by `CropConfig` flags)
/// - **E1+E5** (`enable_adaptive_headroom`): Multi-tier vertical
///   anchoring based on face position.
/// - **E2** (`enable_landscape_expansion`): Wider aspect ratio for
///   narrow subjects (up to `max_landscape_aspect`).
/// - **E3** (`enable_horizontal_softening`): Gentle horizontal pull
///   toward photo center for off-center subjects.
///
/// # Parameters
/// - `photo_width` / `photo_height`: Source photo dimensions.
/// - `bbox`: Person bounding box (margin-expanded).
/// - `face_bbox`: Dominant face bbox, or `None` if no face detected.
/// - `config`: Crop config with feature flags.
pub fn calculate_landscape_21_9_crop_with_face(
    photo_width: u32,
    photo_height: u32,
    bbox: &BBox,
    face_bbox: Option<&BBox>,
    config: &CropConfig,
) -> Option<CropRegion> {
    let pw = photo_width as f32;
    let ph = photo_height as f32;

    // E2: Dynamic aspect ratio for narrow subjects.
    let aspect_ratio = if config.enable_landscape_expansion {
        calculate_landscape_aspect_ratio(bbox, photo_width, config)
    } else {
        21.0 / 9.0
    };

    // Compute crop dimensions.
    let mut crop_height = ph;
    let mut crop_width = crop_height * aspect_ratio;
    if crop_width > pw {
        crop_width = pw;
        crop_height = crop_width / aspect_ratio;
    }

    // E1+E5: Adaptive vertical anchor.
    let crop_y = if config.enable_adaptive_headroom {
        calculate_crop_y_anchor(bbox, face_bbox, crop_height, config)
    } else {
        bbox.center_y() - (crop_height * config.headroom_ratio)
    };

    // E3: Horizontal softening.
    let raw_crop_x = bbox.center_x() - (crop_width / 2.0);
    let crop_x = if config.enable_horizontal_softening {
        apply_horizontal_softening(raw_crop_x, crop_width, bbox.center_x(), pw, config)
    } else {
        raw_crop_x
    };

    // Clamp to photo bounds.
    let crop_x = crop_x.max(0.0).min((pw - crop_width).max(0.0));
    let crop_y = crop_y.max(0.0).min((ph - crop_height).max(0.0));

    let crop = CropRegion {
        x: crop_x,
        y: crop_y,
        width: crop_width,
        height: crop_height,
    };

    if person_is_reasonably_visible_threshold(
        bbox,
        &crop,
        config.visibility_threshold,
        photo_width,
        photo_height,
    ) {
        Some(crop)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Milestone 3: Portrait Crops (Configurable Bbox-Center Upward Bias)
// ---------------------------------------------------------------------------

/// Internal generic portrait crop calculator used by both 9:21 and 9:16 variants.
///
/// **Algorithm:**
/// 1. Start with `crop_width = photo_width`; derive `crop_height = crop_width / aspect_ratio`.
/// 2. If `crop_height > photo_height`, reduce to `photo_height` and recalculate width.
/// 3. Position bbox center at `config.headroom_ratio` from crop top (default 40%).
/// 4. Center person horizontally.
/// 5. Clamp to photo bounds.
/// 6. Return `None` if person visibility < `config.visibility_threshold`.
fn calculate_portrait_crop(
    photo_width: u32,
    photo_height: u32,
    bbox: &BBox,
    aspect_ratio: f32, // width/height, e.g. 9.0/21.0 for 9:21
    config: &CropConfig,
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

    // Step 3: Bbox-center upward bias — place bbox center at headroom_ratio from crop top.
    let crop_y = bbox.center_y() - (crop_height * config.headroom_ratio);

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

    // Step 6: Visibility gate using configurable threshold.
    if person_is_reasonably_visible_threshold(
        bbox,
        &crop,
        config.visibility_threshold,
        photo_width,
        photo_height,
    ) {
        Some(crop)
    } else {
        None
    }
}

/// Calculate a 9:21 ultrawide portrait crop with configurable bbox-center upward bias.
///
/// See [`calculate_portrait_crop`] for the full algorithm description.
///
/// # Parameters
/// - `photo_width` / `photo_height`: Source photo dimensions.
/// - `bbox`: Person bounding box (should already include margin if applicable).
/// - `config`: Cropping configuration controlling headroom and visibility threshold.
///
/// # Returns
/// `Some(CropRegion)` when the person is sufficiently visible, `None` otherwise.
pub fn calculate_portrait_9_21_crop(
    photo_width: u32,
    photo_height: u32,
    bbox: &BBox,
    config: &CropConfig,
) -> Option<CropRegion> {
    const ASPECT_RATIO: f32 = 9.0 / 21.0; // ~0.4286 (width/height)
    calculate_portrait_crop(photo_width, photo_height, bbox, ASPECT_RATIO, config)
}

/// Calculate a 9:16 standard portrait crop with configurable bbox-center upward bias.
///
/// See [`calculate_portrait_crop`] for the full algorithm description.
///
/// # Parameters
/// - `photo_width` / `photo_height`: Source photo dimensions.
/// - `bbox`: Person bounding box (should already include margin if applicable).
/// - `config`: Cropping configuration controlling headroom and visibility threshold.
///
/// # Returns
/// `Some(CropRegion)` when the person is sufficiently visible, `None` otherwise.
pub fn calculate_portrait_9_16_crop(
    photo_width: u32,
    photo_height: u32,
    bbox: &BBox,
    config: &CropConfig,
) -> Option<CropRegion> {
    const ASPECT_RATIO: f32 = 9.0 / 16.0; // 0.5625 (width/height)
    calculate_portrait_crop(photo_width, photo_height, bbox, ASPECT_RATIO, config)
}

/// Enhanced portrait crop with face-aware anchoring and horizontal
/// softening (E1, E3, E5). No landscape expansion (E2 is
/// landscape-only).
fn calculate_portrait_crop_enhanced(
    photo_width: u32,
    photo_height: u32,
    bbox: &BBox,
    face_bbox: Option<&BBox>,
    aspect_ratio: f32,
    config: &CropConfig,
) -> Option<CropRegion> {
    let pw = photo_width as f32;
    let ph = photo_height as f32;

    let mut crop_width = pw;
    let mut crop_height = crop_width / aspect_ratio;
    if crop_height > ph {
        crop_height = ph;
        crop_width = crop_height * aspect_ratio;
    }

    // E1+E5: Adaptive vertical anchor.
    let crop_y = if config.enable_adaptive_headroom {
        calculate_crop_y_anchor(bbox, face_bbox, crop_height, config)
    } else {
        bbox.center_y() - (crop_height * config.headroom_ratio)
    };

    // E3: Horizontal softening.
    let raw_crop_x = bbox.center_x() - (crop_width / 2.0);
    let crop_x = if config.enable_horizontal_softening {
        apply_horizontal_softening(raw_crop_x, crop_width, bbox.center_x(), pw, config)
    } else {
        raw_crop_x
    };

    let crop_x = crop_x.max(0.0).min((pw - crop_width).max(0.0));
    let crop_y = crop_y.max(0.0).min((ph - crop_height).max(0.0));

    let crop = CropRegion {
        x: crop_x,
        y: crop_y,
        width: crop_width,
        height: crop_height,
    };

    if person_is_reasonably_visible_threshold(
        bbox,
        &crop,
        config.visibility_threshold,
        photo_width,
        photo_height,
    ) {
        Some(crop)
    } else {
        None
    }
}

/// Enhanced 9:21 portrait crop with face-aware anchoring (E1, E3, E5).
///
/// See [`calculate_portrait_9_21_crop`] for the base algorithm and
/// [`calculate_landscape_21_9_crop_with_face`] for enhancement docs.
pub fn calculate_portrait_9_21_crop_with_face(
    photo_width: u32,
    photo_height: u32,
    bbox: &BBox,
    face_bbox: Option<&BBox>,
    config: &CropConfig,
) -> Option<CropRegion> {
    const ASPECT_RATIO: f32 = 9.0 / 21.0;
    calculate_portrait_crop_enhanced(
        photo_width,
        photo_height,
        bbox,
        face_bbox,
        ASPECT_RATIO,
        config,
    )
}

/// Enhanced 9:16 portrait crop with face-aware anchoring (E1, E3, E5).
///
/// See [`calculate_portrait_9_16_crop`] for the base algorithm and
/// [`calculate_landscape_21_9_crop_with_face`] for enhancement docs.
pub fn calculate_portrait_9_16_crop_with_face(
    photo_width: u32,
    photo_height: u32,
    bbox: &BBox,
    face_bbox: Option<&BBox>,
    config: &CropConfig,
) -> Option<CropRegion> {
    const ASPECT_RATIO: f32 = 9.0 / 16.0;
    calculate_portrait_crop_enhanced(
        photo_width,
        photo_height,
        bbox,
        face_bbox,
        ASPECT_RATIO,
        config,
    )
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
/// Cropping behaviour (headroom, visibility threshold) is governed by `config`.
///
/// # Parameters
/// - `photo_width` / `photo_height`: Source photo dimensions in pixels.
/// - `bbox`: Raw detected bounding box in `[x1, y1, x2, y2]` pixel coordinates.
/// - `margin_percent`: Optional padding around bbox as percentage of bbox size (0 = no margin).
/// - `config`: Cropping configuration controlling headroom ratio and visibility threshold.
///
/// # Returns
/// A `Vec<String>` of format names (`"21:9"`, `"9:21"`, `"9:16"`) that are viable.
/// Returns an empty vec if no format can accommodate the person.
///
/// # Example
/// ```rust,ignore
/// let bbox = BBox { x1: 200.0, y1: 50.0, x2: 500.0, y2: 1900.0 };
/// let config = CropConfig::default();
/// let formats = detect_suitable_formats(3024, 4032, &bbox, 5.0, &config);
/// // likely ["21:9", "9:21", "9:16"] for a tall portrait person in a portrait photo
/// ```
pub fn detect_suitable_formats(
    photo_width: u32,
    photo_height: u32,
    bbox: &BBox,
    margin_percent: f32,
    config: &CropConfig,
) -> Vec<String> {
    let working_bbox = apply_margin_to_bbox(bbox, margin_percent, photo_width, photo_height);

    let bbox_is_wide = working_bbox.width() > working_bbox.height();

    let mut suitable: Vec<String> = Vec::new();

    // Landscape 21:9 is always attempted regardless of bbox orientation.
    if calculate_landscape_21_9_crop(photo_width, photo_height, &working_bbox, config).is_some() {
        suitable.push("21:9".to_string());
    }

    // Portrait formats are only attempted when the person's bbox is portrait-oriented.
    if !bbox_is_wide {
        if calculate_portrait_9_21_crop(photo_width, photo_height, &working_bbox, config).is_some()
        {
            suitable.push("9:21".to_string());
        }
        if calculate_portrait_9_16_crop(photo_width, photo_height, &working_bbox, config).is_some()
        {
            suitable.push("9:16".to_string());
        }
    }

    suitable
}

// ---------------------------------------------------------------------------
// E4: Reflection / Overlap Deduplication
// ---------------------------------------------------------------------------

/// Compute Intersection-over-Union (IoU) for two bounding boxes
/// given as `[x1, y1, x2, y2]` arrays.
///
/// Returns `0.0` when boxes do not overlap or either has zero area.
fn compute_iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let [ax1, ay1, ax2, ay2] = a;
    let [bx1, by1, bx2, by2] = b;

    let inter_x1 = ax1.max(bx1);
    let inter_y1 = ay1.max(by1);
    let inter_x2 = ax2.min(bx2);
    let inter_y2 = ay2.min(by2);

    let inter_w = (inter_x2 - inter_x1).max(0.0);
    let inter_h = (inter_y2 - inter_y1).max(0.0);
    let inter_area = inter_w * inter_h;

    let area_a = (ax2 - ax1).max(0.0) * (ay2 - ay1).max(0.0);
    let area_b = (bx2 - bx1).max(0.0) * (by2 - by1).max(0.0);
    let union_area = area_a + area_b - inter_area;

    if union_area <= 0.0 {
        0.0
    } else {
        inter_area / union_area
    }
}

/// Remove duplicate person detections caused by reflections, mirrors,
/// or glass surfaces using greedy Non-Maximum Suppression (NMS).
///
/// Detections are sorted by confidence (descending). For each kept
/// detection, any subsequent detection whose IoU exceeds
/// `iou_threshold` is suppressed. The surviving subset is returned
/// in confidence-descending order.
///
/// # Parameters
/// - `detections`: Person detections (class-filtered). Not modified.
/// - `iou_threshold`: IoU above which a lower-confidence detection
///   is suppressed. Typical: `0.50`.
///
/// # Returns
/// A new `Vec` containing only the surviving detections (cloned).
/// Empty input → empty output.
///
/// # Example
/// ```rust,ignore
/// let deduped = deduplicate_person_detections(&persons, 0.50);
/// let compound = calculate_compound_bbox(&deduped);
/// ```
pub fn deduplicate_person_detections(
    detections: &[crate::image_utils::Detection],
    iou_threshold: f32,
) -> Vec<crate::image_utils::Detection> {
    if detections.len() <= 1 {
        return detections.to_vec();
    }

    let mut sorted: Vec<&crate::image_utils::Detection> = detections.iter().collect();
    sorted.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut suppressed = vec![false; sorted.len()];
    let mut kept: Vec<crate::image_utils::Detection> = Vec::new();

    for i in 0..sorted.len() {
        if suppressed[i] {
            continue;
        }
        kept.push((*sorted[i]).clone());
        for j in (i + 1)..sorted.len() {
            if suppressed[j] {
                continue;
            }
            if compute_iou(sorted[i].bbox, sorted[j].bbox) > iou_threshold {
                suppressed[j] = true;
            }
        }
    }

    kept
}

// ---------------------------------------------------------------------------
// Multi-person: Compound Bounding Box
// ---------------------------------------------------------------------------

/// Calculate a single compound bounding box that encompasses all provided detections.
///
/// When multiple people are detected in a photo, this function produces a single
/// bbox that tightly wraps all of them. The resulting compound bbox can then be
/// passed to any of the cropping functions to produce a crop that frames the whole
/// group rather than just the highest-confidence individual.
///
/// # Parameters
/// - `detections`: Slice of detections, each carrying a `bbox: [x1, y1, x2, y2]` field.
///
/// # Returns
/// - `Some(BBox)` encompassing all detections if the slice is non-empty.
/// - `None` if `detections` is empty (no bbox to compute).
///
/// # Example
/// ```rust,ignore
/// let detections = vec![det_a, det_b, det_c];
/// if let Some(compound) = calculate_compound_bbox(&detections) {
///     let crop = calculate_landscape_21_9_crop(w, h, &compound, &config);
/// }
/// ```
pub fn calculate_compound_bbox(detections: &[crate::image_utils::Detection]) -> Option<BBox> {
    if detections.is_empty() {
        return None;
    }

    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;

    for det in detections {
        let [x1, y1, x2, y2] = det.bbox;
        if x1 < min_x {
            min_x = x1;
        }
        if y1 < min_y {
            min_y = y1;
        }
        if x2 > max_x {
            max_x = x2;
        }
        if y2 > max_y {
            max_y = y2;
        }
    }

    Some(BBox {
        x1: min_x,
        y1: min_y,
        x2: max_x,
        y2: max_y,
    })
}

// ---------------------------------------------------------------------------
// Face-aware artistic cropping
// ---------------------------------------------------------------------------

/// A (face, person) correlation pair produced by [`correlate_faces_to_persons`].
///
/// Represents a single face that was detected within a person's bounding box,
/// along with the metadata needed for dominant-face selection and crop computation.
///
/// # Deprecation Notice
/// This type is deprecated. The new face-aware pipeline uses `multi_format_cropping::BBox`
/// directly for both face and person bboxes.
///
/// # Example
/// ```rust,ignore
/// let pairs = correlate_faces_to_persons(&faces, &persons);
/// if let Some(dominant) = select_dominant_face(&pairs, &strategy, img_w, img_h) {
///     let crop = compute_artistic_crop(img_w, img_h, &dominant, &persons, &config, base_config);
/// }
/// ```
#[deprecated(
    since = "2.1.0",
    note = "Use multi_format_cropping::BBox directly with face_aware_cropping functions."
)]
#[derive(Debug, Clone)]
pub struct FacePersonPair {
    /// Zero-based index of the person detection this face belongs to.
    pub person_id: usize,
    /// Zero-based index within the original face detection list.
    pub face_id: usize,
    /// Detection confidence of the face (from the face model output).
    pub confidence: f32,
    /// Face bounding box in original image pixel coordinates (XYXY).
    pub face_bbox: BBox,
    /// Person bounding box in original image pixel coordinates (XYXY).
    pub person_bbox: BBox,
}

#[allow(deprecated)]
impl FacePersonPair {
    /// Returns the centroid (cx, cy) of the face bounding box in pixel coordinates.
    #[inline]
    pub fn face_centroid(&self) -> (f32, f32) {
        (self.face_bbox.center_x(), self.face_bbox.center_y())
    }

    /// Returns the area of the face bounding box in square pixels.
    #[inline]
    pub fn face_area(&self) -> f32 {
        self.face_bbox.width() * self.face_bbox.height()
    }
}

/// Correlate detected faces with detected persons using bounding-box containment.
///
/// For each face, checks whether it falls fully inside any person bounding box.
/// A face is assigned to the **first** person whose bbox fully contains it.
///
/// # Deprecation Notice
/// This function is deprecated. The new pipeline passes face bboxes directly to
/// [`crate::face_aware_cropping::apply_face_aware_adjustment`] without correlation.
///
/// # Arguments
/// * `faces`   — All face detections (from the YOLOv11x-Face model).
/// * `persons` — All person detections (from the YOLOv10 model), in XYXY pixel coords.
///
/// # Returns
/// A list of [`FacePersonPair`] sorted by (person_id, face confidence descending).
///
/// # Example
/// ```rust,ignore
/// let pairs = correlate_faces_to_persons(&face_detections, &person_bboxes);
/// assert!(pairs.iter().all(|p| p.person_id < persons.len()));
/// ```
#[deprecated(
    since = "2.1.0",
    note = "Pass face bboxes directly to face_aware_cropping::apply_face_aware_adjustment()."
)]
#[allow(deprecated)]
pub fn correlate_faces_to_persons(
    faces: &[crate::models::Detection],
    persons: &[BBox],
) -> Vec<FacePersonPair> {
    let mut pairs: Vec<FacePersonPair> = vec![];

    for (face_id, face) in faces.iter().enumerate() {
        let face_bbox = BBox {
            x1: face.bbox.x_min,
            y1: face.bbox.y_min,
            x2: face.bbox.x_max,
            y2: face.bbox.y_max,
        };

        // Assign to the first person bbox that fully contains this face.
        for (person_id, person_bbox) in persons.iter().enumerate() {
            if is_bbox_within(
                face_bbox.x1,
                face_bbox.y1,
                face_bbox.x2,
                face_bbox.y2,
                person_bbox,
            ) {
                pairs.push(FacePersonPair {
                    person_id,
                    face_id,
                    confidence: face.confidence,
                    face_bbox: face_bbox.clone(),
                    person_bbox: person_bbox.clone(),
                });
                break; // Each face is assigned to at most one person.
            }
        }
    }

    // Sort: primary by person_id (ascending), secondary by confidence (descending).
    pairs.sort_by(|a, b| {
        a.person_id.cmp(&b.person_id).then_with(|| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    });

    pairs
}

/// Return `true` if the rectangle `(x1, y1, x2, y2)` is fully contained within `container`.
///
/// "Fully contained" means all four corners of the inner rectangle are inside (or
/// on the boundary of) the container. This is a strict containment check — faces
/// that straddle a person's boundary are excluded to avoid false correlations.
///
/// # Example
/// ```rust,ignore
/// let person = BBox { x1: 0.0, y1: 0.0, x2: 200.0, y2: 400.0 };
/// assert!(is_bbox_within(50.0, 10.0, 150.0, 100.0, &person));
/// assert!(!is_bbox_within(-10.0, 10.0, 150.0, 100.0, &person));
/// ```
pub fn is_bbox_within(x1: f32, y1: f32, x2: f32, y2: f32, container: &BBox) -> bool {
    x1 >= container.x1 && y1 >= container.y1 && x2 <= container.x2 && y2 <= container.y2
}

/// Merge two bounding boxes into their smallest enclosing rectangle.
///
/// The resulting bbox is the minimum bounding rectangle that contains both inputs.
///
/// # Example
/// ```rust,ignore
/// let merged = merge_bboxes(
///     &BBox { x1: 10.0, y1: 10.0, x2: 100.0, y2: 200.0 },
///     &BBox { x1: 80.0, y1: 50.0, x2: 180.0, y2: 250.0 },
/// );
/// assert_eq!(merged.x1, 10.0);
/// assert_eq!(merged.x2, 180.0);
/// ```
pub fn merge_bboxes(a: &BBox, b: &BBox) -> BBox {
    BBox {
        x1: a.x1.min(b.x1),
        y1: a.y1.min(b.y1),
        x2: a.x2.max(b.x2),
        y2: a.y2.max(b.y2),
    }
}

/// Expand a bounding box by a fixed pixel margin in all four directions.
///
/// Unlike the percentage-based [`apply_margin_to_bbox`], this function uses an
/// absolute pixel margin suitable for face-region expansion (faces vary less in
/// size relative to the image than full-body person bboxes).
///
/// The result is clamped to `(0, 0, photo_width, photo_height)`.
///
/// # Arguments
/// * `bbox`         — Source bounding box.
/// * `margin_px`    — Margin in pixels to add on each side.
/// * `photo_width`  — Image width (clamp limit).
/// * `photo_height` — Image height (clamp limit).
///
/// # Example
/// ```rust,ignore
/// let expanded = expand_bbox_px(&face_bbox, 20, 1920, 1080);
/// ```
pub fn expand_bbox_px(bbox: &BBox, margin_px: u32, photo_width: u32, photo_height: u32) -> BBox {
    let m = margin_px as f32;
    BBox {
        x1: (bbox.x1 - m).max(0.0),
        y1: (bbox.y1 - m).max(0.0),
        x2: (bbox.x2 + m).min(photo_width as f32),
        y2: (bbox.y2 + m).min(photo_height as f32),
    }
}

// ---------------------------------------------------------------------------
// E1 + E5: Adaptive Headroom & Face-Anchor Positioning
// ---------------------------------------------------------------------------

/// Select the dominant face for initial crop anchoring (largest by area).
///
/// When multiple faces are detected, the largest face by bounding box
/// area is the most visually prominent subject and the best vertical
/// anchor for initial crop positioning.
///
/// This differs from `face_aware_cropping::find_dominant_face` (which
/// uses centrality for post-positioning adjustment). For initial crop
/// anchoring, area is the better heuristic.
///
/// # Returns
/// `None` if `face_bboxes` is empty.
pub fn select_dominant_face_for_crop(face_bboxes: &[BBox]) -> Option<&BBox> {
    face_bboxes.iter().max_by(|a, b| {
        let area_a = a.width() * a.height();
        let area_b = b.width() * b.height();
        area_a
            .partial_cmp(&area_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

/// Compute a pose-adaptive headroom ratio based on where the face
/// sits within the person bounding box.
///
/// Returns a value in `[0.0, 1.0]` representing the fraction of
/// crop height to allocate above the vertical anchor point.
///
/// # Three-tier logic
///
/// | Tier | `face_rel_y` range | Headroom returned | Rationale |
/// |------|--------------------|-------------------|-----------|
/// | 1    | `< 0.20`           | `config.headroom_ratio` | Arms-raised/reclining face-at-edge; caller uses face-anchor (E5) instead |
/// | 2    | `[0.20, 0.50)`     | `0.35` | Normal standing — face higher than body center → tighter headroom |
/// | 3    | `≥ 0.50` or no face| `config.headroom_ratio` | Sitting/bending/no face → original algorithm |
///
/// `face_rel_y` = `(face_center_y − person_bbox.y1) / person_bbox.height()`.
///
/// # Parameters
/// - `person_bbox`: Person detection bounding box.
/// - `face_bbox`: Dominant face bbox, or `None` if no face detected.
/// - `config`: Crop configuration (provides fallback `headroom_ratio`).
pub fn calculate_adaptive_headroom(
    person_bbox: &BBox,
    face_bbox: Option<&BBox>,
    config: &CropConfig,
) -> f32 {
    let face = match face_bbox {
        Some(f) => f,
        None => return config.headroom_ratio,
    };

    let person_height = person_bbox.height();
    if person_height <= 0.0 {
        return config.headroom_ratio;
    }

    let face_rel_y = (face.center_y() - person_bbox.y1) / person_height;

    if face_rel_y < 0.20 {
        // Tier 1: face-anchor mode — caller uses E5 formula; return safe fallback.
        config.headroom_ratio
    } else if face_rel_y < 0.50 {
        // Tier 2: normal standing — tighter headroom anchored to face center.
        0.35
    } else {
        // Tier 3: sitting/bending or face below center — original headroom.
        config.headroom_ratio
    }
}

/// Compute the vertical crop position (crop_y) using adaptive
/// face-aware anchoring.
///
/// Integrates E1 (multi-tier headroom) and E5 (face-anchor for
/// top-20% face positions) into a single crop_y computation.
///
/// # Algorithm
///
/// 1. If `face_bbox` is `None`: body-center formula
///    `person_center_y − crop_height × headroom_ratio`.
/// 2. Compute `face_rel_y` (face center's relative vertical position
///    within the person bbox; 0.0 = top, 1.0 = bottom).
/// 3. **Tier 1** (`face_rel_y < 0.20`, E5): Anchor crop top to
///    `face.y1 − 5% × crop_height` (tiny breathing gap above forehead).
/// 4. **Tier 2** (`0.20 ≤ face_rel_y < 0.50`, E1): Anchor on
///    face center with adaptive headroom (0.35).
/// 5. **Tier 3** (`face_rel_y ≥ 0.50`): Original body-center formula.
///
/// The returned value is **not clamped** — caller must clamp to
/// `[0.0, photo_height − crop_height]`.
///
/// # Parameters
/// - `person_bbox`: Person bounding box.
/// - `face_bbox`: Dominant face bbox, or `None`.
/// - `crop_height`: Height of the crop region in pixels.
/// - `config`: Crop config (`headroom_ratio` used for fallback).
///
/// # Returns
/// Raw `crop_y` value (may be negative; caller clamps).
pub fn calculate_crop_y_anchor(
    person_bbox: &BBox,
    face_bbox: Option<&BBox>,
    crop_height: f32,
    config: &CropConfig,
) -> f32 {
    let face = match face_bbox {
        Some(f) => f,
        None => {
            return person_bbox.center_y() - (crop_height * config.headroom_ratio);
        }
    };

    let person_height = person_bbox.height();
    if person_height <= 0.0 {
        return person_bbox.center_y() - (crop_height * config.headroom_ratio);
    }

    let face_rel_y = (face.center_y() - person_bbox.y1) / person_height;

    if face_rel_y < 0.20 {
        // Tier 1 (E5): Face-anchor — place crop top just above forehead.
        let breathing_gap = 0.05 * crop_height;
        face.y1 - breathing_gap
    } else if face_rel_y < 0.50 {
        // Tier 2 (E1): Face-center anchor with tighter headroom.
        let headroom = calculate_adaptive_headroom(person_bbox, Some(face), config);
        face.center_y() - (crop_height * headroom)
    } else {
        // Tier 3 (E1 fallback): Body-center with original headroom.
        person_bbox.center_y() - (crop_height * config.headroom_ratio)
    }
}

/// Select the dominant face from a list of face-person pairs using the given strategy.
///
/// When `pairs` is empty, returns `None`.
///
/// # Deprecation Notice
/// This function is deprecated. Use [`crate::face_aware_cropping::find_dominant_face`]
/// which works directly with `BBox` slices and always uses the `MostCentral` strategy.
///
/// # Arguments
/// * `pairs`      — Correlated face-person pairs (from [`correlate_faces_to_persons`]).
/// * `strategy`   — Selection algorithm.
/// * `img_w`      — Image width in pixels (used for centrality calculation).
/// * `img_h`      — Image height in pixels (used for centrality calculation).
///
/// # Returns
/// Reference to the dominant [`FacePersonPair`], or `None` if `pairs` is empty.
///
/// # Example
/// ```rust,ignore
/// use icarus_v2::config::FaceSelectionStrategy;
/// let dominant = select_dominant_face(&pairs, &FaceSelectionStrategy::MostCentral, 1920, 1080);
/// ```
#[deprecated(
    since = "2.1.0",
    note = "Use face_aware_cropping::find_dominant_face() which uses MostCentral strategy."
)]
#[allow(deprecated)]
pub fn select_dominant_face<'a>(
    pairs: &'a [FacePersonPair],
    strategy: &crate::config::FaceSelectionStrategy,
    img_w: u32,
    img_h: u32,
) -> Option<&'a FacePersonPair> {
    use crate::config::FaceSelectionStrategy;

    if pairs.is_empty() {
        return None;
    }

    let img_cx = img_w as f32 / 2.0;
    let img_cy = img_h as f32 / 2.0;
    let max_dist = ((img_cx * img_cx) + (img_cy * img_cy)).sqrt();

    match strategy {
        FaceSelectionStrategy::Largest => pairs.iter().max_by(|a, b| {
            a.face_area()
                .partial_cmp(&b.face_area())
                .unwrap_or(std::cmp::Ordering::Equal)
        }),

        FaceSelectionStrategy::HighestConfidence => pairs.iter().max_by(|a, b| {
            a.confidence
                .partial_cmp(&b.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        }),

        FaceSelectionStrategy::MostCentral => pairs.iter().min_by(|a, b| {
            let (ax, ay) = a.face_centroid();
            let (bx, by) = b.face_centroid();
            let da = ((ax - img_cx).powi(2) + (ay - img_cy).powi(2)).sqrt();
            let db = ((bx - img_cx).powi(2) + (by - img_cy).powi(2)).sqrt();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        }),

        FaceSelectionStrategy::WeightedScore => {
            // Compute per-pair score: 50% size + 30% confidence + 20% centrality.
            // All three components are normalised to [0, 1].
            let max_area = pairs
                .iter()
                .map(|p| p.face_area())
                .fold(0.0f32, f32::max)
                .max(1.0); // avoid division by zero

            pairs.iter().max_by(|a, b| {
                let score = |p: &FacePersonPair| {
                    let size_score = p.face_area() / max_area;
                    let conf_score = p.confidence; // already in [0,1]
                    let (px, py) = p.face_centroid();
                    let dist = ((px - img_cx).powi(2) + (py - img_cy).powi(2)).sqrt();
                    let central_score = if max_dist > 0.0 {
                        1.0 - (dist / max_dist)
                    } else {
                        1.0
                    };
                    0.5 * size_score + 0.3 * conf_score + 0.2 * central_score
                };

                score(a)
                    .partial_cmp(&score(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        }
    }
}

/// Compute a face-centric artistic crop region for the given aspect ratio.
///
/// Positions the crop to satisfy three competing objectives simultaneously:
/// 1. **Face visibility**: The face (plus margin) must be entirely inside the crop.
/// 2. **Body composition**: The person bbox is included proportionally per a fixed head-to-body ratio.
/// 3. **Artistic framing**: The face centroid is biased toward the crop center based on `mode`.
///
/// # Deprecation Notice
/// This function is deprecated. Use [`crate::face_aware_cropping::apply_face_aware_adjustment`]
/// instead, which provides cleaner separation of concerns and always returns a valid `CropRegion`.
///
/// # Arguments
/// * `photo_w` / `photo_h` — Source photo dimensions.
/// * `dominant_face` — The selected dominant face-person pair.
/// * `aspect_ratio` — Target crop width/height ratio (e.g. 21.0/9.0).
/// * `artistic`     — Artistic crop configuration.
/// * `base_config`  — Base crop config (visibility threshold, headroom ratio).
///
/// # Returns
/// `Some(CropRegion)` when all constraints are met, `None` otherwise.
///
/// # Example
/// ```rust,ignore
/// let crop = compute_artistic_crop(
///     3024, 4032, &dominant_face, 9.0 / 16.0, &artistic_config, &crop_config
/// );
/// ```
#[deprecated(
    since = "2.1.0",
    note = "Use face_aware_cropping::apply_face_aware_adjustment() instead."
)]
#[allow(deprecated)]
pub fn compute_artistic_crop(
    photo_w: u32,
    photo_h: u32,
    dominant_face: &FacePersonPair,
    aspect_ratio: f32, // width / height
    artistic: &crate::config::ArtisticCropConfig,
    base_config: &CropConfig,
) -> Option<CropRegion> {
    let pw = photo_w as f32;
    let ph = photo_h as f32;

    // Derive legacy params from the new config fields.
    let (face_centroid_bias, margin_multiplier, min_body_visibility) = match artistic.artistic_mode
    {
        crate::config::ArtisticMode::Conservative => (0.3f32, 1.5f32, 0.6f32),
        crate::config::ArtisticMode::Balanced => (0.6, 1.0, 0.4),
        crate::config::ArtisticMode::Aggressive => (0.9, 0.7, 0.2),
    };
    let face_margin = artistic.face_safety_margin_px;
    // Fixed head-to-body ratio (was configurable in old code, now mode-derived).
    let head_to_body = match artistic.artistic_mode {
        crate::config::ArtisticMode::Conservative => 0.4f32,
        crate::config::ArtisticMode::Balanced => 0.5,
        crate::config::ArtisticMode::Aggressive => 0.6,
    };
    // Step 1: Expand the face bbox by the pixel margin (used for visibility check below).

    // Step 2: Compute the target crop dimensions for this aspect ratio.
    // Prefer using the full photo width (portrait-like) or height (landscape-like).
    let (crop_w, crop_h) = if aspect_ratio >= 1.0 {
        // Landscape: start with full photo width.
        let cw = pw;
        let ch = cw / aspect_ratio;
        if ch <= ph {
            (cw, ch)
        } else {
            let ch2 = ph;
            (ch2 * aspect_ratio, ch2)
        }
    } else {
        // Portrait: start with full photo width.
        let cw = pw;
        let ch = cw / aspect_ratio;
        if ch <= ph {
            (cw, ch)
        } else {
            let ch2 = ph;
            (ch2 * aspect_ratio, ch2)
        }
    };

    // Step 3: Determine the face centroid.
    let (face_cx, face_cy) = dominant_face.face_centroid();

    // Step 4: Compute the person-body anchor point.
    // Blend face centroid with person bbox center using head_to_body_ratio.
    let person_cx = dominant_face.person_bbox.center_x();
    let person_cy = dominant_face.person_bbox.center_y();

    // Anchor Y: blend face cy and person cy. When head_to_body = 0.5, the anchor
    // sits halfway between the face centroid and the person center (body-aware).
    let anchor_cx = face_cx * head_to_body + person_cx * (1.0 - head_to_body);
    let anchor_cy = face_cy * head_to_body + person_cy * (1.0 - head_to_body);

    // Step 5: Position the crop.
    // face_centroid_bias controls where in the crop the face centroid lands:
    // 0.0 → face at top of crop, 1.0 → face at bottom.
    // We target: crop_y = anchor_cy - (crop_h * headroom_position)
    let headroom_position =
        base_config.headroom_ratio * (1.0 - face_centroid_bias) + 0.35 * face_centroid_bias;

    // Apply centroid bias: blend anchor_cx toward face_cx by centroid_bias.
    let biased_cx = anchor_cx * (1.0 - face_centroid_bias) + face_cx * face_centroid_bias;
    let biased_cy = anchor_cy * (1.0 - face_centroid_bias) + face_cy * face_centroid_bias;

    let crop_x_raw = biased_cx - (crop_w / 2.0);
    let crop_y_raw = biased_cy - (crop_h * headroom_position);

    // Step 6: Clamp to photo bounds.
    let crop_x = crop_x_raw.max(0.0).min((pw - crop_w).max(0.0));
    let crop_y = crop_y_raw.max(0.0).min((ph - crop_h).max(0.0));

    let crop = CropRegion {
        x: crop_x,
        y: crop_y,
        width: crop_w,
        height: crop_h,
    };

    // Step 7: Verify the face zone is fully visible with margins (face must never be cut).
    let effective_margin = (face_margin as f32 * margin_multiplier) as u32;
    let required_face_zone =
        expand_bbox_px(&dominant_face.face_bbox, effective_margin, photo_w, photo_h);

    if !is_region_visible(&required_face_zone, &crop) {
        return None;
    }

    // Step 8: Check person body visibility against the mode's minimum threshold.
    if !person_is_reasonably_visible_threshold(
        &dominant_face.person_bbox,
        &crop,
        min_body_visibility,
        photo_w,
        photo_h,
    ) {
        return None;
    }

    Some(crop)
}

/// Returns `true` if the bounding box `zone` is fully contained within the `crop` region.
///
/// Used to verify face visibility after crop positioning.
///
/// # Example
/// ```rust,ignore
/// let face_zone = BBox { x1: 100.0, y1: 100.0, x2: 200.0, y2: 200.0 };
/// let crop = CropRegion { x: 50.0, y: 50.0, width: 500.0, height: 400.0 };
/// assert!(is_region_visible(&face_zone, &crop));
/// ```
pub fn is_region_visible(zone: &BBox, crop: &CropRegion) -> bool {
    let crop_x2 = crop.x + crop.width;
    let crop_y2 = crop.y + crop.height;

    zone.x1 >= crop.x && zone.y1 >= crop.y && zone.x2 <= crop_x2 && zone.y2 <= crop_y2
}

/// Build the compound bounding box spanning all detected faces (for group shots).
///
/// In group shots, rather than selecting a single dominant face, we compute a
/// compound face bbox that encompasses all correlated faces, then use that as
/// the framing anchor.
///
/// # Deprecation Notice
/// This function is deprecated.
///
/// # Arguments
/// * `pairs` — All face-person correlation pairs.
///
/// # Returns
/// `Some(BBox)` spanning all face bboxes, or `None` if `pairs` is empty.
///
/// # Example
/// ```rust,ignore
/// let group_bbox = compound_face_bbox(&pairs);
/// ```
#[deprecated(since = "2.1.0", note = "Use merge_bboxes() on BBox slices directly.")]
#[allow(deprecated)]
pub fn compound_face_bbox(pairs: &[FacePersonPair]) -> Option<BBox> {
    if pairs.is_empty() {
        return None;
    }

    let merged = pairs
        .iter()
        .map(|p| &p.face_bbox)
        .cloned()
        .reduce(|acc, b| merge_bboxes(&acc, &b));

    merged
}

/// Detect which output formats support face-centric framing for this photo.
///
/// Extends [`detect_suitable_formats`] with face-aware crop positioning. When
/// face detections are available, the crop is positioned using [`compute_artistic_crop`]
/// rather than the person-bbox-only algorithm.
///
/// # Deprecation Notice
/// This function is deprecated. Use [`crate::face_aware_cropping::apply_face_aware_adjustment`]
/// combined with [`detect_suitable_formats`] instead.
///
/// # Arguments
/// * `photo_w` / `photo_h` — Source photo dimensions.
/// * `person_bbox`  — Compound or single person bounding box.
/// * `face_pairs`   — Correlated face-person pairs (may be empty).
/// * `margin_pct`   — Percentage margin for person bbox expansion.
/// * `artistic`     — Artistic crop configuration.
/// * `base_config`  — Base crop configuration.
///
/// # Returns
/// List of format name strings that produce valid face-centric crops.
///
/// # Example
/// ```rust,ignore
/// let formats = detect_suitable_formats_with_faces(
///     3024, 4032, &person_bbox, &pairs, 0.0, &artistic_config, &base_config
/// );
/// ```
#[deprecated(
    since = "2.1.0",
    note = "Use detect_suitable_formats() + face_aware_cropping::apply_face_aware_adjustment() instead."
)]
#[allow(deprecated)]
pub fn detect_suitable_formats_with_faces(
    photo_w: u32,
    photo_h: u32,
    person_bbox: &BBox,
    face_pairs: &[FacePersonPair],
    margin_pct: f32,
    artistic: &crate::config::ArtisticCropConfig,
    base_config: &CropConfig,
) -> Vec<String> {
    if face_pairs.is_empty() {
        // Fall back to person-bbox-only format detection.
        return detect_suitable_formats(photo_w, photo_h, person_bbox, margin_pct, base_config);
    }

    // Select the dominant face using MostCentral strategy (always, per new design).
    #[allow(deprecated)]
    let dominant = select_dominant_face(
        face_pairs,
        &crate::config::FaceSelectionStrategy::MostCentral,
        photo_w,
        photo_h,
    );

    let Some(dominant) = dominant else {
        return detect_suitable_formats(photo_w, photo_h, person_bbox, margin_pct, base_config);
    };

    // Attempt each candidate aspect ratio.
    let candidate_ratios: &[(&str, f32)] = &[
        ("21:9", 21.0 / 9.0),
        ("9:16", 9.0 / 16.0),
        ("9:21", 9.0 / 21.0),
    ];

    candidate_ratios
        .iter()
        .filter_map(|(name, ratio)| {
            #[allow(deprecated)]
            let result =
                compute_artistic_crop(photo_w, photo_h, dominant, *ratio, artistic, base_config)
                    .map(|_| name.to_string());
            result
        })
        .collect()
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
        let config = CropConfig::default();
        let result = calculate_landscape_21_9_crop(4000, 3000, &bbox, &config);
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
        let config = CropConfig::default();
        let result = calculate_landscape_21_9_crop(1920, 1080, &bbox, &config);
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
        let config = CropConfig::default();
        let result = calculate_portrait_9_21_crop(photo_w, photo_h, &bbox, &config);
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
        let config = CropConfig::default();
        let result = calculate_portrait_9_16_crop(photo_w, photo_h, &bbox, &config);
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
        let config = CropConfig::default();
        let formats = detect_suitable_formats(1920, 1080, &bbox, 0.0, &config);
        // Wide bbox → portrait formats must be skipped
        assert!(!formats.contains(&"9:21".to_string()));
        assert!(!formats.contains(&"9:16".to_string()));
    }

    #[test]
    fn test_tall_bbox_can_return_portrait_formats() {
        let photo_w = 3024u32;
        let photo_h = 4032u32;
        let bbox = tall_person_bbox();
        let config = CropConfig::default();
        let formats = detect_suitable_formats(photo_w, photo_h, &bbox, 0.0, &config);
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
        let config = CropConfig::default();
        // With margin, the bbox expands; portrait formats should still be viable
        let formats_with_margin = detect_suitable_formats(photo_w, photo_h, &bbox, 10.0, &config);
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
        let config = CropConfig::default();
        let formats = detect_suitable_formats(1920, 1080, &bbox, 0.0, &config);
        assert!(
            formats.contains(&"21:9".to_string()),
            "Landscape format should be viable for this photo/person: {:?}",
            formats
        );
    }

    // ── E4 Tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_compute_iou_full_overlap() {
        let a = [100.0, 200.0, 400.0, 800.0];
        let result = compute_iou(a, a);
        assert!(
            (result - 1.0).abs() < 0.001,
            "full overlap should be 1.0, got {result}"
        );
    }

    #[test]
    fn test_compute_iou_no_overlap() {
        let a = [0.0, 0.0, 100.0, 100.0];
        let b = [200.0, 200.0, 300.0, 300.0];
        let result = compute_iou(a, b);
        assert_eq!(result, 0.0, "disjoint boxes should be 0.0");
    }

    #[test]
    fn test_compute_iou_partial_overlap() {
        let a = [0.0, 0.0, 200.0, 200.0];
        let b = [100.0, 100.0, 300.0, 300.0];
        // intersection: 100×100=10000; union: 40000+40000-10000=70000
        let expected = 10000.0 / 70000.0;
        let result = compute_iou(a, b);
        assert!(
            (result - expected).abs() < 0.001,
            "expected {expected:.4}, got {result:.4}"
        );
    }

    #[test]
    fn test_deduplicate_removes_reflection() {
        use crate::image_utils::Detection;
        let det_a = Detection {
            bbox: [100.0, 200.0, 400.0, 800.0],
            confidence: 0.90,
            label: "person".to_string(),
            class_id: 0,
        };
        let det_b = Detection {
            bbox: [110.0, 205.0, 410.0, 805.0],
            confidence: 0.85,
            label: "person".to_string(),
            class_id: 0,
        };
        // These boxes overlap heavily — det_b should be suppressed
        let result = deduplicate_person_detections(&[det_a, det_b], 0.50);
        assert_eq!(result.len(), 1, "reflection should be suppressed");
        assert!(
            (result[0].confidence - 0.90).abs() < 0.001,
            "higher-confidence detection should survive"
        );
    }

    #[test]
    fn test_deduplicate_preserves_separate_persons() {
        use crate::image_utils::Detection;
        let det_a = Detection {
            bbox: [100.0, 200.0, 400.0, 800.0],
            confidence: 0.90,
            label: "person".to_string(),
            class_id: 0,
        };
        let det_b = Detection {
            bbox: [600.0, 200.0, 900.0, 800.0],
            confidence: 0.85,
            label: "person".to_string(),
            class_id: 0,
        };
        let result = deduplicate_person_detections(&[det_a, det_b], 0.50);
        assert_eq!(result.len(), 2, "distinct persons should both be kept");
    }

    #[test]
    fn test_deduplicate_empty_input() {
        use crate::image_utils::Detection;
        let result = deduplicate_person_detections(&[] as &[Detection], 0.50);
        assert!(result.is_empty());
    }

    // ── E1+E5 Tests ──────────────────────────────────────────────────────

    #[test]
    fn test_select_dominant_face_largest() {
        let face_a = BBox {
            x1: 100.0,
            y1: 100.0,
            x2: 200.0,
            y2: 200.0,
        }; // area=10000
        let face_b = BBox {
            x1: 300.0,
            y1: 300.0,
            x2: 500.0,
            y2: 500.0,
        }; // area=40000
        let faces = [face_a, face_b];
        let result = select_dominant_face_for_crop(&faces);
        assert!(result.is_some());
        assert!(
            (result.unwrap().x1 - 300.0).abs() < 0.001,
            "largest face should be selected"
        );
    }

    #[test]
    fn test_select_dominant_face_empty() {
        let result = select_dominant_face_for_crop(&[]);
        assert!(result.is_none());
    }

    #[test]
    fn test_adaptive_headroom_tier1_top_20_percent() {
        // face_rel_y = (450-200)/3600 = 0.069 → Tier 1
        let person = BBox {
            x1: 500.0,
            y1: 200.0,
            x2: 1500.0,
            y2: 3800.0,
        };
        let face = BBox {
            x1: 700.0,
            y1: 250.0,
            x2: 1300.0,
            y2: 650.0,
        }; // center_y=450
        let config = CropConfig::default();
        let result = calculate_adaptive_headroom(&person, Some(&face), &config);
        assert!(
            (result - config.headroom_ratio).abs() < 0.001,
            "tier 1 returns fallback ratio"
        );
    }

    #[test]
    fn test_adaptive_headroom_tier2_normal_range() {
        // face_rel_y = (1100-200)/3600 = 0.25 → Tier 2
        let person = BBox {
            x1: 500.0,
            y1: 200.0,
            x2: 1500.0,
            y2: 3800.0,
        };
        let face = BBox {
            x1: 700.0,
            y1: 900.0,
            x2: 1300.0,
            y2: 1300.0,
        }; // center_y=1100
        let config = CropConfig::default();
        let result = calculate_adaptive_headroom(&person, Some(&face), &config);
        assert!(
            (result - 0.35).abs() < 0.001,
            "tier 2 should return 0.35, got {result}"
        );
    }

    #[test]
    fn test_adaptive_headroom_tier3_sitting_pose() {
        // face_rel_y = (2200-200)/3600 = 0.556 → Tier 3
        let person = BBox {
            x1: 500.0,
            y1: 200.0,
            x2: 1500.0,
            y2: 3800.0,
        };
        let face = BBox {
            x1: 700.0,
            y1: 2000.0,
            x2: 1300.0,
            y2: 2400.0,
        }; // center_y=2200
        let config = CropConfig::default();
        let result = calculate_adaptive_headroom(&person, Some(&face), &config);
        assert!(
            (result - config.headroom_ratio).abs() < 0.001,
            "tier 3 should return config.headroom_ratio"
        );
    }

    #[test]
    fn test_adaptive_headroom_no_face() {
        let person = BBox {
            x1: 500.0,
            y1: 200.0,
            x2: 1500.0,
            y2: 3800.0,
        };
        let config = CropConfig::default();
        let result = calculate_adaptive_headroom(&person, None, &config);
        assert!(
            (result - config.headroom_ratio).abs() < 0.001,
            "no face should return config.headroom_ratio"
        );
    }

    #[test]
    fn test_crop_y_anchor_tier1_face_anchor() {
        // face_rel_y = (450-200)/3600 = 0.069 → Tier 1 → face.y1 - 5%*crop_h
        let person = BBox {
            x1: 500.0,
            y1: 200.0,
            x2: 1500.0,
            y2: 3800.0,
        };
        let face = BBox {
            x1: 700.0,
            y1: 250.0,
            x2: 1300.0,
            y2: 650.0,
        }; // y1=250
        let crop_height = 1296.0;
        let config = CropConfig::default();
        let result = calculate_crop_y_anchor(&person, Some(&face), crop_height, &config);
        let expected = 250.0 - (0.05 * 1296.0); // = 250 - 64.8 = 185.2
        assert!(
            (result - expected).abs() < 0.5,
            "tier 1 face anchor expected {expected:.1}, got {result:.1}"
        );
    }

    #[test]
    fn test_crop_y_anchor_no_face() {
        // No face → person center anchor
        let person = BBox {
            x1: 500.0,
            y1: 200.0,
            x2: 1500.0,
            y2: 3800.0,
        }; // center_y=2000
        let crop_height = 1296.0;
        let config = CropConfig::default(); // headroom=0.40
        let result = calculate_crop_y_anchor(&person, None, crop_height, &config);
        let expected = 2000.0 - (1296.0 * 0.40); // = 2000 - 518.4 = 1481.6
        assert!(
            (result - expected).abs() < 0.5,
            "no face expected {expected:.1}, got {result:.1}"
        );
    }

    // ── E2 Tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_landscape_aspect_narrow_subject() {
        // person_width_ratio = 400/4000 = 0.10
        // t = 0.10/0.40 = 0.25
        // expected = 2.333 + 0.75 * (2.778-2.333) = 2.333 + 0.333 = 2.667
        let bbox = BBox {
            x1: 1800.0,
            y1: 200.0,
            x2: 2200.0,
            y2: 3800.0,
        };
        let config = CropConfig::default();
        let result = calculate_landscape_aspect_ratio(&bbox, 4000, &config);
        let expected = 21.0_f32 / 9.0 + 0.75 * (config.max_landscape_aspect - 21.0 / 9.0);
        assert!(
            (result - expected).abs() < 0.01,
            "narrow subject expected {expected:.4}, got {result:.4}"
        );
    }

    #[test]
    fn test_landscape_aspect_wide_subject() {
        // person_width_ratio = 1600/4000 = 0.40 → standard 21:9
        let bbox = BBox {
            x1: 800.0,
            y1: 200.0,
            x2: 2400.0,
            y2: 3800.0,
        };
        let config = CropConfig::default();
        let result = calculate_landscape_aspect_ratio(&bbox, 4000, &config);
        assert!(
            (result - 21.0 / 9.0).abs() < 0.01,
            "wide subject should get standard 21:9"
        );
    }

    #[test]
    fn test_landscape_aspect_very_wide_subject() {
        // person_width_ratio = 0.80 > 0.40
        let bbox = BBox {
            x1: 400.0,
            y1: 200.0,
            x2: 3600.0,
            y2: 3800.0,
        };
        let config = CropConfig::default();
        let result = calculate_landscape_aspect_ratio(&bbox, 4000, &config);
        assert!((result - 21.0 / 9.0).abs() < 0.01);
    }

    // ── E3 Tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_softening_centered_subject() {
        // off_center_frac = 0.0 → below threshold → unchanged
        let config = CropConfig::default();
        let result = apply_horizontal_softening(500.0, 2000.0, 2000.0, 4000.0, &config);
        assert!(
            (result - 500.0).abs() < 0.001,
            "centered subject should be unchanged"
        );
    }

    #[test]
    fn test_softening_far_right_subject() {
        // person_center_x=3600, photo_center=2000
        // off_center_frac = 1600/2000 = 0.80
        // activation = (0.80-0.50)/0.50 = 0.60
        // blend = 0.60 * 0.30 = 0.18
        // centered_x = 2000-1000 = 1000
        // expected = 2800*0.82 + 1000*0.18 = 2296 + 180 = 2476
        let config = CropConfig::default();
        let result = apply_horizontal_softening(2800.0, 2000.0, 3600.0, 4000.0, &config);
        let expected = 2476.0;
        assert!(
            (result - expected).abs() < 1.0,
            "far right subject expected ~{expected:.0}, got {result:.1}"
        );
        assert!(result < 2800.0, "softened crop should be shifted left");
    }

    #[test]
    fn test_softening_strength_zero_disables() {
        let config = CropConfig {
            softening_strength: 0.0,
            ..CropConfig::default()
        };
        let result = apply_horizontal_softening(2800.0, 2000.0, 3600.0, 4000.0, &config);
        assert!(
            (result - 2800.0).abs() < 0.001,
            "zero strength should not change crop_x"
        );
    }

    #[test]
    fn test_softening_at_threshold_boundary() {
        // off_center_frac = |3000-2000|/2000 = 0.50 → exactly at threshold → no softening
        let config = CropConfig::default();
        let result = apply_horizontal_softening(1000.0, 2000.0, 3000.0, 4000.0, &config);
        assert!(
            (result - 1000.0).abs() < 0.001,
            "exactly at threshold should not soften"
        );
    }
}
