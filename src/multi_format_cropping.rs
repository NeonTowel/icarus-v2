/// Multi-Format Intelligent Cropping Module
///
/// Provides automatic detection of suitable wallpaper formats (landscape 21:9,
/// portrait 9:21, portrait 9:16) from a single photo based on the detected
/// person's bounding box and photo dimensions.
///
/// # Algorithm Overview
///
/// 1. **Bbox Orientation**: Wide bbox → landscape only; tall/square bbox → try all formats.
/// 2. **Landscape 21:9 (Bbox-Center Anchor)**: Crop center placed at the person bbox vertical
///    center (`target_y_frac = 0.50`), keeping the full subject visible.
/// 3. **Portrait 9:21 & 9:16 (Bbox-Center Anchor)**: Same configurable ratio.
/// 4. **Margin**: Optional symmetric padding (percentage of bbox dimensions) before positioning.
/// 5. **Visibility Gate**: Person must be ≥`visibility_threshold` visible in the final crop,
///    or format is skipped (configurable, default 50%).
/// 6. **Eye-Safety Guard**: After crop placement, [`crate::face_aware_cropping::enforce_eye_safety`]
///    nudges vertically by up to 3% of crop height to prevent facial clipping.
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
        let x1 = self.x.clamp(0.0, photo_width as f32);
        let y1 = self.y.clamp(0.0, photo_height as f32);
        let x2 = (self.x + self.width).clamp(0.0, photo_width as f32);
        let y2 = (self.y + self.height).clamp(0.0, photo_height as f32);
        [x1, y1, x2, y2]
    }
}

pub fn place_focal_point(
    photo_width: u32,
    photo_height: u32,
    crop_width: f32,
    crop_height: f32,
    focal: &crate::focal_point::FocalPoint,
    target_x_frac: f32,
    target_y_frac: f32,
) -> CropRegion {
    let mut x = focal.x - target_x_frac * crop_width;
    let mut y = focal.y - target_y_frac * crop_height;

    let max_x = (photo_width as f32 - crop_width).max(0.0);
    let max_y = (photo_height as f32 - crop_height).max(0.0);

    x = x.clamp(0.0, max_x);
    y = y.clamp(0.0, max_y);

    CropRegion {
        x,
        y,
        width: crop_width,
        height: crop_height,
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

/// Detected pose type inferred from person and face geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoseType {
    /// Head is in the upper portion of the person bbox or the bbox is tall and narrow.
    Standing,
    /// Head is lower in the person bbox or the bbox is short and wide.
    Sitting,
    /// Pose could not be classified confidently.
    Unknown,
}

/// Target crop format category for pose-adaptive headroom selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CropFormat {
    /// 21:9 landscape crop.
    Landscape,
    /// 9:16 portrait crop.
    Portrait,
    /// 9:21 mobile crop.
    Mobile,
}

/// Horizontal facing direction used for rule-of-thirds portrait framing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Subject is facing toward the right side of the photo.
    FacingRight,
    /// Subject is facing toward the left side of the photo.
    FacingLeft,
    /// Subject is facing the camera or the signal is ambiguous.
    Frontal,
}

/// Detect whether the subject is standing, sitting, or ambiguous.
///
/// # Example
/// ```rust,ignore
/// let pose = detect_pose_type(&person_bbox, Some(&face_bbox));
/// ```
pub fn detect_pose_type(person_bbox: &BBox, face_bbox: Option<&BBox>) -> PoseType {
    if person_bbox.height() <= 0.0 || person_bbox.width() <= 0.0 {
        return PoseType::Unknown;
    }

    if let Some(face) = face_bbox {
        if face.height() > 0.0 {
            let face_rel_y = (face.center_y() - person_bbox.y1) / person_bbox.height();
            if face_rel_y < 0.25 {
                return PoseType::Standing;
            }
            if face_rel_y > 0.35 {
                return PoseType::Sitting;
            }
            return PoseType::Unknown;
        }
    }

    let aspect_ratio = person_bbox.height() / person_bbox.width();
    if aspect_ratio > 2.5 {
        PoseType::Standing
    } else if aspect_ratio < 1.5 {
        PoseType::Sitting
    } else {
        PoseType::Unknown
    }
}

/// Select a headroom ratio for the detected pose and target crop format.
///
/// # Example
/// ```rust,ignore
/// let headroom = calculate_headroom_for_pose(PoseType::Standing, CropFormat::Portrait);
/// ```
pub fn calculate_headroom_for_pose(pose: PoseType, format: CropFormat) -> f32 {
    match (format, pose) {
        (CropFormat::Landscape, PoseType::Standing) => 0.48,
        (CropFormat::Landscape, PoseType::Sitting) => 0.50,
        (CropFormat::Landscape, PoseType::Unknown) => 0.45,
        (CropFormat::Portrait, PoseType::Standing) => 0.45,
        (CropFormat::Portrait, PoseType::Sitting) => 0.50,
        (CropFormat::Portrait, PoseType::Unknown) => 0.45,
        (CropFormat::Mobile, PoseType::Standing) => 0.43,
        (CropFormat::Mobile, PoseType::Sitting) => 0.48,
        (CropFormat::Mobile, PoseType::Unknown) => 0.45,
    }
}

/// Detect whether the subject is facing left, right, or front.
///
/// # Example
/// ```rust,ignore
/// let direction = detect_pose_direction(&person_bbox, Some(&face_bbox), photo_width as f32);
/// ```
pub fn detect_pose_direction(person_bbox: &BBox, face_bbox: &BBox) -> Direction {
    if person_bbox.width() <= 0.0 {
        return Direction::Frontal;
    }

    if face_bbox.width() > 0.0 {
        let face_rel_x = (face_bbox.center_x() - person_bbox.x1) / person_bbox.width();
        if face_rel_x < 0.35 {
            return Direction::FacingLeft;
        }
        if face_rel_x > 0.65 {
            return Direction::FacingRight;
        }
    }

    Direction::Frontal
}

/// Calculate the left edge of a portrait crop using rule-of-thirds placement.
///
/// # Example
/// ```rust,ignore
/// let crop_x = calculate_portrait_crop_x(&person_bbox, 1200.0, Direction::FacingRight, 3024.0);
/// ```
pub fn calculate_portrait_crop_x(
    person_bbox: &BBox,
    crop_width: f32,
    direction: Direction,
    photo_width: f32,
) -> f32 {
    let person_center_x = person_bbox.center_x();
    let max_x = (photo_width - crop_width).max(0.0);

    let crop_x = match direction {
        Direction::FacingRight => person_center_x - (crop_width / 3.0),
        Direction::FacingLeft => person_center_x - (crop_width * 2.0 / 3.0),
        Direction::Frontal => person_center_x - (crop_width / 2.0),
    };

    crop_x.clamp(0.0, max_x)
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
    face_bboxes: &[BBox],
    focal: &crate::focal_point::FocalPoint,
    config: &CropConfig,
    artistic: &crate::config::ArtisticCropConfig,
) -> Option<CropRegion> {
    let pw = photo_width as f32;
    let ph = photo_height as f32;

    let aspect_ratio = 21.0 / 9.0;

    // Compute crop dimensions.
    let mut crop_height = ph;
    let mut crop_width = crop_height * aspect_ratio;
    if crop_width > pw {
        crop_width = pw;
        crop_height = crop_width / aspect_ratio;
    }

    let mut target_x_frac = 0.50;
    if config.enable_directional_thirds && !face_bboxes.is_empty() {
        let direction = detect_pose_direction(bbox, &face_bboxes[0]);
        target_x_frac = match direction {
            Direction::FacingRight => 0.33,
            Direction::FacingLeft => 0.67,
            Direction::Frontal => 0.50,
        };
    }

    let mut target_y_frac = config.target_y_frac_landscape + artistic.target_y_offset();
    target_y_frac = target_y_frac.clamp(0.10, 0.50);

    let crop = place_focal_point(
        photo_width,
        photo_height,
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

/// Calculate a 9:21 ultrawide portrait crop with configurable bbox-center upward bias.
///
/// See the shared portrait crop helper in this module for the full algorithm description.
///
/// # Parameters
/// - `photo_width` / `photo_height`: Source photo dimensions.
/// - `bbox`: Person bounding box (should already include margin if applicable).
/// - `config`: Cropping configuration controlling headroom and visibility threshold.
///
/// # Returns
/// `Some(CropRegion)` when the person is sufficiently visible, `None` otherwise.
pub fn calculate_portrait_9_21_crop_with_face(
    photo_width: u32,
    photo_height: u32,
    bbox: &BBox,
    face_bboxes: &[BBox],
    focal: &crate::focal_point::FocalPoint,
    config: &CropConfig,
    artistic: &crate::config::ArtisticCropConfig,
) -> Option<CropRegion> {
    const ASPECT_RATIO: f32 = 9.0 / 21.0;
    calculate_portrait_crop_enhanced(
        photo_width,
        photo_height,
        bbox,
        face_bboxes,
        focal,
        ASPECT_RATIO,
        config.target_y_frac_mobile,
        config,
        artistic,
    )
}

pub fn calculate_portrait_9_16_crop_with_face(
    photo_width: u32,
    photo_height: u32,
    bbox: &BBox,
    face_bboxes: &[BBox],
    focal: &crate::focal_point::FocalPoint,
    config: &CropConfig,
    artistic: &crate::config::ArtisticCropConfig,
) -> Option<CropRegion> {
    const ASPECT_RATIO: f32 = 9.0 / 16.0;
    calculate_portrait_crop_enhanced(
        photo_width,
        photo_height,
        bbox,
        face_bboxes,
        focal,
        ASPECT_RATIO,
        config.target_y_frac_portrait,
        config,
        artistic,
    )
}

#[allow(clippy::too_many_arguments)]
fn calculate_portrait_crop_enhanced(
    photo_width: u32,
    photo_height: u32,
    bbox: &BBox,
    face_bboxes: &[BBox],
    focal: &crate::focal_point::FocalPoint,
    aspect_ratio: f32,
    target_y_frac_base: f32,
    config: &CropConfig,
    artistic: &crate::config::ArtisticCropConfig,
) -> Option<CropRegion> {
    let pw = photo_width as f32;
    let ph = photo_height as f32;

    let mut crop_width = pw;
    let mut crop_height = crop_width / aspect_ratio;
    if crop_height > ph {
        crop_height = ph;
        crop_width = crop_height * aspect_ratio;
    }

    let mut target_x_frac = 0.50;
    if config.enable_directional_thirds && !face_bboxes.is_empty() {
        let direction = detect_pose_direction(bbox, &face_bboxes[0]);
        target_x_frac = match direction {
            Direction::FacingRight => 0.33,
            Direction::FacingLeft => 0.67,
            Direction::Frontal => 0.50,
        };
    }

    let mut target_y_frac = target_y_frac_base + artistic.target_y_offset();
    target_y_frac = target_y_frac.clamp(0.10, 0.50);

    let crop = place_focal_point(
        photo_width,
        photo_height,
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
        photo_width,
        photo_height,
    ) {
        Some(crop)
    } else {
        None
    }
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

    let focal = crate::focal_point::compute_focal_point(
        Some(&working_bbox),
        &[],
        photo_width,
        photo_height,
    );

    // Landscape 21:9 is always attempted regardless of bbox orientation.
    if calculate_landscape_21_9_crop_with_face(
        photo_width,
        photo_height,
        &working_bbox,
        &[],
        &focal,
        config,
        &crate::config::ArtisticCropConfig::default(),
    )
    .is_some()
    {
        suitable.push("21:9".to_string());
    }

    // Portrait formats are only attempted when the person's bbox is portrait-oriented.
    if !bbox_is_wide {
        let focal = crate::focal_point::compute_focal_point(
            Some(&working_bbox),
            &[],
            photo_width,
            photo_height,
        );
        if calculate_portrait_9_21_crop_with_face(
            photo_width,
            photo_height,
            &working_bbox,
            &[],
            &focal,
            config,
            &crate::config::ArtisticCropConfig::default(),
        )
        .is_some()
        {
            suitable.push("9:21".to_string());
        }
        if calculate_portrait_9_16_crop_with_face(
            photo_width,
            photo_height,
            &working_bbox,
            &[],
            &focal,
            config,
            &crate::config::ArtisticCropConfig::default(),
        )
        .is_some()
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
    _artistic: &crate::config::ArtisticCropConfig,
    base_config: &CropConfig,
) -> Vec<String> {
    if face_pairs.is_empty() {
        // Fall back to person-bbox-only format detection.
        return detect_suitable_formats(photo_w, photo_h, person_bbox, margin_pct, base_config);
    }
    vec![]
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

    // --- Milestone 1: pose detection and pose-aware headroom ---

    #[test]
    fn test_detect_pose_standing_with_face() {
        let person = BBox {
            x1: 100.0,
            y1: 100.0,
            x2: 200.0,
            y2: 500.0,
        };
        let face = BBox {
            x1: 120.0,
            y1: 140.0,
            x2: 180.0,
            y2: 180.0,
        };
        assert_eq!(detect_pose_type(&person, Some(&face)), PoseType::Standing);
    }

    #[test]
    fn test_detect_pose_sitting_with_face() {
        let person = BBox {
            x1: 100.0,
            y1: 100.0,
            x2: 200.0,
            y2: 500.0,
        };
        let face = BBox {
            x1: 120.0,
            y1: 280.0,
            x2: 180.0,
            y2: 340.0,
        };
        assert_eq!(detect_pose_type(&person, Some(&face)), PoseType::Sitting);
    }

    #[test]
    fn test_detect_pose_unknown_in_ambiguous_face_zone() {
        let person = BBox {
            x1: 100.0,
            y1: 100.0,
            x2: 200.0,
            y2: 500.0,
        };
        let face = BBox {
            x1: 120.0,
            y1: 205.0,
            x2: 180.0,
            y2: 245.0,
        };
        assert_eq!(detect_pose_type(&person, Some(&face)), PoseType::Unknown);
    }

    #[test]
    fn test_detect_pose_standing_without_face_from_aspect() {
        let person = BBox {
            x1: 100.0,
            y1: 100.0,
            x2: 220.0,
            y2: 700.0,
        };
        assert_eq!(detect_pose_type(&person, None), PoseType::Standing);
    }

    #[test]
    fn test_detect_pose_sitting_without_face_from_aspect() {
        let person = BBox {
            x1: 100.0,
            y1: 100.0,
            x2: 420.0,
            y2: 260.0,
        };
        assert_eq!(detect_pose_type(&person, None), PoseType::Sitting);
    }

    #[test]
    fn test_headroom_for_pose_landscape_standing() {
        let headroom = calculate_headroom_for_pose(PoseType::Standing, CropFormat::Landscape);
        assert!((headroom - 0.48).abs() < 0.001);
    }

    #[test]
    fn test_headroom_for_pose_portrait_sitting() {
        let headroom = calculate_headroom_for_pose(PoseType::Sitting, CropFormat::Portrait);
        assert!((headroom - 0.50).abs() < 0.001);
    }

    #[test]
    fn test_headroom_for_pose_mobile_unknown() {
        let headroom = calculate_headroom_for_pose(PoseType::Unknown, CropFormat::Mobile);
        assert!((headroom - 0.45).abs() < 0.001);
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
        // raw x = 100 - 200 = -100 -> clamps to 0
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
        // raw x = 900 - 200 = 700. Max x = 1000 - 400 = 600. Clamps to 600
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
        // x = 600 - 0.33*300 = 501. y = 300 - 0.30*400 = 180
        assert_eq!(c.x, 501.0);
        assert_eq!(c.y, 180.0);
    }
}
