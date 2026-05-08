use crate::multi_format_cropping::BBox;

// ---------------------------------------------------------------------------
// SubjectAnchor — the primary positioning type (bbox-center based)
// ---------------------------------------------------------------------------

/// The computed anchor point for crop positioning, derived from the person bbox center.
///
/// This is the canonical anchor: vertical placement is the bbox vertical midpoint,
/// horizontal placement is the bbox horizontal midpoint. The eye-safety validator
/// in [`crate::face_aware_cropping::enforce_eye_safety`] may nudge the final crop
/// vertically within a 3% budget, but it does not change this anchor.
///
/// # Example
/// ```rust,ignore
/// let anchor = compute_subject_anchor(&person_bboxes, &face_bboxes, 1920, 1080);
/// // anchor.x == person_bbox.center_x()
/// // anchor.y == person_bbox.center_y()
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubjectAnchor {
    /// Horizontal center of the subject bbox (pixels from left).
    pub x: f32,
    /// Vertical center of the subject bbox (pixels from top).
    pub y: f32,
}

/// Compute the subject anchor as the bbox geometric center.
///
/// Decision tree:
/// 1. If `person_bbox` is `Some`, return its center `(center_x, center_y)`.
/// 2. Otherwise fall back to the image center.
///
/// Face detections are accepted as a parameter but are **not used** for vertical
/// anchoring. They are reserved for the eye-safety guard-rail applied downstream
/// by [`crate::face_aware_cropping::enforce_eye_safety`].
///
/// # Parameters
/// - `person_bbox`: Optional person bounding box (margin-expanded).
/// - `_face_bboxes`: Face detections — unused here, present for API symmetry.
/// - `image_width` / `image_height`: Photo dimensions for the fallback center.
///
/// # Example
/// ```rust,ignore
/// let anchor = compute_subject_anchor(
///     Some(&person_bbox), &face_bboxes, 1920, 1080,
/// );
/// assert_eq!(anchor.y, person_bbox.center_y());
/// ```
pub fn compute_subject_anchor(
    person_bbox: Option<&BBox>,
    _face_bboxes: &[BBox],
    image_width: u32,
    image_height: u32,
) -> SubjectAnchor {
    match person_bbox {
        Some(bbox) => SubjectAnchor {
            x: bbox.center_x(),
            y: bbox.center_y(),
        },
        None => SubjectAnchor {
            x: image_width as f32 / 2.0,
            y: image_height as f32 / 2.0,
        },
    }
}

// ---------------------------------------------------------------------------
// FocalPoint — wire adapter for place_focal_point()
// ---------------------------------------------------------------------------

/// Classification of how the focal point was determined.
///
/// Used only as metadata on [`FocalPoint`]; the crop placement algorithm
/// treats all kinds identically (it reads `.x` and `.y` only).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FocalKind {
    /// Anchor is the geometric center of the person bbox.
    BboxCenter,
    /// Fallback: anchor is the image center (no person detected).
    ImageCenter,
}

/// Wire type consumed by [`crate::multi_format_cropping::place_focal_point`].
///
/// Construct via [`compute_focal_point`] (which delegates to [`compute_subject_anchor`]).
/// The `kind` field is metadata only and is never inspected by placement logic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FocalPoint {
    pub x: f32,
    pub y: f32,
    pub kind: FocalKind,
}

/// Compute the focal point for crop placement.
///
/// This is a thin adapter over [`compute_subject_anchor`] that wraps the result
/// in the [`FocalPoint`] type expected by [`crate::multi_format_cropping::place_focal_point`].
///
/// The anchor is always the **person bbox geometric center** (or image center when no
/// person is detected). Face detections are passed through to
/// [`crate::face_aware_cropping::enforce_eye_safety`] by the caller; they do not
/// influence vertical placement here.
///
/// # Parameters
/// - `person_bbox`: Optional person bounding box (margin-expanded).
/// - `face_bboxes`: Face detections — forwarded to eye-safety guard, not used here.
/// - `image_width` / `image_height`: Photo dimensions for the fallback center.
///
/// # Example
/// ```rust,ignore
/// let focal = compute_focal_point(Some(&person_bbox), &face_bboxes, 1920, 1080);
/// let crop = place_focal_point(1920, 1080, crop_w, crop_h, &focal, 0.5, 0.5);
/// ```
pub fn compute_focal_point(
    person_bbox: Option<&BBox>,
    face_bboxes: &[BBox],
    image_width: u32,
    image_height: u32,
) -> FocalPoint {
    let anchor = compute_subject_anchor(person_bbox, face_bboxes, image_width, image_height);
    let kind = if person_bbox.is_some() {
        FocalKind::BboxCenter
    } else {
        FocalKind::ImageCenter
    };
    FocalPoint {
        x: anchor.x,
        y: anchor.y,
        kind,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multi_format_cropping::BBox;
    use serde::Deserialize;
    use std::fs;

    #[derive(Debug, Deserialize)]
    struct FocalPointFixture {
        #[allow(dead_code)]
        artboard: String,
        image_width: u32,
        image_height: u32,
        person_bbox: BBoxJson,
        face_bboxes: Vec<BBoxJson>,
        expected_focal_point: PointJson,
        #[allow(dead_code)]
        expected_format: String,
        #[allow(dead_code)]
        expected_target_y_frac: f32,
        tolerance_frac_x: f32,
        tolerance_frac_y: f32,
        #[allow(dead_code)]
        notes: String,
    }

    #[derive(Debug, Deserialize)]
    struct BBoxJson {
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
    }

    #[derive(Debug, Deserialize)]
    struct PointJson {
        x: f32,
        y: f32,
    }

    fn run_fixture(name: &str) {
        let path = format!("tests/fixtures/finetuning/{}.json", name);
        let content = fs::read_to_string(&path).expect("failed to read fixture");
        let fixture: FocalPointFixture =
            serde_json::from_str(&content).expect("failed to parse json");

        let person_bbox = BBox {
            x1: fixture.person_bbox.x1,
            y1: fixture.person_bbox.y1,
            x2: fixture.person_bbox.x2,
            y2: fixture.person_bbox.y2,
        };

        let face_bboxes: Vec<BBox> = fixture
            .face_bboxes
            .into_iter()
            .map(|f| BBox {
                x1: f.x1,
                y1: f.y1,
                x2: f.x2,
                y2: f.y2,
            })
            .collect();

        let focal = compute_focal_point(
            Some(&person_bbox),
            &face_bboxes,
            fixture.image_width,
            fixture.image_height,
        );

        let tol_x = fixture.image_width as f32 * fixture.tolerance_frac_x;
        let tol_y = fixture.image_height as f32 * fixture.tolerance_frac_y;

        assert!(
            (focal.x - fixture.expected_focal_point.x).abs() <= tol_x,
            "x {} != expected {} (tol {})",
            focal.x,
            fixture.expected_focal_point.x,
            tol_x
        );
        assert!(
            (focal.y - fixture.expected_focal_point.y).abs() <= tol_y,
            "y {} != expected {} (tol {})",
            focal.y,
            fixture.expected_focal_point.y,
            tol_y
        );
    }

    #[test]
    fn test_finetuning_01() {
        run_fixture("finetuning_01");
    }
    #[test]
    fn test_finetuning_02() {
        run_fixture("finetuning_02");
    }
    #[test]
    fn test_finetuning_03() {
        run_fixture("finetuning_03");
    }
    #[test]
    fn test_finetuning_04() {
        run_fixture("finetuning_04");
    }
    #[test]
    fn test_finetuning_05() {
        run_fixture("finetuning_05");
    }
    #[test]
    fn test_finetuning_06() {
        run_fixture("finetuning_06");
    }
    #[test]
    fn test_finetuning_07() {
        run_fixture("finetuning_07");
    }
    #[test]
    fn test_finetuning_08() {
        run_fixture("finetuning_08");
    }
    #[test]
    fn test_finetuning_09() {
        run_fixture("finetuning_09");
    }
    #[test]
    fn test_finetuning_10() {
        run_fixture("finetuning_10");
    }

    // --- Unit tests for compute_subject_anchor ---

    #[test]
    fn test_subject_anchor_uses_bbox_center() {
        let person = BBox {
            x1: 100.0,
            y1: 100.0,
            x2: 200.0,
            y2: 400.0, // center_x=150, center_y=250
        };
        let anchor = compute_subject_anchor(Some(&person), &[], 1000, 1000);
        assert_eq!(anchor.x, 150.0);
        assert_eq!(anchor.y, 250.0);
    }

    #[test]
    fn test_subject_anchor_ignores_face_bboxes() {
        // Face is at a very different position; anchor must still be bbox center.
        let person = BBox {
            x1: 100.0,
            y1: 100.0,
            x2: 200.0,
            y2: 400.0, // center_y = 250
        };
        let face = BBox {
            x1: 120.0,
            y1: 110.0, // eye-line would be ~y=124 if old logic
            x2: 180.0,
            y2: 160.0,
        };
        let anchor = compute_subject_anchor(Some(&person), &[face], 1000, 1000);
        assert_eq!(anchor.y, 250.0, "face bbox must not affect vertical anchor");
    }

    #[test]
    fn test_subject_anchor_no_person_falls_back_to_image_center() {
        let anchor = compute_subject_anchor(None, &[], 1000, 800);
        assert_eq!(anchor.x, 500.0);
        assert_eq!(anchor.y, 400.0);
    }

    #[test]
    fn test_subject_anchor_face_outside_person_is_still_bbox_center() {
        // Old logic: face outside person → BodyTopThird. New: always bbox center.
        let person = BBox {
            x1: 100.0,
            y1: 100.0,
            x2: 200.0,
            y2: 200.0, // center_y = 150
        };
        let face = BBox {
            x1: 300.0,
            y1: 300.0,
            x2: 350.0,
            y2: 350.0,
        };
        let anchor = compute_subject_anchor(Some(&person), &[face], 1000, 1000);
        assert_eq!(anchor.x, 150.0);
        assert_eq!(anchor.y, 150.0);
    }

    // --- Unit tests for compute_focal_point (wire adapter) ---

    #[test]
    fn test_focal_point_kind_bbox_center_when_person_present() {
        let person = BBox {
            x1: 100.0,
            y1: 100.0,
            x2: 200.0,
            y2: 400.0,
        };
        let f = compute_focal_point(Some(&person), &[], 1000, 1000);
        assert_eq!(f.kind, FocalKind::BboxCenter);
        assert_eq!(f.x, 150.0);
        assert_eq!(f.y, 250.0);
    }

    #[test]
    fn test_focal_point_kind_image_center_when_no_person() {
        let f = compute_focal_point(None, &[], 1000, 1000);
        assert_eq!(f.kind, FocalKind::ImageCenter);
        assert_eq!(f.x, 500.0);
        assert_eq!(f.y, 500.0);
    }

    #[test]
    fn test_focal_point_multi_face_ignored_for_vertical_anchor() {
        // Old: ≥2 faces → GroupFaces (face eye-line average). New: bbox center.
        let person = BBox {
            x1: 100.0,
            y1: 100.0,
            x2: 500.0,
            y2: 700.0, // center_y = 400
        };
        let face_a = BBox {
            x1: 150.0,
            y1: 110.0,
            x2: 250.0,
            y2: 190.0,
        };
        let face_b = BBox {
            x1: 350.0,
            y1: 110.0,
            x2: 450.0,
            y2: 190.0,
        };
        let f = compute_focal_point(Some(&person), &[face_a, face_b], 1000, 1000);
        assert_eq!(f.kind, FocalKind::BboxCenter);
        assert_eq!(f.y, 400.0, "vertical anchor must be bbox center, not face eye-line average");
    }
}
