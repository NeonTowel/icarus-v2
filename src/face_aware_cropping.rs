use crate::multi_format_cropping::{BBox, CropRegion};

const EYE_LINE_FRACTION: f32 = 0.45;
const FALLBACK_SHIFT_BUDGET: f32 = 0.15;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FaceAnchor {
    Top,
    Bottom,
    Left,
    Right,
}

/// Enforces the legacy vertical eye-zone guard for baseline crops.
pub fn enforce_eye_safety(
    crop: &CropRegion,
    face_bboxes: &[BBox],
    _image_width: u32,
    image_height: u32,
) -> CropRegion {
    let Some(face) = dominant_intersecting_face(crop, face_bboxes) else {
        return crop.clone();
    };
    let eye_top = face.y1 + 0.20 * face.height();
    let eye_bottom = face.y1 + 0.70 * face.height();
    let shift = if eye_top < crop.y {
        eye_top - crop.y
    } else if eye_bottom > crop.y + crop.height {
        eye_bottom - (crop.y + crop.height)
    } else {
        return crop.clone();
    };
    shift_y(crop, shift, image_height, 0.06, |adjusted| {
        eye_top >= adjusted.y && eye_bottom <= adjusted.y + adjusted.height
    })
}

/// Protect an estimated eye line using person-to-face geometry.
///
/// A face near an extreme edge of a tall or wide person box establishes a body
/// axis. The crop moves only along that axis. For a top-anchored upright body,
/// it moves upward, intentionally sacrificing lower-body coverage first.
pub fn enforce_pose_aware_eye_safety(
    crop: &CropRegion,
    person_bbox: Option<&BBox>,
    face_bboxes: &[BBox],
    image_width: u32,
    image_height: u32,
) -> CropRegion {
    let Some(face) = person_bbox
        .and_then(|person| dominant_associated_face(person, face_bboxes))
        .or_else(|| dominant_intersecting_face(crop, face_bboxes))
    else {
        return crop.clone();
    };

    match person_bbox.and_then(|person| infer_face_anchor(person, face)) {
        Some(FaceAnchor::Top) => {
            let eye_y = face.y1 + EYE_LINE_FRACTION * face.height();
            if eye_y < crop.y {
                shift_y(crop, eye_y - crop.y, image_height, 1.0, |adjusted| {
                    eye_y >= adjusted.y
                })
            } else {
                crop.clone()
            }
        }
        Some(FaceAnchor::Bottom) => {
            let eye_y = face.y2 - EYE_LINE_FRACTION * face.height();
            if eye_y > crop.y + crop.height {
                shift_y(
                    crop,
                    eye_y - (crop.y + crop.height),
                    image_height,
                    1.0,
                    |adjusted| eye_y <= adjusted.y + adjusted.height,
                )
            } else {
                crop.clone()
            }
        }
        Some(FaceAnchor::Left) => {
            let eye_x = face.x1 + EYE_LINE_FRACTION * face.width();
            if eye_x < crop.x {
                shift_x(crop, eye_x - crop.x, image_width, 1.0, |adjusted| {
                    eye_x >= adjusted.x
                })
            } else {
                crop.clone()
            }
        }
        Some(FaceAnchor::Right) => {
            let eye_x = face.x2 - EYE_LINE_FRACTION * face.width();
            if eye_x > crop.x + crop.width {
                shift_x(
                    crop,
                    eye_x - (crop.x + crop.width),
                    image_width,
                    1.0,
                    |adjusted| eye_x <= adjusted.x + adjusted.width,
                )
            } else {
                crop.clone()
            }
        }
        None => fallback_eye_guard(crop, face, image_height),
    }
}

fn fallback_eye_guard(crop: &CropRegion, face: &BBox, image_height: u32) -> CropRegion {
    let eye_y = face.y1 + EYE_LINE_FRACTION * face.height();
    if eye_y < crop.y {
        shift_y(
            crop,
            eye_y - crop.y,
            image_height,
            FALLBACK_SHIFT_BUDGET,
            |adjusted| eye_y >= adjusted.y,
        )
    } else if eye_y > crop.y + crop.height {
        shift_y(
            crop,
            eye_y - (crop.y + crop.height),
            image_height,
            FALLBACK_SHIFT_BUDGET,
            |adjusted| eye_y <= adjusted.y + adjusted.height,
        )
    } else {
        crop.clone()
    }
}

fn dominant_intersecting_face<'a>(crop: &CropRegion, faces: &'a [BBox]) -> Option<&'a BBox> {
    faces
        .iter()
        .filter(|face| intersects(crop, face))
        .max_by(|left, right| face_area(left).total_cmp(&face_area(right)))
}

fn dominant_associated_face<'a>(person: &BBox, faces: &'a [BBox]) -> Option<&'a BBox> {
    faces
        .iter()
        .filter(|face| face_belongs_to_person(person, face))
        .max_by(|left, right| face_area(left).total_cmp(&face_area(right)))
}

fn infer_face_anchor(person: &BBox, face: &BBox) -> Option<FaceAnchor> {
    let width = person.width();
    let height = person.height();
    if width <= 0.0 || height <= 0.0 || !face_belongs_to_person(person, face) {
        return None;
    }
    if height >= width * 1.15 {
        anchor_from_position(
            (face.center_y() - person.y1) / height,
            FaceAnchor::Top,
            FaceAnchor::Bottom,
        )
    } else if width >= height * 1.15 {
        anchor_from_position(
            (face.center_x() - person.x1) / width,
            FaceAnchor::Left,
            FaceAnchor::Right,
        )
    } else {
        None
    }
}

fn anchor_from_position(position: f32, near: FaceAnchor, far: FaceAnchor) -> Option<FaceAnchor> {
    if position <= 0.38 {
        Some(near)
    } else if position >= 0.62 {
        Some(far)
    } else {
        None
    }
}

fn face_belongs_to_person(person: &BBox, face: &BBox) -> bool {
    let overlap_width = (person.x2.min(face.x2) - person.x1.max(face.x1)).max(0.0);
    let overlap_height = (person.y2.min(face.y2) - person.y1.max(face.y1)).max(0.0);
    let area = face_area(face);
    area > 0.0 && overlap_width * overlap_height / area >= 0.50
}

fn shift_y(
    crop: &CropRegion,
    shift: f32,
    image_height: u32,
    budget_fraction: f32,
    is_safe: impl Fn(&CropRegion) -> bool,
) -> CropRegion {
    let y = (crop.y
        + shift.clamp(
            -budget_fraction * crop.height,
            budget_fraction * crop.height,
        ))
    .clamp(0.0, (image_height as f32 - crop.height).max(0.0));
    let adjusted = CropRegion { y, ..crop.clone() };
    if is_safe(&adjusted) {
        adjusted
    } else {
        crop.clone()
    }
}

fn shift_x(
    crop: &CropRegion,
    shift: f32,
    image_width: u32,
    budget_fraction: f32,
    is_safe: impl Fn(&CropRegion) -> bool,
) -> CropRegion {
    let x = (crop.x + shift.clamp(-budget_fraction * crop.width, budget_fraction * crop.width))
        .clamp(0.0, (image_width as f32 - crop.width).max(0.0));
    let adjusted = CropRegion { x, ..crop.clone() };
    if is_safe(&adjusted) {
        adjusted
    } else {
        crop.clone()
    }
}

fn intersects(crop: &CropRegion, face: &BBox) -> bool {
    face.x1 < crop.x + crop.width
        && face.x2 > crop.x
        && face.y1 < crop.y + crop.height
        && face.y2 > crop.y
}

fn face_area(face: &BBox) -> f32 {
    face.width().max(0.0) * face.height().max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn crop(x: f32, y: f32) -> CropRegion {
        CropRegion {
            x,
            y,
            width: 400.0,
            height: 400.0,
        }
    }

    fn bbox(x1: f32, y1: f32, x2: f32, y2: f32) -> BBox {
        BBox { x1, y1, x2, y2 }
    }

    #[test]
    fn baseline_guard_is_unchanged() {
        let adjusted = enforce_eye_safety(
            &crop(0.0, 100.0),
            &[bbox(100.0, 60.0, 200.0, 160.0)],
            1000,
            1000,
        );
        assert_eq!(adjusted.y, 80.0);
    }

    #[test]
    fn top_anchored_person_moves_crop_up_to_keep_eye_line() {
        let person = bbox(100.0, 0.0, 300.0, 800.0);
        let adjusted = enforce_pose_aware_eye_safety(
            &crop(0.0, 150.0),
            Some(&person),
            &[bbox(150.0, 40.0, 250.0, 140.0)],
            1000,
            1000,
        );
        assert_eq!(adjusted.y, 85.0);
    }

    #[test]
    fn left_anchored_horizontal_person_moves_crop_left() {
        let person = bbox(0.0, 100.0, 800.0, 300.0);
        let adjusted = enforce_pose_aware_eye_safety(
            &crop(150.0, 0.0),
            Some(&person),
            &[bbox(40.0, 150.0, 140.0, 250.0)],
            1000,
            1000,
        );
        assert_eq!(adjusted.x, 85.0);
    }

    #[test]
    fn bottom_anchored_person_moves_crop_down() {
        let person = bbox(100.0, 0.0, 300.0, 800.0);
        let adjusted = enforce_pose_aware_eye_safety(
            &crop(0.0, 0.0),
            Some(&person),
            &[bbox(150.0, 660.0, 250.0, 760.0)],
            1000,
            1000,
        );
        assert_eq!(adjusted.y, 315.0);
    }

    #[test]
    fn ambiguous_geometry_uses_bounded_vertical_fallback() {
        let person = bbox(100.0, 100.0, 500.0, 500.0);
        let adjusted = enforce_pose_aware_eye_safety(
            &crop(0.0, 200.0),
            Some(&person),
            &[bbox(200.0, 100.0, 300.0, 200.0)],
            1000,
            1000,
        );
        assert_eq!(adjusted.y, 145.0);
    }

    #[test]
    fn guard_keeps_original_crop_when_source_bound_hides_eye_line() {
        let person = bbox(100.0, 0.0, 300.0, 800.0);
        let adjusted = enforce_pose_aware_eye_safety(
            &crop(0.0, 0.0),
            Some(&person),
            &[bbox(150.0, 0.0, 250.0, 100.0)],
            1000,
            400,
        );
        assert_eq!(adjusted.y, 0.0);
    }
}
