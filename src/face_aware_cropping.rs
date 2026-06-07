use crate::multi_format_cropping::{BBox, CropRegion};

/// Enforces eye safety for the computed crop region.
///
/// If any face's eye zone (top 20% to 70% of the face bbox, covering forehead/eyes) is clipped by the crop's
/// top or bottom edge, this function attempts to nudge the crop vertically by the
/// minimal amount needed to bring the eye zone fully inside, clamped to the image
/// bounds and capped at 6% of the crop height (increased to reduce head cutoffs).
///
/// If the required shift exceeds the budget, or if the nudge causes the crop to exceed
/// image boundaries while still clipping, the original crop is returned unmodified.
pub fn enforce_eye_safety(
    crop: &CropRegion,
    face_bboxes: &[BBox],
    _image_width: u32,
    image_height: u32,
) -> CropRegion {
    if face_bboxes.is_empty() {
        return crop.clone();
    }

    // Find the dominant face inside the crop.
    // We pick the one with the largest area that intersects the crop.
    let mut dominant_face: Option<&BBox> = None;
    let mut max_area = 0.0;

    let cx1 = crop.x;
    let cx2 = crop.x + crop.width;
    let cy1 = crop.y;
    let cy2 = crop.y + crop.height;

    for face in face_bboxes {
        // Intersection check
        if face.x1 < cx2 && face.x2 > cx1 && face.y1 < cy2 && face.y2 > cy1 {
            let area = face.width() * face.height();
            if area > max_area {
                max_area = area;
                dominant_face = Some(face);
            }
        }
    }

    let Some(face) = dominant_face else {
        return crop.clone();
    };

    let eye_top = face.y1 + 0.20 * face.height();
    let eye_bottom = face.y1 + 0.70 * face.height();

    let mut shift_y = 0.0;
    if eye_top < crop.y {
        // Crop is too low, need to shift crop UP (decrease crop.y)
        shift_y = eye_top - crop.y; // this is negative
    } else if eye_bottom > crop.y + crop.height {
        // Crop is too high, need to shift crop DOWN (increase crop.y)
        shift_y = eye_bottom - (crop.y + crop.height); // this is positive
    }

    let budget = 0.06 * crop.height;
    shift_y = shift_y.clamp(-budget, budget);

    let mut new_y = crop.y + shift_y;
    let max_y = (image_height as f32 - crop.height).max(0.0);
    new_y = new_y.clamp(0.0, max_y);

    let mut new_crop = crop.clone();
    new_crop.y = new_y;

    // If still clipping after budgeted/clamped shift, return original
    if eye_top < new_crop.y || eye_bottom > new_crop.y + new_crop.height {
        return crop.clone();
    }

    new_crop
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enforce_eye_safety_no_op() {
        let crop = CropRegion {
            x: 0.0,
            y: 100.0,
            width: 400.0,
            height: 400.0,
        };
        // Face fully inside
        let face = BBox {
            x1: 100.0,
            y1: 200.0,
            x2: 200.0,
            y2: 300.0,
        };
        let new_crop = enforce_eye_safety(&crop, &[face], 1000, 1000);
        assert_eq!(new_crop.y, 100.0);
    }

    #[test]
    fn test_enforce_eye_safety_nudge_up() {
        let crop = CropRegion {
            x: 0.0,
            y: 100.0,
            width: 400.0,
            height: 400.0,
        };
        // With new 20% zone: y1=60 h=100 → eye_top=80; shift=-20 (within 6% =24 budget)
        let face = BBox {
            x1: 100.0,
            y1: 60.0,
            x2: 200.0,
            y2: 160.0,
        };
        let new_crop = enforce_eye_safety(&crop, &[face], 1000, 1000);
        assert_eq!(new_crop.y, 80.0); // shifted up by 20
    }

    #[test]
    fn test_enforce_eye_safety_cap_reached() {
        let crop = CropRegion {
            x: 0.0,
            y: 100.0,
            width: 400.0,
            height: 400.0,
        }; // budget = 24 (6%)
           // eye_top = 50 + 0.2*100 = 70 (needs 30 shift, exceeds 24 budget)
        let face = BBox {
            x1: 100.0,
            y1: 50.0,
            x2: 200.0,
            y2: 150.0,
        };
        let new_crop = enforce_eye_safety(&crop, &[face], 1000, 1000);
        // Should return original crop because cap reached and still clipping
        assert_eq!(new_crop.y, 100.0);
    }
}
