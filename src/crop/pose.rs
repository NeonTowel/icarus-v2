//! Pose detection and crop-positioning helpers.
//!
//! Detects whether a subject is standing/sitting, which horizontal direction they
//! face, and derives the appropriate headroom ratio and crop X offset.

use super::geometry::BBox;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Detected pose type inferred from person and face geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoseType {
    Standing,
    Sitting,
    Unknown,
}

/// Target crop format category for pose-adaptive headroom selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CropFormat {
    Landscape,
    Portrait,
    Mobile,
}

/// Horizontal facing direction for rule-of-thirds portrait framing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    FacingRight,
    FacingLeft,
    Frontal,
}

// ---------------------------------------------------------------------------
// Detection functions
// ---------------------------------------------------------------------------

/// Detect whether the subject is standing, sitting, or unknown.
pub fn detect_pose_type(person_bbox: &BBox, face_bbox: Option<&BBox>) -> PoseType {
    if person_bbox.height() <= 0.0 || person_bbox.width() <= 0.0 {
        return PoseType::Unknown;
    }
    if let Some(face) = face_bbox {
        if face.height() > 0.0 {
            let rel_y = (face.center_y() - person_bbox.y1) / person_bbox.height();
            return if rel_y < 0.25 {
                PoseType::Standing
            } else if rel_y > 0.35 {
                PoseType::Sitting
            } else {
                PoseType::Unknown
            };
        }
    }
    let ar = person_bbox.height() / person_bbox.width();
    if ar > 2.5 {
        PoseType::Standing
    } else if ar < 1.5 {
        PoseType::Sitting
    } else {
        PoseType::Unknown
    }
}

/// Select a headroom ratio for the detected pose and target crop format.
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

/// Detect which horizontal direction a subject is facing.
pub fn detect_pose_direction(person_bbox: &BBox, face_bbox: &BBox) -> Direction {
    if person_bbox.width() <= 0.0 {
        return Direction::Frontal;
    }
    if face_bbox.width() > 0.0 {
        let rel_x = (face_bbox.center_x() - person_bbox.x1) / person_bbox.width();
        if rel_x < 0.35 {
            return Direction::FacingLeft;
        }
        if rel_x > 0.65 {
            return Direction::FacingRight;
        }
    }
    Direction::Frontal
}

/// Calculate the left edge of a portrait crop using rule-of-thirds placement.
pub fn calculate_portrait_crop_x(
    person_bbox: &BBox,
    crop_width: f32,
    direction: Direction,
    photo_width: f32,
) -> f32 {
    let cx = person_bbox.center_x();
    let max_x = (photo_width - crop_width).max(0.0);
    let crop_x = match direction {
        Direction::FacingRight => cx - (crop_width / 3.0),
        Direction::FacingLeft => cx - (crop_width * 2.0 / 3.0),
        Direction::Frontal => cx - (crop_width / 2.0),
    };
    crop_x.clamp(0.0, max_x)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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
        let h = calculate_headroom_for_pose(PoseType::Standing, CropFormat::Landscape);
        assert!((h - 0.48).abs() < 0.001);
    }

    #[test]
    fn test_headroom_for_pose_portrait_sitting() {
        let h = calculate_headroom_for_pose(PoseType::Sitting, CropFormat::Portrait);
        assert!((h - 0.50).abs() < 0.001);
    }

    #[test]
    fn test_headroom_for_pose_mobile_unknown() {
        let h = calculate_headroom_for_pose(PoseType::Unknown, CropFormat::Mobile);
        assert!((h - 0.45).abs() < 0.001);
    }
}
