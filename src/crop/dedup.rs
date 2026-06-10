//! Multi-person deduplication and compound bounding-box computation.
//!
//! Used when multiple persons are detected in one image to suppress reflections
//! and compute a single compound bbox for group framing.

use super::geometry::{compute_iou, BBox};

/// Remove duplicate person detections caused by reflections/mirrors using greedy NMS.
///
/// Detections are sorted by confidence (descending). Any later detection whose IoU
/// with a kept detection exceeds `iou_threshold` is suppressed.
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
            if !suppressed[j] && compute_iou(sorted[i].bbox, sorted[j].bbox) > iou_threshold {
                suppressed[j] = true;
            }
        }
    }
    kept
}

/// Calculate a single compound bbox encompassing all provided detections.
///
/// Returns `None` if `detections` is empty.
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

/// Select the dominant face for crop anchoring (largest by area).
///
/// Area-based selection is better than centrality for initial crop positioning.
pub fn select_dominant_face_for_crop(face_bboxes: &[BBox]) -> Option<&BBox> {
    face_bboxes.iter().max_by(|a, b| {
        (a.width() * a.height())
            .partial_cmp(&(b.width() * b.height()))
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image_utils::Detection;

    #[test]
    fn test_deduplicate_removes_reflection() {
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
        let result = deduplicate_person_detections(&[det_a, det_b], 0.50);
        assert_eq!(result.len(), 1);
        assert!((result[0].confidence - 0.90).abs() < 0.001);
    }

    #[test]
    fn test_deduplicate_preserves_separate_persons() {
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
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_deduplicate_empty_input() {
        let result = deduplicate_person_detections(&[] as &[Detection], 0.50);
        assert!(result.is_empty());
    }

    #[test]
    fn test_select_dominant_face_largest() {
        let face_a = BBox {
            x1: 100.0,
            y1: 100.0,
            x2: 200.0,
            y2: 200.0,
        }; // area 10000
        let face_b = BBox {
            x1: 300.0,
            y1: 300.0,
            x2: 500.0,
            y2: 500.0,
        }; // area 40000
        let faces = [face_a, face_b];
        let result = select_dominant_face_for_crop(&faces);
        assert!(result.is_some());
        assert!((result.unwrap().x1 - 300.0).abs() < 0.001);
    }

    #[test]
    fn test_select_dominant_face_empty() {
        assert!(select_dominant_face_for_crop(&[]).is_none());
    }
}
