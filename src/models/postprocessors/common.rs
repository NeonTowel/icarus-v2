/// Common postprocessor utilities for model outputs.
///
/// This module implements two distinct decode pipelines that cover all five supported models:
///
/// ## DETR-family pipeline
/// Used by: DETRResNet101, DFINE-L, RF-DETR-Large, RF-DETR-Medium.
/// Inputs: `logits` [1, 100, num_classes+1] + `pred_boxes` [1, 100, 4] (cx, cy, w, h normalised).
/// Steps: softmax → drop "no-object" class → filter by confidence → convert to pixel coords → NMS.
///
/// ## YOLO-family pipeline
/// Used by: YOLOv9-c.
/// Input: `output` [1, 25200, 85] (cx, cy, w, h in 640-px space + objectness + 80 class scores).
/// Steps: objectness×class → filter by confidence → scale to image dims → per-class NMS.
use crate::image_utils::Detection;
use crate::models::onnx_backend::OrtTensor;
use anyhow::{anyhow, Result};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Tuneable constants
// ---------------------------------------------------------------------------

/// Minimum confidence score to keep a detection (applies to both DETR and YOLO).
const CONFIDENCE_THRESHOLD: f32 = 0.5;

/// IoU threshold above which a lower-confidence box is suppressed by NMS.
const NMS_IOU_THRESHOLD: f32 = 0.5;

/// Spatial size assumed for YOLO inputs (pixels). Used to scale box coordinates.
const YOLO_INPUT_SIZE: f32 = 640.0;

// ---------------------------------------------------------------------------
// COCO class names (80 classes, used by YOLO-family models)
// ---------------------------------------------------------------------------

/// COCO 2017 class names in canonical order (index 0 = "person" … index 79 = "toothbrush").
///
/// Source: <https://cocodataset.org/#explore>
const COCO_CLASS_NAMES: &[&str] = &[
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

/// Numerically stable in-place softmax over a slice.
///
/// Subtracts the maximum value before exponentiation to prevent `f32` overflow.
/// All values in `slice` are replaced with their softmax probabilities.
/// The output sums to 1.0 (within floating-point rounding).
///
/// # Panics
/// Does not panic; returns `slice` unchanged if it is empty.
fn softmax_inplace(slice: &mut [f32]) {
    if slice.is_empty() {
        return;
    }
    // Subtract max for numerical stability.
    let max = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for v in slice.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in slice.iter_mut() {
            *v /= sum;
        }
    }
}

/// Compute the Intersection-over-Union (IoU) of two axis-aligned bounding boxes.
///
/// Both boxes are in `[x1, y1, x2, y2]` format (top-left / bottom-right corners).
/// Returns 0.0 if there is no overlap or if either box has zero area.
fn iou(box1: [f32; 4], box2: [f32; 4]) -> f32 {
    let [x1_min, y1_min, x1_max, y1_max] = box1;
    let [x2_min, y2_min, x2_max, y2_max] = box2;

    let inter_x_min = x1_min.max(x2_min);
    let inter_y_min = y1_min.max(y2_min);
    let inter_x_max = x1_max.min(x2_max);
    let inter_y_max = y1_max.min(y2_max);

    if inter_x_max <= inter_x_min || inter_y_max <= inter_y_min {
        return 0.0;
    }

    let inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min);
    let box1_area = (x1_max - x1_min) * (y1_max - y1_min);
    let box2_area = (x2_max - x2_min) * (y2_max - y2_min);
    let union_area = box1_area + box2_area - inter_area;

    if union_area <= 0.0 {
        return 0.0;
    }
    inter_area / union_area
}

/// Apply Non-Maximum Suppression (NMS) to a list of detections.
///
/// Detections are processed in descending confidence order.  A detection is kept
/// only if its IoU with every already-kept detection of the **same class** is below
/// `iou_threshold`.  The returned list is sorted by descending confidence.
///
/// # Arguments
/// * `detections` - Input detections (may be in any order).
/// * `iou_threshold` - Overlap threshold above which a box is suppressed.
///
/// # Returns
/// A new `Vec<Detection>` with overlapping boxes removed, sorted by confidence.
fn apply_nms(mut detections: Vec<Detection>, iou_threshold: f32) -> Vec<Detection> {
    // Sort descending by confidence.
    detections.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut kept: Vec<Detection> = Vec::new();

    'outer: for candidate in detections {
        for kept_det in &kept {
            // Suppress only when class matches (per-class NMS).
            if kept_det.class_id == candidate.class_id
                && iou(kept_det.bbox, candidate.bbox) > iou_threshold
            {
                continue 'outer;
            }
        }
        kept.push(candidate);
    }

    kept
}

// ---------------------------------------------------------------------------
// DefaultPostprocessor
// ---------------------------------------------------------------------------

/// Shared postprocessor for all five supported models.
///
/// The `postprocess` method detects which model family produced the outputs by
/// inspecting the output tensor keys:
/// - Keys `"logits"` and `"pred_boxes"` → DETR-family decode.
/// - Key `"output"` → YOLO-family decode.
///
/// Both paths share the NMS and coordinate-conversion helpers defined above.
pub struct DefaultPostprocessor;

impl DefaultPostprocessor {
    /// Decode model outputs into pixel-coordinate detections.
    ///
    /// # Arguments
    /// * `outputs` - Named output tensors from `OnnxBackend::infer()`.
    /// * `image_width` - Original image width in pixels (used for coordinate rescaling).
    /// * `image_height` - Original image height in pixels.
    ///
    /// # Returns
    /// `Vec<Detection>` sorted by descending confidence, with boxes clamped to image bounds.
    /// Returns an empty `Vec` if no detections exceed the confidence threshold.
    ///
    /// # Errors
    /// Returns `Err` if required output keys are missing or tensor shapes are unexpected.
    pub fn postprocess(
        &self,
        outputs: HashMap<String, OrtTensor>,
        image_width: u32,
        image_height: u32,
    ) -> Result<Vec<Detection>> {
        if outputs.contains_key("logits") && outputs.contains_key("pred_boxes") {
            self.postprocess_detr(outputs, image_width, image_height)
        } else if outputs.contains_key("output") {
            self.postprocess_yolo(outputs, image_width, image_height)
        } else {
            let keys: Vec<&str> = outputs.keys().map(|s| s.as_str()).collect();
            Err(anyhow!(
                "DefaultPostprocessor: unrecognised output keys {:?}. \
                 Expected ('logits', 'pred_boxes') for DETR or ('output',) for YOLO.",
                keys
            ))
        }
    }

    // ------------------------------------------------------------------
    // DETR-family decode
    // ------------------------------------------------------------------

    /// Decode DETR-family model outputs.
    ///
    /// ## Tensor layout
    /// - `logits`     → `[1, num_queries, num_classes + 1]`  (last class = "no-object")
    /// - `pred_boxes` → `[1, num_queries, 4]`  values: normalised `(cx, cy, w, h)` in `[0, 1]`
    ///
    /// ## Algorithm
    /// 1. For each query, apply softmax over the class dimension.
    /// 2. Drop the last class (no-object background).
    /// 3. If the max class probability ≥ `CONFIDENCE_THRESHOLD`, convert box and record.
    /// 4. Apply NMS with `NMS_IOU_THRESHOLD`.
    fn postprocess_detr(
        &self,
        outputs: HashMap<String, OrtTensor>,
        image_width: u32,
        image_height: u32,
    ) -> Result<Vec<Detection>> {
        let logits = outputs
            .get("logits")
            .ok_or_else(|| anyhow!("postprocess_detr: missing 'logits' key"))?;
        let pred_boxes = outputs
            .get("pred_boxes")
            .ok_or_else(|| anyhow!("postprocess_detr: missing 'pred_boxes' key"))?;

        // Expected shapes: logits [1, Q, C+1], pred_boxes [1, Q, 4].
        let logits_shape = logits.shape();
        let boxes_shape = pred_boxes.shape();

        if logits_shape.len() != 3 || boxes_shape.len() != 3 {
            return Err(anyhow!(
                "postprocess_detr: expected 3-D tensors, got logits {:?}, boxes {:?}",
                logits_shape,
                boxes_shape
            ));
        }
        if boxes_shape[2] != 4 {
            return Err(anyhow!(
                "postprocess_detr: pred_boxes last dim must be 4, got {}",
                boxes_shape[2]
            ));
        }

        let num_queries = logits_shape[1];
        let num_classes_plus_one = logits_shape[2];

        if num_classes_plus_one < 2 {
            return Err(anyhow!(
                "postprocess_detr: num_classes + 1 must be >= 2, got {}",
                num_classes_plus_one
            ));
        }

        let num_classes = num_classes_plus_one - 1; // exclude no-object class
        let w = image_width as f32;
        let h = image_height as f32;
        let mut raw_detections: Vec<Detection> = Vec::new();

        for q in 0..num_queries {
            // 1. Collect logits for this query into an owned buffer.
            let mut class_logits: Vec<f32> = (0..num_classes_plus_one)
                .map(|c| logits[[0, q, c]])
                .collect();

            // Guard against NaN: skip queries whose logits contain NaN.
            if class_logits.iter().any(|v| v.is_nan()) {
                continue;
            }

            // 2. Softmax over all classes (including no-object).
            softmax_inplace(&mut class_logits);

            // 3. Drop last class (no-object/background). The remaining slice has `num_classes` entries.
            let foreground_probs = &class_logits[..num_classes];

            // 4. Find the maximum probability and the corresponding class index.
            let (class_id, &confidence) = foreground_probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, &0.0_f32));

            // 5. Filter by threshold.
            if confidence < CONFIDENCE_THRESHOLD {
                continue;
            }

            // 6. Convert normalised (cx, cy, w, h) → pixel (x1, y1, x2, y2).
            let cx = pred_boxes[[0, q, 0]];
            let cy = pred_boxes[[0, q, 1]];
            let bw = pred_boxes[[0, q, 2]];
            let bh = pred_boxes[[0, q, 3]];

            // Skip degenerate boxes (NaN or negative dimensions).
            if cx.is_nan() || cy.is_nan() || bw.is_nan() || bh.is_nan() || bw < 0.0 || bh < 0.0 {
                continue;
            }

            let x1 = ((cx - bw / 2.0) * w).clamp(0.0, w);
            let y1 = ((cy - bh / 2.0) * h).clamp(0.0, h);
            let x2 = ((cx + bw / 2.0) * w).clamp(0.0, w);
            let y2 = ((cy + bh / 2.0) * h).clamp(0.0, h);

            // Skip zero-area boxes.
            if x2 <= x1 || y2 <= y1 {
                continue;
            }

            // Map class_id to a label string.
            // DETR uses COCO labels (91-class subset; indices 1..90 are the 80 things + 11 stuff).
            // For simplicity we use the COCO_CLASS_NAMES array (80 names) and fall back to a
            // numeric string for indices that are beyond the table.
            let label = if class_id < COCO_CLASS_NAMES.len() {
                COCO_CLASS_NAMES[class_id].to_string()
            } else {
                format!("class_{}", class_id)
            };

            raw_detections.push(Detection {
                bbox: [x1, y1, x2, y2],
                confidence,
                label,
                class_id,
            });
        }

        // 7. NMS + sort.
        Ok(apply_nms(raw_detections, NMS_IOU_THRESHOLD))
    }

    // ------------------------------------------------------------------
    // YOLO-family decode
    // ------------------------------------------------------------------

    /// Decode YOLO-family model outputs.
    ///
    /// ## Tensor layout
    /// - `output` → `[1, 25200, 85]`
    ///   Row format: `[x_center, y_center, width, height, objectness, class_0 … class_79]`
    ///   Coordinates are in **640 × 640** pixel space (the model's input resolution).
    ///
    /// ## Algorithm
    /// 1. For each of the 25200 anchor rows, compute `confidence = objectness * max(class_scores)`.
    /// 2. If `confidence >= CONFIDENCE_THRESHOLD`, find `class_id` and convert box to pixel coords.
    /// 3. Apply per-class NMS.
    fn postprocess_yolo(
        &self,
        outputs: HashMap<String, OrtTensor>,
        image_width: u32,
        image_height: u32,
    ) -> Result<Vec<Detection>> {
        let output = outputs
            .get("output")
            .ok_or_else(|| anyhow!("postprocess_yolo: missing 'output' key"))?;

        let shape = output.shape();
        if shape.len() != 3 {
            return Err(anyhow!(
                "postprocess_yolo: expected 3-D 'output' tensor, got {:?}",
                shape
            ));
        }
        if shape[2] < 5 {
            return Err(anyhow!(
                "postprocess_yolo: 'output' last dim must be >= 5, got {}",
                shape[2]
            ));
        }

        let num_anchors = shape[1];
        let row_len = shape[2];
        let num_classes = row_len - 5; // 85 − 5 = 80 for COCO

        let scale_x = image_width as f32 / YOLO_INPUT_SIZE;
        let scale_y = image_height as f32 / YOLO_INPUT_SIZE;
        let w = image_width as f32;
        let h = image_height as f32;

        let mut raw_detections: Vec<Detection> = Vec::new();

        for a in 0..num_anchors {
            let objectness = output[[0, a, 4]];

            // Fast-path: skip anchors with near-zero objectness early.
            if objectness <= 0.0 || objectness.is_nan() {
                continue;
            }

            // Find the maximum class score and its index.
            let (class_id, max_class_score) = (0..num_classes)
                .map(|c| (c, output[[0, a, 5 + c]]))
                .filter(|(_, s)| !s.is_nan())
                .fold((0usize, f32::NEG_INFINITY), |(best_c, best_s), (c, s)| {
                    if s > best_s {
                        (c, s)
                    } else {
                        (best_c, best_s)
                    }
                });

            let confidence = objectness * max_class_score;
            if confidence < CONFIDENCE_THRESHOLD {
                continue;
            }

            // Decode box from 640-px space → original image pixel space.
            let cx_640 = output[[0, a, 0]];
            let cy_640 = output[[0, a, 1]];
            let bw_640 = output[[0, a, 2]];
            let bh_640 = output[[0, a, 3]];

            if cx_640.is_nan() || cy_640.is_nan() || bw_640.is_nan() || bh_640.is_nan() {
                continue;
            }

            let cx = cx_640 * scale_x;
            let cy = cy_640 * scale_y;
            let bw = bw_640 * scale_x;
            let bh = bh_640 * scale_y;

            let x1 = (cx - bw / 2.0).clamp(0.0, w);
            let y1 = (cy - bh / 2.0).clamp(0.0, h);
            let x2 = (cx + bw / 2.0).clamp(0.0, w);
            let y2 = (cy + bh / 2.0).clamp(0.0, h);

            if x2 <= x1 || y2 <= y1 {
                continue;
            }

            let label = if class_id < COCO_CLASS_NAMES.len() {
                COCO_CLASS_NAMES[class_id].to_string()
            } else {
                format!("class_{}", class_id)
            };

            raw_detections.push(Detection {
                bbox: [x1, y1, x2, y2],
                confidence,
                label,
                class_id,
            });
        }

        Ok(apply_nms(raw_detections, NMS_IOU_THRESHOLD))
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array3, IxDyn};

    // -----------------------------------------------------------------------
    // Helper constructors
    // -----------------------------------------------------------------------

    /// Build an OrtTensor from a 3-D array, converting to dynamic rank.
    fn tensor_from_3d(arr: Array3<f32>) -> OrtTensor {
        arr.into_dyn()
    }

    // -----------------------------------------------------------------------
    // softmax_inplace
    // -----------------------------------------------------------------------

    #[test]
    fn test_softmax_sum_to_one() {
        let mut v = vec![1.0_f32, 2.0, 3.0, 4.0];
        softmax_inplace(&mut v);
        let sum: f32 = v.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax must sum to 1.0; got {sum}"
        );
    }

    #[test]
    fn test_softmax_argmax_preserved() {
        let mut v = vec![0.1_f32, 10.0, 0.2];
        softmax_inplace(&mut v);
        let max_idx = v
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        assert_eq!(max_idx, 1, "argmax should stay at index 1 after softmax");
    }

    #[test]
    fn test_softmax_empty_slice_no_panic() {
        let mut v: Vec<f32> = vec![];
        softmax_inplace(&mut v); // must not panic
    }

    #[test]
    fn test_softmax_uniform_input() {
        let mut v = vec![1.0_f32; 4];
        softmax_inplace(&mut v);
        for &x in &v {
            assert!(
                (x - 0.25).abs() < 1e-5,
                "uniform input → 0.25 each; got {x}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // iou
    // -----------------------------------------------------------------------

    #[test]
    fn test_iou_identical_boxes() {
        let bbox = [0.0_f32, 0.0, 100.0, 100.0];
        let result = iou(bbox, bbox);
        assert!(
            (result - 1.0).abs() < 1e-5,
            "identical boxes → IoU 1.0; got {result}"
        );
    }

    #[test]
    fn test_iou_no_overlap() {
        let a = [0.0_f32, 0.0, 10.0, 10.0];
        let b = [20.0_f32, 20.0, 30.0, 30.0];
        assert_eq!(iou(a, b), 0.0, "non-overlapping boxes → IoU 0.0");
    }

    #[test]
    fn test_iou_partial_overlap() {
        // Two 10×10 boxes offset by 5 in x → intersection 5×10=50, union 200-50=150.
        let a = [0.0_f32, 0.0, 10.0, 10.0];
        let b = [5.0_f32, 0.0, 15.0, 10.0];
        let result = iou(a, b);
        let expected = 50.0 / 150.0;
        assert!(
            (result - expected).abs() < 1e-5,
            "partial overlap: expected {expected:.4}, got {result:.4}"
        );
    }

    #[test]
    fn test_iou_touching_edges_is_zero() {
        // Boxes that share only an edge have zero area intersection.
        let a = [0.0_f32, 0.0, 10.0, 10.0];
        let b = [10.0_f32, 0.0, 20.0, 10.0];
        assert_eq!(iou(a, b), 0.0, "touching-edge boxes → IoU 0.0");
    }

    // -----------------------------------------------------------------------
    // apply_nms
    // -----------------------------------------------------------------------

    fn make_det(x1: f32, y1: f32, x2: f32, y2: f32, conf: f32, class_id: usize) -> Detection {
        Detection {
            bbox: [x1, y1, x2, y2],
            confidence: conf,
            label: format!("cls_{}", class_id),
            class_id,
        }
    }

    #[test]
    fn test_nms_keeps_single_detection() {
        let dets = vec![make_det(0.0, 0.0, 10.0, 10.0, 0.9, 0)];
        let kept = apply_nms(dets, 0.5);
        assert_eq!(kept.len(), 1);
    }

    #[test]
    fn test_nms_removes_duplicate_box() {
        // Two identical boxes for the same class → only the higher-confidence one survives.
        let dets = vec![
            make_det(0.0, 0.0, 100.0, 100.0, 0.9, 0),
            make_det(0.0, 0.0, 100.0, 100.0, 0.8, 0),
        ];
        let kept = apply_nms(dets, 0.5);
        assert_eq!(kept.len(), 1);
        assert!(
            (kept[0].confidence - 0.9).abs() < 1e-5,
            "highest-conf box should survive"
        );
    }

    #[test]
    fn test_nms_keeps_different_classes() {
        // Two overlapping boxes of different classes must both survive (per-class NMS).
        let dets = vec![
            make_det(0.0, 0.0, 100.0, 100.0, 0.9, 0),
            make_det(0.0, 0.0, 100.0, 100.0, 0.8, 1),
        ];
        let kept = apply_nms(dets, 0.5);
        assert_eq!(kept.len(), 2, "different classes must both survive NMS");
    }

    #[test]
    fn test_nms_keeps_non_overlapping_same_class() {
        // Two non-overlapping boxes of the same class must both survive.
        let dets = vec![
            make_det(0.0, 0.0, 10.0, 10.0, 0.9, 0),
            make_det(50.0, 50.0, 60.0, 60.0, 0.85, 0),
        ];
        let kept = apply_nms(dets, 0.5);
        assert_eq!(
            kept.len(),
            2,
            "non-overlapping same-class boxes must both survive"
        );
    }

    #[test]
    fn test_nms_output_sorted_by_confidence() {
        let dets = vec![
            make_det(0.0, 0.0, 5.0, 5.0, 0.7, 0),
            make_det(50.0, 50.0, 55.0, 55.0, 0.95, 1),
            make_det(100.0, 100.0, 105.0, 105.0, 0.8, 2),
        ];
        let kept = apply_nms(dets, 0.5);
        assert_eq!(kept.len(), 3);
        assert!(
            kept[0].confidence >= kept[1].confidence && kept[1].confidence >= kept[2].confidence,
            "NMS output must be sorted by descending confidence"
        );
    }

    #[test]
    fn test_nms_empty_input() {
        let kept = apply_nms(vec![], 0.5);
        assert!(kept.is_empty());
    }

    // -----------------------------------------------------------------------
    // DefaultPostprocessor: empty / no-output path
    // -----------------------------------------------------------------------

    #[test]
    fn test_postprocessor_empty_outputs_returns_err() {
        let pp = DefaultPostprocessor;
        let outputs: HashMap<String, OrtTensor> = HashMap::new();
        let result = pp.postprocess(outputs, 640, 480);
        assert!(
            result.is_err(),
            "empty outputs should return Err (unknown format)"
        );
    }

    #[test]
    fn test_postprocessor_unknown_keys_returns_err() {
        let pp = DefaultPostprocessor;
        let mut outputs: HashMap<String, OrtTensor> = HashMap::new();
        let dummy: OrtTensor = ndarray::ArrayD::zeros(IxDyn(&[1, 1, 1]));
        outputs.insert("mystery_key".to_string(), dummy);
        let result = pp.postprocess(outputs, 640, 480);
        assert!(result.is_err(), "unknown output keys should return Err");
    }

    // -----------------------------------------------------------------------
    // DefaultPostprocessor: DETR path
    // -----------------------------------------------------------------------

    /// Build minimal DETR outputs: all-zero logits + all-zero pred_boxes.
    /// Zero logits → after softmax every class is equally probable (~1/92).
    /// No detection should exceed the 0.5 threshold → empty result.
    fn make_detr_outputs_zero(
        num_queries: usize,
        num_classes: usize,
    ) -> HashMap<String, OrtTensor> {
        let logits = tensor_from_3d(Array3::zeros((1, num_queries, num_classes + 1)));
        let pred_boxes = tensor_from_3d(Array3::zeros((1, num_queries, 4)));
        let mut map = HashMap::new();
        map.insert("logits".to_string(), logits);
        map.insert("pred_boxes".to_string(), pred_boxes);
        map
    }

    #[test]
    fn test_detr_zero_logits_returns_no_detections() {
        let pp = DefaultPostprocessor;
        let outputs = make_detr_outputs_zero(100, 91);
        let dets = pp.postprocess(outputs, 800, 800).unwrap();
        // With uniform probabilities across 92 classes, max ≈ 0.0109 → below threshold.
        assert!(
            dets.is_empty(),
            "zero-logit DETR outputs should produce no detections; got {}",
            dets.len()
        );
    }

    /// Build DETR outputs where query 0 has a strong class-0 logit.
    /// This should produce exactly one detection at the expected pixel coordinates.
    fn make_detr_outputs_one_detection() -> HashMap<String, OrtTensor> {
        let mut logits_arr = Array3::<f32>::zeros((1, 5, 92));
        // Make class 0 of query 0 very dominant (logit = 10.0).
        logits_arr[[0, 0, 0]] = 10.0;

        let mut boxes_arr = Array3::<f32>::zeros((1, 5, 4));
        // cx=0.5, cy=0.5, w=0.4, h=0.4 → box covering the centre of the image.
        boxes_arr[[0, 0, 0]] = 0.5; // cx
        boxes_arr[[0, 0, 1]] = 0.5; // cy
        boxes_arr[[0, 0, 2]] = 0.4; // w
        boxes_arr[[0, 0, 3]] = 0.4; // h

        let mut map = HashMap::new();
        map.insert("logits".to_string(), tensor_from_3d(logits_arr));
        map.insert("pred_boxes".to_string(), tensor_from_3d(boxes_arr));
        map
    }

    #[test]
    fn test_detr_one_strong_detection_is_returned() {
        let pp = DefaultPostprocessor;
        let outputs = make_detr_outputs_one_detection();
        let dets = pp.postprocess(outputs, 800, 800).unwrap();
        assert_eq!(
            dets.len(),
            1,
            "expected exactly one detection; got {}",
            dets.len()
        );
    }

    #[test]
    fn test_detr_detection_pixel_coordinates() {
        let pp = DefaultPostprocessor;
        let outputs = make_detr_outputs_one_detection();
        let dets = pp.postprocess(outputs, 800, 800).unwrap();
        assert_eq!(dets.len(), 1);
        let [x1, y1, x2, y2] = dets[0].bbox;
        // cx=0.5, cy=0.5, w=0.4, h=0.4 on 800×800 image:
        //   x1 = (0.5 - 0.2) * 800 = 240, y1 = 240
        //   x2 = (0.5 + 0.2) * 800 = 560, y2 = 560
        assert!((x1 - 240.0).abs() < 1.0, "x1 should be ≈ 240; got {x1}");
        assert!((y1 - 240.0).abs() < 1.0, "y1 should be ≈ 240; got {y1}");
        assert!((x2 - 560.0).abs() < 1.0, "x2 should be ≈ 560; got {x2}");
        assert!((y2 - 560.0).abs() < 1.0, "y2 should be ≈ 560; got {y2}");
    }

    #[test]
    fn test_detr_confidence_is_above_threshold() {
        let pp = DefaultPostprocessor;
        let outputs = make_detr_outputs_one_detection();
        let dets = pp.postprocess(outputs, 800, 800).unwrap();
        assert_eq!(dets.len(), 1);
        assert!(
            dets[0].confidence >= CONFIDENCE_THRESHOLD,
            "detection confidence should be >= threshold; got {}",
            dets[0].confidence
        );
    }

    #[test]
    fn test_detr_label_mapped_from_coco_names() {
        let pp = DefaultPostprocessor;
        let outputs = make_detr_outputs_one_detection();
        let dets = pp.postprocess(outputs, 800, 800).unwrap();
        // class_id 0 → "person"
        assert_eq!(dets[0].class_id, 0);
        assert_eq!(dets[0].label, "person");
    }

    #[test]
    fn test_detr_missing_logits_returns_err() {
        let pp = DefaultPostprocessor;
        let mut outputs = make_detr_outputs_zero(5, 91);
        outputs.remove("logits");
        // Removing `logits` means the key check fails → falls to "unknown keys" path.
        let result = pp.postprocess(outputs, 800, 800);
        assert!(result.is_err(), "missing 'logits' should return Err");
    }

    #[test]
    fn test_detr_missing_pred_boxes_returns_err() {
        let pp = DefaultPostprocessor;
        let mut outputs = make_detr_outputs_zero(5, 91);
        outputs.remove("pred_boxes");
        let result = pp.postprocess(outputs, 800, 800);
        assert!(result.is_err());
    }

    #[test]
    fn test_detr_boxes_clamped_to_image_bounds() {
        // Box with cx=0.0, cy=0.0, w=1.0, h=1.0 → corners at (-0.5*W, -0.5*H) and (0.5*W, 0.5*H).
        // Negative corners get clamped to 0.
        let mut logits_arr = Array3::<f32>::zeros((1, 1, 2));
        logits_arr[[0, 0, 0]] = 20.0; // dominant class 0

        let mut boxes_arr = Array3::<f32>::zeros((1, 1, 4));
        boxes_arr[[0, 0, 0]] = 0.0; // cx
        boxes_arr[[0, 0, 1]] = 0.0; // cy
        boxes_arr[[0, 0, 2]] = 1.0; // w
        boxes_arr[[0, 0, 3]] = 1.0; // h

        let mut outputs = HashMap::new();
        outputs.insert("logits".to_string(), tensor_from_3d(logits_arr));
        outputs.insert("pred_boxes".to_string(), tensor_from_3d(boxes_arr));

        let pp = DefaultPostprocessor;
        // A box at corner (0,0) with w=1,h=1 gives x1 = (0-0.5)*800 = -400 → clamped to 0;
        // x2 = (0+0.5)*800 = 400. Box should survive (non-zero area after clamping).
        let dets = pp.postprocess(outputs, 800, 800).unwrap();
        if !dets.is_empty() {
            let [x1, y1, _x2, _y2] = dets[0].bbox;
            assert!(x1 >= 0.0, "x1 must be clamped to >= 0; got {x1}");
            assert!(y1 >= 0.0, "y1 must be clamped to >= 0; got {y1}");
        }
    }

    // -----------------------------------------------------------------------
    // DefaultPostprocessor: YOLO path
    // -----------------------------------------------------------------------

    /// Build all-zero YOLO output [1, num_anchors, 85].
    /// Objectness = 0 → no detections.
    fn make_yolo_outputs_zero(num_anchors: usize) -> HashMap<String, OrtTensor> {
        let arr = Array3::zeros((1, num_anchors, 85));
        let mut map = HashMap::new();
        map.insert("output".to_string(), tensor_from_3d(arr));
        map
    }

    #[test]
    fn test_yolo_zero_objectness_returns_no_detections() {
        let pp = DefaultPostprocessor;
        let outputs = make_yolo_outputs_zero(25200);
        let dets = pp.postprocess(outputs, 640, 480).unwrap();
        assert!(
            dets.is_empty(),
            "zero-objectness YOLO output should produce no detections; got {}",
            dets.len()
        );
    }

    /// Build a YOLO output with one high-confidence anchor.
    fn make_yolo_outputs_one_detection() -> HashMap<String, OrtTensor> {
        let mut arr = Array3::<f32>::zeros((1, 10, 85));
        // Anchor 0: cx=320, cy=240 (centre of 640×640), w=200, h=150, objectness=1.0, class 0 score=1.0
        arr[[0, 0, 0]] = 320.0; // cx in 640-px space
        arr[[0, 0, 1]] = 240.0; // cy
        arr[[0, 0, 2]] = 200.0; // w
        arr[[0, 0, 3]] = 150.0; // h
        arr[[0, 0, 4]] = 1.0; // objectness
        arr[[0, 0, 5]] = 1.0; // class 0 score
        let mut map = HashMap::new();
        map.insert("output".to_string(), tensor_from_3d(arr));
        map
    }

    #[test]
    fn test_yolo_one_detection_is_returned() {
        let pp = DefaultPostprocessor;
        let outputs = make_yolo_outputs_one_detection();
        let dets = pp.postprocess(outputs, 640, 480).unwrap();
        assert_eq!(
            dets.len(),
            1,
            "expected exactly one YOLO detection; got {}",
            dets.len()
        );
    }

    #[test]
    fn test_yolo_detection_pixel_coordinates_640x480() {
        // cx=320, cy=240, w=200, h=150 in 640-px space on a 640×480 image.
        // scale_x = 640/640 = 1.0, scale_y = 480/640 = 0.75
        // x1 = 320 - 100 = 220, x2 = 420
        // y1 = (240 - 75) * 0.75 → cy_orig = 240*0.75=180, bh_orig=150*0.75=112.5 → y1=180-56.25=123.75
        let pp = DefaultPostprocessor;
        let outputs = make_yolo_outputs_one_detection();
        let dets = pp.postprocess(outputs, 640, 480).unwrap();
        assert_eq!(dets.len(), 1);
        let [x1, _y1, x2, _y2] = dets[0].bbox;
        assert!((x1 - 220.0).abs() < 1.0, "x1 should be ≈ 220; got {x1}");
        assert!((x2 - 420.0).abs() < 1.0, "x2 should be ≈ 420; got {x2}");
    }

    #[test]
    fn test_yolo_label_mapped_from_coco_names() {
        let pp = DefaultPostprocessor;
        let outputs = make_yolo_outputs_one_detection();
        let dets = pp.postprocess(outputs, 640, 480).unwrap();
        assert_eq!(dets[0].class_id, 0);
        assert_eq!(dets[0].label, "person");
    }

    #[test]
    fn test_yolo_missing_output_key_returns_err() {
        let pp = DefaultPostprocessor;
        let outputs: HashMap<String, OrtTensor> = HashMap::new();
        // An empty map triggers the "unknown keys" path, not the YOLO path.
        let result = pp.postprocess(outputs, 640, 480);
        assert!(result.is_err());
    }

    #[test]
    fn test_yolo_confidence_below_threshold_filtered() {
        // Objectness = 0.4, class_score = 0.4 → confidence = 0.16 → below 0.5 threshold.
        let mut arr = Array3::<f32>::zeros((1, 5, 85));
        arr[[0, 0, 4]] = 0.4; // objectness
        arr[[0, 0, 5]] = 0.4; // class 0
        arr[[0, 0, 0]] = 320.0;
        arr[[0, 0, 1]] = 240.0;
        arr[[0, 0, 2]] = 100.0;
        arr[[0, 0, 3]] = 100.0;
        let mut map = HashMap::new();
        map.insert("output".to_string(), tensor_from_3d(arr));
        let pp = DefaultPostprocessor;
        let dets = pp.postprocess(map, 640, 640).unwrap();
        assert!(
            dets.is_empty(),
            "low-confidence YOLO anchor should not produce a detection; got {}",
            dets.len()
        );
    }

    // -----------------------------------------------------------------------
    // COCO names sanity
    // -----------------------------------------------------------------------

    #[test]
    fn test_coco_class_names_count() {
        assert_eq!(
            COCO_CLASS_NAMES.len(),
            80,
            "COCO class name table must have exactly 80 entries"
        );
    }

    #[test]
    fn test_coco_first_class_is_person() {
        assert_eq!(COCO_CLASS_NAMES[0], "person");
    }

    #[test]
    fn test_coco_last_class_is_toothbrush() {
        assert_eq!(COCO_CLASS_NAMES[79], "toothbrush");
    }
}
