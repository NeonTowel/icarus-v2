//! Detection JSON serialization helpers.
//!
//! Writes per-image detection + face results as structured JSON for downstream
//! tooling, debugging, or pipeline integration.

use anyhow::{Context, Result};
use std::path::Path;

use crate::batch_processor::Detection;
use crate::multi_format_cropping::BBox;

/// Serialize person detections and face bounding boxes to a JSON file.
///
/// Output schema:
/// ```json
/// {
///   "persons": [{ "person_id": 0, "label": "person", "class_id": 0, "confidence": 0.9, "bbox": [...] }],
///   "faces":   [{ "face_id": 0, "bbox": [x1, y1, x2, y2] }]
/// }
/// ```
pub fn save_detections_json(
    detections: &[Detection],
    face_bboxes: &[BBox],
    output_path: &Path,
) -> Result<()> {
    use serde_json::json;

    let person_records: Vec<serde_json::Value> = detections
        .iter()
        .enumerate()
        .map(|(index, detection)| {
            json!({
                "person_id": index,
                "label": detection.label,
                "class_id": detection.class_id,
                "confidence": detection.confidence,
                "bbox": detection.bbox,
            })
        })
        .collect();

    let face_records: Vec<serde_json::Value> = face_bboxes
        .iter()
        .enumerate()
        .map(|(index, bbox)| {
            json!({
                "face_id": index,
                "bbox": [bbox.x1, bbox.y1, bbox.x2, bbox.y2],
            })
        })
        .collect();

    let output = json!({
        "persons": person_records,
        "faces": face_records,
    });

    let serialised = serde_json::to_string_pretty(&output)
        .context("Failed to serialise detections+faces to JSON")?;

    std::fs::write(output_path, serialised)
        .with_context(|| format!("Failed to write JSON to {:?}", output_path))?;
    Ok(())
}
