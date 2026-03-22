/// Postprocessor: Trait for converting model output tensors to structured detections.
///
/// Each model family produces outputs in a different format:
/// - DETR: `logits` [1, 100, 92] + `pred_boxes` [1, 100, 4] (normalised cx/cy/w/h)
/// - YOLO: `output` [1, 25200, 85] (xywh + obj_conf + 80 class scores)
///
/// Postprocessors are placeholders in Phase 2; Phase 3 adds the decode/NMS logic.
use super::onnx_backend::OrtTensor;
use crate::image_utils::Detection;
use anyhow::Result;
use std::collections::HashMap;

/// Contract for model-specific output decoding.
///
/// Like `Preprocessor`, postprocessors are `Send + Sync` to support multi-threaded use.
pub trait Postprocessor: Send + Sync {
    /// Decode raw model output tensors into human-readable detections.
    ///
    /// # Arguments
    /// * `outputs` - Named output tensors from `OnnxBackend::infer()`.
    /// * `image_width` - Original image width (pixels), needed to rescale normalised coordinates.
    /// * `image_height` - Original image height (pixels).
    ///
    /// # Returns
    /// A (possibly empty) list of detections ordered by descending confidence.
    fn postprocess(
        &self,
        outputs: HashMap<String, OrtTensor>,
        image_width: u32,
        image_height: u32,
    ) -> Result<Vec<Detection>>;

    /// Human-readable name for this postprocessor (used in logging).
    fn name(&self) -> &str;
}

/// Generic no-op postprocessor.
///
/// Returns an empty detection list unconditionally. This allows the pipeline to compile
/// and run end-to-end while Phase 3 fills in real decode logic.
pub struct DefaultPostprocessor;

impl Postprocessor for DefaultPostprocessor {
    fn postprocess(
        &self,
        _outputs: HashMap<String, OrtTensor>,
        _image_width: u32,
        _image_height: u32,
    ) -> Result<Vec<Detection>> {
        // TODO (Phase 3): implement per-model output decoding + NMS.
        Ok(vec![])
    }

    fn name(&self) -> &str {
        "default"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_postprocessor_placeholder() {
        let postprocessor = DefaultPostprocessor;
        let outputs: HashMap<String, OrtTensor> = HashMap::new();
        let detections = postprocessor.postprocess(outputs, 640, 480).unwrap();
        assert_eq!(detections.len(), 0);
    }

    #[test]
    fn test_default_postprocessor_name() {
        assert_eq!(DefaultPostprocessor.name(), "default");
    }
}
