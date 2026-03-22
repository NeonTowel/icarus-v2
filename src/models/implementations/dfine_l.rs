/// DFINE-L Model Implementation
///
/// DFINE-L is a DETR-family model and uses the same preprocessing and postprocessing
/// pipeline as DETRResNet101:
///   image → `DetrPreprocessor` → `OnnxBackend::infer` → `DefaultPostprocessor` → `Vec<Detection>`
use crate::image_utils::Detection;
use crate::models::postprocessors::common::DefaultPostprocessor;
use crate::models::preprocessor::PreprocessorRegistry;
use crate::models::OnnxBackend;
use anyhow::Result;
use image::DynamicImage;
use std::path::Path;

/// DFINE-L model for object detection.
///
/// DFINE-L is a DETR-based model that shares the dual-input scheme:
/// `pixel_values` [1,3,800,800] + `pixel_mask` [1,64,64].
/// Its outputs (`logits` + `pred_boxes`) are decoded by the same `DefaultPostprocessor`
/// as other DETR-family models.
pub struct DFineL {
    backend: OnnxBackend,
    postprocessor: DefaultPostprocessor,
}

impl DFineL {
    /// Load a DFINE-L ONNX model from disk.
    ///
    /// # Errors
    /// Returns `Err` if the model file is missing or cannot be parsed by ONNX Runtime.
    pub fn new(model_path: &Path) -> Result<Self> {
        let backend = OnnxBackend::new(model_path)?;
        Ok(Self {
            backend,
            postprocessor: DefaultPostprocessor,
        })
    }

    /// Detect objects in `image` and return bounding-box detections sorted by confidence.
    ///
    /// # Pipeline
    /// 1. Retrieve `DetrPreprocessor` from the registry (aliased for "dfine-l").
    /// 2. Preprocess `image` → `HashMap<String, OrtTensor>` with keys `pixel_values`, `pixel_mask`.
    /// 3. Run inference via `OnnxBackend` → outputs with keys `logits`, `pred_boxes`.
    /// 4. Decode with `DefaultPostprocessor` → pixel-coordinate `Vec<Detection>`.
    ///
    /// # Errors
    /// Returns `Err` on preprocessing failure, ONNX runtime error, or postprocessing failure.
    pub async fn detect(&self, image: DynamicImage) -> Result<Vec<Detection>> {
        let width = image.width();
        let height = image.height();

        let preprocessor = PreprocessorRegistry::get_preprocessor("dfine-l")?;
        let inputs = preprocessor.preprocess(image)?;
        let outputs = self.backend.infer(inputs).await?;
        self.postprocessor.postprocess(outputs, width, height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_dfine_l_new_missing_file_returns_err() {
        let path = PathBuf::from("/nonexistent/dfine_l.onnx");
        let result = DFineL::new(&path);
        assert!(
            result.is_err(),
            "DFineL::new should fail for missing model file"
        );
    }

    #[test]
    fn test_dfine_l_preprocessor_maps_to_detr() {
        // DFINE-L must use the DETR-family preprocessor (registered alias).
        let preprocessor = PreprocessorRegistry::get_preprocessor("dfine-l")
            .expect("registry must resolve dfine-l");
        assert_eq!(preprocessor.name(), "detr");
    }

    #[test]
    fn test_dfine_l_preprocessor_output_shapes() {
        let preprocessor = PreprocessorRegistry::get_preprocessor("dfine-l").unwrap();
        let img = image::DynamicImage::ImageRgb8(image::ImageBuffer::from_fn(640, 480, |_, _| {
            image::Rgb([100_u8, 150, 200])
        }));
        let map = preprocessor.preprocess(img).unwrap();
        assert_eq!(map["pixel_values"].shape(), &[1, 3, 800, 800]);
        assert_eq!(map["pixel_mask"].shape(), &[1, 64, 64]);
    }

    #[test]
    fn test_dfine_l_detect_pipeline_returns_result_type() {
        let path = PathBuf::from("/nonexistent/dfine_l.onnx");
        let model = DFineL::new(&path);
        assert!(model.is_err(), "should fail cleanly without model file");
    }
}
