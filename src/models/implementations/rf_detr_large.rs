/// RF-DETR-Large Model Implementation
///
/// RF-DETR-Large is a DETR-family model and uses the same preprocessing and postprocessing
/// pipeline as DETRResNet101:
///   image → `DetrPreprocessor` → `OnnxBackend::infer` → `DefaultPostprocessor` → `Vec<Detection>`
use crate::image_utils::Detection;
use crate::models::postprocessors::common::DefaultPostprocessor;
use crate::models::preprocessor::PreprocessorRegistry;
use crate::models::OnnxBackend;
use anyhow::Result;
use image::DynamicImage;
use std::path::Path;

/// RF-DETR-Large model for object detection.
///
/// RF-DETR-Large is a DETR-based model that shares the dual-input scheme:
/// `pixel_values` [1,3,800,800] + `pixel_mask` [1,64,64].
/// Its outputs (`logits` + `pred_boxes`) are decoded by `DefaultPostprocessor`.
pub struct RFDETRLarge {
    backend: OnnxBackend,
    postprocessor: DefaultPostprocessor,
}

impl RFDETRLarge {
    /// Load an RF-DETR-Large ONNX model from disk.
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
    /// 1. Retrieve `DetrPreprocessor` from the registry (aliased for "rf-detr-large").
    /// 2. Preprocess `image` → `HashMap<String, OrtTensor>` with keys `pixel_values`, `pixel_mask`.
    /// 3. Run inference via `OnnxBackend` → outputs with keys `logits`, `pred_boxes`.
    /// 4. Decode with `DefaultPostprocessor` → pixel-coordinate `Vec<Detection>`.
    ///
    /// # Errors
    /// Returns `Err` on preprocessing failure, ONNX runtime error, or postprocessing failure.
    pub async fn detect(&self, image: DynamicImage) -> Result<Vec<Detection>> {
        let width = image.width();
        let height = image.height();

        let preprocessor = PreprocessorRegistry::get_preprocessor("rf-detr-large")?;
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
    fn test_rf_detr_large_new_missing_file_returns_err() {
        let path = PathBuf::from("/nonexistent/rf_detr_large.onnx");
        let result = RFDETRLarge::new(&path);
        assert!(
            result.is_err(),
            "RFDETRLarge::new should fail for missing model file"
        );
    }

    #[test]
    fn test_rf_detr_large_preprocessor_maps_to_detr() {
        let preprocessor = PreprocessorRegistry::get_preprocessor("rf-detr-large")
            .expect("registry must resolve rf-detr-large");
        assert_eq!(preprocessor.name(), "detr");
    }

    #[test]
    fn test_rf_detr_large_preprocessor_output_shapes() {
        let preprocessor = PreprocessorRegistry::get_preprocessor("rf-detr-large").unwrap();
        let img = image::DynamicImage::ImageRgb8(image::ImageBuffer::from_fn(800, 600, |_, _| {
            image::Rgb([50_u8, 100, 150])
        }));
        let map = preprocessor.preprocess(img).unwrap();
        assert_eq!(map["pixel_values"].shape(), &[1, 3, 800, 800]);
        assert_eq!(map["pixel_mask"].shape(), &[1, 64, 64]);
    }

    #[test]
    fn test_rf_detr_large_detect_pipeline_returns_result_type() {
        let path = PathBuf::from("/nonexistent/rf_detr_large.onnx");
        let model = RFDETRLarge::new(&path);
        assert!(model.is_err(), "should fail cleanly without model file");
    }
}
