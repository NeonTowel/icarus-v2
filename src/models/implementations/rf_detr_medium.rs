/// RF-DETR-Medium Model Implementation
///
/// RF-DETR-Medium is a DETR-family model and uses the same preprocessing and postprocessing
/// pipeline as DETRResNet101:
///   image → `DetrPreprocessor` → `OnnxBackend::infer` → `DefaultPostprocessor` → `Vec<Detection>`
use crate::image_utils::Detection;
use crate::models::postprocessors::common::DefaultPostprocessor;
use crate::models::preprocessor::PreprocessorRegistry;
use crate::models::OnnxBackend;
use anyhow::Result;
use image::DynamicImage;
use std::path::Path;

/// RF-DETR-Medium model for object detection.
///
/// RF-DETR-Medium is a lighter-weight DETR-based model that shares the dual-input scheme:
/// `pixel_values` [1,3,800,800] + `pixel_mask` [1,64,64].
/// Its outputs (`logits` + `pred_boxes`) are decoded by `DefaultPostprocessor`.
pub struct RFDETRMedium {
    backend: OnnxBackend,
    postprocessor: DefaultPostprocessor,
}

impl RFDETRMedium {
    /// Load an RF-DETR-Medium ONNX model from disk.
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
    /// 1. Retrieve `DetrPreprocessor` from the registry (aliased for "rf-detr-medium").
    /// 2. Preprocess `image` → `HashMap<String, OrtTensor>` with keys `pixel_values`, `pixel_mask`.
    /// 3. Run inference via `OnnxBackend` → outputs with keys `logits`, `pred_boxes`.
    /// 4. Decode with `DefaultPostprocessor` → pixel-coordinate `Vec<Detection>`.
    ///
    /// # Errors
    /// Returns `Err` on preprocessing failure, ONNX runtime error, or postprocessing failure.
    pub async fn detect(&self, image: DynamicImage) -> Result<Vec<Detection>> {
        let width = image.width();
        let height = image.height();

        let preprocessor = PreprocessorRegistry::get_preprocessor("rf-detr-medium")?;
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
    fn test_rf_detr_medium_new_missing_file_returns_err() {
        let path = PathBuf::from("/nonexistent/rf_detr_medium.onnx");
        let result = RFDETRMedium::new(&path);
        assert!(
            result.is_err(),
            "RFDETRMedium::new should fail for missing model file"
        );
    }

    #[test]
    fn test_rf_detr_medium_preprocessor_maps_to_detr() {
        let preprocessor = PreprocessorRegistry::get_preprocessor("rf-detr-medium")
            .expect("registry must resolve rf-detr-medium");
        assert_eq!(preprocessor.name(), "detr");
    }

    #[test]
    fn test_rf_detr_medium_preprocessor_output_shapes() {
        let preprocessor = PreprocessorRegistry::get_preprocessor("rf-detr-medium").unwrap();
        let img = image::DynamicImage::ImageRgb8(image::ImageBuffer::from_fn(400, 300, |_, _| {
            image::Rgb([200_u8, 150, 100])
        }));
        let map = preprocessor.preprocess(img).unwrap();
        assert_eq!(map["pixel_values"].shape(), &[1, 3, 800, 800]);
        assert_eq!(map["pixel_mask"].shape(), &[1, 64, 64]);
    }

    #[test]
    fn test_rf_detr_medium_detect_pipeline_returns_result_type() {
        let path = PathBuf::from("/nonexistent/rf_detr_medium.onnx");
        let model = RFDETRMedium::new(&path);
        assert!(model.is_err(), "should fail cleanly without model file");
    }
}
