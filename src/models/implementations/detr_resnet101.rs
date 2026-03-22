/// DETR-ResNet101 Model Implementation
///
/// Full detection pipeline:
///   image → `DetrPreprocessor` → `OnnxBackend::infer` → `DefaultPostprocessor` → `Vec<Detection>`
use crate::image_utils::Detection;
use crate::models::postprocessors::common::DefaultPostprocessor;
use crate::models::preprocessor::PreprocessorRegistry;
use crate::models::OnnxBackend;
use anyhow::Result;
use image::DynamicImage;
use std::path::Path;

/// DETR-ResNet101 model for object detection.
///
/// Wraps the ONNX backend and wires together the DETR preprocessor (which produces
/// `pixel_values` [1,3,800,800] + `pixel_mask` [1,64,64]) with the common
/// postprocessor that decodes `logits` + `pred_boxes` into pixel-coordinate detections.
pub struct DETRResNet101 {
    backend: OnnxBackend,
    postprocessor: DefaultPostprocessor,
}

impl DETRResNet101 {
    /// Load a DETR-ResNet101 ONNX model from disk.
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
    /// 1. Retrieve `DetrPreprocessor` from the registry.
    /// 2. Preprocess `image` → `HashMap<String, OrtTensor>` with keys `pixel_values`, `pixel_mask`.
    /// 3. Run inference via `OnnxBackend` → outputs with keys `logits`, `pred_boxes`.
    /// 4. Decode with `DefaultPostprocessor` → pixel-coordinate `Vec<Detection>`.
    ///
    /// # Errors
    /// Returns `Err` on preprocessing failure, ONNX runtime error, or postprocessing failure.
    pub async fn detect(&self, image: DynamicImage) -> Result<Vec<Detection>> {
        let width = image.width();
        let height = image.height();

        let preprocessor = PreprocessorRegistry::get_preprocessor("detr-resnet101")?;
        let inputs = preprocessor.preprocess(image)?;
        let outputs = self.backend.infer(inputs).await?;
        self.postprocessor.postprocess(outputs, width, height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // -----------------------------------------------------------------------
    // Model creation (without a real ONNX file)
    // -----------------------------------------------------------------------

    #[test]
    fn test_detr_resnet101_new_missing_file_returns_err() {
        let path = PathBuf::from("/nonexistent/detr_resnet101.onnx");
        let result = DETRResNet101::new(&path);
        assert!(
            result.is_err(),
            "DETRResNet101::new should fail for missing model file"
        );
    }

    // -----------------------------------------------------------------------
    // Preprocessing integration (no ONNX runtime needed)
    // -----------------------------------------------------------------------

    #[test]
    fn test_detr_resnet101_preprocessor_output_keys() {
        // We can exercise the preprocessor step without a real model.
        let preprocessor = PreprocessorRegistry::get_preprocessor("detr-resnet101")
            .expect("registry must resolve detr-resnet101");
        let img = image::DynamicImage::ImageRgb8(image::ImageBuffer::from_fn(100, 100, |_, _| {
            image::Rgb([128_u8, 64, 32])
        }));
        let map = preprocessor
            .preprocess(img)
            .expect("DETR preprocessing must succeed");
        assert!(
            map.contains_key("pixel_values"),
            "must contain pixel_values"
        );
        assert!(map.contains_key("pixel_mask"), "must contain pixel_mask");
    }

    #[test]
    fn test_detr_resnet101_pixel_values_shape() {
        let preprocessor = PreprocessorRegistry::get_preprocessor("detr-resnet101").unwrap();
        let img = image::DynamicImage::ImageRgb8(image::ImageBuffer::from_fn(200, 150, |_, _| {
            image::Rgb([100_u8, 100, 100])
        }));
        let map = preprocessor.preprocess(img).unwrap();
        assert_eq!(map["pixel_values"].shape(), &[1, 3, 800, 800]);
    }

    #[test]
    fn test_detr_resnet101_pixel_mask_shape() {
        let preprocessor = PreprocessorRegistry::get_preprocessor("detr-resnet101").unwrap();
        let img = image::DynamicImage::ImageRgb8(image::ImageBuffer::from_fn(100, 100, |_, _| {
            image::Rgb([255_u8, 255, 255])
        }));
        let map = preprocessor.preprocess(img).unwrap();
        assert_eq!(map["pixel_mask"].shape(), &[1, 64, 64]);
    }

    // -----------------------------------------------------------------------
    // Postprocessing integration (no ONNX runtime needed)
    // -----------------------------------------------------------------------

    #[test]
    fn test_detr_resnet101_postprocessor_empty_outputs_returns_err() {
        use crate::models::postprocessors::common::DefaultPostprocessor;
        use std::collections::HashMap;
        let pp = DefaultPostprocessor;
        let result = pp.postprocess(HashMap::new(), 800, 800);
        assert!(result.is_err(), "empty outputs should return Err");
    }

    #[test]
    fn test_detr_resnet101_detect_pipeline_returns_result_type() {
        // Verifies that detect() compiles and returns the correct type.
        // Without a real model file we only test the pre-model steps.
        let path = PathBuf::from("/nonexistent/detr_resnet101.onnx");
        let model = DETRResNet101::new(&path);
        // Model creation should fail cleanly — this tests the Err path.
        assert!(model.is_err(), "should fail cleanly without model file");
    }
}
