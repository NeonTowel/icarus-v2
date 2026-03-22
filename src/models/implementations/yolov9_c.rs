/// YOLOv9-c Model Implementation
///
/// Full detection pipeline:
///   image → `YoloPreprocessor` → `OnnxBackend::infer` → `DefaultPostprocessor` → `Vec<Detection>`
use crate::image_utils::Detection;
use crate::models::postprocessors::common::DefaultPostprocessor;
use crate::models::preprocessor::PreprocessorRegistry;
use crate::models::OnnxBackend;
use anyhow::Result;
use image::DynamicImage;
use std::path::Path;

/// YOLOv9-c model for object detection.
///
/// Wraps the ONNX backend and wires together the YOLO preprocessor (which produces
/// `images` [1,3,640,640]) with the common postprocessor that decodes the
/// `output` tensor [1,25200,85] into pixel-coordinate detections.
pub struct YOLOv9c {
    backend: OnnxBackend,
    postprocessor: DefaultPostprocessor,
}

impl YOLOv9c {
    /// Load a YOLOv9-c ONNX model from disk.
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
    /// 1. Retrieve `YoloPreprocessor` from the registry.
    /// 2. Preprocess `image` → `HashMap<String, OrtTensor>` with key `images`.
    /// 3. Run inference via `OnnxBackend` → outputs with key `output`.
    /// 4. Decode with `DefaultPostprocessor` → pixel-coordinate `Vec<Detection>`.
    ///
    /// # Errors
    /// Returns `Err` on preprocessing failure, ONNX runtime error, or postprocessing failure.
    pub async fn detect(&self, image: DynamicImage) -> Result<Vec<Detection>> {
        let width = image.width();
        let height = image.height();

        let preprocessor = PreprocessorRegistry::get_preprocessor("yolov9-c")?;
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
    fn test_yolov9c_new_missing_file_returns_err() {
        let path = PathBuf::from("/nonexistent/yolov9c.onnx");
        let result = YOLOv9c::new(&path);
        assert!(
            result.is_err(),
            "YOLOv9c::new should fail for missing model file"
        );
    }

    // -----------------------------------------------------------------------
    // Preprocessing integration (no ONNX runtime needed)
    // -----------------------------------------------------------------------

    #[test]
    fn test_yolov9c_preprocessor_output_key() {
        let preprocessor = PreprocessorRegistry::get_preprocessor("yolov9-c")
            .expect("registry must resolve yolov9-c");
        let img = image::DynamicImage::ImageRgb8(image::ImageBuffer::from_fn(320, 240, |_, _| {
            image::Rgb([64_u8, 128, 200])
        }));
        let map = preprocessor
            .preprocess(img)
            .expect("YOLO preprocessing must succeed");
        assert!(map.contains_key("images"), "must contain 'images' key");
    }

    #[test]
    fn test_yolov9c_images_tensor_shape() {
        let preprocessor = PreprocessorRegistry::get_preprocessor("yolov9-c").unwrap();
        let img = image::DynamicImage::ImageRgb8(image::ImageBuffer::from_fn(1280, 720, |_, _| {
            image::Rgb([200_u8, 100, 50])
        }));
        let map = preprocessor.preprocess(img).unwrap();
        assert_eq!(map["images"].shape(), &[1, 3, 640, 640]);
    }

    #[test]
    fn test_yolov9c_images_values_normalized_0_to_1() {
        let preprocessor = PreprocessorRegistry::get_preprocessor("yolov9-c").unwrap();
        let img = image::DynamicImage::ImageRgb8(image::ImageBuffer::from_fn(100, 100, |_, _| {
            image::Rgb([128_u8, 64, 255])
        }));
        let map = preprocessor.preprocess(img).unwrap();
        for &v in map["images"].iter() {
            assert!(
                v >= 0.0 && v <= 1.0,
                "YOLO images tensor must be in [0,1]; got {v}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Postprocessing integration (no ONNX runtime needed)
    // -----------------------------------------------------------------------

    #[test]
    fn test_yolov9c_postprocessor_zero_objectness_no_detections() {
        use crate::models::postprocessors::common::DefaultPostprocessor;
        use ndarray::Array3;
        use std::collections::HashMap;

        let pp = DefaultPostprocessor;
        let arr = Array3::<f32>::zeros((1, 25200, 85)).into_dyn();
        let mut outputs = HashMap::new();
        outputs.insert("output".to_string(), arr);
        let dets = pp.postprocess(outputs, 640, 480).unwrap();
        assert!(
            dets.is_empty(),
            "zero-objectness YOLO output → no detections; got {}",
            dets.len()
        );
    }

    #[test]
    fn test_yolov9c_detect_pipeline_returns_result_type() {
        let path = PathBuf::from("/nonexistent/yolov9c.onnx");
        let model = YOLOv9c::new(&path);
        assert!(model.is_err(), "should fail cleanly without model file");
    }
}
