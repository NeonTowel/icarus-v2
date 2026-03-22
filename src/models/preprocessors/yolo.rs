/// YOLO-specific preprocessor implementation.
///
/// This module houses the `YoloPreprocessor` that lives in the `preprocessors` sub-module.
/// In Phase 2 it is a stub; Phase 3 will add the real pixel-processing logic.
///
/// See `src/models/preprocessor.rs` for the authoritative `YoloPreprocessor` implementation
/// that is exposed through the `PreprocessorRegistry`. This file re-exports it so the
/// `preprocessors` sub-module has a coherent namespace.
pub use crate::models::preprocessor::YoloPreprocessor;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::preprocessor::Preprocessor;

    #[test]
    fn test_yolo_preprocessor_creation() {
        let preprocessor = YoloPreprocessor::new();
        // Zero-sized struct — no heap allocation expected.
        assert_eq!(std::mem::size_of_val(&preprocessor), 0);
    }

    #[test]
    fn test_yolo_preprocessor_name() {
        let p = YoloPreprocessor::new();
        assert_eq!(p.name(), "yolo");
    }
}
