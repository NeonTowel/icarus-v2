/// Error types for Icarus-v2
use thiserror::Error;

/// Result type alias using anyhow for flexibility
pub type Result<T> = anyhow::Result<T>;

/// Icarus-v2 specific errors
#[derive(Error, Debug)]
pub enum Error {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Invalid model input shape: expected {expected:?}, got {actual:?}")]
    InvalidInputShape {
        expected: Vec<i64>,
        actual: Vec<i64>,
    },

    #[error("Preprocessing failed: {0}")]
    PreprocessingError(String),

    #[error("Model inference failed: {0}")]
    InferenceError(String),

    #[error("Postprocessing failed: {0}")]
    PostprocessingError(String),

    #[error("Image processing failed: {0}")]
    ImageError(String),

    #[error("ONNX error: {0}")]
    OnnxError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),
}

impl From<image::ImageError> for Error {
    fn from(err: image::ImageError) -> Self {
        Error::ImageError(err.to_string())
    }
}
