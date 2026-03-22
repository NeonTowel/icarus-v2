/// ONNX Backend: Handles model inference with support for multiple named inputs.
///
/// This module provides the core inference engine. The key design decision is accepting
/// `HashMap<String, ndarray::ArrayD<f32>>` inputs rather than a single tensor — this is
/// the fix for the Icarus-v1 architecture flaw where DETR models (which require both
/// `pixel_values` *and* `pixel_mask`) could not be accommodated.
use anyhow::{anyhow, Context, Result};
use ndarray::ArrayD;
use ort::{session::Session, value::TensorRef};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// A multi-dimensional f32 tensor (dynamic rank). Used for both inputs and outputs.
///
/// Dynamic rank (`ArrayD`) is used intentionally: DETR outputs are 3-D while YOLO outputs
/// can be 3-D or higher. Callers are responsible for knowing which shape to expect.
pub type OrtTensor = ArrayD<f32>;

/// ONNX model backend for inference.
///
/// Wraps an `ort::Session` and exposes a named-input / named-output inference interface.
/// The session is held behind a `Mutex` because `ort::Session::run` requires `&mut self`.
///
/// # Example (unit tests use a mock; real usage requires a valid `.onnx` file)
/// ```rust,ignore
/// let backend = OnnxBackend::new(Path::new("model.onnx"))?;
/// let mut inputs = HashMap::new();
/// inputs.insert("pixel_values".to_string(), pixel_values_array);
/// inputs.insert("pixel_mask".to_string(), pixel_mask_array);
/// let outputs = backend.infer(inputs)?;
/// let logits = &outputs["logits"];
/// ```
// Custom Debug impl because ort::Session doesn't implement Debug.
pub struct OnnxBackend {
    /// The ONNX Runtime session. `Mutex` required because `Session::run` takes `&mut self`.
    /// Wrapped in Arc to allow cloning into spawn_blocking closures.
    session: Arc<std::sync::Mutex<Session>>,
    /// Cached model path string for error messages and diagnostics.
    model_path: String,
}

impl OnnxBackend {
    /// Load an ONNX model from disk and initialise the ORT session.
    ///
    /// Returns `Err` if the file does not exist or cannot be parsed by ONNX Runtime.
    /// Does **not** panic on invalid models.
    ///
    /// # Arguments
    /// * `model_path` - Absolute or relative path to a `.onnx` file.
    pub fn new(model_path: &Path) -> Result<Self> {
        let path_str = model_path.to_string_lossy().to_string();

        if !model_path.exists() {
            return Err(anyhow!("Model file not found: {}", path_str));
        }

        // Build the session. `Session::builder()` initialises the ORT environment
        // lazily on first call (thread-safe global init inside the ort crate).
        let mut session_builder = Session::builder()
            .context("Failed to create ONNX Runtime session builder")?;
        
        let session = session_builder.commit_from_file(model_path)
            .with_context(|| format!("Failed to load ONNX model from: {}", path_str))?;

        Ok(Self {
            session: Arc::new(std::sync::Mutex::new(session)),
            model_path: path_str,
        })
    }

    /// Run inference with one or more named input tensors (async version).
    ///
    /// Accepts a `HashMap` of `(input_name, tensor)` pairs so that models with
    /// multiple inputs (e.g. DETR's `pixel_values` + `pixel_mask`) are supported
    /// without any special-casing in the caller.
    ///
    /// Returns a `HashMap` of `(output_name, tensor)` pairs. All output tensors are
    /// returned as owned `ArrayD<f32>` so callers can use them after the session lock
    /// is released.
    ///
    /// # Important
    /// This method uses `tokio::task::spawn_blocking()` to run the CPU-intensive
    /// ONNX Runtime inference on a blocking thread pool, preventing the async executor
    /// from being starved. This is necessary because `Session::run()` is synchronous
    /// and can block for several seconds on large models.
    ///
    /// # Arguments
    /// * `inputs` - Named input tensors. Keys must match the model's expected input names
    ///   exactly (case-sensitive, as stored in the ONNX graph).
    ///
    /// # Errors
    /// - Returns `Err` if any input name is not recognised by the model.
    /// - Returns `Err` if an output tensor cannot be extracted as `f32`.
    /// - Returns `Err` if the session is poisoned (should never happen in practice).
    /// - Returns `Err` if the blocking task is cancelled.
    ///
    /// # Notes
    /// Shape validation (ensuring tensor dimensions match model expectations) is
    /// intentionally deferred to Phase 3 preprocessors. The ONNX Runtime will return
    /// a descriptive error if shapes are wrong.
    pub async fn infer(&self, inputs: HashMap<String, OrtTensor>) -> Result<HashMap<String, OrtTensor>> {
        // Convert input HashMap to Vec of (name, owned_array) tuples that can be moved
        // into spawn_blocking.
        let input_vec: Vec<(String, OrtTensor)> = inputs
            .into_iter()
            .collect();

        let session = Arc::clone(&self.session);
        let model_path = self.model_path.clone();

        // Run inference on a blocking thread pool to avoid starving the async executor.
        // Session::run() is CPU-intensive and synchronous, so it must not be called
        // from the main tokio thread.
        let outputs = tokio::task::spawn_blocking(move || {
            // Rebuild the input Vec for ORT's run() method
            let ort_inputs: Result<Vec<(
                std::borrow::Cow<'static, str>,
                ort::session::SessionInputValue<'_>,
            )>> = input_vec
                .iter()
                .map(|(name, array)| {
                    let ort_tensor = TensorRef::from_array_view(array.view()).with_context(|| {
                        format!("Failed to create ORT tensor view for input '{}'", name)
                    })?;
                    Ok((
                        std::borrow::Cow::Owned(name.clone()),
                        ort::session::SessionInputValue::from(ort_tensor),
                    ))
                })
                .collect();

            let ort_inputs = ort_inputs?;

            // Acquire the session lock and run inference.
            let mut session_guard = session.lock().map_err(|_| {
                anyhow!(
                    "ONNX session mutex was poisoned for model: {}",
                    model_path
                )
            })?;

            let ort_outputs = session_guard
                .run(ort_inputs)
                .with_context(|| format!("ONNX inference failed for model: {}", model_path))?;

            // Extract and clone outputs into owned data before releasing the session lock.
            // Store as (name, flat_data, shape) tuples so we can reconstruct ArrayD later.
            let mut outputs: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
            for (name, value) in &ort_outputs {
                let array = value
                    .try_extract_array::<f32>()
                    .with_context(|| {
                        format!(
                            "Failed to extract f32 array from output '{}' of model: {}",
                            name, model_path
                        )
                    })?
                    .to_owned();
                let shape: Vec<usize> = array.shape().to_vec();
                let flat_vec = array.into_raw_vec_and_offset().0;
                outputs.insert(name.to_string(), (flat_vec, shape));
            }

            Ok::<HashMap<String, (Vec<f32>, Vec<usize>)>, anyhow::Error>(outputs)
        })
        .await
        .context("ONNX inference task was cancelled")??;

        // Reconstruct ArrayD<f32> from flat vectors and stored shapes
        let mut result: HashMap<String, OrtTensor> = HashMap::new();
        for (name, (flat_vec, shape)) in outputs {
            let array = ndarray::ArrayD::<f32>::from_shape_vec(
                ndarray::IxDyn(&shape),
                flat_vec,
            ).context(format!("Failed to reconstruct output array for '{}'", name))?;
            result.insert(name, array);
        }

        Ok(result)
    }

    /// Return the file-system path of the loaded model (for diagnostics and logging).
    pub fn model_path(&self) -> &str {
        &self.model_path
    }

    /// Return the names of the model's expected inputs, in declaration order.
    ///
    /// Useful for diagnostic logging and for constructing the `inputs` HashMap correctly.
    pub fn input_names(&self) -> Result<Vec<String>> {
        let session = self.session.lock().map_err(|_| {
            anyhow!(
                "ONNX session mutex was poisoned for model: {}",
                self.model_path
            )
        })?;
        Ok(session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect())
    }

    /// Return the names of the model's outputs, in declaration order.
    pub fn output_names(&self) -> Result<Vec<String>> {
        let session = self.session.lock().map_err(|_| {
            anyhow!(
                "ONNX session mutex was poisoned for model: {}",
                self.model_path
            )
        })?;
        Ok(session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect())
    }
}

// OnnxBackend is Send + Sync because the inner Session is protected by a Mutex.
// The ort crate marks Session as Send but not Sync; the Mutex provides the Sync guarantee.
unsafe impl Send for OnnxBackend {}
unsafe impl Sync for OnnxBackend {}

impl std::fmt::Debug for OnnxBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxBackend")
            .field("model_path", &self.model_path)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_backend_rejects_missing_file() {
        let path = PathBuf::from("/nonexistent/path/model.onnx");
        let result = OnnxBackend::new(&path);
        assert!(result.is_err(), "Expected error for missing model file");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("not found"),
            "Error message should mention 'not found', got: {err_msg}"
        );
    }

    #[test]
    fn test_backend_model_path_accessor() {
        // Can't create a real session without a file, but we can test the error path
        // gives a useful message containing the path.
        let path = PathBuf::from("/tmp/fake_model_path_test.onnx");
        let result = OnnxBackend::new(&path);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("fake_model_path_test.onnx"),
            "Error should mention the model path; got: {err}"
        );
    }

    #[test]
    fn test_ort_tensor_type_alias() {
        // Verify the type alias compiles and produces the expected ndarray type.
        let tensor: OrtTensor = ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&[1, 3, 224, 224]));
        assert_eq!(tensor.shape(), &[1, 3, 224, 224]);
    }

    #[test]
    fn test_infer_requires_valid_session() {
        // Without a real ONNX model file, we can only verify that calling infer()
        // on a missing-file path yields a clear error before we even try inference.
        // Real inference integration tests live in tests/integration_tests.rs.
        let path = PathBuf::from("/nonexistent/model.onnx");
        let backend_result = OnnxBackend::new(&path);
        assert!(
            backend_result.is_err(),
            "Backend creation should fail for nonexistent file"
        );
    }
}
