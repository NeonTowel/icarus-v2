/// Model Inspector: Queries ONNX models to discover input/output requirements dynamically.
///
/// This module solves a core problem from Icarus-v1: the old code *assumed* every model
/// only needed a single `pixel_values` input. `ModelInspector` instead asks the model
/// itself what it needs, enabling correct preprocessing for all model families.
///
/// # Design
/// `ModelInspector::inspect()` opens a temporary ONNX session (not reused for inference)
/// purely to read metadata. This is intentionally separate from `OnnxBackend` to keep
/// introspection and inference concerns decoupled.
use anyhow::{anyhow, Context, Result};
use ort::{session::Session, value::ValueType};
use std::path::Path;

/// Human-readable data-type strings produced by [`tensor_element_type_to_str`].
/// These are intentionally simple strings rather than an enum so downstream code
/// can log/display them without pulling in ORT types.
fn tensor_element_type_to_str(ty: &ort::value::TensorElementType) -> &'static str {
    use ort::value::TensorElementType::*;
    match ty {
        Float32 => "float32",
        Float64 => "float64",
        Int8 => "int8",
        Int16 => "int16",
        Int32 => "int32",
        Int64 => "int64",
        Uint8 => "uint8",
        Uint16 => "uint16",
        Uint32 => "uint32",
        Uint64 => "uint64",
        Bool => "bool",
        Float16 => "float16",
        Bfloat16 => "bfloat16",
        String => "string",
        Complex64 => "complex64",
        Complex128 => "complex128",
        Float8E4M3FN => "float8e4m3fn",
        Float8E4M3FNUZ => "float8e4m3fnuz",
        Float8E5M2 => "float8e5m2",
        Float8E5M2FNUZ => "float8e5m2fnuz",
        Uint4 => "uint4",
        Int4 => "int4",
        Undefined => "undefined",
    }
}

/// Metadata describing a single tensor input or output of an ONNX model.
///
/// Shapes may contain `-1` for dynamic/unknown dimensions (common in batch and spatial dims).
#[derive(Debug, Clone, PartialEq)]
pub struct TensorMetadata {
    /// Exact name as declared in the ONNX graph (case-sensitive).
    pub name: String,

    /// Shape of the tensor.  `-1` means "dynamic / unknown at model-definition time".
    /// For example, DETR's `pixel_values` is typically `[1, 3, 800, 800]` but some
    /// exported variants use `-1` for the batch dimension.
    pub shape: Vec<i64>,

    /// Human-readable element type string (e.g. `"float32"`, `"int64"`).
    pub dtype: String,
}

/// Complete metadata for an ONNX model: its required inputs and produced outputs.
///
/// Returned by [`ModelInspector::inspect`]. Use this to drive preprocessing decisions
/// without hardcoding per-model input requirements in library code.
///
/// # Example
/// ```rust,ignore
/// let meta = ModelInspector::inspect(Path::new("detr_resnet101.onnx"))?;
/// // Prints: ["pixel_values", "pixel_mask"]
/// println!("{:?}", meta.input_names());
/// ```
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// Input tensors required by the model, in declaration order.
    pub inputs: Vec<TensorMetadata>,

    /// Output tensors produced by the model, in declaration order.
    pub outputs: Vec<TensorMetadata>,
}

impl ModelMetadata {
    /// Convenience: return only the input names in declaration order.
    pub fn input_names(&self) -> Vec<&str> {
        self.inputs.iter().map(|i| i.name.as_str()).collect()
    }

    /// Convenience: return only the output names in declaration order.
    pub fn output_names(&self) -> Vec<&str> {
        self.outputs.iter().map(|o| o.name.as_str()).collect()
    }

    /// Return `true` if the model declares an input with the given name.
    pub fn has_input(&self, name: &str) -> bool {
        self.inputs.iter().any(|i| i.name == name)
    }

    /// Return `true` if the model declares an output with the given name.
    pub fn has_output(&self, name: &str) -> bool {
        self.outputs.iter().any(|o| o.name == name)
    }
}

/// Inspects ONNX models to discover their input/output tensor requirements.
///
/// This struct is a namespace — all methods are static (associated functions).
/// There is no need to instantiate it.
pub struct ModelInspector;

impl ModelInspector {
    /// Open an ONNX model and return its full input/output metadata.
    ///
    /// This method creates a temporary `ort::Session` purely for introspection.
    /// It does **not** allocate GPU/CPU resources for inference; it only reads
    /// the graph definition embedded in the model file.
    ///
    /// # Errors
    /// - Returns `Err` if `model_path` does not exist.
    /// - Returns `Err` if the file is not a valid ONNX model.
    /// - Returns `Err` if the file is not readable (permissions, I/O errors).
    ///
    /// Does **not** panic on any of the above conditions.
    ///
    /// # Example
    /// ```rust,ignore
    /// let meta = ModelInspector::inspect(Path::new("yolov9_c.onnx"))?;
    /// assert_eq!(meta.inputs[0].name, "images");
    /// assert_eq!(meta.inputs[0].shape, vec![1, 3, 640, 640]);
    /// ```
    pub fn inspect(model_path: &Path) -> Result<ModelMetadata> {
        let path_str = model_path.to_string_lossy();

        if !model_path.exists() {
            return Err(anyhow!("Model file not found: {}", path_str));
        }

        // Open a lightweight session for introspection only. No GPU execution providers
        // are registered, so this is CPU-only and fast even for large models.
        let session = Session::builder()
            .context("Failed to create ORT session builder for inspection")?
            .commit_from_file(model_path)
            .with_context(|| format!("Failed to open ONNX model for inspection: {}", path_str))?;

        let inputs = session
            .inputs()
            .iter()
            .map(|outlet| Self::outlet_to_tensor_metadata(outlet))
            .collect::<Result<Vec<_>>>()
            .with_context(|| format!("Failed to read input metadata from: {}", path_str))?;

        let outputs = session
            .outputs()
            .iter()
            .map(|outlet| Self::outlet_to_tensor_metadata(outlet))
            .collect::<Result<Vec<_>>>()
            .with_context(|| format!("Failed to read output metadata from: {}", path_str))?;

        Ok(ModelMetadata { inputs, outputs })
    }

    /// Extract the input names from a model without loading full metadata.
    ///
    /// Convenience wrapper around [`inspect`](Self::inspect) for callers that only need names.
    pub fn get_input_names(model_path: &Path) -> Result<Vec<String>> {
        let metadata = Self::inspect(model_path)?;
        Ok(metadata.inputs.into_iter().map(|i| i.name).collect())
    }

    /// Extract the output names from a model without loading full metadata.
    ///
    /// Convenience wrapper around [`inspect`](Self::inspect) for callers that only need names.
    pub fn get_output_names(model_path: &Path) -> Result<Vec<String>> {
        let metadata = Self::inspect(model_path)?;
        Ok(metadata.outputs.into_iter().map(|o| o.name).collect())
    }

    /// Convert an `ort::Outlet` (a named typed port on an ONNX graph node) into our
    /// `TensorMetadata` struct. Only `Tensor`-typed outlets are fully supported; other
    /// value types (Sequence, Map, Optional) are represented with an empty shape and a
    /// descriptive dtype string.
    fn outlet_to_tensor_metadata(outlet: &ort::value::Outlet) -> Result<TensorMetadata> {
        let name = outlet.name().to_string();
        let (shape, dtype) = match outlet.dtype() {
            ValueType::Tensor { ty, shape, .. } => {
                let shape_vec: Vec<i64> = shape.iter().copied().collect();
                let dtype_str = tensor_element_type_to_str(ty).to_string();
                (shape_vec, dtype_str)
            }
            ValueType::Sequence(inner) => {
                // Sequences of tensors: report the inner type, empty shape.
                let inner_str = format!("sequence<{inner}>");
                (vec![], inner_str)
            }
            ValueType::Map { key, value } => {
                let map_str = format!(
                    "map<{key},{value}>",
                    key = tensor_element_type_to_str(key),
                    value = tensor_element_type_to_str(value)
                );
                (vec![], map_str)
            }
            ValueType::Optional(inner) => {
                let opt_str = format!("optional<{inner}>");
                (vec![], opt_str)
            }
        };

        Ok(TensorMetadata { name, shape, dtype })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_inspector_rejects_missing_file() {
        let path = PathBuf::from("/nonexistent/model.onnx");
        let result = ModelInspector::inspect(&path);
        assert!(result.is_err(), "Expected Err for missing model file");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("not found"),
            "Error should say 'not found'; got: {err}"
        );
    }

    #[test]
    #[ignore = "requires ORT native library at runtime; run with --include-ignored in an env with libonnxruntime installed"]
    fn test_inspector_rejects_non_onnx_file() {
        // Create a temp file with non-ONNX content and verify we get a clear error.
        // This test is ignored by default because it requires the ORT native library
        // (libonnxruntime.so) to be installed. In CI, set ORT_DYLIB_PATH or install
        // onnxruntime to enable it.
        use std::io::Write;
        let mut tmp = tempfile::NamedTempFile::new().expect("Could not create tempfile for test");
        writeln!(tmp, "this is not an onnx model").unwrap();
        let path = tmp.path().to_path_buf();

        let result = ModelInspector::inspect(&path);
        assert!(result.is_err(), "Expected Err when loading a non-ONNX file");
    }

    #[test]
    fn test_model_metadata_convenience_methods() {
        // Build synthetic metadata and verify the helpers work correctly.
        let meta = ModelMetadata {
            inputs: vec![
                TensorMetadata {
                    name: "pixel_values".to_string(),
                    shape: vec![1, 3, 800, 800],
                    dtype: "float32".to_string(),
                },
                TensorMetadata {
                    name: "pixel_mask".to_string(),
                    shape: vec![1, 64, 64],
                    dtype: "int64".to_string(),
                },
            ],
            outputs: vec![
                TensorMetadata {
                    name: "logits".to_string(),
                    shape: vec![1, 100, 92],
                    dtype: "float32".to_string(),
                },
                TensorMetadata {
                    name: "pred_boxes".to_string(),
                    shape: vec![1, 100, 4],
                    dtype: "float32".to_string(),
                },
            ],
        };

        assert_eq!(meta.input_names(), vec!["pixel_values", "pixel_mask"]);
        assert_eq!(meta.output_names(), vec!["logits", "pred_boxes"]);
        assert!(meta.has_input("pixel_values"));
        assert!(meta.has_input("pixel_mask"));
        assert!(!meta.has_input("images"));
        assert!(meta.has_output("logits"));
        assert!(!meta.has_output("nonexistent"));
    }

    #[test]
    fn test_tensor_metadata_struct() {
        let meta = TensorMetadata {
            name: "pixel_values".to_string(),
            shape: vec![1, 3, 800, 800],
            dtype: "float32".to_string(),
        };
        assert_eq!(meta.name, "pixel_values");
        assert_eq!(meta.shape, vec![1, 3, 800, 800]);
        assert_eq!(meta.dtype, "float32");
    }

    #[test]
    fn test_get_input_names_rejects_missing_file() {
        let path = PathBuf::from("/nonexistent/model.onnx");
        assert!(ModelInspector::get_input_names(&path).is_err());
    }

    #[test]
    fn test_get_output_names_rejects_missing_file() {
        let path = PathBuf::from("/nonexistent/model.onnx");
        assert!(ModelInspector::get_output_names(&path).is_err());
    }
}
