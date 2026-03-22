/// Preprocessor: Trait and registry for model-specific image preprocessing.
///
/// # Architecture
/// The `Preprocessor` trait is the contract between raw image data and the typed tensors
/// that `OnnxBackend::infer()` expects. Each model family (DETR, YOLO, etc.) implements
/// this trait differently because their tensor names, shapes, and normalization schemes vary.
///
/// This is the architectural fix for Icarus-v1's single-input assumption: instead of
/// one generic pipeline that always produces `pixel_values`, each preprocessor returns
/// a `HashMap<String, OrtTensor>` with exactly the named inputs the model requires.
///
/// # Extension points
/// - New model families can be added by implementing `Preprocessor` and registering
///   the name in `PreprocessorRegistry::get_preprocessor`.
use super::onnx_backend::OrtTensor;
use crate::image_utils;
use anyhow::{anyhow, Result};
use image::DynamicImage;
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// DETR normalization constants (ImageNet statistics)
// ---------------------------------------------------------------------------

/// ImageNet channel means (R, G, B), in the same order as pixel layout.
/// Applied after scaling pixels to [0.0, 1.0] by dividing by 255.0.
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];

/// ImageNet channel standard deviations (R, G, B).
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Target spatial resolution for DETR-family models.
const DETR_SIZE: u32 = 800;

/// Downsampled pixel-mask spatial resolution for DETR-family models.
///
/// DETR's ResNet backbone applies three pooling/stride operations, each halving
/// spatial dimensions, from a base of 512×512 feature maps down to 64×64 (factor 8).
/// See: DETR paper (Carion et al. 2020), Figure 2. The authoritative value is
/// confirmed by the ONNX model's declared input shape for `pixel_mask`.
const DETR_MASK_SIZE: usize = 64;

/// Target spatial resolution for YOLO-family models (v5, v8, v9).
const YOLO_SIZE: u32 = 640;

/// Contract for model-specific image preprocessing.
///
/// Implementors receive a raw `DynamicImage` and return a map of named tensors ready
/// for inference.  The keys in the returned map **must** match the model's declared
/// input names (as returned by `ModelInspector::inspect`).
///
/// # Thread safety
/// Implementors are required to be `Send + Sync` so that preprocessor instances can
/// be shared across threads (e.g. a web-server serving multiple requests concurrently).
/// In practice this is easily satisfied by keeping preprocessors stateless (zero-sized
/// structs or structs containing only immutable configuration data).
pub trait Preprocessor: Send + Sync {
    /// Preprocess `image` and return the named input tensors for this model.
    ///
    /// # Arguments
    /// * `image` - The raw decoded image, any format accepted by the `image` crate.
    ///   Implementors are responsible for colour conversion, resizing, and normalisation.
    ///
    /// # Returns
    /// `HashMap<input_name, tensor>` where:
    /// - Keys are exact model input names (e.g. `"pixel_values"`, `"pixel_mask"`, `"images"`).
    /// - Tensors are 4-D `f32` arrays with shape matching the model's expectation.
    ///
    /// # Errors
    /// Returns `Err` if:
    /// - The image cannot be decoded or resized.
    /// - A required tensor cannot be allocated.
    fn preprocess(&self, image: DynamicImage) -> Result<HashMap<String, OrtTensor>>;

    /// A stable, human-readable name for this preprocessor (used in logging/diagnostics).
    ///
    /// Should be lowercase, e.g. `"detr"` or `"yolo"`.
    fn name(&self) -> &str;
}

/// Registry that maps model name strings to their concrete `Preprocessor` implementations.
///
/// Model names follow the naming convention established in Phase 1:
/// all lowercase, hyphen-separated (e.g. `"detr-resnet101"`, `"yolov9-c"`).
///
/// # Phase 3 note
/// The returned preprocessors are currently **placeholder stubs** — calling `preprocess()`
/// on them will return `Err`. Phase 3 will replace the stub bodies with real logic.
pub struct PreprocessorRegistry;

impl PreprocessorRegistry {
    /// Return a boxed `Preprocessor` for the given model name.
    ///
    /// Model name matching is **case-sensitive** to stay consistent with the rest of
    /// the codebase (see Phase 1 decision log).
    ///
    /// # Supported models (Phase 2 — stubs; Phase 3 will add real logic)
    /// | Model name          | Preprocessor         | Inputs                                   |
    /// |---------------------|----------------------|------------------------------------------|
    /// | `detr-resnet101`    | `DetrPreprocessor`   | `pixel_values`, `pixel_mask`             |
    /// | `yolov9-c`          | `YoloPreprocessor`   | `images`                                 |
    /// | `dfine-l`           | `DetrPreprocessor`   | `pixel_values`, `pixel_mask`             |
    /// | `rf-detr-large`     | `DetrPreprocessor`   | `pixel_values`, `pixel_mask`             |
    /// | `rf-detr-medium`    | `DetrPreprocessor`   | `pixel_values`, `pixel_mask`             |
    ///
    /// # Errors
    /// Returns `Err` for unknown model names so callers get a clear message rather than
    /// a silent fallback to a wrong preprocessor.
    pub fn get_preprocessor(model_name: &str) -> Result<Box<dyn Preprocessor>> {
        match model_name {
            "detr-resnet101" => Ok(Box::new(DetrPreprocessor::new())),
            "yolov9-c" => Ok(Box::new(YoloPreprocessor::new())),
            // DFINE-L uses the same DETR-style dual-input scheme.
            "dfine-l" => Ok(Box::new(DetrPreprocessor::new())),
            "rf-detr-large" => Ok(Box::new(DetrPreprocessor::new())),
            "rf-detr-medium" => Ok(Box::new(DetrPreprocessor::new())),
            unknown => Err(anyhow!(
                "No preprocessor registered for model '{}'. \
                 Known models: detr-resnet101, yolov9-c, dfine-l, rf-detr-large, rf-detr-medium",
                unknown
            )),
        }
    }

    /// Return the list of all model names that have a registered preprocessor.
    ///
    /// Useful for CLI help text and validation.
    pub fn registered_model_names() -> &'static [&'static str] {
        &[
            "detr-resnet101",
            "yolov9-c",
            "dfine-l",
            "rf-detr-large",
            "rf-detr-medium",
        ]
    }
}

// ---------------------------------------------------------------------------
// Internal helper: RGB interleaved → channel-first (CHW) conversion
// ---------------------------------------------------------------------------

/// Convert a flat, row-major interleaved-RGB pixel buffer into a channel-first
/// tensor of shape `[channels, height, width]`.
///
/// # Arguments
/// * `pixels`  - Flat buffer `[R0, G0, B0, R1, G1, B1, …]` from `image_utils::extract_pixels`.
///               Length must equal `height * width * 3`.
/// * `height`  - Image height in pixels.
/// * `width`   - Image width in pixels.
///
/// # Returns
/// `ArrayD<f32>` with shape `[3, height, width]` in standard C-contiguous layout
/// (all R channel data, then G, then B; within each channel, row-major).
///
/// # Errors
/// Returns `Err` if the `pixels` buffer length is inconsistent with the declared dimensions.
fn rgb_hwc_to_chw(pixels: Vec<f32>, height: usize, width: usize) -> Result<ArrayD<f32>> {
    let expected_len = height * width * 3;
    if pixels.len() != expected_len {
        return Err(anyhow!(
            "rgb_hwc_to_chw: pixel buffer length {} does not match expected {} \
             ({}×{}×3)",
            pixels.len(),
            expected_len,
            height,
            width
        ));
    }

    // Separate interleaved RGB into three independent channel planes.
    // Layout of `pixels`: [R₀₀, G₀₀, B₀₀, R₀₁, G₀₁, B₀₁, … Rₕw, Gₕw, Bₕw]
    // We want three planes each of size height×width.
    let n_pixels = height * width;
    let mut r_plane = Vec::with_capacity(n_pixels);
    let mut g_plane = Vec::with_capacity(n_pixels);
    let mut b_plane = Vec::with_capacity(n_pixels);

    for chunk in pixels.chunks_exact(3) {
        r_plane.push(chunk[0]);
        g_plane.push(chunk[1]);
        b_plane.push(chunk[2]);
    }

    // Concatenate into a single CHW buffer: [R…, G…, B…]
    let mut chw_data = Vec::with_capacity(3 * n_pixels);
    chw_data.extend_from_slice(&r_plane);
    chw_data.extend_from_slice(&g_plane);
    chw_data.extend_from_slice(&b_plane);

    // Wrap in ArrayD with shape [3, height, width].
    ArrayD::from_shape_vec(IxDyn(&[3, height, width]), chw_data).map_err(|e| {
        anyhow!(
            "rgb_hwc_to_chw: failed to construct ArrayD from CHW buffer: {}",
            e
        )
    })
}

// ---------------------------------------------------------------------------
// DetrPreprocessor
// ---------------------------------------------------------------------------

/// DETR-family preprocessor.
///
/// Handles models that require **two** named inputs:
///
/// | Input name      | Shape            | Type | Normalization                         |
/// |-----------------|------------------|------|---------------------------------------|
/// | `pixel_values`  | `[1, 3, 800, 800]` | f32  | ImageNet mean/std per channel         |
/// | `pixel_mask`    | `[1, 64, 64]`    | f32  | All-ones (every pixel is valid)       |
///
/// ## Critical difference from Icarus-v1
/// The `pixel_mask` shape is `[1, 64, 64]`, **not** `[1, 800, 800]`. The 64×64
/// resolution comes from DETR's ResNet encoder downsampling the 800×800 input by a
/// factor of 8 × (stride-2 layers) → effective stride 32, giving 800/32 = 25 feature
/// grid cells per axis in the backbone, and then a separate pooling path used for the
/// attention mask yields 64×64. The authoritative value is the ONNX model's declared
/// input shape; the 64-cell resolution is confirmed in the Icarus-v2 spec.
///
/// ## Processing steps
/// 1. Convert image to RGB (drops alpha channel, handles greyscale).
/// 2. Resize to 800×800 using Lanczos3 filter.
/// 3. Extract flat interleaved-RGB pixel buffer (values in `[0, 255]`).
/// 4. Scale to `[0.0, 1.0]` by dividing by 255.0.
/// 5. Apply per-channel ImageNet normalization: `(pixel − mean) / std`.
/// 6. Rearrange from HWC to CHW layout → shape `[3, 800, 800]`.
/// 7. Prepend batch dimension → shape `[1, 3, 800, 800]`.
/// 8. Create `pixel_mask` as all-ones with shape `[1, 64, 64]`.
pub struct DetrPreprocessor;

impl DetrPreprocessor {
    /// Create a new DETR preprocessor.
    ///
    /// The struct is zero-sized; all configuration (ImageNet constants, target
    /// resolution) is stored as module-level constants to avoid heap allocations.
    pub fn new() -> Self {
        Self
    }
}

impl Default for DetrPreprocessor {
    fn default() -> Self {
        Self::new()
    }
}

impl Preprocessor for DetrPreprocessor {
    /// Preprocess `image` for DETR-family models.
    ///
    /// # Returns
    /// `HashMap` with keys:
    /// - `"pixel_values"` → `ArrayD<f32>` shape `[1, 3, 800, 800]`
    /// - `"pixel_mask"`   → `ArrayD<f32>` shape `[1, 64, 64]`
    ///
    /// # Errors
    /// Returns `Err` if resizing or tensor construction fails. Does not panic.
    fn preprocess(&self, image: DynamicImage) -> Result<HashMap<String, OrtTensor>> {
        let size = DETR_SIZE as usize;

        // Step 1 & 2: Convert to RGB, then resize to 800×800.
        // `image_utils::resize_image` uses Lanczos3 and handles aspect-ratio stretching.
        let rgb_image = image_utils::to_rgb(&image);
        let resized = image_utils::resize_image(&rgb_image, DETR_SIZE, DETR_SIZE);

        // Step 3: Extract flat interleaved-RGB buffer [R₀₀,G₀₀,B₀₀, R₀₁,G₀₁,B₀₁, …].
        // Values are in the raw [0.0, 255.0] range at this point.
        let mut pixels = image_utils::extract_pixels(&resized);

        // Step 4 & 5: Scale to [0, 1] then apply ImageNet (mean, std) normalization.
        // `normalize_pixels` applies `(pixel[c] - mean[c]) / std[c]` in-place.
        // But it expects values already in [0, 1], so we divide first.
        for p in pixels.iter_mut() {
            *p /= 255.0;
        }
        image_utils::normalize_pixels(&mut pixels, &IMAGENET_MEAN, &IMAGENET_STD);

        // Step 6: HWC [height, width, 3] → CHW [3, height, width].
        let chw = rgb_hwc_to_chw(pixels, size, size)?;

        // Step 7: Insert batch dimension → [1, 3, 800, 800].
        // `insert_axis` adds a length-1 dimension at position 0.
        let pixel_values = chw
            .insert_axis(ndarray::Axis(0))
            .into_shape_with_order(IxDyn(&[1, 3, size, size]))
            .map_err(|e| {
                anyhow!(
                    "DetrPreprocessor: failed to reshape to [1,3,800,800]: {}",
                    e
                )
            })?;

        // Step 8: Create pixel_mask all-ones [1, 64, 64].
        // Shape [1, DETR_MASK_SIZE, DETR_MASK_SIZE]: batch=1, 64 rows, 64 columns.
        // Filled with 1.0 to indicate all 64×64 attention positions are valid.
        let pixel_mask = ArrayD::ones(IxDyn(&[1, DETR_MASK_SIZE, DETR_MASK_SIZE]));

        let mut map = HashMap::new();
        map.insert("pixel_values".to_string(), pixel_values);
        map.insert("pixel_mask".to_string(), pixel_mask);
        Ok(map)
    }

    fn name(&self) -> &str {
        "detr"
    }
}

// ---------------------------------------------------------------------------
// YoloPreprocessor
// ---------------------------------------------------------------------------

/// YOLO-family preprocessor (YOLOv5, YOLOv8, YOLOv9).
///
/// Handles models that require a **single** named input:
///
/// | Input name | Shape              | Type | Normalization          |
/// |------------|--------------------|------|------------------------|
/// | `images`   | `[1, 3, 640, 640]` | f32  | Divide by 255 → [0, 1] |
///
/// ## Processing steps
/// 1. Convert image to RGB.
/// 2. Resize to 640×640 using Lanczos3 filter.
/// 3. Extract flat interleaved-RGB pixel buffer.
/// 4. Normalize: divide all values by 255.0.
/// 5. Rearrange from HWC to CHW layout → shape `[3, 640, 640]`.
/// 6. Prepend batch dimension → shape `[1, 3, 640, 640]`.
///
/// ## Why no ImageNet normalization?
/// YOLO models are trained with simple 0–1 normalization, not per-channel mean/std
/// subtraction. Applying ImageNet stats would shift the distribution away from the
/// training domain and degrade detection accuracy.
pub struct YoloPreprocessor;

impl YoloPreprocessor {
    /// Create a new YOLO preprocessor.
    pub fn new() -> Self {
        Self
    }
}

impl Default for YoloPreprocessor {
    fn default() -> Self {
        Self::new()
    }
}

impl Preprocessor for YoloPreprocessor {
    /// Preprocess `image` for YOLO-family models.
    ///
    /// # Returns
    /// `HashMap` with key:
    /// - `"images"` → `ArrayD<f32>` shape `[1, 3, 640, 640]`
    ///
    /// # Errors
    /// Returns `Err` if resizing or tensor construction fails. Does not panic.
    fn preprocess(&self, image: DynamicImage) -> Result<HashMap<String, OrtTensor>> {
        let size = YOLO_SIZE as usize;

        // Step 1 & 2: Ensure RGB, then resize to 640×640.
        let rgb_image = image_utils::to_rgb(&image);
        let resized = image_utils::resize_image(&rgb_image, YOLO_SIZE, YOLO_SIZE);

        // Step 3: Extract flat interleaved-RGB buffer; values in [0.0, 255.0].
        let mut pixels = image_utils::extract_pixels(&resized);

        // Step 4: Normalize by dividing by 255.0 → values in [0.0, 1.0].
        for p in pixels.iter_mut() {
            *p /= 255.0;
        }

        // Step 5: HWC [height, width, 3] → CHW [3, height, width].
        let chw = rgb_hwc_to_chw(pixels, size, size)?;

        // Step 6: Insert batch dimension → [1, 3, 640, 640].
        let images = chw
            .insert_axis(ndarray::Axis(0))
            .into_shape_with_order(IxDyn(&[1, 3, size, size]))
            .map_err(|e| {
                anyhow!(
                    "YoloPreprocessor: failed to reshape to [1,3,640,640]: {}",
                    e
                )
            })?;

        let mut map = HashMap::new();
        map.insert("images".to_string(), images);
        Ok(map)
    }

    fn name(&self) -> &str {
        "yolo"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Registry tests -------------------------------------------------------

    #[test]
    fn test_registry_returns_detr_preprocessor() {
        let result = PreprocessorRegistry::get_preprocessor("detr-resnet101");
        assert!(result.is_ok(), "Expected Ok for 'detr-resnet101'");
        let p = result.unwrap();
        assert_eq!(p.name(), "detr");
    }

    #[test]
    fn test_registry_returns_yolo_preprocessor() {
        let result = PreprocessorRegistry::get_preprocessor("yolov9-c");
        assert!(result.is_ok(), "Expected Ok for 'yolov9-c'");
        let p = result.unwrap();
        assert_eq!(p.name(), "yolo");
    }

    #[test]
    fn test_registry_dfine_maps_to_detr() {
        let result = PreprocessorRegistry::get_preprocessor("dfine-l");
        assert!(result.is_ok());
        assert_eq!(result.unwrap().name(), "detr");
    }

    #[test]
    fn test_registry_rf_detr_large_maps_to_detr() {
        let result = PreprocessorRegistry::get_preprocessor("rf-detr-large");
        assert!(result.is_ok());
        assert_eq!(result.unwrap().name(), "detr");
    }

    #[test]
    fn test_registry_rf_detr_medium_maps_to_detr() {
        let result = PreprocessorRegistry::get_preprocessor("rf-detr-medium");
        assert!(result.is_ok());
        assert_eq!(result.unwrap().name(), "detr");
    }

    #[test]
    fn test_registry_unknown_model_returns_err() {
        let result = PreprocessorRegistry::get_preprocessor("unknown-model-xyz");
        assert!(result.is_err(), "Expected Err for unknown model name");
        // Use err() + unwrap() via anyhow to avoid the Debug bound on Box<dyn Preprocessor>.
        let err_msg = result.err().unwrap().to_string();
        assert!(
            err_msg.contains("unknown-model-xyz"),
            "Error message should include the offending name; got: {err_msg}"
        );
    }

    #[test]
    fn test_registry_case_sensitive_rejection() {
        // Uppercase variant must not silently succeed (consistency with Phase 1 naming).
        let result = PreprocessorRegistry::get_preprocessor("DETR-ResNet101");
        assert!(result.is_err(), "Registry should be case-sensitive");
    }

    #[test]
    fn test_registered_model_names_coverage() {
        // Every name in the static list must resolve successfully.
        for name in PreprocessorRegistry::registered_model_names() {
            let result = PreprocessorRegistry::get_preprocessor(name);
            assert!(
                result.is_ok(),
                "registered name '{name}' failed to resolve a preprocessor"
            );
        }
    }

    // -- Basic trait tests ---------------------------------------------------

    #[test]
    fn test_detr_preprocessor_name() {
        let p = DetrPreprocessor::new();
        assert_eq!(p.name(), "detr");
    }

    #[test]
    fn test_yolo_preprocessor_name() {
        let p = YoloPreprocessor::new();
        assert_eq!(p.name(), "yolo");
    }

    #[test]
    fn test_preprocessor_is_send_sync() {
        // Compile-time check: Preprocessor implementors must be Send + Sync.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<DetrPreprocessor>();
        assert_send_sync::<YoloPreprocessor>();
    }

    #[test]
    fn test_detr_zero_sized() {
        // Constants live at module level; structs remain zero-sized (no heap allocation).
        assert_eq!(std::mem::size_of::<DetrPreprocessor>(), 0);
        assert_eq!(std::mem::size_of::<YoloPreprocessor>(), 0);
    }

    // -- Phase 3: DetrPreprocessor shape tests --------------------------------

    /// Build a solid-colour 100×100 RGB test image.
    fn make_rgb_image(w: u32, h: u32, r: u8, g: u8, b: u8) -> DynamicImage {
        let img = image::ImageBuffer::from_fn(w, h, |_, _| image::Rgb([r, g, b]));
        DynamicImage::ImageRgb8(img)
    }

    /// Build a solid-colour 100×100 RGBA test image (to verify alpha stripping).
    fn make_rgba_image(w: u32, h: u32, r: u8, g: u8, b: u8, a: u8) -> DynamicImage {
        let img = image::ImageBuffer::from_fn(w, h, |_, _| image::Rgba([r, g, b, a]));
        DynamicImage::ImageRgba8(img)
    }

    /// Build a greyscale test image (to verify RGB conversion).
    fn make_luma_image(w: u32, h: u32, luma: u8) -> DynamicImage {
        let img = image::ImageBuffer::from_fn(w, h, |_, _| image::Luma([luma]));
        DynamicImage::ImageLuma8(img)
    }

    // ---------- DetrPreprocessor: output key presence ----------

    #[test]
    fn test_detr_output_contains_pixel_values_key() {
        let p = DetrPreprocessor::new();
        let img = make_rgb_image(200, 150, 128, 64, 32);
        let map = p.preprocess(img).expect("DETR preprocess must succeed");
        assert!(
            map.contains_key("pixel_values"),
            "Output must contain 'pixel_values'"
        );
    }

    #[test]
    fn test_detr_output_contains_pixel_mask_key() {
        let p = DetrPreprocessor::new();
        let img = make_rgb_image(200, 150, 128, 64, 32);
        let map = p.preprocess(img).expect("DETR preprocess must succeed");
        assert!(
            map.contains_key("pixel_mask"),
            "Output must contain 'pixel_mask'"
        );
    }

    #[test]
    fn test_detr_output_has_exactly_two_keys() {
        let p = DetrPreprocessor::new();
        let img = make_rgb_image(100, 100, 100, 100, 100);
        let map = p.preprocess(img).expect("DETR preprocess must succeed");
        assert_eq!(
            map.len(),
            2,
            "DETR output must have exactly 2 keys (pixel_values, pixel_mask); got {}",
            map.len()
        );
    }

    // ---------- DetrPreprocessor: pixel_values shape ----------

    #[test]
    fn test_detr_pixel_values_shape_is_1_3_800_800() {
        let p = DetrPreprocessor::new();
        let img = make_rgb_image(320, 240, 0, 128, 255);
        let map = p.preprocess(img).expect("DETR preprocess must succeed");
        let pv = &map["pixel_values"];
        assert_eq!(
            pv.shape(),
            &[1, 3, 800, 800],
            "pixel_values must have shape [1, 3, 800, 800]; got {:?}",
            pv.shape()
        );
    }

    #[test]
    fn test_detr_pixel_values_shape_from_tiny_input() {
        // Even a 1×1 image must be upsampled to 800×800 correctly.
        let p = DetrPreprocessor::new();
        let img = make_rgb_image(1, 1, 200, 100, 50);
        let map = p
            .preprocess(img)
            .expect("DETR preprocess on 1×1 must succeed");
        assert_eq!(map["pixel_values"].shape(), &[1, 3, 800, 800]);
    }

    #[test]
    fn test_detr_pixel_values_shape_from_large_input() {
        // Large input must be downsampled without error.
        let p = DetrPreprocessor::new();
        let img = make_rgb_image(3000, 2000, 50, 50, 50);
        let map = p
            .preprocess(img)
            .expect("DETR preprocess on large image must succeed");
        assert_eq!(map["pixel_values"].shape(), &[1, 3, 800, 800]);
    }

    // ---------- DetrPreprocessor: pixel_mask shape ----------

    #[test]
    fn test_detr_pixel_mask_shape_is_1_64_64() {
        let p = DetrPreprocessor::new();
        let img = make_rgb_image(640, 480, 200, 100, 50);
        let map = p.preprocess(img).expect("DETR preprocess must succeed");
        let pm = &map["pixel_mask"];
        assert_eq!(
            pm.shape(),
            &[1, 64, 64],
            "pixel_mask must have shape [1, 64, 64]; got {:?} — this is the v1 bug fix!",
            pm.shape()
        );
    }

    // ---------- DetrPreprocessor: pixel_mask content ----------

    #[test]
    fn test_detr_pixel_mask_all_ones() {
        let p = DetrPreprocessor::new();
        let img = make_rgb_image(100, 100, 128, 128, 128);
        let map = p.preprocess(img).expect("DETR preprocess must succeed");
        let pm = &map["pixel_mask"];
        // Every element must be 1.0 (all pixels are valid; no padding applied).
        let all_ones = pm.iter().all(|&v| v == 1.0_f32);
        assert!(
            all_ones,
            "pixel_mask must be all 1.0; found a non-1.0 value"
        );
    }

    // ---------- DetrPreprocessor: pixel_values normalization ----------

    #[test]
    fn test_detr_pixel_values_range_after_imagenet_norm() {
        // After ImageNet normalization, values should be roughly in [-2.5, 2.5].
        // For solid-color images the exact values are predictable:
        //   pixel=128 → raw=128/255≈0.502
        //   channel R: (0.502 - 0.485) / 0.229 ≈  0.074
        //   channel G: (0.502 - 0.456) / 0.224 ≈  0.205
        //   channel B: (0.502 - 0.406) / 0.225 ≈  0.427
        let p = DetrPreprocessor::new();
        let img = make_rgb_image(100, 100, 128, 128, 128);
        let map = p.preprocess(img).expect("DETR preprocess must succeed");
        let pv = &map["pixel_values"];
        for &v in pv.iter() {
            assert!(
                v > -3.0 && v < 3.0,
                "pixel_values after ImageNet norm should be in [-3, 3]; got {v}"
            );
        }
    }

    #[test]
    fn test_detr_pixel_values_white_image_normalization() {
        // White image (255, 255, 255):
        //   raw = 1.0 for all channels
        //   R: (1.0 - 0.485) / 0.229 ≈ 2.249
        //   G: (1.0 - 0.456) / 0.224 ≈ 2.429
        //   B: (1.0 - 0.406) / 0.225 ≈ 2.640
        let p = DetrPreprocessor::new();
        let img = make_rgb_image(50, 50, 255, 255, 255);
        let map = p
            .preprocess(img)
            .expect("DETR preprocess on white image must succeed");
        let pv = &map["pixel_values"];

        // Every value should be positive and > 2.0 for a white image.
        for &v in pv.iter() {
            assert!(
                v > 2.0,
                "white-image pixel_values should all be > 2.0 after ImageNet norm; got {v}"
            );
        }
    }

    #[test]
    fn test_detr_pixel_values_black_image_normalization() {
        // Black image (0, 0, 0):
        //   raw = 0.0 for all channels
        //   R: (0.0 - 0.485) / 0.229 ≈ -2.118
        //   G: (0.0 - 0.456) / 0.224 ≈ -2.036
        //   B: (0.0 - 0.406) / 0.225 ≈ -1.804
        let p = DetrPreprocessor::new();
        let img = make_rgb_image(50, 50, 0, 0, 0);
        let map = p
            .preprocess(img)
            .expect("DETR preprocess on black image must succeed");
        let pv = &map["pixel_values"];

        // Every value should be negative for a black image.
        for &v in pv.iter() {
            assert!(
                v < 0.0,
                "black-image pixel_values should all be < 0.0 after ImageNet norm; got {v}"
            );
        }
    }

    // ---------- DetrPreprocessor: input format handling ----------

    #[test]
    fn test_detr_handles_rgba_image() {
        // Alpha channel must be stripped; output shape must be correct.
        let p = DetrPreprocessor::new();
        let img = make_rgba_image(100, 100, 200, 100, 50, 128);
        let map = p
            .preprocess(img)
            .expect("DETR preprocess on RGBA image must succeed");
        assert_eq!(map["pixel_values"].shape(), &[1, 3, 800, 800]);
        assert_eq!(map["pixel_mask"].shape(), &[1, 64, 64]);
    }

    #[test]
    fn test_detr_handles_greyscale_image() {
        // Greyscale must be converted to RGB; output shape must be correct.
        let p = DetrPreprocessor::new();
        let img = make_luma_image(100, 100, 128);
        let map = p
            .preprocess(img)
            .expect("DETR preprocess on greyscale image must succeed");
        assert_eq!(map["pixel_values"].shape(), &[1, 3, 800, 800]);
        assert_eq!(map["pixel_mask"].shape(), &[1, 64, 64]);
    }

    #[test]
    fn test_detr_handles_non_square_image() {
        // A wide landscape image should be stretched to 800×800 without error.
        let p = DetrPreprocessor::new();
        let img = make_rgb_image(1920, 1080, 100, 150, 200);
        let map = p
            .preprocess(img)
            .expect("DETR preprocess on non-square image must succeed");
        assert_eq!(map["pixel_values"].shape(), &[1, 3, 800, 800]);
    }

    // -- Phase 3: YoloPreprocessor shape tests --------------------------------

    // ---------- YoloPreprocessor: output key presence ----------

    #[test]
    fn test_yolo_output_contains_images_key() {
        let p = YoloPreprocessor::new();
        let img = make_rgb_image(200, 150, 100, 200, 50);
        let map = p.preprocess(img).expect("YOLO preprocess must succeed");
        assert!(map.contains_key("images"), "Output must contain 'images'");
    }

    #[test]
    fn test_yolo_output_has_exactly_one_key() {
        let p = YoloPreprocessor::new();
        let img = make_rgb_image(100, 100, 100, 100, 100);
        let map = p.preprocess(img).expect("YOLO preprocess must succeed");
        assert_eq!(
            map.len(),
            1,
            "YOLO output must have exactly 1 key ('images'); got {}",
            map.len()
        );
    }

    // ---------- YoloPreprocessor: shape ----------

    #[test]
    fn test_yolo_images_shape_is_1_3_640_640() {
        let p = YoloPreprocessor::new();
        let img = make_rgb_image(320, 240, 0, 128, 255);
        let map = p.preprocess(img).expect("YOLO preprocess must succeed");
        let images = &map["images"];
        assert_eq!(
            images.shape(),
            &[1, 3, 640, 640],
            "images must have shape [1, 3, 640, 640]; got {:?}",
            images.shape()
        );
    }

    #[test]
    fn test_yolo_images_shape_from_tiny_input() {
        let p = YoloPreprocessor::new();
        let img = make_rgb_image(1, 1, 200, 100, 50);
        let map = p
            .preprocess(img)
            .expect("YOLO preprocess on 1×1 must succeed");
        assert_eq!(map["images"].shape(), &[1, 3, 640, 640]);
    }

    #[test]
    fn test_yolo_images_shape_from_large_input() {
        let p = YoloPreprocessor::new();
        let img = make_rgb_image(4000, 3000, 50, 50, 50);
        let map = p
            .preprocess(img)
            .expect("YOLO preprocess on large image must succeed");
        assert_eq!(map["images"].shape(), &[1, 3, 640, 640]);
    }

    // ---------- YoloPreprocessor: normalization ----------

    #[test]
    fn test_yolo_pixel_values_range_0_to_1() {
        // After ÷255 normalization all values must be in [0.0, 1.0].
        let p = YoloPreprocessor::new();
        let img = make_rgb_image(100, 100, 128, 64, 32);
        let map = p.preprocess(img).expect("YOLO preprocess must succeed");
        let images = &map["images"];
        for &v in images.iter() {
            assert!(
                v >= 0.0 && v <= 1.0,
                "YOLO images tensor value out of [0, 1] range: {v}"
            );
        }
    }

    #[test]
    fn test_yolo_white_image_normalization() {
        // White pixels (255) → normalized to 1.0 exactly.
        let p = YoloPreprocessor::new();
        let img = make_rgb_image(50, 50, 255, 255, 255);
        let map = p
            .preprocess(img)
            .expect("YOLO preprocess on white image must succeed");
        for &v in map["images"].iter() {
            assert!(
                (v - 1.0_f32).abs() < 1e-5,
                "white-image YOLO values should be ≈ 1.0; got {v}"
            );
        }
    }

    #[test]
    fn test_yolo_black_image_normalization() {
        // Black pixels (0) → normalized to 0.0 exactly.
        let p = YoloPreprocessor::new();
        let img = make_rgb_image(50, 50, 0, 0, 0);
        let map = p
            .preprocess(img)
            .expect("YOLO preprocess on black image must succeed");
        for &v in map["images"].iter() {
            assert!(
                v.abs() < 1e-5,
                "black-image YOLO values should be ≈ 0.0; got {v}"
            );
        }
    }

    // ---------- YoloPreprocessor: input format handling ----------

    #[test]
    fn test_yolo_handles_rgba_image() {
        let p = YoloPreprocessor::new();
        let img = make_rgba_image(100, 100, 200, 100, 50, 255);
        let map = p
            .preprocess(img)
            .expect("YOLO preprocess on RGBA image must succeed");
        assert_eq!(map["images"].shape(), &[1, 3, 640, 640]);
    }

    #[test]
    fn test_yolo_handles_greyscale_image() {
        let p = YoloPreprocessor::new();
        let img = make_luma_image(100, 100, 200);
        let map = p
            .preprocess(img)
            .expect("YOLO preprocess on greyscale image must succeed");
        assert_eq!(map["images"].shape(), &[1, 3, 640, 640]);
    }

    #[test]
    fn test_yolo_handles_non_square_image() {
        let p = YoloPreprocessor::new();
        let img = make_rgb_image(1920, 1080, 100, 150, 200);
        let map = p
            .preprocess(img)
            .expect("YOLO preprocess on non-square image must succeed");
        assert_eq!(map["images"].shape(), &[1, 3, 640, 640]);
    }

    // -- Phase 3: CHW channel ordering correctness ---------------------------

    #[test]
    fn test_detr_chw_channel_ordering_is_correct() {
        // Use a solid red image (255, 0, 0).
        // After normalization channel 0 (R) should have mean ≈ 2.249,
        // channels 1 (G) and 2 (B) should be the negative-mean values (≈ -2.03, ≈ -1.80).
        // This proves channels are not swapped.
        let p = DetrPreprocessor::new();
        let img = make_rgb_image(100, 100, 255, 0, 0); // pure red
        let map = p
            .preprocess(img)
            .expect("DETR preprocess on red image must succeed");
        let pv = &map["pixel_values"];

        // Channel 0 = R: (1.0 - 0.485) / 0.229 ≈ 2.249  → should be positive
        // Channel 1 = G: (0.0 - 0.456) / 0.224 ≈ -2.036  → should be negative
        // Channel 2 = B: (0.0 - 0.406) / 0.225 ≈ -1.804  → should be negative

        // Sample the first pixel of each channel plane (index [0, c, 0, 0]).
        let r_val = pv[[0, 0, 0, 0]];
        let g_val = pv[[0, 1, 0, 0]];
        let b_val = pv[[0, 2, 0, 0]];

        assert!(
            r_val > 2.0,
            "R channel on red image should be > 2.0; got {r_val}"
        );
        assert!(
            g_val < 0.0,
            "G channel on red image should be < 0.0; got {g_val}"
        );
        assert!(
            b_val < 0.0,
            "B channel on red image should be < 0.0; got {b_val}"
        );
    }

    #[test]
    fn test_yolo_chw_channel_ordering_is_correct() {
        // Use a solid green image (0, 255, 0) with YOLO ÷255 normalization.
        // Channel 0 (R) = 0.0, Channel 1 (G) = 1.0, Channel 2 (B) = 0.0.
        let p = YoloPreprocessor::new();
        let img = make_rgb_image(100, 100, 0, 255, 0); // pure green
        let map = p
            .preprocess(img)
            .expect("YOLO preprocess on green image must succeed");
        let images = &map["images"];

        let r_val = images[[0, 0, 0, 0]];
        let g_val = images[[0, 1, 0, 0]];
        let b_val = images[[0, 2, 0, 0]];

        assert!(
            r_val.abs() < 1e-5,
            "R channel on green image should be ≈ 0.0; got {r_val}"
        );
        assert!(
            (g_val - 1.0).abs() < 1e-5,
            "G channel on green image should be ≈ 1.0; got {g_val}"
        );
        assert!(
            b_val.abs() < 1e-5,
            "B channel on green image should be ≈ 0.0; got {b_val}"
        );
    }

    // -- Phase 3: registry dispatch end-to-end -------------------------------

    #[test]
    fn test_registry_dispatch_detr_resnet101_produces_correct_shapes() {
        let p = PreprocessorRegistry::get_preprocessor("detr-resnet101")
            .expect("registry must resolve detr-resnet101");
        let img = make_rgb_image(640, 480, 128, 128, 128);
        let map = p
            .preprocess(img)
            .expect("detr-resnet101 preprocess via registry must succeed");
        assert_eq!(map["pixel_values"].shape(), &[1, 3, 800, 800]);
        assert_eq!(map["pixel_mask"].shape(), &[1, 64, 64]);
    }

    #[test]
    fn test_registry_dispatch_yolov9c_produces_correct_shape() {
        let p = PreprocessorRegistry::get_preprocessor("yolov9-c")
            .expect("registry must resolve yolov9-c");
        let img = make_rgb_image(640, 480, 128, 128, 128);
        let map = p
            .preprocess(img)
            .expect("yolov9-c preprocess via registry must succeed");
        assert_eq!(map["images"].shape(), &[1, 3, 640, 640]);
    }

    #[test]
    fn test_registry_dispatch_dfine_l_produces_detr_shapes() {
        let p = PreprocessorRegistry::get_preprocessor("dfine-l")
            .expect("registry must resolve dfine-l");
        let img = make_rgb_image(200, 200, 100, 100, 100);
        let map = p
            .preprocess(img)
            .expect("dfine-l preprocess via registry must succeed");
        assert_eq!(map["pixel_values"].shape(), &[1, 3, 800, 800]);
        assert_eq!(map["pixel_mask"].shape(), &[1, 64, 64]);
    }

    // -- Internal helper: rgb_hwc_to_chw unit tests --------------------------

    #[test]
    fn test_rgb_hwc_to_chw_shape() {
        // 2×2 image, 3 channels → shape [3, 2, 2]
        let pixels = vec![
            1.0, 2.0, 3.0, // pixel (0,0): R=1 G=2 B=3
            4.0, 5.0, 6.0, // pixel (0,1): R=4 G=5 B=6
            7.0, 8.0, 9.0, // pixel (1,0): R=7 G=8 B=9
            10., 11., 12., // pixel (1,1): R=10 G=11 B=12
        ];
        let chw = rgb_hwc_to_chw(pixels, 2, 2).expect("rgb_hwc_to_chw must succeed");
        assert_eq!(chw.shape(), &[3, 2, 2]);
    }

    #[test]
    fn test_rgb_hwc_to_chw_channel_placement() {
        // 1×2 image (1 row, 2 columns) — verifies channel separation.
        let pixels = vec![
            10.0, 20.0, 30.0, // pixel 0: R=10, G=20, B=30
            40.0, 50.0, 60.0, // pixel 1: R=40, G=50, B=60
        ];
        let chw = rgb_hwc_to_chw(pixels, 1, 2).expect("rgb_hwc_to_chw must succeed");
        // R channel (index 0): [10, 40]
        assert_eq!(chw[[0, 0, 0]], 10.0_f32);
        assert_eq!(chw[[0, 0, 1]], 40.0_f32);
        // G channel (index 1): [20, 50]
        assert_eq!(chw[[1, 0, 0]], 20.0_f32);
        assert_eq!(chw[[1, 0, 1]], 50.0_f32);
        // B channel (index 2): [30, 60]
        assert_eq!(chw[[2, 0, 0]], 30.0_f32);
        assert_eq!(chw[[2, 0, 1]], 60.0_f32);
    }

    #[test]
    fn test_rgb_hwc_to_chw_wrong_buffer_size_returns_err() {
        // 2×2 image needs 12 floats; supply only 9 → must return Err.
        let pixels = vec![1.0_f32; 9];
        let result = rgb_hwc_to_chw(pixels, 2, 2);
        assert!(
            result.is_err(),
            "rgb_hwc_to_chw must return Err for mismatched buffer length"
        );
    }
}
