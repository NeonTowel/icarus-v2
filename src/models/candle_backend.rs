/// Candle Backend: Core abstractions for pure-Rust ML inference with Hugging Face Candle.
///
/// This module provides:
/// - The [`Model`] trait that all Candle-based detectors implement.
/// - The [`Detection`] struct representing a single bounding-box prediction.
/// - Preprocessing helpers: [`resize_and_normalize`], [`batch_images`].
/// - Postprocessing helpers: [`apply_nms`], [`convert_cxcywh_to_xyxy`].
/// - Model weight download: [`load_model_from_hub`].
///
/// # Design rationale
/// The `Model` trait mirrors the existing ONNX pipeline pattern (preprocess → forward →
/// postprocess) but operates entirely on Candle `Tensor` values.  This lets the rest of
/// the codebase stay generic over the backend while we migrate model-by-model.
///
/// # Thread safety
/// All trait implementors must be `Send + Sync` so they can be shared across async tasks
/// without cloning the (potentially large) weight tensors.
use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_transformers::object_detection::{non_maximum_suppression, Bbox};
use hf_hub::api::sync::Api;
use image::{imageops::FilterType, DynamicImage};
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Detection struct
// ---------------------------------------------------------------------------

/// A single object detection result.
///
/// Bounding box coordinates are in pixel space, matching the original input image
/// dimensions (not the model's internal resolution).
///
/// # Example
/// ```rust,ignore
/// let det = Detection {
///     bbox: BBox { x_min: 10.0, y_min: 20.0, x_max: 100.0, y_max: 200.0 },
///     class_id: 0,
///     confidence: 0.92,
///     class_name: "person".to_string(),
/// };
/// ```
#[derive(Clone, Debug)]
pub struct Detection {
    /// Bounding box in pixel-space XYXY format: (x_min, y_min, x_max, y_max).
    pub bbox: BBox,
    /// Zero-based COCO class index.
    pub class_id: usize,
    /// Detection confidence score in (0.0, 1.0].
    pub confidence: f32,
    /// Human-readable class name (e.g. "person", "car").
    pub class_name: String,
}

/// Bounding box in pixel-space XYXY format.
#[derive(Clone, Debug, Default)]
pub struct BBox {
    pub x_min: f32,
    pub y_min: f32,
    pub x_max: f32,
    pub y_max: f32,
}

impl BBox {
    /// Clamp coordinates to be within the given image dimensions.
    pub fn clamp(&self, img_w: f32, img_h: f32) -> Self {
        Self {
            x_min: self.x_min.clamp(0.0, img_w),
            y_min: self.y_min.clamp(0.0, img_h),
            x_max: self.x_max.clamp(0.0, img_w),
            y_max: self.y_max.clamp(0.0, img_h),
        }
    }

    /// Returns `true` if this bbox has positive area.
    pub fn is_valid(&self) -> bool {
        self.x_max > self.x_min && self.y_max > self.y_min
    }
}

// ---------------------------------------------------------------------------
// ImageClassifier trait
// ---------------------------------------------------------------------------

/// The core inference contract for image classification models.
pub trait ImageClassifier: Send + Sync {
    /// Classify a cropped image and return the predicted tier (1-4).
    fn classify(&self, image: &image::DynamicImage) -> anyhow::Result<u8>;

    /// The name of this classifier.
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Model trait
// ---------------------------------------------------------------------------

/// The core inference contract for all Candle-based detectors.
///
/// Each model family (YOLOv9, DETR, etc.) implements this trait and provides its own
/// architecture-specific preprocessing and postprocessing logic.
///
/// # Implementing a new model
/// 1. Implement `forward` to run the neural network on a preprocessed tensor.
/// 2. Implement `preprocess` to resize/normalise raw images into the tensor format the
///    network expects.
/// 3. Implement `postprocess` to decode raw network outputs into [`Detection`] values.
/// 4. Override `classes` with the list of class names your model was trained on.
/// 5. Override `input_size` with the (width, height) the model expects.
pub trait Model: Send + Sync {
    /// Run the neural network on a preprocessed input batch.
    ///
    /// # Arguments
    /// * `xs` — input tensor with shape `[batch, channels, height, width]` (NCHW).
    ///
    /// # Returns
    /// A `(logits, boxes)` pair in whatever format the model produces.  Exact semantics
    /// depend on the architecture (see implementor docs).
    fn forward(&self, xs: &Tensor) -> CandleResult<(Tensor, Tensor)>;

    /// Preprocess a batch of raw images into the input tensor the model expects.
    ///
    /// Implementors must:
    /// - Resize images to `self.input_size()`.
    /// - Apply model-specific normalisation (e.g. divide by 255 for YOLO, ImageNet
    ///   mean/std for DETR-family).
    /// - Return an `NCHW` tensor on the correct device.
    fn preprocess(&self, images: &[DynamicImage]) -> CandleResult<Tensor>;

    /// Convert raw network outputs into a list of [`Detection`] values.
    ///
    /// Implementors must apply confidence thresholding and non-maximum suppression.
    ///
    /// # Arguments
    /// * `logits` — class score tensor.
    /// * `boxes`  — bounding box tensor in the model's native format.
    fn postprocess(&self, logits: Tensor, boxes: Tensor) -> CandleResult<Vec<Detection>>;

    /// The class names this model was trained on.
    ///
    /// The index of each name must match the class-id index used in `postprocess`.
    fn classes(&self) -> &[&str];

    /// The (width, height) in pixels that this model's input layer expects.
    fn input_size(&self) -> (usize, usize);

    /// Run inference on a single image and return all detected objects.
    ///
    /// The **default** implementation calls `preprocess → forward → postprocess` in
    /// sequence (the legacy three-step path). ORT-backed YOLO models override this
    /// with a direct ndarray path that:
    /// - Eliminates the Candle tensor round-trip (`DynamicImage → Tensor → Vec<f32>`).
    /// - Keeps original `(width, height)` on the call stack rather than in a
    ///   `thread_local!` side-channel.
    ///
    /// Callers (e.g. `batch_processor::run_detection`) should prefer this method over
    /// the three-step dance.
    fn infer(&self, image: &DynamicImage) -> CandleResult<Vec<Detection>> {
        let tensor = self.preprocess(std::slice::from_ref(image))?;
        let (logits, boxes) = self.forward(&tensor)?;
        self.postprocess(logits, boxes)
    }
}

// ---------------------------------------------------------------------------
// Preprocessing utilities
// ---------------------------------------------------------------------------

/// Build an NCHW `[1, 3, model_h, model_w]` f32 ndarray directly from image pixels.
///
/// Resizes to `(model_w, model_h)` using nearest-neighbour interpolation, converts to
/// RGB8, and de-interleaves HWC → NCHW with pixel values normalised to `[0.0, 1.0]`.
///
/// This produces **byte-for-byte identical values** to the Candle round-trip:
/// `Tensor::from_vec(HWC u8) → permute(CHW) → F32/255 → unsqueeze(NCHW)`.
///
/// Parity is verified by `test_preprocess_to_nchw_array_matches_candle_path`.
///
/// # Memory
/// One allocation of `3 × model_w × model_h × 4` bytes (≈4.7 MB for 640×640).
///
/// # Example
/// ```rust,ignore
/// let arr = preprocess_to_nchw_array(&img, 640, 640);
/// assert_eq!(arr.shape(), &[1, 3, 640, 640]);
/// ```
pub fn preprocess_to_nchw_array(
    img: &DynamicImage,
    model_w: u32,
    model_h: u32,
) -> ndarray::Array<f32, ndarray::Ix4> {
    let resized = img.resize_exact(model_w, model_h, FilterType::Nearest);
    let rgb = resized.to_rgb8();
    let data = rgb.into_raw(); // HWC Vec<u8>: index = h*W*3 + w*3 + c

    let pixel_count = (model_w * model_h) as usize;
    let mut nchw_data = vec![0.0f32; 3 * pixel_count];

    // De-interleave HWC → NCHW: R-plane, then G-plane, then B-plane.
    for i in 0..pixel_count {
        nchw_data[i] = data[i * 3] as f32 / 255.0; // channel 0 (R)
        nchw_data[pixel_count + i] = data[i * 3 + 1] as f32 / 255.0; // channel 1 (G)
        nchw_data[2 * pixel_count + i] = data[i * 3 + 2] as f32 / 255.0; // channel 2 (B)
    }

    ndarray::Array::from_shape_vec((1, 3, model_h as usize, model_w as usize), nchw_data)
        .expect("shape is exactly 1 × 3 × model_h × model_w — infallible for fixed constants")
}

/// Resize `img` to `(width, height)` using Lanczos3 (high-quality, no aspect-ratio
/// preservation — the model input is always square).
///
/// # Example
/// ```rust,ignore
/// let resized = resize_image(&img, (640, 640));
/// ```
pub fn resize_image(img: &DynamicImage, size: (u32, u32)) -> DynamicImage {
    img.resize_exact(size.0, size.1, FilterType::Lanczos3)
}

/// Encode a single RGB image as a Candle tensor and normalise pixel values to [0, 1].
///
/// The returned tensor has shape `[3, height, width]` (CHW), dtype `f32`, on `device`.
///
/// # Errors
/// Returns `Err` if the tensor allocation fails on the target device.
///
/// # Example
/// ```rust,ignore
/// let device = Device::Cpu;
/// let t = normalize_image(&img, &device)?;
/// assert_eq!(t.dims(), &[3, 640, 640]);
/// ```
pub fn normalize_image(img: &DynamicImage, device: &Device) -> CandleResult<Tensor> {
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let data: Vec<u8> = rgb.into_raw();

    // Build HWC tensor [h, w, 3] then permute to CHW [3, h, w].
    let t = Tensor::from_vec(data, (h, w, 3), device)?.permute((2, 0, 1))?;
    // Cast to f32 and scale to [0, 1].
    t.to_dtype(DType::F32)? * (1.0 / 255.0)
}

/// Stack a list of CHW tensors into a batch tensor with shape `[n, C, H, W]`.
///
/// All input tensors must have the same shape.
///
/// # Errors
/// Returns `Err` if `tensors` is empty or shapes are inconsistent.
///
/// # Example
/// ```rust,ignore
/// let batch = batch_images(vec![t1, t2])?;
/// assert_eq!(batch.dims()[0], 2); // batch size
/// ```
pub fn batch_images(tensors: Vec<Tensor>) -> CandleResult<Tensor> {
    // Stack along a new leading dimension to produce [n, C, H, W].
    let refs: Vec<&Tensor> = tensors.iter().collect();
    Tensor::stack(&refs, 0)
}

// ---------------------------------------------------------------------------
// Postprocessing utilities
// ---------------------------------------------------------------------------

/// Convert YOLO centre-format `(cx, cy, w, h)` to XYXY `(x_min, y_min, x_max, y_max)`.
///
/// All input values are in model-input pixel coordinates (e.g. [0, 640]).
/// `img_w` / `img_h` are the model input dimensions and used to scale back to the
/// original image when calling code also provides the original dimensions.
///
/// # Example
/// ```rust
/// use icarus_v2::models::candle_backend::convert_cxcywh_to_xyxy;
/// let (x1, y1, x2, y2) = convert_cxcywh_to_xyxy(320.0, 240.0, 100.0, 80.0, 640.0, 480.0);
/// assert!((x1 - 270.0).abs() < 1e-4);
/// ```
pub fn convert_cxcywh_to_xyxy(
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
    _img_w: f32,
    _img_h: f32,
) -> (f32, f32, f32, f32) {
    let x_min = cx - w / 2.0;
    let y_min = cy - h / 2.0;
    let x_max = cx + w / 2.0;
    let y_max = cy + h / 2.0;
    (x_min, y_min, x_max, y_max)
}

/// Apply non-maximum suppression to a list of detections grouped by class.
///
/// This is a thin wrapper around `candle_transformers::object_detection::non_maximum_suppression`
/// that operates on the [`Detection`] type used throughout this crate.
///
/// The algorithm:
/// 1. Groups detections by `class_id`.
/// 2. Within each class, sorts by confidence descending.
/// 3. Drops detections whose IoU with an already-kept detection exceeds `threshold`.
///
/// # Arguments
/// * `detections` — unsorted detections, may span multiple classes.
/// * `threshold`  — IoU threshold above which a box is suppressed (typical: 0.45).
///
/// # Returns
/// Filtered list, still potentially spanning multiple classes.
///
/// # Example
/// ```rust,ignore
/// let kept = apply_nms(raw_detections, 0.45);
/// ```
pub fn apply_nms(detections: Vec<Detection>, threshold: f32) -> Vec<Detection> {
    if detections.is_empty() {
        return detections;
    }

    // Determine how many classes are present.
    let n_classes = detections.iter().map(|d| d.class_id).max().unwrap_or(0) + 1;

    // Build per-class Bbox vecs that candle-transformers NMS expects.
    let mut bbox_by_class: Vec<Vec<Bbox<usize>>> = (0..n_classes).map(|_| vec![]).collect();
    for det in &detections {
        bbox_by_class[det.class_id].push(Bbox {
            xmin: det.bbox.x_min,
            ymin: det.bbox.y_min,
            xmax: det.bbox.x_max,
            ymax: det.bbox.y_max,
            confidence: det.confidence,
            // Store original index so we can reconstruct the Detection afterwards.
            data: det.class_id,
        });
    }

    non_maximum_suppression(&mut bbox_by_class, threshold);

    // Rebuild Detection list from the surviving bboxes.
    let mut kept: Vec<Detection> = vec![];
    for (class_id, bboxes_for_class) in bbox_by_class.iter().enumerate() {
        for b in bboxes_for_class {
            // Find the original Detection with matching class and bbox (by confidence).
            if let Some(orig) = detections
                .iter()
                .find(|d| d.class_id == class_id && (d.confidence - b.confidence).abs() < 1e-6)
            {
                kept.push(orig.clone());
            }
        }
    }

    kept
}

// ---------------------------------------------------------------------------
// Model weight download
// ---------------------------------------------------------------------------

/// Download a model file from HuggingFace Hub and return its local cache path.
///
/// The `model_name` argument should be in the format `"owner/repo"` (e.g.
/// `"hustvl/yolov9-c"`).  The `filename` argument is the specific file within that
/// repository (e.g. `"yolov9c.safetensors"`).
///
/// Files are cached in the HuggingFace Hub local cache directory (typically
/// `~/.cache/huggingface/hub/`).  Subsequent calls for the same file return the cached
/// path immediately without re-downloading.
///
/// # Arguments
/// * `model_name` — HuggingFace repo ID in the form `"owner/repo"`.
/// * `filename`   — file name inside the repo.
/// * `_device`    — device hint (reserved for future GPU-pinned memory support).
///
/// # Errors
/// Returns `Err` if the download fails or the cache directory is not writable.
///
/// # Example
/// ```rust,ignore
/// let path = load_model_from_hub("lmz/candle-yolo-v8", "yolov8s.safetensors", &Device::Cpu).await?;
/// let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &device)? };
/// ```
pub async fn load_model_from_hub(
    model_name: &str,
    filename: &str,
    _device: &Device,
) -> anyhow::Result<PathBuf> {
    let model_name = model_name.to_string();
    let filename = filename.to_string();

    // Run the blocking HF Hub API call on a dedicated thread pool to avoid blocking the
    // Tokio executor (hf-hub's sync API performs network I/O).
    let path = tokio::task::spawn_blocking(move || -> anyhow::Result<PathBuf> {
        let api = Api::new()?;
        let repo = api.model(model_name);
        let local_path = repo.get(&filename)?;
        Ok(local_path)
    })
    .await
    .map_err(|e| anyhow::anyhow!("model download task panicked: {e}"))??;

    Ok(path)
}

// ---------------------------------------------------------------------------
// DETR 91-class label mapping (IDs 0–90, used by facebook/detr-resnet-50)
// ---------------------------------------------------------------------------

/// DETR uses 91 COCO classes indexed 0–90.
///
/// ID 0 = "N/A" (background/no-object — skip in postprocessing).
/// IDs 1–90 = real object categories from the COCO 2017 detection benchmark.
///
/// Source: `facebook/detr-resnet-50` config.json `id2label` mapping.
pub const DETR_COCO_CLASSES_91: &[&str] = &[
    "N/A",            // 0
    "person",         // 1
    "bicycle",        // 2
    "car",            // 3
    "motorcycle",     // 4
    "airplane",       // 5
    "bus",            // 6
    "train",          // 7
    "truck",          // 8
    "boat",           // 9
    "traffic light",  // 10
    "fire hydrant",   // 11
    "street sign",    // 12
    "stop sign",      // 13
    "parking meter",  // 14
    "bench",          // 15
    "bird",           // 16
    "cat",            // 17
    "dog",            // 18
    "horse",          // 19
    "sheep",          // 20
    "cow",            // 21
    "elephant",       // 22
    "bear",           // 23
    "zebra",          // 24
    "giraffe",        // 25
    "hat",            // 26
    "backpack",       // 27
    "umbrella",       // 28
    "shoe",           // 29
    "eye glasses",    // 30
    "handbag",        // 31
    "tie",            // 32
    "suitcase",       // 33
    "frisbee",        // 34
    "skis",           // 35
    "snowboard",      // 36
    "sports ball",    // 37
    "kite",           // 38
    "baseball bat",   // 39
    "baseball glove", // 40
    "skateboard",     // 41
    "surfboard",      // 42
    "tennis racket",  // 43
    "bottle",         // 44
    "plate",          // 45
    "wine glass",     // 46
    "cup",            // 47
    "fork",           // 48
    "knife",          // 49
    "spoon",          // 50
    "bowl",           // 51
    "banana",         // 52
    "apple",          // 53
    "sandwich",       // 54
    "orange",         // 55
    "broccoli",       // 56
    "carrot",         // 57
    "hot dog",        // 58
    "pizza",          // 59
    "donut",          // 60
    "cake",           // 61
    "chair",          // 62
    "couch",          // 63
    "potted plant",   // 64
    "bed",            // 65
    "mirror",         // 66
    "dining table",   // 67
    "window",         // 68
    "desk",           // 69
    "toilet",         // 70
    "door",           // 71
    "tv",             // 72
    "laptop",         // 73
    "mouse",          // 74
    "remote",         // 75
    "keyboard",       // 76
    "cell phone",     // 77
    "microwave",      // 78
    "oven",           // 79
    "toaster",        // 80
    "sink",           // 81
    "refrigerator",   // 82
    "blender",        // 83
    "book",           // 84
    "clock",          // 85
    "vase",           // 86
    "scissors",       // 87
    "teddy bear",     // 88
    "hair drier",     // 89
    "toothbrush",     // 90
];

// ---------------------------------------------------------------------------
// COCO class names (80 classes, used by YOLO family)
// ---------------------------------------------------------------------------

/// COCO dataset class names in canonical index order.
///
/// Index 0 = "person", index 79 = "toothbrush".  These are the 80 classes used by
/// YOLOv8/v9 and most COCO-trained object detectors.
pub const COCO_CLASSES: [&str; 80] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bbox_clamp_keeps_within_bounds() {
        let b = BBox {
            x_min: -10.0,
            y_min: 5.0,
            x_max: 700.0,
            y_max: 500.0,
        };
        let clamped = b.clamp(640.0, 480.0);
        assert_eq!(clamped.x_min, 0.0);
        assert_eq!(clamped.y_min, 5.0);
        assert_eq!(clamped.x_max, 640.0);
        assert_eq!(clamped.y_max, 480.0);
    }

    #[test]
    fn test_bbox_is_valid_positive_area() {
        let valid = BBox {
            x_min: 10.0,
            y_min: 10.0,
            x_max: 100.0,
            y_max: 100.0,
        };
        assert!(valid.is_valid());
    }

    #[test]
    fn test_bbox_is_valid_zero_width() {
        let zero_w = BBox {
            x_min: 50.0,
            y_min: 10.0,
            x_max: 50.0,
            y_max: 100.0,
        };
        assert!(!zero_w.is_valid());
    }

    #[test]
    fn test_convert_cxcywh_to_xyxy_centre_box() {
        // A box centred at (320, 240) with w=100, h=80 on a 640x480 canvas.
        let (x1, y1, x2, y2) = convert_cxcywh_to_xyxy(320.0, 240.0, 100.0, 80.0, 640.0, 480.0);
        assert!((x1 - 270.0).abs() < 1e-4, "x_min should be 270; got {x1}");
        assert!((y1 - 200.0).abs() < 1e-4, "y_min should be 200; got {y1}");
        assert!((x2 - 370.0).abs() < 1e-4, "x_max should be 370; got {x2}");
        assert!((y2 - 280.0).abs() < 1e-4, "y_max should be 280; got {y2}");
    }

    #[test]
    fn test_detr_coco_classes_91_has_91_entries() {
        assert_eq!(
            DETR_COCO_CLASSES_91.len(),
            91,
            "DETR_COCO_CLASSES_91 must have exactly 91 entries (IDs 0–90)"
        );
    }

    #[test]
    fn test_detr_coco_classes_91_first_is_na() {
        assert_eq!(DETR_COCO_CLASSES_91[0], "N/A", "ID 0 is background");
    }

    #[test]
    fn test_detr_coco_classes_91_person_at_index_1() {
        assert_eq!(DETR_COCO_CLASSES_91[1], "person");
    }

    #[test]
    fn test_detr_coco_classes_91_last_is_toothbrush() {
        assert_eq!(DETR_COCO_CLASSES_91[90], "toothbrush");
    }

    #[test]
    fn test_coco_classes_has_80_entries() {
        assert_eq!(
            COCO_CLASSES.len(),
            80,
            "COCO_CLASSES must have exactly 80 entries"
        );
    }

    #[test]
    fn test_coco_classes_first_is_person() {
        assert_eq!(COCO_CLASSES[0], "person");
    }

    #[test]
    fn test_coco_classes_last_is_toothbrush() {
        assert_eq!(COCO_CLASSES[79], "toothbrush");
    }

    #[test]
    fn test_apply_nms_empty_input() {
        let result = apply_nms(vec![], 0.45);
        assert!(result.is_empty());
    }

    #[test]
    fn test_apply_nms_single_detection_always_kept() {
        let det = Detection {
            bbox: BBox {
                x_min: 10.0,
                y_min: 10.0,
                x_max: 100.0,
                y_max: 100.0,
            },
            class_id: 0,
            confidence: 0.9,
            class_name: "person".to_string(),
        };
        let kept = apply_nms(vec![det], 0.45);
        assert_eq!(kept.len(), 1, "single detection must always survive NMS");
    }

    #[test]
    fn test_apply_nms_suppresses_overlapping_boxes() {
        // Two heavily overlapping boxes for the same class — only the higher-confidence one
        // should survive.
        let det_high = Detection {
            bbox: BBox {
                x_min: 10.0,
                y_min: 10.0,
                x_max: 100.0,
                y_max: 100.0,
            },
            class_id: 0,
            confidence: 0.9,
            class_name: "person".to_string(),
        };
        let det_low = Detection {
            bbox: BBox {
                x_min: 12.0,
                y_min: 12.0,
                x_max: 102.0,
                y_max: 102.0,
            },
            class_id: 0,
            confidence: 0.7,
            class_name: "person".to_string(),
        };
        let kept = apply_nms(vec![det_high, det_low], 0.45);
        assert_eq!(
            kept.len(),
            1,
            "highly overlapping boxes should be reduced to 1"
        );
        assert!(
            (kept[0].confidence - 0.9).abs() < 1e-4,
            "highest-confidence box must be kept; got {}",
            kept[0].confidence
        );
    }

    #[test]
    fn test_normalize_image_shape_and_range() {
        let device = Device::Cpu;
        let img = DynamicImage::ImageRgb8(image::ImageBuffer::from_fn(32, 32, |_, _| {
            image::Rgb([128u8, 64, 255])
        }));
        let t = normalize_image(&img, &device).expect("normalize_image must succeed");
        assert_eq!(t.dims(), &[3, 32, 32], "shape must be [C, H, W]");

        // All values must be in [0, 1].
        let values: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
        for v in &values {
            assert!(*v >= 0.0 && *v <= 1.0, "pixel value out of [0,1]: {v}");
        }
    }

    #[test]
    fn test_preprocess_to_nchw_array_shape_640x640() {
        let img = DynamicImage::new_rgb8(320, 240);
        let arr = preprocess_to_nchw_array(&img, 640, 640);
        assert_eq!(
            arr.shape(),
            &[1, 3, 640, 640],
            "preprocess_to_nchw_array must return [N=1, C=3, H=640, W=640]"
        );
    }

    #[test]
    fn test_preprocess_to_nchw_array_values_in_unit_range() {
        let img = DynamicImage::ImageRgb8(image::ImageBuffer::from_fn(32, 32, |_, _| {
            image::Rgb([0u8, 128u8, 255u8])
        }));
        let arr = preprocess_to_nchw_array(&img, 64, 64);
        for &v in arr.iter() {
            assert!(
                (0.0..=1.0).contains(&v),
                "pixel value {v} is outside [0.0, 1.0]"
            );
        }
    }

    /// Defensive test: ensure the function works for non-square model dimensions.
    #[test]
    fn test_preprocess_to_nchw_array_non_square_shape() {
        let img = DynamicImage::new_rgb8(100, 75);
        let arr = preprocess_to_nchw_array(&img, 320, 240);
        assert_eq!(
            arr.shape(),
            &[1, 3, 240, 320],
            "output shape must be [1, 3, model_h, model_w]"
        );
    }

    /// Verify byte-for-byte parity with the Candle round-trip for the shared helper.
    #[test]
    fn test_preprocess_to_nchw_array_matches_candle_path() {
        use image::imageops::FilterType;

        let img = DynamicImage::ImageRgb8(image::ImageBuffer::from_fn(100, 75, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 128u8])
        }));

        // --- Candle round-trip (legacy path) ---
        let resized = img.resize_exact(64, 64, FilterType::Nearest);
        let rgb = resized.to_rgb8();
        let raw_u8: Vec<u8> = rgb.into_raw();
        let t = Tensor::from_vec(raw_u8, (64usize, 64usize, 3), &Device::Cpu)
            .unwrap()
            .permute((2, 0, 1))
            .unwrap();
        let t = (t.to_dtype(DType::F32).unwrap() * (1.0f64 / 255.0)).unwrap();
        let t = t.unsqueeze(0).unwrap();
        let candle_data: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();

        // --- Shared helper ---
        let direct_arr = preprocess_to_nchw_array(&img, 64, 64);
        let direct_data: Vec<f32> = direct_arr.iter().copied().collect();

        assert_eq!(candle_data.len(), direct_data.len());
        let max_diff = candle_data
            .iter()
            .zip(direct_data.iter())
            .map(|(c, d)| (c - d).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-6,
            "max diff {max_diff}: shared helper must be numerically identical to Candle path"
        );
    }

    #[test]
    fn test_batch_images_stacks_correctly() {
        let device = Device::Cpu;
        let make_t = |val: u8| {
            let img = DynamicImage::ImageRgb8(image::ImageBuffer::from_fn(8, 8, |_, _| {
                image::Rgb([val, val, val])
            }));
            normalize_image(&img, &device).unwrap()
        };
        let t1 = make_t(0);
        let t2 = make_t(255);
        let batch = batch_images(vec![t1, t2]).expect("batch_images must succeed");
        assert_eq!(
            batch.dims(),
            &[2, 3, 8, 8],
            "batch shape must be [N, C, H, W]"
        );
    }
}
