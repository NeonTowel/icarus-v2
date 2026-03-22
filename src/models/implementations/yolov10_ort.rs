/// YOLOv10 ONNX Runtime implementation of the [`Model`] trait.
///
/// Uses the `ort` crate (ONNX Runtime) directly to run the `onnx-community/yolov10n`
/// model. That model has NMS built-in and outputs `[1, 300, 6]` tensors where each
/// row is `[x1, y1, x2, y2, score, class_id]` in the model's 640×640 input space.
///
/// Our only custom logic is:
/// 1. Resize image to 640×640 (simple resize, no letterboxing) and normalise to [0,1].
/// 2. After inference, scale bbox coordinates back to original image dimensions (~5 LOC).
///
/// All heavy lifting (model loading, graph optimisation, NMS, confidence thresholding)
/// is delegated to ONNX Runtime.
///
/// # Model source
/// `onnx-community/yolov10n` → `onnx/model.onnx` (~9 MB).
/// Downloaded on first use from HuggingFace Hub; cached in `~/.cache/huggingface/`.
///
/// # Thread safety
/// [`YOLOv10Ort`] is `Send + Sync`. The ONNX session is immutable after construction
/// (`ort::Session::run` takes `&self`), so no Mutex is needed for inference.
/// Image dimensions are stored in a `Mutex<Option<(u32, u32)>>` because `preprocess`
/// and `postprocess` share state across two separate `&self` calls.
use candle_core::{DType, Device, Result as CandleResult, Tensor};
use hf_hub::api::sync::Api;
use image::{imageops::FilterType, DynamicImage};
use ndarray::Array;
use ort::{
    inputs,
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};
use std::sync::Mutex;

use crate::models::candle_backend::{BBox, Detection, Model, COCO_CLASSES};

// ---------------------------------------------------------------------------
// YOLOv10Ort struct
// ---------------------------------------------------------------------------

/// YOLOv10n detector backed by ONNX Runtime.
///
/// # Example
/// ```rust,ignore
/// let device = candle_core::Device::Cpu;
/// let model = YOLOv10Ort::from_hub(&device)?;
/// let tensor = model.preprocess(&[img])?;
/// let (logits, boxes) = model.forward(&tensor)?;
/// let detections = model.postprocess(logits, boxes)?;
/// ```
pub struct YOLOv10Ort {
    /// ONNX Runtime session; `run()` requires `&mut Session`, so wrapped in Mutex.
    session: Mutex<Session>,
    /// Original image dimensions captured in `preprocess`, consumed in `postprocess`.
    pending_dims: Mutex<Option<(u32, u32)>>,
}

impl YOLOv10Ort {
    /// Download the YOLOv10n ONNX model from HuggingFace Hub and initialise the session.
    ///
    /// The `device` parameter is accepted for API compatibility but is not used —
    /// ONNX Runtime manages its own execution providers (CPU by default).
    ///
    /// # Errors
    /// Returns `Err` if the download fails or the ONNX session cannot be created.
    pub fn from_hub(_device: &Device) -> anyhow::Result<Self> {
        let api = Api::new()?;
        // onnx-community/yolov10n: end-to-end YOLOv10n with NMS built in.
        // Output: `output0` shaped [1, 300, 6] with [x1, y1, x2, y2, score, class_id]
        // in 640×640 model input coordinate space.
        let repo = api.model("onnx-community/yolov10n".to_string());
        let model_path = repo.get("onnx/model.onnx")?;

        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("ort Session builder failed: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("ort optimisation level failed: {e}"))?
            .commit_from_file(&model_path)
            .map_err(|e| anyhow::anyhow!("ort session load failed: {e}"))?;

        Ok(Self {
            session: Mutex::new(session),
            pending_dims: Mutex::new(None),
        })
    }
}

// ---------------------------------------------------------------------------
// Model trait implementation
// ---------------------------------------------------------------------------

/// Input size the model expects.
const MODEL_W: u32 = 640;
const MODEL_H: u32 = 640;

impl Model for YOLOv10Ort {
    /// Resize image to 640×640, normalise to [0,1], stash original dimensions.
    ///
    /// Returns a 1×3×640×640 Candle tensor (used as a pass-through in `forward`).
    fn preprocess(&self, images: &[DynamicImage]) -> CandleResult<Tensor> {
        let img = images
            .first()
            .ok_or_else(|| candle_core::Error::Msg("preprocess: empty image slice".into()))?;

        // Capture original dimensions for bbox scaling in postprocess.
        *self.pending_dims.lock().unwrap() = Some((img.width(), img.height()));

        // Resize to 640×640 (simple resize, matching the model's expected preprocessing).
        let resized = img.resize_exact(MODEL_W, MODEL_H, FilterType::Nearest);
        let rgb = resized.to_rgb8();

        // Build CHW tensor and normalise to [0, 1].
        let data: Vec<u8> = rgb.into_raw();
        let t = Tensor::from_vec(data, (MODEL_H as usize, MODEL_W as usize, 3), &Device::Cpu)?
            .permute((2, 0, 1))?; // HWC → CHW
        let t = (t.to_dtype(DType::F32)? * (1.0 / 255.0))?; // [0,255] → [0,1]
        t.unsqueeze(0) // CHW → NCHW
    }

    /// Pass-through: no computation here; inference runs in `postprocess`.
    fn forward(&self, xs: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        Ok((xs.clone(), xs.clone()))
    }

    /// Run ONNX inference and scale bbox coordinates back to original image space.
    ///
    /// The model output `output0` has shape `[1, 300, 6]`. Each row is
    /// `[x1, y1, x2, y2, score, class_id]` in 640×640 model space.
    /// We apply `(x * orig_w / 640, y * orig_h / 640)` to scale back.
    ///
    /// # Errors
    /// Returns `Err` if `preprocess` was not called first, or if the ONNX session fails.
    fn postprocess(&self, logits: Tensor, _boxes: Tensor) -> CandleResult<Vec<Detection>> {
        let (orig_w, orig_h) = self.pending_dims.lock().unwrap().take().ok_or_else(|| {
            candle_core::Error::Msg(
                "postprocess: no image dimensions — call preprocess first".into(),
            )
        })?;

        // Convert Candle tensor → ndarray for the ORT input.
        let data: Vec<f32> = logits.flatten_all()?.to_vec1()?;
        let array =
            Array::from_shape_vec((1usize, 3usize, MODEL_H as usize, MODEL_W as usize), data)
                .map_err(|e| candle_core::Error::Msg(format!("ndarray reshape failed: {e}")))?;

        // Run ONNX inference (NMS is built into the model — no custom NMS needed).
        let tensor_ref = TensorRef::from_array_view(&array)
            .map_err(|e| candle_core::Error::Msg(format!("ort TensorRef failed: {e}")))?;
        let mut session_guard = self.session.lock().unwrap();
        let outputs = session_guard
            .run(inputs!["images" => tensor_ref])
            .map_err(|e| candle_core::Error::Msg(format!("ort inference failed: {e}")))?;

        let (_shape, raw) = outputs["output0"]
            .try_extract_tensor::<f32>()
            .map_err(|e| candle_core::Error::Msg(format!("ort extract output0 failed: {e}")))?;

        // output0 shape: [1, 300, 6] → flat vec of 1800 f32 values.
        // Each detection: [x1, y1, x2, y2, score, class_id] in 640×640 space.
        let scale_x = orig_w as f32 / MODEL_W as f32;
        let scale_y = orig_h as f32 / MODEL_H as f32;
        let values: Vec<f32> = raw.iter().copied().collect();

        let detections = values
            .chunks_exact(6)
            .filter_map(|row| {
                let (x1, y1, x2, y2, score, class_raw) =
                    (row[0], row[1], row[2], row[3], row[4], row[5]);

                // Skip padding rows (model outputs exactly 300 slots; unfilled = zeros).
                if score <= 0.0 {
                    return None;
                }
                // Skip invalid confidence values.
                if !(0.0..=1.0).contains(&score) {
                    return None;
                }

                let class_id = class_raw as usize;
                if class_id >= COCO_CLASSES.len() {
                    return None;
                }

                // Scale bboxes from 640×640 model space → original image space.
                Some(Detection {
                    bbox: BBox {
                        x_min: (x1 * scale_x).max(0.0),
                        y_min: (y1 * scale_y).max(0.0),
                        x_max: (x2 * scale_x).min(orig_w as f32),
                        y_max: (y2 * scale_y).min(orig_h as f32),
                    },
                    class_id,
                    confidence: score,
                    class_name: COCO_CLASSES[class_id].to_string(),
                })
            })
            .collect();

        Ok(detections)
    }

    fn classes(&self) -> &[&str] {
        &COCO_CLASSES
    }

    fn input_size(&self) -> (usize, usize) {
        (MODEL_W as usize, MODEL_H as usize)
    }
}
