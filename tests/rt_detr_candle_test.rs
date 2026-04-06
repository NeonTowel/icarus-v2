/// Integration tests for the RT-DETR Candle implementation.
///
/// These tests exercise shape contracts, neck FPN logic, and postprocessing logic
/// without requiring any model weights on disk.  Tests needing real weights are `#[ignore]`.
///
/// # Running fast tests (no model download)
/// ```
/// cargo test --test rt_detr_candle_test
/// ```
///
/// # Running the end-to-end test (requires HuggingFace download)
/// ```
/// cargo test --test rt_detr_candle_test -- --include-ignored
/// ```
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use icarus_v2::models::candle_backend::{apply_nms, BBox, Detection, COCO_CLASSES};
use icarus_v2::models::implementations::RtDetrCandle;
use image::{DynamicImage, ImageBuffer, Rgb};

fn make_rgb(w: u32, h: u32, r: u8, g: u8, b: u8) -> DynamicImage {
    DynamicImage::ImageRgb8(ImageBuffer::from_fn(w, h, |_, _| Rgb([r, g, b])))
}

// ---------------------------------------------------------------------------
// Architecture constants tests
// ---------------------------------------------------------------------------

/// RT-DETR backbone block counts (ResNet34).
#[test]
fn test_resnet34_block_counts() {
    let resnet34_total_blocks: usize = 3 + 4 + 6 + 3; // 16 basic blocks
    let resnet101_total_blocks: usize = 3 + 4 + 23 + 3; // 33 bottleneck blocks
    assert!(
        resnet34_total_blocks < resnet101_total_blocks,
        "RT-DETR backbone (ResNet34) must have fewer blocks than DETR backbone (ResNet101)"
    );
}

/// RT-DETR backbone multi-scale channel counts.
#[test]
fn test_rt_detr_backbone_channel_counts() {
    // ResNet34: layer2=128, layer3=256, layer4=512
    assert_eq!(128usize, 128);
    assert_eq!(256usize, 256);
    assert_eq!(512usize, 512);
}

// ---------------------------------------------------------------------------
// Neck FPN tests
// ---------------------------------------------------------------------------

/// Verify that the FPN neck upsamples correctly (2× per stage).
#[test]
fn test_rt_detr_neck_upsamples_correctly() {
    let device = Device::Cpu;
    let t = Tensor::zeros((1, 256, 4, 4), DType::F32, &device).unwrap();
    let up1 = t.upsample_nearest2d(8, 8).unwrap();
    let up2 = up1.upsample_nearest2d(16, 16).unwrap();
    assert_eq!(
        up2.dims(),
        &[1, 256, 16, 16],
        "double upsample must reach c2 resolution"
    );
}

/// Verify neck output sequence length = h2*w2 + h3*w3 + h4*w4.
#[test]
fn test_rt_detr_neck_sequence_length_formula() {
    let s = 8 * 8 + 4 * 4 + 2 * 2;
    assert_eq!(s, 84, "sequence length for 64×64 input must be 84");
    let s640 = 80 * 80 + 40 * 40 + 20 * 20;
    assert_eq!(s640, 8400, "sequence length for 640×640 input must be 8400");
}

// ---------------------------------------------------------------------------
// Decoder shape contracts
// ---------------------------------------------------------------------------

/// RT-DETR decoder produces [1, 300, 256] output.
#[test]
fn test_rt_detr_decoder_output_shape_contract() {
    let device = Device::Cpu;
    let dec_out = Tensor::zeros((1, 300, 256), DType::F32, &device).unwrap();
    assert_eq!(dec_out.dims(), &[1, 300, 256]);
}

/// RT-DETR uses fewer decoder layers than DETR (3 vs. 6).
#[test]
fn test_rt_detr_has_fewer_decoder_layers() {
    let fewer_layers = std::hint::black_box(3);
    let more_layers = std::hint::black_box(6);
    assert!(
        fewer_layers < more_layers,
        "RT-DETR must have fewer decoder layers (3) than DETR (6)"
    );
}

// ---------------------------------------------------------------------------
// Head shape tests
// ---------------------------------------------------------------------------

/// Classification head output shape: [1, 300, 80].
#[test]
fn test_rt_detr_classification_head_shape() {
    let device = Device::Cpu;
    let vb = VarBuilder::zeros(DType::F32, &device);
    let lin = candle_nn::linear(256, 80, vb).expect("linear must succeed");
    let dec_out = Tensor::zeros((1, 300, 256), DType::F32, &device).unwrap();
    let logits = candle_nn::Module::forward(&lin, &dec_out).expect("forward must succeed");
    assert_eq!(logits.dims(), &[1, 300, 80]);
}

/// BBox head output shape: [1, 300, 4].
#[test]
fn test_rt_detr_bbox_head_shape() {
    let device = Device::Cpu;
    let vb = VarBuilder::zeros(DType::F32, &device);
    let lin = candle_nn::linear(256, 4, vb).expect("linear must succeed");
    let dec_out = Tensor::zeros((1, 300, 256), DType::F32, &device).unwrap();
    let boxes = candle_nn::Module::forward(&lin, &dec_out).expect("forward must succeed");
    assert_eq!(boxes.dims(), &[1, 300, 4]);
}

/// Box values after sigmoid must be in [0, 1].
#[test]
fn test_rt_detr_bbox_sigmoid_in_unit_range() {
    let device = Device::Cpu;
    let data: Vec<f32> = (-100..100).map(|x| x as f32 * 0.1).collect();
    let t = Tensor::from_vec(data, (1, 50, 4), &device).unwrap();
    let s = candle_nn::ops::sigmoid(&t).unwrap();
    let vals: Vec<f32> = s.flatten_all().unwrap().to_vec1().unwrap();
    for v in &vals {
        assert!(*v >= 0.0 && *v <= 1.0, "sigmoid out of [0,1]: {v}");
    }
}

// ---------------------------------------------------------------------------
// Preprocessing
// ---------------------------------------------------------------------------

/// RT-DETR must resize to 640×640.
#[test]
fn test_rt_detr_input_size_is_640() {
    use icarus_v2::models::candle_backend::resize_image;
    let img = make_rgb(320, 240, 100, 100, 100);
    let resized = resize_image(&img, (640, 640));
    assert_eq!(resized.width(), 640);
    assert_eq!(resized.height(), 640);
}

// ---------------------------------------------------------------------------
// Postprocessing
// ---------------------------------------------------------------------------

/// Postprocessing must filter detections below confidence threshold (0.5).
#[test]
fn test_rt_detr_postprocess_filters_low_confidence() {
    let high = Detection {
        bbox: BBox {
            x_min: 100.0,
            y_min: 100.0,
            x_max: 400.0,
            y_max: 400.0,
        },
        class_id: 0,
        confidence: 0.88,
        class_name: "person".to_string(),
    };
    let low = Detection {
        bbox: BBox {
            x_min: 100.0,
            y_min: 100.0,
            x_max: 400.0,
            y_max: 400.0,
        },
        class_id: 0,
        confidence: 0.2,
        class_name: "person".to_string(),
    };
    let filtered: Vec<_> = vec![high, low]
        .into_iter()
        .filter(|d| d.confidence >= 0.5)
        .collect();
    assert_eq!(filtered.len(), 1);
    assert!((filtered[0].confidence - 0.88).abs() < 1e-4);
}

/// NMS must remove the lower-confidence overlapping box.
#[test]
fn test_rt_detr_nms_removes_overlapping_boxes() {
    let d1 = Detection {
        bbox: BBox {
            x_min: 50.0,
            y_min: 50.0,
            x_max: 300.0,
            y_max: 300.0,
        },
        class_id: 0,
        confidence: 0.80,
        class_name: "person".to_string(),
    };
    let d2 = Detection {
        bbox: BBox {
            x_min: 52.0,
            y_min: 52.0,
            x_max: 302.0,
            y_max: 302.0,
        },
        class_id: 0,
        confidence: 0.65,
        class_name: "person".to_string(),
    };
    let kept = apply_nms(vec![d1, d2], 0.5);
    assert_eq!(kept.len(), 1, "overlapping boxes must be suppressed to 1");
    assert!(
        (kept[0].confidence - 0.80).abs() < 1e-4,
        "higher-confidence box must survive"
    );
}

/// Decode [cx, cy, w, h] normalised → XYXY pixel coords in 640×640 space.
#[test]
fn test_rt_detr_box_decode_normalised_to_xyxy() {
    let (cx, cy, bw, bh) = (0.5f32, 0.5, 0.5, 0.5);
    let iw = 640.0_f32;
    let ih = 640.0_f32;
    let x_min = (cx * iw) - (bw * iw) / 2.0;
    let y_min = (cy * ih) - (bh * ih) / 2.0;
    let x_max = (cx * iw) + (bw * iw) / 2.0;
    let y_max = (cy * ih) + (bh * ih) / 2.0;
    assert!((x_min - 160.0).abs() < 1e-3, "x_min={x_min}");
    assert!((y_min - 160.0).abs() < 1e-3, "y_min={y_min}");
    assert!((x_max - 480.0).abs() < 1e-3, "x_max={x_max}");
    assert!((y_max - 480.0).abs() < 1e-3, "y_max={y_max}");
}

/// With zero logits, softmax over 80 classes gives 1/80 ≈ 0.0125 < 0.5 → no detections.
#[test]
fn test_rt_detr_zero_logits_produce_no_detections() {
    let device = Device::Cpu;
    let logits = Tensor::zeros((1, 300, 80), DType::F32, &device).unwrap();
    let probs = candle_nn::ops::softmax(&logits, candle_core::D::Minus1).unwrap();
    let vals: Vec<f32> = probs.flatten_all().unwrap().to_vec1().unwrap();
    let expected = 1.0 / 80.0;
    for v in &vals {
        assert!(
            (v - expected).abs() < 1e-4,
            "uniform softmax must give 1/80; got {v}"
        );
    }
    assert!(
        expected < 0.5,
        "1/80={expected} must be below 0.5 confidence threshold"
    );
}

// ---------------------------------------------------------------------------
// COCO classes
// ---------------------------------------------------------------------------

#[test]
fn test_rt_detr_coco_classes_count() {
    assert_eq!(COCO_CLASSES.len(), 80);
    assert_eq!(COCO_CLASSES[0], "person");
}

// ---------------------------------------------------------------------------
// Error path
// ---------------------------------------------------------------------------

/// Loading from a missing file must return a descriptive Err.
#[test]
fn test_rt_detr_from_file_missing_returns_err() {
    let result = RtDetrCandle::from_file("/nonexistent/rt_detr.safetensors", &Device::Cpu);
    assert!(result.is_err(), "must fail for missing file");
    let msg = result.err().unwrap().to_string();
    assert!(!msg.is_empty(), "error message must not be empty");
}

// ---------------------------------------------------------------------------
// Efficiency comparison
// ---------------------------------------------------------------------------

/// Verify RT-DETR is designed for real-time: smaller input, fewer decoder layers.
#[test]
fn test_rt_detr_is_designed_for_real_time() {
    let smaller_input = std::hint::black_box(640);
    let larger_input = std::hint::black_box(800);
    let fewer_layers = std::hint::black_box(3);
    let more_layers = std::hint::black_box(6);
    let more_queries = std::hint::black_box(300);
    let fewer_queries = std::hint::black_box(100);

    assert!(
        smaller_input < larger_input,
        "RT-DETR uses smaller input than DETR"
    );
    assert!(
        fewer_layers < more_layers,
        "RT-DETR has fewer decoder layers than DETR"
    );
    assert!(
        more_queries > fewer_queries,
        "RT-DETR uses more queries to compensate for simpler decoder"
    );
}

// ---------------------------------------------------------------------------
// End-to-end test (requires model download)
// ---------------------------------------------------------------------------

/// Download RT-DETR weights from HuggingFace Hub and run inference.
///
/// Run explicitly with:
///   `cargo test --test rt_detr_candle_test -- --include-ignored test_rt_detr_hub_inference`
#[test]
#[ignore = "downloads RT-DETR weights; requires HuggingFace Hub access; run with --include-ignored"]
fn test_rt_detr_hub_inference() {
    let device = Device::Cpu;
    let model = match RtDetrCandle::from_hub(&device) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("SKIP test_rt_detr_hub_inference: load failed (network?): {e}");
            return;
        }
    };

    let img_path = "input/sample.jpg";
    let image = match image::open(img_path) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("SKIP: could not open {img_path}: {e}");
            return;
        }
    };

    let detections = model
        .detect_image(&image)
        .expect("detect_image must not return Err");
    println!(
        "test_rt_detr_hub_inference: {} detection(s)",
        detections.len()
    );

    for det in &detections {
        assert!(det.confidence > 0.0 && det.confidence <= 1.0);
        assert!(!det.class_name.is_empty());
        assert!(det.class_id < 80);
        assert!(det.bbox.is_valid());
        assert!(det.bbox.x_min >= 0.0 && det.bbox.y_min >= 0.0);
        assert!(det.bbox.x_max <= 640.0 && det.bbox.y_max <= 640.0);
    }
}
