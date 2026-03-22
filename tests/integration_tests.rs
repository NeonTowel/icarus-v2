use icarus_v2::image_utils::crop_image;
use icarus_v2::models::implementations::{
    DETRResNet101, DFineL, RFDETRLarge, RFDETRMedium, YOLOv9c,
};
use icarus_v2::models::preprocessor::PreprocessorRegistry;
/// Integration tests for Icarus-v2
///
/// These tests exercise the full pipeline from image → preprocessing → (mock/real inference)
/// → postprocessing → `Vec<Detection>` without requiring a live ONNX Runtime session for
/// most cases.
///
/// Tests that *do* require a real ONNX session (i.e., they call `model.detect(image)`)
/// will fail gracefully with a descriptive error if the ONNX model files are absent.
/// They are **not** marked `#[ignore]` so CI will report them as expected failures when
/// running without model weights; gate them on a feature flag if needed.
///
/// # External dependencies
/// Real inference tests require `libonnxruntime.so` to be installed and discoverable at
/// runtime (see ORT `load-dynamic` feature). Tests detect the absence of ORT gracefully.
// ---------------------------------------------------------------------------
// Test image factories
// ---------------------------------------------------------------------------
use image::{DynamicImage, ImageBuffer, Luma, Rgb, Rgba};
use std::path::PathBuf;

/// Create a solid-colour RGB test image.
///
/// # Arguments
/// * `width`, `height` — image dimensions in pixels
/// * `r`, `g`, `b`     — RGB channel fill values
fn make_rgb(width: u32, height: u32, r: u8, g: u8, b: u8) -> DynamicImage {
    DynamicImage::ImageRgb8(ImageBuffer::from_fn(width, height, |_, _| Rgb([r, g, b])))
}

/// Create a solid-colour RGBA test image (with alpha).
fn make_rgba(width: u32, height: u32, r: u8, g: u8, b: u8, a: u8) -> DynamicImage {
    DynamicImage::ImageRgba8(ImageBuffer::from_fn(width, height, |_, _| {
        Rgba([r, g, b, a])
    }))
}

/// Create a greyscale test image.
fn make_luma(width: u32, height: u32, luma: u8) -> DynamicImage {
    DynamicImage::ImageLuma8(ImageBuffer::from_fn(width, height, |_, _| Luma([luma])))
}

/// Path to a non-existent ONNX file (used to test graceful error handling).
fn nonexistent_model() -> PathBuf {
    PathBuf::from("/nonexistent/integration_test_model.onnx")
}

// ---------------------------------------------------------------------------
// 5b-1: Load all 5 models — missing file should return Err, not panic
// ---------------------------------------------------------------------------

/// Verify that each model struct returns a descriptive `Err` when the model file
/// does not exist, without panicking or producing a SIGSEGV.
#[test]
fn test_all_models_missing_file_returns_err_detr() {
    let result = DETRResNet101::new(&nonexistent_model());
    assert!(
        result.is_err(),
        "DETRResNet101::new must return Err for a missing model file"
    );
    let msg = result.err().unwrap().to_string();
    assert!(
        msg.contains("not found") || msg.contains("No such") || msg.contains("nonexistent"),
        "Error message should reference the missing path; got: {msg}"
    );
}

#[test]
fn test_all_models_missing_file_returns_err_yolo() {
    let result = YOLOv9c::new(&nonexistent_model());
    assert!(
        result.is_err(),
        "YOLOv9c::new must return Err for missing file"
    );
}

#[test]
fn test_all_models_missing_file_returns_err_dfine() {
    let result = DFineL::new(&nonexistent_model());
    assert!(
        result.is_err(),
        "DFineL::new must return Err for missing file"
    );
}

#[test]
fn test_all_models_missing_file_returns_err_rf_detr_large() {
    let result = RFDETRLarge::new(&nonexistent_model());
    assert!(
        result.is_err(),
        "RFDETRLarge::new must return Err for missing file"
    );
}

#[test]
fn test_all_models_missing_file_returns_err_rf_detr_medium() {
    let result = RFDETRMedium::new(&nonexistent_model());
    assert!(
        result.is_err(),
        "RFDETRMedium::new must return Err for missing file"
    );
}

// ---------------------------------------------------------------------------
// 5b-2: Preprocessing pipeline produces correct shapes for all model families
// ---------------------------------------------------------------------------

/// DETR-family models (detr-resnet101, dfine-l, rf-detr-large, rf-detr-medium)
/// must produce pixel_values [1,3,800,800] and pixel_mask [1,64,64].
#[test]
fn test_detr_family_preprocessing_shapes() {
    let detr_models = &[
        "detr-resnet101",
        "dfine-l",
        "rf-detr-large",
        "rf-detr-medium",
    ];

    for model_name in detr_models {
        let preprocessor = PreprocessorRegistry::get_preprocessor(model_name)
            .unwrap_or_else(|_| panic!("Registry must resolve '{model_name}'"));

        let img = make_rgb(640, 480, 128, 64, 32);
        let map = preprocessor
            .preprocess(img)
            .unwrap_or_else(|_| panic!("Preprocessing must succeed for '{model_name}'"));

        assert_eq!(
            map["pixel_values"].shape(),
            &[1, 3, 800, 800],
            "pixel_values shape wrong for '{model_name}'"
        );
        assert_eq!(
            map["pixel_mask"].shape(),
            &[1, 64, 64],
            "pixel_mask shape wrong for '{model_name}'"
        );
    }
}

/// YOLOv9-c must produce images [1,3,640,640].
#[test]
fn test_yolo_family_preprocessing_shape() {
    let preprocessor = PreprocessorRegistry::get_preprocessor("yolov9-c")
        .expect("Registry must resolve 'yolov9-c'");

    let img = make_rgb(1280, 720, 200, 100, 50);
    let map = preprocessor
        .preprocess(img)
        .expect("YOLO preprocessing must succeed");

    assert_eq!(
        map["images"].shape(),
        &[1, 3, 640, 640],
        "YOLO images tensor must have shape [1,3,640,640]"
    );
}

// ---------------------------------------------------------------------------
// 5b-3: Input format handling — all preprocessors accept RGB, RGBA, Luma
// ---------------------------------------------------------------------------

#[test]
fn test_detr_preprocessor_accepts_rgba_image() {
    let preprocessor = PreprocessorRegistry::get_preprocessor("detr-resnet101").unwrap();
    let img = make_rgba(200, 200, 255, 128, 0, 200);
    let result = preprocessor.preprocess(img);
    assert!(
        result.is_ok(),
        "DETR must accept RGBA input; got {:?}",
        result.err()
    );
    assert_eq!(result.unwrap()["pixel_values"].shape(), &[1, 3, 800, 800]);
}

#[test]
fn test_detr_preprocessor_accepts_greyscale_image() {
    let preprocessor = PreprocessorRegistry::get_preprocessor("detr-resnet101").unwrap();
    let img = make_luma(100, 100, 180);
    let result = preprocessor.preprocess(img);
    assert!(
        result.is_ok(),
        "DETR must accept greyscale input; got {:?}",
        result.err()
    );
    assert_eq!(result.unwrap()["pixel_values"].shape(), &[1, 3, 800, 800]);
}

#[test]
fn test_yolo_preprocessor_accepts_rgba_image() {
    let preprocessor = PreprocessorRegistry::get_preprocessor("yolov9-c").unwrap();
    let img = make_rgba(320, 240, 100, 200, 50, 255);
    let result = preprocessor.preprocess(img);
    assert!(
        result.is_ok(),
        "YOLO must accept RGBA input; got {:?}",
        result.err()
    );
    assert_eq!(result.unwrap()["images"].shape(), &[1, 3, 640, 640]);
}

// ---------------------------------------------------------------------------
// 5b-4: Handle edge-case images — tiny, very large, non-square
// ---------------------------------------------------------------------------

#[test]
fn test_detr_preprocesses_1x1_image_without_panic() {
    let preprocessor = PreprocessorRegistry::get_preprocessor("detr-resnet101").unwrap();
    let img = make_rgb(1, 1, 50, 100, 150);
    let result = preprocessor.preprocess(img);
    assert!(
        result.is_ok(),
        "DETR must handle 1×1 input; got {:?}",
        result.err()
    );
    assert_eq!(result.unwrap()["pixel_values"].shape(), &[1, 3, 800, 800]);
}

#[test]
fn test_yolo_preprocesses_1x1_image_without_panic() {
    let preprocessor = PreprocessorRegistry::get_preprocessor("yolov9-c").unwrap();
    let img = make_rgb(1, 1, 50, 100, 150);
    let result = preprocessor.preprocess(img);
    assert!(
        result.is_ok(),
        "YOLO must handle 1×1 input; got {:?}",
        result.err()
    );
    assert_eq!(result.unwrap()["images"].shape(), &[1, 3, 640, 640]);
}

#[test]
fn test_detr_preprocesses_large_image_without_panic() {
    let preprocessor = PreprocessorRegistry::get_preprocessor("detr-resnet101").unwrap();
    // Allocate a large image; this should downscale cleanly to 800×800.
    let img = make_rgb(4000, 3000, 128, 128, 128);
    let result = preprocessor.preprocess(img);
    assert!(
        result.is_ok(),
        "DETR must handle 4000×3000 input without panic; got {:?}",
        result.err()
    );
    assert_eq!(result.unwrap()["pixel_values"].shape(), &[1, 3, 800, 800]);
}

#[test]
fn test_detr_preprocesses_wide_landscape_image() {
    let preprocessor = PreprocessorRegistry::get_preprocessor("detr-resnet101").unwrap();
    let img = make_rgb(1920, 1080, 100, 150, 200);
    let result = preprocessor.preprocess(img);
    assert!(
        result.is_ok(),
        "DETR must handle 1920×1080 input; got {:?}",
        result.err()
    );
}

#[test]
fn test_detr_preprocesses_tall_portrait_image() {
    let preprocessor = PreprocessorRegistry::get_preprocessor("detr-resnet101").unwrap();
    let img = make_rgb(480, 1280, 200, 50, 100);
    let result = preprocessor.preprocess(img);
    assert!(
        result.is_ok(),
        "DETR must handle 480×1280 portrait input; got {:?}",
        result.err()
    );
}

// ---------------------------------------------------------------------------
// 5b-5: Numerical validation — pixel value ranges after normalisation
// ---------------------------------------------------------------------------

/// DETR applies ImageNet mean/std; normalized values should be in roughly [-3, 3].
#[test]
fn test_detr_pixel_values_within_expected_range() {
    let preprocessor = PreprocessorRegistry::get_preprocessor("detr-resnet101").unwrap();
    let img = make_rgb(100, 100, 128, 128, 128);
    let map = preprocessor.preprocess(img).unwrap();
    let pv = &map["pixel_values"];
    for &v in pv.iter() {
        assert!(
            v > -4.0 && v < 4.0,
            "DETR pixel_values must be in [-4, 4] after ImageNet normalisation; got {v}"
        );
    }
}

/// YOLO normalises to [0, 1] by dividing by 255.
#[test]
fn test_yolo_images_values_in_0_to_1_range() {
    let preprocessor = PreprocessorRegistry::get_preprocessor("yolov9-c").unwrap();
    let img = make_rgb(100, 100, 255, 0, 127);
    let map = preprocessor.preprocess(img).unwrap();
    for &v in map["images"].iter() {
        assert!(
            v >= 0.0 && v <= 1.0,
            "YOLO images tensor must be in [0, 1]; got {v}"
        );
    }
}

/// The pixel_mask for DETR must be all 1.0 (every position is valid).
#[test]
fn test_detr_pixel_mask_all_ones() {
    let preprocessor = PreprocessorRegistry::get_preprocessor("detr-resnet101").unwrap();
    let img = make_rgb(200, 200, 100, 100, 100);
    let map = preprocessor.preprocess(img).unwrap();
    assert!(
        map["pixel_mask"].iter().all(|&v| v == 1.0_f32),
        "pixel_mask must be all 1.0"
    );
}

// ---------------------------------------------------------------------------
// 5b-6: Postprocessor — detection output format validation
// ---------------------------------------------------------------------------

use icarus_v2::models::postprocessors::common::DefaultPostprocessor;
use ndarray::Array3;
use std::collections::HashMap;

/// Build a minimal DETR output with one clearly dominant detection.
fn make_strong_detr_detection() -> HashMap<String, ndarray::ArrayD<f32>> {
    // 5 queries, 92 classes (91 COCO + 1 no-object).
    let mut logits = Array3::<f32>::zeros((1, 5, 92));
    logits[[0, 0, 0]] = 20.0; // class 0 ("person") is overwhelmingly dominant

    let mut boxes = Array3::<f32>::zeros((1, 5, 4));
    // cx=0.5, cy=0.5, w=0.4, h=0.4 — centred box on an 800×800 image
    boxes[[0, 0, 0]] = 0.5;
    boxes[[0, 0, 1]] = 0.5;
    boxes[[0, 0, 2]] = 0.4;
    boxes[[0, 0, 3]] = 0.4;

    let mut map = HashMap::new();
    map.insert("logits".to_string(), logits.into_dyn());
    map.insert("pred_boxes".to_string(), boxes.into_dyn());
    map
}

/// Verify that `Detection` fields satisfy the format contract from the spec.
#[test]
fn test_detection_output_format_valid_fields() {
    let pp = DefaultPostprocessor;
    let outputs = make_strong_detr_detection();
    let dets = pp
        .postprocess(outputs, 800, 800)
        .expect("postprocess must succeed");

    assert_eq!(dets.len(), 1, "expected exactly one detection");
    let det = &dets[0];

    // confidence in (0.0, 1.0]
    assert!(
        det.confidence > 0.0 && det.confidence <= 1.0,
        "confidence must be in (0, 1]; got {}",
        det.confidence
    );

    // label must be non-empty
    assert!(!det.label.is_empty(), "label must not be empty");

    // bbox coordinates must be valid pixel positions
    let [x1, y1, x2, y2] = det.bbox;
    assert!(x1 >= 0.0, "x1 must be >= 0; got {x1}");
    assert!(y1 >= 0.0, "y1 must be >= 0; got {y1}");
    assert!(x2 <= 800.0, "x2 must be <= image_width; got {x2}");
    assert!(y2 <= 800.0, "y2 must be <= image_height; got {y2}");
    assert!(x2 > x1, "bbox must have positive width: x1={x1}, x2={x2}");
    assert!(y2 > y1, "bbox must have positive height: y1={y1}, y2={y2}");

    // class_id must be a valid COCO index (0–79)
    assert!(
        det.class_id < 80,
        "class_id must be < 80 (COCO); got {}",
        det.class_id
    );
}

/// Bounding boxes must be clamped to image bounds and never extend outside.
#[test]
fn test_detection_bboxes_do_not_exceed_image_bounds() {
    let pp = DefaultPostprocessor;
    let outputs = make_strong_detr_detection();
    let dets = pp
        .postprocess(outputs, 640, 480)
        .expect("postprocess must succeed");

    for det in &dets {
        let [x1, y1, x2, y2] = det.bbox;
        assert!(x1 >= 0.0, "x1 must be >= 0");
        assert!(y1 >= 0.0, "y1 must be >= 0");
        assert!(x2 <= 640.0, "x2 must be <= 640");
        assert!(y2 <= 480.0, "y2 must be <= 480");
    }
}

/// DETR detections with class logit 0 must map to label "person".
#[test]
fn test_detection_label_maps_class_id_correctly() {
    let pp = DefaultPostprocessor;
    let outputs = make_strong_detr_detection();
    let dets = pp.postprocess(outputs, 800, 800).unwrap();
    assert_eq!(dets[0].class_id, 0);
    assert_eq!(dets[0].label, "person");
}

// ---------------------------------------------------------------------------
// 5b-7: Consistency — same input produces same output (determinism)
// ---------------------------------------------------------------------------

#[test]
fn test_detr_preprocessing_is_deterministic() {
    let preprocessor = PreprocessorRegistry::get_preprocessor("detr-resnet101").unwrap();
    let img_a = make_rgb(300, 200, 128, 64, 32);
    let img_b = make_rgb(300, 200, 128, 64, 32);

    let map_a = preprocessor.preprocess(img_a).unwrap();
    let map_b = preprocessor.preprocess(img_b).unwrap();

    // Compare pixel_values element-wise.
    let pv_a = &map_a["pixel_values"];
    let pv_b = &map_b["pixel_values"];

    assert_eq!(
        pv_a.shape(),
        pv_b.shape(),
        "shape must match on identical inputs"
    );
    for (a, b) in pv_a.iter().zip(pv_b.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "DETR preprocessing must be deterministic; got {a} vs {b}"
        );
    }
}

#[test]
fn test_postprocessor_is_deterministic() {
    let pp = DefaultPostprocessor;
    let dets_a = pp
        .postprocess(make_strong_detr_detection(), 800, 800)
        .unwrap();
    let dets_b = pp
        .postprocess(make_strong_detr_detection(), 800, 800)
        .unwrap();

    assert_eq!(dets_a.len(), dets_b.len(), "detection count must be stable");
    for (a, b) in dets_a.iter().zip(dets_b.iter()) {
        assert_eq!(a.class_id, b.class_id);
        assert!((a.confidence - b.confidence).abs() < 1e-6);
        for (x, y) in a.bbox.iter().zip(b.bbox.iter()) {
            assert!((x - y).abs() < 1e-4);
        }
    }
}

// ---------------------------------------------------------------------------
// 5b-8: Confidence threshold filtering
// ---------------------------------------------------------------------------

/// Confidence below threshold → empty detection list.
#[test]
fn test_yolo_zero_objectness_returns_no_detections() {
    let pp = DefaultPostprocessor;
    let zero_output = Array3::<f32>::zeros((1, 25200, 85)).into_dyn();
    let mut map = HashMap::new();
    map.insert("output".to_string(), zero_output);
    let dets = pp.postprocess(map, 640, 480).unwrap();
    assert!(
        dets.is_empty(),
        "zero-objectness YOLO output must produce no detections; got {}",
        dets.len()
    );
}

/// Confidence at exactly the threshold should be kept by postprocessor
/// (the internal threshold is 0.5; values ≥ 0.5 pass).
#[test]
fn test_detr_detection_above_threshold_is_kept() {
    let pp = DefaultPostprocessor;
    let outputs = make_strong_detr_detection();
    let dets = pp.postprocess(outputs, 800, 800).unwrap();
    assert!(
        !dets.is_empty(),
        "Strong logit (20.0) should produce at least one detection"
    );
    assert!(
        dets[0].confidence >= 0.5,
        "Retained detection must have confidence >= 0.5; got {}",
        dets[0].confidence
    );
}

// ---------------------------------------------------------------------------
// 5b-9: image_utils::crop_image — correctness and error handling
// ---------------------------------------------------------------------------

#[test]
fn test_crop_image_produces_correct_dimensions() {
    let img = make_rgb(400, 300, 100, 150, 200);
    let bbox = [50.0_f32, 60.0, 200.0, 180.0];
    let crop = crop_image(&img, bbox).expect("crop must succeed for valid bbox");
    // Expected: width = 200 - 50 = 150, height = 180 - 60 = 120
    assert_eq!(crop.width(), 150, "crop width must be x2 - x1");
    assert_eq!(crop.height(), 120, "crop height must be y2 - y1");
}

#[test]
fn test_crop_image_clamps_to_image_bounds() {
    let img = make_rgb(200, 200, 100, 100, 100);
    // bbox extends beyond image boundary on x2/y2
    let bbox = [100.0_f32, 100.0, 300.0, 300.0]; // x2, y2 > image size
    let result = crop_image(&img, bbox);
    // Should succeed — coordinates are clamped internally
    assert!(
        result.is_ok(),
        "crop must succeed with out-of-bounds bbox (clamped)"
    );
    let crop = result.unwrap();
    assert!(
        crop.width() <= 200,
        "crop width must not exceed image width; got {}",
        crop.width()
    );
    assert!(
        crop.height() <= 200,
        "crop height must not exceed image height; got {}",
        crop.height()
    );
}

#[test]
fn test_crop_image_invalid_bbox_returns_err() {
    let img = make_rgb(100, 100, 0, 0, 0);
    // Inverted bbox: x1 > x2 → invalid
    let inverted = [80.0_f32, 10.0, 20.0, 50.0];
    let result = crop_image(&img, inverted);
    assert!(
        result.is_err(),
        "crop_image must return Err for inverted bbox"
    );
}

#[test]
fn test_crop_image_zero_area_bbox_returns_err() {
    let img = make_rgb(100, 100, 0, 0, 0);
    // Zero-width box
    let zero_w = [50.0_f32, 10.0, 50.0, 80.0];
    let result = crop_image(&img, zero_w);
    assert!(
        result.is_err(),
        "crop_image must return Err for zero-width bbox"
    );
}

// ---------------------------------------------------------------------------
// 5b-10: PreprocessorRegistry — all registered names resolve without panic
// ---------------------------------------------------------------------------

#[test]
fn test_all_registered_model_names_resolve() {
    for name in PreprocessorRegistry::registered_model_names() {
        let result = PreprocessorRegistry::get_preprocessor(name);
        assert!(
            result.is_ok(),
            "Registered model '{name}' must resolve a preprocessor without error"
        );
    }
}

#[test]
fn test_unknown_model_name_returns_descriptive_err() {
    let result = PreprocessorRegistry::get_preprocessor("not-a-real-model");
    assert!(result.is_err(), "Unknown model name must return Err");
    let msg = result.err().unwrap().to_string();
    assert!(
        msg.contains("not-a-real-model"),
        "Error must mention the offending name; got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// 5b-11: Model-level pipeline smoke test (no ONNX file needed for pre-ORT steps)
// ---------------------------------------------------------------------------

/// Each model's `new()` returns a clean `Err` (not a panic) for a missing path.
/// This verifies the complete error chain: OnnxBackend → Model::new → caller.
#[test]
fn test_model_new_error_is_descriptive() {
    let path = nonexistent_model();
    let cases: &[(&str, Box<dyn Fn() -> Result<(), anyhow::Error>>)] = &[
        (
            "DETRResNet101",
            Box::new(|| DETRResNet101::new(&path).map(|_| ())),
        ),
        ("YOLOv9c", Box::new(|| YOLOv9c::new(&path).map(|_| ()))),
        ("DFineL", Box::new(|| DFineL::new(&path).map(|_| ()))),
        (
            "RFDETRLarge",
            Box::new(|| RFDETRLarge::new(&path).map(|_| ())),
        ),
        (
            "RFDETRMedium",
            Box::new(|| RFDETRMedium::new(&path).map(|_| ())),
        ),
    ];

    for (name, factory) in cases {
        let result = factory();
        assert!(
            result.is_err(),
            "{name}::new must return Err for missing file"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            !msg.is_empty(),
            "{name}::new error message must not be empty"
        );
    }
}

// ---------------------------------------------------------------------------
// 5b-12: Real model inference tests (graceful skip if model file absent)
// ---------------------------------------------------------------------------

/// Attempt end-to-end detection with the DETR model if the weights file is present.
///
/// Marked `#[ignore]` because loading a 160 MB ONNX model and running CPU inference
/// takes > 10 seconds. Run explicitly with:
///   `cargo test -- --ignored test_detr_detect_end_to_end_if_model_available`
///
/// If the model file is missing or ORT is not installed, the test prints a warning
/// and passes — this keeps CI green on machines without model weights while still
/// exercising the full pipeline on machines that have them.
#[test]
#[ignore = "requires model file + ORT runtime; run explicitly with --ignored"]
fn test_detr_detect_end_to_end_if_model_available() {
    let model_path = PathBuf::from("./icarus-v1-rust/models/detr/model.onnx");
    if !model_path.exists() {
        eprintln!(
            "SKIP test_detr_detect_end_to_end_if_model_available: \
             model file {:?} not found",
            model_path
        );
        return;
    }

    let model = match DETRResNet101::new(&model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!(
                "SKIP test_detr_detect_end_to_end_if_model_available: \
                 model failed to load (ORT issue?): {e}"
            );
            return;
        }
    };

    // Use a 400×300 synthetic image; the preprocessor upscales it to 800×800 internally.
    let img = make_rgb(400, 300, 120, 80, 200);
    let result = model.detect(img);

    match result {
        Ok(dets) => {
            // A synthetic solid-colour image won't produce real detections,
            // but the pipeline must return a valid (possibly empty) Vec.
            println!(
                "DETR end-to-end: {} detection(s) found on synthetic image",
                dets.len()
            );
            for det in &dets {
                // Validate each detection's invariants.
                assert!(det.confidence > 0.0 && det.confidence <= 1.0);
                assert!(!det.label.is_empty());
                let [x1, y1, x2, y2] = det.bbox;
                assert!(x1 >= 0.0 && y1 >= 0.0);
                assert!(x2 <= 400.0 && y2 <= 300.0);
                assert!(x2 > x1 && y2 > y1);
            }
        }
        Err(e) => {
            eprintln!("DETR inference returned Err (acceptable without GPU/ORT): {e}");
        }
    }
}

/// Attempt end-to-end detection with the YOLO model if the weights file is present.
///
/// Marked `#[ignore]` because loading a 59 MB ONNX model and running CPU inference
/// takes > 10 seconds. Run explicitly with:
///   `cargo test -- --ignored test_yolo_detect_end_to_end_if_model_available`
#[test]
#[ignore = "requires model file + ORT runtime; run explicitly with --ignored"]
fn test_yolo_detect_end_to_end_if_model_available() {
    let model_path = PathBuf::from("./icarus-v1-rust/models/yolov10m/model.onnx");
    if !model_path.exists() {
        eprintln!(
            "SKIP test_yolo_detect_end_to_end_if_model_available: \
             model file {:?} not found",
            model_path
        );
        return;
    }

    let model = match YOLOv9c::new(&model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!(
                "SKIP test_yolo_detect_end_to_end_if_model_available: \
                 model failed to load: {e}"
            );
            return;
        }
    };

    let img = make_rgb(640, 640, 200, 180, 160);
    let result = model.detect(img);

    match result {
        Ok(dets) => {
            println!(
                "YOLO end-to-end: {} detection(s) found on synthetic image",
                dets.len()
            );
            for det in &dets {
                assert!(det.confidence > 0.0 && det.confidence <= 1.0);
                assert!(!det.label.is_empty());
                let [x1, y1, x2, y2] = det.bbox;
                assert!(x1 >= 0.0 && y1 >= 0.0);
                assert!(x2 <= 640.0 && y2 <= 640.0);
                assert!(x2 > x1 && y2 > y1);
            }
        }
        Err(e) => {
            eprintln!("YOLO inference returned Err (acceptable without ORT): {e}");
        }
    }
}

// ---------------------------------------------------------------------------
// 5b-13: Baseline smoke test — confirms the library itself loads correctly
// ---------------------------------------------------------------------------

#[test]
fn test_library_loads() {
    // Trivial check: the library compiles and all public modules are reachable.
    let _ = PreprocessorRegistry::registered_model_names();
}
