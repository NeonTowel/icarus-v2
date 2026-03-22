/// Integration tests for Icarus-v2 (Candle backend).
///
/// The ONNX-based preprocessing/postprocessor pipeline was removed in the Candle
/// migration (v2.0).  Integration-level concerns are now covered by the model-specific
/// test suites:
///
/// - `detr_candle_test.rs`  — DETR-ResNet50 shapes, preprocessing, postprocessing
/// - `rt_detr_candle_test.rs` — RT-DETR shapes, neck FPN, postprocessing
/// - `rf_detr_candle_test.rs` — RF-DETR stub / deferral verification
/// - `yolov9_candle_test.rs`  — YOLOv9-c shapes and inference
///
/// This file retains cross-cutting tests: image_utils, crop_image edge cases.
use icarus_v2::image_utils::crop_image;
use image::{DynamicImage, ImageBuffer, Rgb};

fn make_rgb(width: u32, height: u32, r: u8, g: u8, b: u8) -> DynamicImage {
    DynamicImage::ImageRgb8(ImageBuffer::from_fn(width, height, |_, _| Rgb([r, g, b])))
}

// ---------------------------------------------------------------------------
// image_utils::crop_image — correctness and error handling
// ---------------------------------------------------------------------------

#[test]
fn test_crop_image_produces_correct_dimensions() {
    let img = make_rgb(400, 300, 100, 150, 200);
    let bbox = [50.0_f32, 60.0, 200.0, 180.0];
    let crop = crop_image(&img, bbox).expect("crop must succeed for valid bbox");
    assert_eq!(crop.width(), 150, "crop width must be x2 - x1");
    assert_eq!(crop.height(), 120, "crop height must be y2 - y1");
}

#[test]
fn test_crop_image_clamps_to_image_bounds() {
    let img = make_rgb(200, 200, 100, 100, 100);
    let bbox = [100.0_f32, 100.0, 300.0, 300.0];
    let result = crop_image(&img, bbox);
    assert!(
        result.is_ok(),
        "crop must succeed with out-of-bounds bbox (clamped)"
    );
    let crop = result.unwrap();
    assert!(
        crop.width() <= 200,
        "crop width must not exceed image width"
    );
    assert!(
        crop.height() <= 200,
        "crop height must not exceed image height"
    );
}

#[test]
fn test_crop_image_invalid_bbox_returns_err() {
    let img = make_rgb(100, 100, 0, 0, 0);
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
    let zero_w = [50.0_f32, 10.0, 50.0, 80.0];
    let result = crop_image(&img, zero_w);
    assert!(
        result.is_err(),
        "crop_image must return Err for zero-width bbox"
    );
}

// ---------------------------------------------------------------------------
// Library smoke test
// ---------------------------------------------------------------------------

#[test]
fn test_library_loads() {
    // Verify the library compiles and public types are accessible.
    let _: Option<icarus_v2::models::Detection> = None;
}
