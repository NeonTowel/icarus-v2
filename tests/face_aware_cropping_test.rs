/// Integration tests for the Face-Aware Crop Adjustment module.
///
/// Tests the complete pipeline from the design document's Test 7 (full pipeline
/// integration), plus edge case coverage for the new modular API.
///
/// All tests use mock BBox data (no real images, no model inference).
use icarus_v2::config::{ArtisticCropConfig, ArtisticMode};
use icarus_v2::face_aware_cropping::{apply_face_aware_adjustment, find_dominant_face};
use icarus_v2::multi_format_cropping::{
    calculate_portrait_9_16_crop, calculate_portrait_9_21_crop, BBox, CropConfig, CropRegion,
};

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

fn make_crop(x: f32, y: f32, w: f32, h: f32) -> CropRegion {
    CropRegion {
        x,
        y,
        width: w,
        height: h,
    }
}

fn make_bbox(x1: f32, y1: f32, x2: f32, y2: f32) -> BBox {
    BBox { x1, y1, x2, y2 }
}

// ---------------------------------------------------------------------------
// Test 7: Full Pipeline Integration (design doc §Testing Strategy)
// ---------------------------------------------------------------------------

/// Full pipeline: person detection → original crop algorithm → face-aware adjustment.
///
/// Uses a 3024×4032 portrait photo (typical phone photo dimensions).
/// The face must be inside the adjusted crop, and dimensions must not change.
#[test]
fn test_full_pipeline_person_face_crop_adjustment() {
    let person = make_bbox(800.0, 200.0, 2200.0, 3800.0);
    let faces = vec![make_bbox(1200.0, 300.0, 1600.0, 700.0)];
    let crop_config = CropConfig::default();
    let artistic_config = ArtisticCropConfig::default();

    // Step 1: Original algorithm calculates crop (9:16 portrait format).
    let original = calculate_portrait_9_16_crop(3024, 4032, &person, &crop_config);
    assert!(
        original.is_some(),
        "9:16 crop must be calculable for tall portrait person"
    );
    let original_crop = original.unwrap();

    // Step 2: Face-aware adjustment.
    let adjusted = apply_face_aware_adjustment(
        &original_crop,
        Some(&person),
        &faces,
        &artistic_config,
        3024,
        4032,
    );

    // Step 3: Verify face is inside adjusted crop.
    let face = &faces[0];
    let crop_x2 = adjusted.x + adjusted.width;
    let crop_y2 = adjusted.y + adjusted.height;
    assert!(
        face.x1 >= adjusted.x && face.x2 <= crop_x2,
        "face must be horizontally inside crop: face=[{:.0},{:.0}], crop_x=[{:.0},{:.0}]",
        face.x1,
        face.x2,
        adjusted.x,
        crop_x2
    );
    assert!(
        face.y1 >= adjusted.y && face.y2 <= crop_y2,
        "face must be vertically inside crop: face=[{:.0},{:.0}], crop_y=[{:.0},{:.0}]",
        face.y1,
        face.y2,
        adjusted.y,
        crop_y2
    );

    // Step 4: Verify crop dimensions unchanged (only position may have moved).
    assert_eq!(adjusted.width, original_crop.width, "width must not change");
    assert_eq!(
        adjusted.height, original_crop.height,
        "height must not change"
    );
}

/// Full pipeline with 9:21 format (taller mobile format).
#[test]
fn test_full_pipeline_9_21_format() {
    let person = make_bbox(800.0, 200.0, 2200.0, 3800.0);
    let faces = vec![make_bbox(1300.0, 400.0, 1700.0, 800.0)];
    let crop_config = CropConfig::default();
    let artistic_config = ArtisticCropConfig::from_mode(ArtisticMode::Conservative);

    let original = calculate_portrait_9_21_crop(3024, 4032, &person, &crop_config);
    assert!(original.is_some());
    let original_crop = original.unwrap();

    let adjusted = apply_face_aware_adjustment(
        &original_crop,
        Some(&person),
        &faces,
        &artistic_config,
        3024,
        4032,
    );

    // Dimensions must not change.
    assert_eq!(adjusted.width, original_crop.width);
    assert_eq!(adjusted.height, original_crop.height);
}

// ---------------------------------------------------------------------------
// New module API tests (face_aware_cropping module)
// ---------------------------------------------------------------------------

/// When there are no faces, the original crop is returned unchanged.
#[test]
fn test_no_faces_returns_original_unchanged() {
    let crop = make_crop(100.0, 100.0, 800.0, 600.0);
    let person = make_bbox(200.0, 150.0, 600.0, 550.0);
    let config = ArtisticCropConfig::default();
    let result = apply_face_aware_adjustment(&crop, Some(&person), &[], &config, 1920, 1080);
    assert_eq!(result, crop);
}

/// When there is no person and no faces, the original crop is returned.
#[test]
fn test_no_person_no_faces_returns_original() {
    let crop = make_crop(100.0, 100.0, 800.0, 600.0);
    let config = ArtisticCropConfig::default();
    let result = apply_face_aware_adjustment(&crop, None, &[], &config, 1920, 1080);
    assert_eq!(result, crop);
}

/// Face is already inside the crop — returns original (no spurious shift).
#[test]
fn test_face_inside_crop_returns_original() {
    let crop = make_crop(100.0, 100.0, 800.0, 600.0); // x: 100..900, y: 100..700
    let person = make_bbox(200.0, 150.0, 600.0, 550.0);
    // Face is well inside the crop (accounts for 15px safety margin).
    let faces = vec![make_bbox(350.0, 250.0, 450.0, 350.0)];
    let config = ArtisticCropConfig::from_mode(ArtisticMode::Balanced); // 15px margin
    let result = apply_face_aware_adjustment(&crop, Some(&person), &faces, &config, 1920, 1080);
    // Face zone: x=[335..465], y=[235..365]. Crop covers x=[100..900], y=[100..700].
    // Zone is inside crop → no shift needed.
    assert_eq!(result.x, crop.x, "no shift when face is already inside");
    assert_eq!(result.y, crop.y, "no shift when face is already inside");
}

/// Face is outside the crop by a small amount within budget — crop shifts.
#[test]
fn test_face_outside_crop_shifts_to_include() {
    // Crop at x=500, width=800 → crop_x2=1300.
    // Face at x1=460, x2=560. With 15px margin: zone_x1=445.
    // Required shift: 445 - 500 = -55px. Budget = 800*0.1=80px. -55 < 80 → OK.
    let crop = make_crop(500.0, 100.0, 800.0, 600.0);
    let person = make_bbox(300.0, 100.0, 1100.0, 650.0);
    let faces = vec![make_bbox(460.0, 200.0, 560.0, 300.0)];
    let config = ArtisticCropConfig::from_mode(ArtisticMode::Balanced);
    let result = apply_face_aware_adjustment(&crop, Some(&person), &faces, &config, 1920, 1080);
    // Crop should have shifted left.
    assert!(result.x < crop.x, "crop should shift left to include face");
    // Face must now be inside.
    let zone_x1 = 460.0 - 15.0; // face x1 minus margin
    assert!(
        result.x <= zone_x1,
        "crop.x={} should be ≤ zone_x1={}",
        result.x,
        zone_x1
    );
    // Dimensions unchanged.
    assert_eq!(result.width, crop.width);
    assert_eq!(result.height, crop.height);
}

/// Face requires shift that exceeds budget → return original.
#[test]
fn test_excessive_shift_budget_returns_original() {
    let crop = make_crop(0.0, 0.0, 800.0, 600.0);
    // Face at x1=1500, far outside crop x2=800. With 15px margin: zone_x2=1515+15=1515.
    // Required shift: 1515 - 800 = 715px. Budget = 800*0.1=80px → exceeds budget.
    let faces = vec![make_bbox(1500.0, 300.0, 1600.0, 400.0)];
    let config = ArtisticCropConfig::from_mode(ArtisticMode::Balanced);
    let result = apply_face_aware_adjustment(&crop, None, &faces, &config, 1920, 1080);
    assert_eq!(result.x, crop.x, "shift budget exceeded → return original");
    assert_eq!(result.y, crop.y, "shift budget exceeded → return original");
}

// ---------------------------------------------------------------------------
// find_dominant_face tests
// ---------------------------------------------------------------------------

/// Empty face list returns None.
#[test]
fn test_find_dominant_face_empty() {
    assert!(find_dominant_face(&[], 1920, 1080).is_none());
}

/// Single face is always the dominant face.
#[test]
fn test_find_dominant_face_single() {
    let faces = vec![make_bbox(100.0, 100.0, 200.0, 200.0)];
    let result = find_dominant_face(&faces, 1920, 1080);
    assert!(result.is_some());
    assert_eq!(result.unwrap().x1, 100.0);
}

/// The face closest to the image center is selected.
#[test]
fn test_find_dominant_face_selects_most_central() {
    // Image center: (500, 500).
    // Face 1 at (50, 50) — distance from center: sqrt(450^2 + 450^2) ≈ 636.
    // Face 2 at (480, 480)-(520, 520), centroid (500, 500) — distance: 0.
    let faces = vec![
        make_bbox(0.0, 0.0, 100.0, 100.0),     // centroid (50, 50) — far
        make_bbox(480.0, 480.0, 520.0, 520.0), // centroid (500, 500) ← at center
    ];
    let dominant = find_dominant_face(&faces, 1000, 1000).unwrap();
    assert!(
        (dominant.center_x() - 500.0).abs() < 1.0,
        "expected centroid near 500, got {}",
        dominant.center_x()
    );
}

// ---------------------------------------------------------------------------
// Config mode mapping tests (integration)
// ---------------------------------------------------------------------------

/// Conservative mode has larger safety margin.
#[test]
fn test_conservative_mode_uses_larger_margin() {
    let conservative = ArtisticCropConfig::from_mode(ArtisticMode::Conservative);
    let balanced = ArtisticCropConfig::from_mode(ArtisticMode::Balanced);
    let aggressive = ArtisticCropConfig::from_mode(ArtisticMode::Aggressive);

    assert!(
        conservative.face_safety_margin_px > balanced.face_safety_margin_px,
        "conservative margin {} > balanced margin {}",
        conservative.face_safety_margin_px,
        balanced.face_safety_margin_px
    );
    assert!(
        balanced.face_safety_margin_px > aggressive.face_safety_margin_px,
        "balanced margin {} > aggressive margin {}",
        balanced.face_safety_margin_px,
        aggressive.face_safety_margin_px
    );
}

/// Default config is Balanced mode.
#[test]
fn test_default_config_is_balanced() {
    let default = ArtisticCropConfig::default();
    assert_eq!(default.artistic_mode, ArtisticMode::Balanced);
    assert_eq!(default.face_safety_margin_px, 15);
    assert!((default.max_shift_fraction - 0.10).abs() < 0.001);
}

/// All modes use the same shift fraction (10%).
#[test]
fn test_all_modes_same_shift_fraction() {
    for mode in [
        ArtisticMode::Conservative,
        ArtisticMode::Balanced,
        ArtisticMode::Aggressive,
    ] {
        let config = ArtisticCropConfig::from_mode(mode.clone());
        assert!(
            (config.max_shift_fraction - 0.10).abs() < 0.001,
            "{:?} should have 10% shift fraction",
            mode
        );
    }
}
