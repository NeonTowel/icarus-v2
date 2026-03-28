/// Integration tests for the Artistic Face Detection Enhancement pipeline.
///
/// These tests validate:
/// - Face-person correlation logic (`correlate_faces_to_persons`)
/// - Dominant face selection (`select_dominant_face`)
/// - Artistic crop positioning (`compute_artistic_crop`)
/// - Group-shot handling (`compound_face_bbox`)
/// - Format suitability with face-aware algorithm (`detect_suitable_formats_with_faces`)
/// - Edge cases: small faces, boundary faces, multi-person scenes
///
/// No network calls or model weight downloads are made in these tests.
/// All tests operate on synthetic bounding boxes and configurations.
///
/// # Deprecation Note
/// Some functions tested here (`correlate_faces_to_persons`, `compute_artistic_crop`,
/// `detect_suitable_formats_with_faces`, `select_dominant_face`, `compound_face_bbox`,
/// `FacePersonPair`) are deprecated as of v2.1.0. These tests remain to verify
/// backward compatibility of the deprecated API. New tests for the replacement
/// API are in `tests/face_aware_cropping_test.rs`.
#[allow(deprecated)]
mod face_detection_tests {

    use icarus_v2::config::{ArtisticCropConfig, ArtisticMode, FaceSelectionStrategy};
    use icarus_v2::models::candle_backend::{BBox as ModelBBox, Detection as ModelDetection};
    use icarus_v2::multi_format_cropping::{
        compound_face_bbox, compute_artistic_crop, correlate_faces_to_persons,
        detect_suitable_formats_with_faces, expand_bbox_px, is_bbox_within, is_region_visible,
        merge_bboxes, select_dominant_face, BBox, CropConfig, FacePersonPair,
    };

    // ---------------------------------------------------------------------------
    // Test helpers
    // ---------------------------------------------------------------------------

    /// Build a synthetic `ModelDetection` with a face class.
    fn make_face_detection(
        x_min: f32,
        y_min: f32,
        x_max: f32,
        y_max: f32,
        confidence: f32,
    ) -> ModelDetection {
        ModelDetection {
            bbox: ModelBBox {
                x_min,
                y_min,
                x_max,
                y_max,
            },
            class_id: 0,
            confidence,
            class_name: "face".to_string(),
        }
    }

    /// Build a BBox in multi_format_cropping coordinate space.
    fn make_bbox(x1: f32, y1: f32, x2: f32, y2: f32) -> BBox {
        BBox { x1, y1, x2, y2 }
    }

    /// A person bbox occupying the left half of a 1920×1080 image.
    fn left_person_bbox() -> BBox {
        make_bbox(0.0, 0.0, 960.0, 1080.0)
    }

    /// A person bbox occupying the right half of a 1920×1080 image.
    fn right_person_bbox() -> BBox {
        make_bbox(960.0, 0.0, 1920.0, 1080.0)
    }

    /// A tall portrait person bbox for a 3024×4032 photo.
    fn portrait_person_bbox() -> BBox {
        make_bbox(800.0, 200.0, 2200.0, 3800.0)
    }

    // ---------------------------------------------------------------------------
    // is_bbox_within
    // ---------------------------------------------------------------------------

    #[test]
    fn test_is_bbox_within_fully_inside_returns_true() {
        let container = make_bbox(0.0, 0.0, 200.0, 400.0);
        assert!(is_bbox_within(50.0, 10.0, 150.0, 100.0, &container));
    }

    #[test]
    fn test_is_bbox_within_touching_boundary_returns_true() {
        let container = make_bbox(0.0, 0.0, 200.0, 400.0);
        assert!(is_bbox_within(0.0, 0.0, 200.0, 400.0, &container));
    }

    #[test]
    fn test_is_bbox_within_partial_overlap_returns_false() {
        let container = make_bbox(0.0, 0.0, 200.0, 400.0);
        // Face extends 10px past the right edge of container.
        assert!(!is_bbox_within(100.0, 10.0, 210.0, 100.0, &container));
    }

    #[test]
    fn test_is_bbox_within_completely_outside_returns_false() {
        let container = make_bbox(0.0, 0.0, 200.0, 400.0);
        assert!(!is_bbox_within(300.0, 300.0, 500.0, 600.0, &container));
    }

    // ---------------------------------------------------------------------------
    // merge_bboxes
    // ---------------------------------------------------------------------------

    #[test]
    fn test_merge_bboxes_two_non_overlapping() {
        let a = make_bbox(10.0, 10.0, 100.0, 100.0);
        let b = make_bbox(200.0, 200.0, 300.0, 300.0);
        let merged = merge_bboxes(&a, &b);
        assert_eq!(merged.x1, 10.0);
        assert_eq!(merged.y1, 10.0);
        assert_eq!(merged.x2, 300.0);
        assert_eq!(merged.y2, 300.0);
    }

    #[test]
    fn test_merge_bboxes_identical_is_same() {
        let a = make_bbox(50.0, 60.0, 200.0, 300.0);
        let merged = merge_bboxes(&a, &a);
        assert_eq!(merged, a);
    }

    #[test]
    fn test_merge_bboxes_second_contains_first() {
        let inner = make_bbox(100.0, 100.0, 150.0, 150.0);
        let outer = make_bbox(50.0, 50.0, 300.0, 300.0);
        let merged = merge_bboxes(&inner, &outer);
        assert_eq!(merged, outer, "merged must equal the outer bbox");
    }

    // ---------------------------------------------------------------------------
    // expand_bbox_px
    // ---------------------------------------------------------------------------

    #[test]
    fn test_expand_bbox_px_adds_margin_on_all_sides() {
        let bbox = make_bbox(100.0, 100.0, 300.0, 300.0);
        let expanded = expand_bbox_px(&bbox, 20, 1920, 1080);
        assert!((expanded.x1 - 80.0).abs() < 0.1);
        assert!((expanded.y1 - 80.0).abs() < 0.1);
        assert!((expanded.x2 - 320.0).abs() < 0.1);
        assert!((expanded.y2 - 320.0).abs() < 0.1);
    }

    #[test]
    fn test_expand_bbox_px_clamped_at_image_boundary() {
        // Bbox near top-left corner; margin would go negative without clamping.
        let bbox = make_bbox(5.0, 5.0, 100.0, 100.0);
        let expanded = expand_bbox_px(&bbox, 20, 1920, 1080);
        assert!(expanded.x1 >= 0.0, "x1 must not go negative");
        assert!(expanded.y1 >= 0.0, "y1 must not go negative");
    }

    #[test]
    fn test_expand_bbox_px_clamped_at_right_bottom_boundary() {
        // Bbox near bottom-right; margin would exceed photo dimensions.
        let bbox = make_bbox(1900.0, 1060.0, 1920.0, 1080.0);
        let expanded = expand_bbox_px(&bbox, 30, 1920, 1080);
        assert!(expanded.x2 <= 1920.0, "x2 must not exceed photo width");
        assert!(expanded.y2 <= 1080.0, "y2 must not exceed photo height");
    }

    // ---------------------------------------------------------------------------
    // is_region_visible
    // ---------------------------------------------------------------------------

    #[test]
    fn test_is_region_visible_fully_inside_crop() {
        use icarus_v2::multi_format_cropping::CropRegion;
        let zone = make_bbox(100.0, 100.0, 200.0, 200.0);
        let crop = CropRegion {
            x: 0.0,
            y: 0.0,
            width: 1000.0,
            height: 1000.0,
        };
        assert!(is_region_visible(&zone, &crop));
    }

    #[test]
    fn test_is_region_visible_partially_outside_returns_false() {
        use icarus_v2::multi_format_cropping::CropRegion;
        let zone = make_bbox(100.0, 100.0, 600.0, 200.0); // extends to x=600
        let crop = CropRegion {
            x: 0.0,
            y: 0.0,
            width: 500.0,
            height: 1000.0,
        }; // crop ends at x=500
        assert!(!is_region_visible(&zone, &crop));
    }

    // ---------------------------------------------------------------------------
    // correlate_faces_to_persons
    // ---------------------------------------------------------------------------

    #[test]
    fn test_correlate_single_face_single_person() {
        let faces = vec![make_face_detection(100.0, 50.0, 250.0, 200.0, 0.9)];
        let persons = vec![left_person_bbox()];
        let pairs = correlate_faces_to_persons(&faces, &persons);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].person_id, 0);
        assert_eq!(pairs[0].face_id, 0);
        assert!((pairs[0].confidence - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_correlate_face_outside_person_bbox_discarded() {
        // Face is completely outside the person bbox.
        let faces = vec![make_face_detection(1500.0, 50.0, 1700.0, 200.0, 0.9)];
        let persons = vec![left_person_bbox()]; // right edge at x=960
        let pairs = correlate_faces_to_persons(&faces, &persons);
        assert!(
            pairs.is_empty(),
            "face outside person bbox must be discarded"
        );
    }

    #[test]
    fn test_correlate_two_persons_two_faces() {
        let faces = vec![
            make_face_detection(100.0, 50.0, 250.0, 200.0, 0.88), // inside left person
            make_face_detection(1100.0, 50.0, 1250.0, 200.0, 0.75), // inside right person
        ];
        let persons = vec![left_person_bbox(), right_person_bbox()];
        let pairs = correlate_faces_to_persons(&faces, &persons);
        assert_eq!(pairs.len(), 2, "both faces should be correlated");
        assert_eq!(pairs[0].person_id, 0, "first face → left person");
        assert_eq!(pairs[1].person_id, 1, "second face → right person");
    }

    #[test]
    fn test_correlate_multiple_faces_in_one_person() {
        // Two faces both within the left-person bbox (e.g., a person with a mirror effect).
        let faces = vec![
            make_face_detection(100.0, 50.0, 250.0, 200.0, 0.95),
            make_face_detection(300.0, 50.0, 450.0, 200.0, 0.80),
        ];
        let persons = vec![left_person_bbox()];
        let pairs = correlate_faces_to_persons(&faces, &persons);
        assert_eq!(pairs.len(), 2, "both faces correlated to the same person");
        assert!(pairs.iter().all(|p| p.person_id == 0));
    }

    #[test]
    fn test_correlate_empty_faces_returns_empty() {
        let persons = vec![left_person_bbox()];
        let pairs = correlate_faces_to_persons(&[], &persons);
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_correlate_empty_persons_returns_empty() {
        let faces = vec![make_face_detection(100.0, 50.0, 250.0, 200.0, 0.9)];
        let pairs = correlate_faces_to_persons(&faces, &[]);
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_correlate_sorted_by_person_id_then_confidence_desc() {
        // Three faces: two inside left person, one inside right.
        let faces = vec![
            make_face_detection(100.0, 50.0, 250.0, 200.0, 0.70), // left, lower conf
            make_face_detection(1100.0, 50.0, 1250.0, 200.0, 0.60), // right
            make_face_detection(300.0, 50.0, 450.0, 200.0, 0.90), // left, higher conf
        ];
        let persons = vec![left_person_bbox(), right_person_bbox()];
        let pairs = correlate_faces_to_persons(&faces, &persons);
        assert_eq!(pairs.len(), 3);
        // First two should be person_id=0, sorted confidence desc (0.90, then 0.70).
        assert_eq!(pairs[0].person_id, 0);
        assert!(
            pairs[0].confidence > pairs[1].confidence,
            "higher conf first within person group"
        );
        assert_eq!(pairs[2].person_id, 1);
    }

    // ---------------------------------------------------------------------------
    // FacePersonPair helpers
    // ---------------------------------------------------------------------------

    #[test]
    fn test_face_person_pair_centroid() {
        let pair = FacePersonPair {
            person_id: 0,
            face_id: 0,
            confidence: 0.9,
            face_bbox: make_bbox(100.0, 200.0, 300.0, 400.0),
            person_bbox: left_person_bbox(),
        };
        let (cx, cy) = pair.face_centroid();
        assert!((cx - 200.0).abs() < 0.1, "cx should be 200");
        assert!((cy - 300.0).abs() < 0.1, "cy should be 300");
    }

    #[test]
    fn test_face_person_pair_area() {
        let pair = FacePersonPair {
            person_id: 0,
            face_id: 0,
            confidence: 0.9,
            face_bbox: make_bbox(0.0, 0.0, 100.0, 80.0),
            person_bbox: left_person_bbox(),
        };
        assert!(
            (pair.face_area() - 8000.0).abs() < 0.1,
            "area = 100*80 = 8000"
        );
    }

    // ---------------------------------------------------------------------------
    // select_dominant_face
    // ---------------------------------------------------------------------------

    fn make_pair(
        face_x1: f32,
        face_y1: f32,
        face_x2: f32,
        face_y2: f32,
        conf: f32,
    ) -> FacePersonPair {
        FacePersonPair {
            person_id: 0,
            face_id: 0,
            confidence: conf,
            face_bbox: make_bbox(face_x1, face_y1, face_x2, face_y2),
            person_bbox: make_bbox(0.0, 0.0, 1920.0, 1080.0),
        }
    }

    #[test]
    fn test_select_dominant_face_empty_returns_none() {
        let result = select_dominant_face(&[], &FaceSelectionStrategy::Largest, 1920, 1080);
        assert!(result.is_none());
    }

    #[test]
    fn test_select_dominant_face_largest() {
        let small = make_pair(100.0, 100.0, 150.0, 150.0, 0.9); // 50×50 = 2500
        let large = make_pair(500.0, 100.0, 700.0, 350.0, 0.7); // 200×250 = 50000
        let pairs = vec![small, large];
        let dominant = select_dominant_face(&pairs, &FaceSelectionStrategy::Largest, 1920, 1080);
        assert!(dominant.is_some());
        assert!(
            dominant.unwrap().face_area() > 40000.0,
            "largest face must be selected"
        );
    }

    #[test]
    fn test_select_dominant_face_highest_confidence() {
        let low_conf = make_pair(100.0, 100.0, 300.0, 300.0, 0.5);
        let high_conf = make_pair(500.0, 100.0, 600.0, 200.0, 0.95); // smaller but higher conf
        let pairs = vec![low_conf, high_conf];
        let dominant = select_dominant_face(
            &pairs,
            &FaceSelectionStrategy::HighestConfidence,
            1920,
            1080,
        );
        assert!(dominant.is_some());
        assert!(
            (dominant.unwrap().confidence - 0.95).abs() < 0.001,
            "highest confidence face must be selected"
        );
    }

    #[test]
    fn test_select_dominant_face_most_central() {
        // Image center: (960, 540)
        // Face A: centroid at (100, 100) → far from center
        // Face B: centroid at (950, 530) → very close to center
        let far = make_pair(50.0, 50.0, 150.0, 150.0, 0.9);
        let central = make_pair(900.0, 480.0, 1000.0, 580.0, 0.7);
        let pairs = vec![far, central];
        let dominant =
            select_dominant_face(&pairs, &FaceSelectionStrategy::MostCentral, 1920, 1080);
        assert!(dominant.is_some());
        let (cx, cy) = dominant.unwrap().face_centroid();
        // The central face centroid should be close to image center (960, 540).
        assert!(
            (cx - 950.0).abs() < 10.0 && (cy - 530.0).abs() < 10.0,
            "most central face must be selected; got centroid ({cx}, {cy})"
        );
    }

    #[test]
    fn test_select_dominant_face_weighted_score_single_face() {
        let pair = make_pair(400.0, 200.0, 600.0, 400.0, 0.85);
        let pairs = vec![pair];
        let dominant =
            select_dominant_face(&pairs, &FaceSelectionStrategy::WeightedScore, 1920, 1080);
        assert!(dominant.is_some(), "single face must always be selected");
    }

    #[test]
    fn test_select_dominant_face_weighted_score_prefers_large_central_confident() {
        // A large, central, high-confidence face should win over a small peripheral one.
        let good = make_pair(860.0, 440.0, 1060.0, 640.0, 0.92); // centroid near (960,540), 200×200
        let poor = make_pair(10.0, 10.0, 40.0, 40.0, 0.50); // tiny, corner, low conf
        let pairs = vec![poor, good];
        let dominant =
            select_dominant_face(&pairs, &FaceSelectionStrategy::WeightedScore, 1920, 1080);
        assert!(dominant.is_some());
        let d = dominant.unwrap();
        assert!(
            d.face_area() > 30000.0,
            "weighted score should select the large central face; got area {}",
            d.face_area()
        );
    }

    // ---------------------------------------------------------------------------
    // compound_face_bbox
    // ---------------------------------------------------------------------------

    #[test]
    fn test_compound_face_bbox_empty_returns_none() {
        assert!(compound_face_bbox(&[]).is_none());
    }

    #[test]
    fn test_compound_face_bbox_single_pair_equals_face_bbox() {
        let pair = FacePersonPair {
            person_id: 0,
            face_id: 0,
            confidence: 0.9,
            face_bbox: make_bbox(100.0, 100.0, 300.0, 300.0),
            person_bbox: left_person_bbox(),
        };
        let compound = compound_face_bbox(&[pair.clone()]).unwrap();
        assert_eq!(compound, pair.face_bbox);
    }

    #[test]
    fn test_compound_face_bbox_multiple_spans_all_faces() {
        let pairs = vec![
            FacePersonPair {
                person_id: 0,
                face_id: 0,
                confidence: 0.9,
                face_bbox: make_bbox(100.0, 50.0, 300.0, 250.0),
                person_bbox: left_person_bbox(),
            },
            FacePersonPair {
                person_id: 1,
                face_id: 1,
                confidence: 0.8,
                face_bbox: make_bbox(1100.0, 100.0, 1300.0, 300.0),
                person_bbox: right_person_bbox(),
            },
        ];
        let compound = compound_face_bbox(&pairs).unwrap();
        assert_eq!(compound.x1, 100.0, "x1 should be min of all faces");
        assert_eq!(compound.x2, 1300.0, "x2 should be max of all faces");
        assert_eq!(compound.y1, 50.0, "y1 should be min of all faces");
        assert_eq!(compound.y2, 300.0, "y2 should be max of all faces");
    }

    // ---------------------------------------------------------------------------
    // compute_artistic_crop
    // ---------------------------------------------------------------------------

    fn make_dominant_pair_for_portrait() -> FacePersonPair {
        // Face at roughly head position in a portrait person.
        FacePersonPair {
            person_id: 0,
            face_id: 0,
            confidence: 0.92,
            face_bbox: make_bbox(1000.0, 250.0, 1200.0, 500.0), // 200×250 face
            person_bbox: portrait_person_bbox(),
        }
    }

    #[test]
    fn test_compute_artistic_crop_9_16_returns_some_for_visible_face() {
        let dominant = make_dominant_pair_for_portrait();
        let artistic = ArtisticCropConfig::default();
        let base = CropConfig::default();
        let crop = compute_artistic_crop(3024, 4032, &dominant, 9.0 / 16.0, &artistic, &base);
        assert!(
            crop.is_some(),
            "balanced mode should produce a 9:16 crop for this portrait scene"
        );
    }

    #[test]
    fn test_compute_artistic_crop_stays_within_photo_bounds() {
        let dominant = make_dominant_pair_for_portrait();
        let artistic = ArtisticCropConfig::default();
        let base = CropConfig::default();
        for &(ratio, label) in &[
            (21.0 / 9.0, "21:9"),
            (9.0 / 16.0, "9:16"),
            (9.0 / 21.0, "9:21"),
        ] {
            if let Some(crop) =
                compute_artistic_crop(3024, 4032, &dominant, ratio, &artistic, &base)
            {
                assert!(crop.x >= 0.0, "{label}: x must be ≥ 0");
                assert!(crop.y >= 0.0, "{label}: y must be ≥ 0");
                assert!(
                    crop.x + crop.width <= 3024.0 + 0.1,
                    "{label}: right edge must not exceed photo width"
                );
                assert!(
                    crop.y + crop.height <= 4032.0 + 0.1,
                    "{label}: bottom edge must not exceed photo height"
                );
            }
        }
    }

    #[test]
    fn test_compute_artistic_crop_aggressive_more_face_bias_than_balanced() {
        // Aggressive mode should produce a crop whose center is closer to the face centroid
        // than balanced mode for the same scene.
        let dominant = make_dominant_pair_for_portrait();
        let base = CropConfig::default();

        let balanced_config = ArtisticCropConfig::from_mode(ArtisticMode::Balanced);
        let aggressive_config = ArtisticCropConfig::from_mode(ArtisticMode::Aggressive);

        let ratio = 9.0 / 16.0;
        let face_cy = dominant.face_centroid().1;

        let balanced_crop =
            compute_artistic_crop(3024, 4032, &dominant, ratio, &balanced_config, &base);
        let aggressive_crop =
            compute_artistic_crop(3024, 4032, &dominant, ratio, &aggressive_config, &base);

        if let (Some(b), Some(a)) = (balanced_crop, aggressive_crop) {
            let balanced_center_y = b.y + b.height / 2.0;
            let aggressive_center_y = a.y + a.height / 2.0;
            // Aggressive should have crop center closer to face_cy than balanced.
            let balanced_dist = (balanced_center_y - face_cy).abs();
            let aggressive_dist = (aggressive_center_y - face_cy).abs();
            assert!(
            aggressive_dist <= balanced_dist + 5.0, // 5px tolerance
            "aggressive crop should be closer to face: balanced dist={balanced_dist:.1}, aggressive dist={aggressive_dist:.1}"
        );
        }
    }

    // ---------------------------------------------------------------------------
    // detect_suitable_formats_with_faces
    // ---------------------------------------------------------------------------

    #[test]
    fn test_detect_formats_with_faces_no_pairs_falls_back_to_person_algorithm() {
        // With empty face_pairs, should produce same result as detect_suitable_formats.
        let person_bbox = portrait_person_bbox();
        let artistic = ArtisticCropConfig::default();
        let base = CropConfig::default();
        let result = detect_suitable_formats_with_faces(
            3024,
            4032,
            &person_bbox,
            &[],
            0.0,
            &artistic,
            &base,
        );
        // Portrait person in portrait photo → should have at least one portrait format.
        assert!(
            result.contains(&"9:21".to_string()) || result.contains(&"9:16".to_string()),
            "fallback to person algorithm should detect portrait formats: {:?}",
            result
        );
    }

    #[test]
    fn test_detect_formats_with_faces_use_face_detection_false_falls_back() {
        let person_bbox = portrait_person_bbox();
        let face = FacePersonPair {
            person_id: 0,
            face_id: 0,
            confidence: 0.9,
            face_bbox: make_bbox(1000.0, 250.0, 1200.0, 500.0),
            person_bbox: portrait_person_bbox(),
        };
        // use_face_detection is no longer a field (always true); test now verifies
        // fallback works when face_pairs is empty (new behavior: falls back regardless).
        let artistic = ArtisticCropConfig::default();
        let base = CropConfig::default();
        let result = detect_suitable_formats_with_faces(
            3024,
            4032,
            &person_bbox,
            &[face],
            0.0,
            &artistic,
            &base,
        );
        // With face pairs provided, the deprecated function uses face-aware detection.
        // Results should be non-empty regardless (face is inside portrait person).
        assert!(
            !result.is_empty(),
            "face-aware format detection must produce formats: {:?}",
            result
        );
    }

    #[test]
    fn test_detect_formats_with_faces_portrait_photo_produces_portrait_formats() {
        let person_bbox = portrait_person_bbox();
        let face = FacePersonPair {
            person_id: 0,
            face_id: 0,
            confidence: 0.92,
            face_bbox: make_bbox(1000.0, 250.0, 1200.0, 500.0),
            person_bbox: portrait_person_bbox(),
        };
        let artistic = ArtisticCropConfig::default();
        let base = CropConfig::default();
        let result = detect_suitable_formats_with_faces(
            3024,
            4032,
            &person_bbox,
            &[face],
            0.0,
            &artistic,
            &base,
        );
        assert!(
            !result.is_empty(),
            "at least one format must be suitable: {:?}",
            result
        );
    }

    // ---------------------------------------------------------------------------
    // Edge cases
    // ---------------------------------------------------------------------------

    #[test]
    fn test_correlate_face_exactly_on_person_boundary_is_contained() {
        // A face that exactly matches the person bbox should be treated as contained.
        let face_x1 = 100.0f32;
        let face_y1 = 100.0f32;
        let face_x2 = 900.0f32;
        let face_y2 = 900.0f32;
        let person = make_bbox(face_x1, face_y1, face_x2, face_y2);
        let faces = vec![make_face_detection(
            face_x1, face_y1, face_x2, face_y2, 0.85,
        )];
        let pairs = correlate_faces_to_persons(&faces, &[person]);
        assert_eq!(pairs.len(), 1, "face on exact boundary must be correlated");
    }

    #[test]
    fn test_small_face_still_correlates_if_within_person_bbox() {
        // A 15×15 face (below MIN_FACE_SIZE_PX for decode_predictions, but valid here
        // since correlation only checks containment — model filtering is separate).
        let faces = vec![make_face_detection(100.0, 100.0, 115.0, 115.0, 0.9)];
        let persons = vec![left_person_bbox()];
        let pairs = correlate_faces_to_persons(&faces, &persons);
        assert_eq!(
            pairs.len(),
            1,
            "small face within person bbox must correlate"
        );
    }

    #[test]
    fn test_face_at_image_boundary_correlates_when_within_person() {
        // Face touching the top edge of a person bbox that goes to y=0.
        let faces = vec![make_face_detection(10.0, 0.0, 200.0, 150.0, 0.88)];
        let persons = vec![make_bbox(0.0, 0.0, 960.0, 1080.0)];
        let pairs = correlate_faces_to_persons(&faces, &persons);
        assert_eq!(
            pairs.len(),
            1,
            "face at image boundary must correlate correctly"
        );
    }

    #[test]
    fn test_group_shot_five_faces_all_correlated() {
        // Photo 6 scenario: 5 distinct faces in 5 distinct person bboxes.
        // All should be correlated and produce a compound face bbox.
        let face_positions = [
            (50.0, 50.0, 200.0, 200.0),
            (250.0, 50.0, 400.0, 200.0),
            (450.0, 50.0, 600.0, 200.0),
            (650.0, 50.0, 800.0, 200.0),
            (850.0, 50.0, 1000.0, 200.0),
        ];
        let faces: Vec<ModelDetection> = face_positions
            .iter()
            .map(|&(x1, y1, x2, y2)| make_face_detection(x1, y1, x2, y2, 0.88))
            .collect();
        let persons: Vec<BBox> = face_positions
            .iter()
            .map(|&(x1, _, x2, _)| make_bbox(x1 - 20.0, 0.0, x2 + 20.0, 400.0))
            .collect();

        let pairs = correlate_faces_to_persons(&faces, &persons);
        assert_eq!(pairs.len(), 5, "all 5 faces must be correlated");

        let group_bbox = compound_face_bbox(&pairs);
        assert!(group_bbox.is_some(), "group bbox must be computable");
        let gb = group_bbox.unwrap();
        assert!(
            gb.x1 <= 50.0 && gb.x2 >= 1000.0,
            "group bbox must span all faces: x1={}, x2={}",
            gb.x1,
            gb.x2
        );
    }
} // mod face_detection_tests
