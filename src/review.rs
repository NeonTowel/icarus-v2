//! Review-dataset generation for the Icarus crop editor.

use anyhow::{Context, Result};
use image::DynamicImage;
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};

use crate::batch_processor::Detection;
use crate::config::{ArtisticCropConfig, CropConfig};
use crate::crop::{
    analyze_joint, apply_margin_to_bbox, calculate_compound_bbox,
    calculate_landscape_21_9_crop_with_face, calculate_portrait_9_16_crop_with_face,
    calculate_portrait_9_21_crop_with_face, deduplicate_person_detections, detect_suitable_formats,
    merge_bboxes, select_dominant_face_for_crop, strategy_for, BBox, CropRegion,
};
use crate::face_aware_cropping::{enforce_eye_safety, enforce_pose_aware_eye_safety};
use crate::face_detection::{detect_faces, FaceDetector};
use crate::image_utils::Detection as ImageDetection;
use crate::models::Model;

const PERSON_CLASS_ID: usize = 0;
const REVIEW_SCHEMA_VERSION: u8 = 1;
const FORMATS: [&str; 3] = ["21:9", "9:16", "9:21"];

#[derive(Debug, Serialize)]
pub struct ReviewManifest {
    pub schema_version: u8,
    pub samples: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct ReviewSample {
    pub schema_version: u8,
    pub id: String,
    pub original_path: String,
    pub source: String,
    pub image_size: [u32; 2],
    pub persons: Vec<ReviewPerson>,
    pub faces: Vec<ReviewFace>,
    pub formats: Vec<ReviewFormat>,
}

#[derive(Debug, Serialize)]
pub struct ReviewPerson {
    pub person_id: usize,
    pub confidence: f32,
    pub bbox: [f32; 4],
}

#[derive(Debug, Serialize)]
pub struct ReviewFace {
    pub face_id: usize,
    pub bbox: [f32; 4],
}

#[derive(Debug, Serialize)]
pub struct ReviewFormat {
    pub name: &'static str,
    pub baseline: ReviewCandidate,
    pub enhanced: ReviewCandidate,
}

#[derive(Debug, Serialize)]
pub struct ReviewCandidate {
    pub status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bbox: Option<[f32; 4]>,
}

/// Generate source previews and candidate-crop JSON for the editor.
///
/// This writes only inside `data_root`; it never creates normal crop outputs.
pub fn generate_dataset(
    image_paths: &[PathBuf],
    input_root: &Path,
    data_root: &Path,
    model: &dyn Model,
    face_detector: &FaceDetector,
    confidence: f32,
) -> Result<ReviewManifest> {
    let sources_root = data_root.join("sources");
    let samples_root = data_root.join("samples");
    fs::create_dir_all(&sources_root)
        .with_context(|| format!("Failed to create {:?}", sources_root))?;
    fs::create_dir_all(&samples_root)
        .with_context(|| format!("Failed to create {:?}", samples_root))?;

    let mut sample_ids = Vec::with_capacity(image_paths.len());
    for image_path in image_paths {
        let id = stable_id(image_path)?;
        let sample = generate_sample(
            image_path,
            input_root,
            &id,
            model,
            face_detector,
            confidence,
        )?;
        let source_path = sources_root.join(format!("{id}.jpg"));
        let sample_path = samples_root.join(format!("{id}.json"));

        save_source_preview(image_path, &source_path)?;
        write_json(&sample_path, &sample)?;
        sample_ids.push(id);
    }

    sample_ids.sort();
    let manifest = ReviewManifest {
        schema_version: REVIEW_SCHEMA_VERSION,
        samples: sample_ids,
    };
    write_json(&data_root.join("manifest.json"), &manifest)?;
    Ok(manifest)
}

fn generate_sample(
    image_path: &Path,
    input_root: &Path,
    id: &str,
    model: &dyn Model,
    face_detector: &FaceDetector,
    confidence: f32,
) -> Result<ReviewSample> {
    let image = image::open(image_path)
        .with_context(|| format!("Failed to open review source {:?}", image_path))?;
    let detections = model
        .infer(&image)
        .map_err(|error| anyhow::anyhow!("Inference failed: {error}"))?
        .into_iter()
        .map(Detection::from)
        .filter(|detection| detection.confidence >= confidence)
        .collect::<Vec<_>>();
    let faces = detect_faces(&image, face_detector)
        .map_err(|error| anyhow::anyhow!("Face detection failed: {error}"))?;
    let persons = detections
        .iter()
        .filter(|detection| detection.class_id == PERSON_CLASS_ID)
        .collect::<Vec<_>>();
    let base_bbox = compute_base_bbox(&persons);
    let crop_bbox = compute_crop_bbox(base_bbox.clone(), &faces);
    let formats = build_candidates(&image, crop_bbox.as_ref(), base_bbox.as_ref(), &faces);
    let original_path = image_path
        .strip_prefix(input_root)
        .unwrap_or(image_path)
        .to_string_lossy()
        .replace('\\', "/");

    Ok(ReviewSample {
        schema_version: REVIEW_SCHEMA_VERSION,
        id: id.to_owned(),
        original_path,
        source: format!("sources/{id}.jpg"),
        image_size: [image.width(), image.height()],
        persons: persons
            .iter()
            .enumerate()
            .map(|(person_id, detection)| ReviewPerson {
                person_id,
                confidence: detection.confidence,
                bbox: detection.bbox,
            })
            .collect(),
        faces: faces
            .iter()
            .enumerate()
            .map(|(face_id, face)| ReviewFace {
                face_id,
                bbox: [face.x1, face.y1, face.x2, face.y2],
            })
            .collect(),
        formats,
    })
}

fn build_candidates(
    image: &DynamicImage,
    crop_bbox: Option<&BBox>,
    base_bbox: Option<&BBox>,
    faces: &[BBox],
) -> Vec<ReviewFormat> {
    let Some(crop_bbox) = crop_bbox else {
        return FORMATS
            .iter()
            .map(|name| ReviewFormat {
                name,
                baseline: unavailable("no_subject"),
                enhanced: unavailable("no_subject"),
            })
            .collect();
    };

    let config = CropConfig::default();
    let artistic = ArtisticCropConfig::default();
    let focal = crate::focal_point::compute_focal_point(
        Some(crop_bbox),
        faces,
        image.width(),
        image.height(),
    );
    let suitable = detect_suitable_formats(image.width(), image.height(), crop_bbox, 0.0, &config);

    FORMATS
        .iter()
        .map(|name| {
            if !suitable.iter().any(|format| format == name) {
                return ReviewFormat {
                    name,
                    baseline: unavailable("not_suitable"),
                    enhanced: unavailable("not_suitable"),
                };
            }

            let baseline =
                calculate_baseline(name, image, crop_bbox, faces, &focal, &config, &artistic)
                    .map(|crop| enforce_eye_safety(&crop, faces, image.width(), image.height()));
            let enhanced = calculate_enhanced(
                name, image, crop_bbox, crop_bbox, faces, &focal, &config, &artistic,
            )
            .map(|crop| {
                enforce_pose_aware_eye_safety(
                    &crop,
                    base_bbox,
                    faces,
                    image.width(),
                    image.height(),
                )
            });

            ReviewFormat {
                name,
                baseline: candidate(baseline, image),
                enhanced: candidate(enhanced, image),
            }
        })
        .collect()
}

fn calculate_baseline(
    format: &str,
    image: &DynamicImage,
    bbox: &BBox,
    faces: &[BBox],
    focal: &crate::focal_point::FocalPoint,
    config: &CropConfig,
    artistic: &ArtisticCropConfig,
) -> Option<CropRegion> {
    match format {
        "21:9" => calculate_landscape_21_9_crop_with_face(
            image.width(),
            image.height(),
            bbox,
            faces,
            focal,
            config,
            artistic,
        ),
        "9:16" => calculate_portrait_9_16_crop_with_face(
            image.width(),
            image.height(),
            bbox,
            faces,
            focal,
            config,
            artistic,
        ),
        "9:21" => calculate_portrait_9_21_crop_with_face(
            image.width(),
            image.height(),
            bbox,
            faces,
            focal,
            config,
            artistic,
        ),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn calculate_enhanced(
    format: &str,
    image: &DynamicImage,
    person_bbox: &BBox,
    working_bbox: &BBox,
    faces: &[BBox],
    focal: &crate::focal_point::FocalPoint,
    config: &CropConfig,
    artistic: &ArtisticCropConfig,
) -> Option<CropRegion> {
    let (aspect_ratio, height_first) = match format {
        "21:9" => (21.0 / 9.0, true),
        "9:16" => (9.0 / 16.0, false),
        "9:21" => (9.0 / 21.0, false),
        _ => return None,
    };
    let joint = analyze_joint(
        person_bbox,
        faces,
        image.width(),
        image.height(),
        aspect_ratio,
        height_first,
        config,
        artistic,
    );
    let effective_bbox = if joint.extra_margin_percent > 0.0 {
        apply_margin_to_bbox(
            working_bbox,
            joint.extra_margin_percent,
            image.width(),
            image.height(),
        )
    } else {
        working_bbox.clone()
    };
    strategy_for(format)?.calculate_with_joint(
        image.width(),
        image.height(),
        &effective_bbox,
        faces,
        focal,
        config,
        artistic,
        Some(&joint),
    )
}

fn candidate(crop: Option<CropRegion>, image: &DynamicImage) -> ReviewCandidate {
    match crop {
        Some(crop) => ReviewCandidate {
            status: "available",
            bbox: Some(crop.to_xyxy_clamped(image.width(), image.height())),
        },
        None => unavailable("not_available"),
    }
}

fn unavailable(status: &'static str) -> ReviewCandidate {
    ReviewCandidate { status, bbox: None }
}

fn compute_base_bbox(persons: &[&Detection]) -> Option<BBox> {
    match persons {
        [] => None,
        [person] => Some(person.bbox.into()),
        _ => {
            let items = persons
                .iter()
                .map(|detection| ImageDetection {
                    bbox: detection.bbox,
                    confidence: detection.confidence,
                    label: detection.label.clone(),
                    class_id: detection.class_id,
                })
                .collect::<Vec<_>>();
            let effective = deduplicate_person_detections(&items, 0.50);
            calculate_compound_bbox(&effective)
        }
    }
}

fn compute_crop_bbox(base: Option<BBox>, faces: &[BBox]) -> Option<BBox> {
    if let Some(base) = base {
        Some(
            faces
                .iter()
                .fold(base, |merged, face| merge_bboxes(&merged, face)),
        )
    } else {
        select_dominant_face_for_crop(faces).cloned()
    }
}

fn stable_id(path: &Path) -> Result<String> {
    let canonical = path
        .canonicalize()
        .with_context(|| format!("Failed to canonicalize review source {:?}", path))?;
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in canonical.to_string_lossy().as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    Ok(format!("{hash:016x}"))
}

fn save_source_preview(input: &Path, output: &Path) -> Result<()> {
    let image = image::open(input).with_context(|| format!("Failed to open {:?}", input))?;
    image
        .save_with_format(output, image::ImageFormat::Jpeg)
        .with_context(|| format!("Failed to save review source {:?}", output))
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let json = serde_json::to_vec_pretty(value).context("Failed to serialize review JSON")?;
    fs::write(path, json).with_context(|| format!("Failed to write {:?}", path))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image() -> DynamicImage {
        DynamicImage::new_rgb8(1200, 1800)
    }

    fn bbox(x1: f32, y1: f32, x2: f32, y2: f32) -> BBox {
        BBox { x1, y1, x2, y2 }
    }

    #[test]
    fn candidates_include_baseline_and_enhanced_for_a_suitable_subject() {
        let person = bbox(400.0, 200.0, 800.0, 1600.0);
        let face = bbox(500.0, 250.0, 700.0, 500.0);
        let candidates = build_candidates(&image(), Some(&person), Some(&person), &[face]);

        assert_eq!(candidates.len(), 3);
        let portrait = candidates
            .iter()
            .find(|format| format.name == "9:16")
            .unwrap();
        assert_eq!(portrait.baseline.status, "available");
        assert_eq!(portrait.enhanced.status, "available");
        assert!(portrait.baseline.bbox.is_some());
        assert!(portrait.enhanced.bbox.is_some());
    }

    #[test]
    fn candidates_report_no_subject_without_detections() {
        let candidates = build_candidates(&image(), None, None, &[]);

        assert!(candidates
            .iter()
            .all(|format| format.baseline.status == "no_subject"));
        assert!(candidates
            .iter()
            .all(|format| format.enhanced.status == "no_subject"));
    }

    #[test]
    fn candidate_coordinates_stay_inside_source_bounds() {
        let person = bbox(50.0, 50.0, 1150.0, 1750.0);
        let candidates = build_candidates(&image(), Some(&person), Some(&person), &[]);

        for candidate in candidates
            .iter()
            .flat_map(|format| [&format.baseline, &format.enhanced])
        {
            if let Some([x1, y1, x2, y2]) = candidate.bbox {
                assert!((0.0..=1200.0).contains(&x1));
                assert!((0.0..=1200.0).contains(&x2));
                assert!((0.0..=1800.0).contains(&y1));
                assert!((0.0..=1800.0).contains(&y2));
                assert!(x2 > x1);
                assert!(y2 > y1);
            }
        }
    }
}
