//! Visualization helpers: annotated image rendering with detection overlays.
//!
//! Draws coloured bounding-box overlays for persons, faces, and crop regions
//! onto a canvas copy of the source image and saves it to disk.

use anyhow::{Context, Result};
use image::DynamicImage;
use std::path::Path;

use crate::multi_format_cropping::{BBox, CropRegion};

const PERSON_CLASS_ID: usize = 0;

/// Draw person detections (green/dark-red), face bboxes (cyan), and crop regions (red)
/// onto a copy of `image` and save the result to `output_path`.
///
/// # Parameters
/// - `image`: Source image (not mutated).
/// - `detections`: All detections from the person model.
/// - `face_bboxes`: Face bounding boxes from the face model.
/// - `crop_regions`: Finalized crop regions to overlay.
/// - `output_path`: Destination path for the annotated image.
pub fn save_visualized_with_faces(
    image: &DynamicImage,
    detections: &[crate::batch_processor::Detection],
    face_bboxes: &[BBox],
    crop_regions: &[CropRegion],
    output_path: &Path,
) -> Result<()> {
    use image::{Rgba, RgbaImage};

    let mut canvas: RgbaImage = image.to_rgba8();
    let width = canvas.width();
    let height = canvas.height();

    const PERSON_COLOR: [u8; 4] = [0, 255, 0, 220];
    const OTHER_COLOR: [u8; 4] = [128, 0, 0, 220];
    const FACE_COLOR: [u8; 4] = [0, 255, 255, 220];
    const CROP_COLOR: [u8; 4] = [255, 0, 0, 220];

    for detection in detections {
        let colour = if detection.class_id == PERSON_CLASS_ID {
            Rgba(PERSON_COLOR)
        } else {
            Rgba(OTHER_COLOR)
        };
        let [x1, y1, x2, y2] = detection.bbox;
        draw_rect(
            &mut canvas,
            (x1 as u32).min(width.saturating_sub(1)),
            (y1 as u32).min(height.saturating_sub(1)),
            (x2 as u32).min(width.saturating_sub(1)),
            (y2 as u32).min(height.saturating_sub(1)),
            colour,
        );
    }

    for face in face_bboxes {
        draw_rect(
            &mut canvas,
            (face.x1 as u32).min(width.saturating_sub(1)),
            (face.y1 as u32).min(height.saturating_sub(1)),
            (face.x2 as u32).min(width.saturating_sub(1)),
            (face.y2 as u32).min(height.saturating_sub(1)),
            Rgba(FACE_COLOR),
        );
    }

    for crop in crop_regions {
        let x2 = crop.x + crop.width;
        let y2 = crop.y + crop.height;
        draw_rect(
            &mut canvas,
            (crop.x as u32).min(width.saturating_sub(1)),
            (crop.y as u32).min(height.saturating_sub(1)),
            (x2 as u32).min(width.saturating_sub(1)),
            (y2 as u32).min(height.saturating_sub(1)),
            Rgba(CROP_COLOR),
        );
    }

    DynamicImage::ImageRgba8(canvas)
        .save(output_path)
        .with_context(|| format!("Failed to save visualized image to {:?}", output_path))?;

    Ok(())
}

fn draw_rect(
    canvas: &mut image::RgbaImage,
    x1: u32,
    y1: u32,
    x2: u32,
    y2: u32,
    colour: image::Rgba<u8>,
) {
    let width = canvas.width();
    let height = canvas.height();
    for thickness in 0..2u32 {
        for x in x1..=x2 {
            for dy in 0..=thickness {
                if y1 + dy < height {
                    canvas.put_pixel(x, y1 + dy, colour);
                }
                if y2 >= dy && y2 - dy < height {
                    canvas.put_pixel(x, y2 - dy, colour);
                }
            }
        }
        for y in y1..=y2 {
            for dx in 0..=thickness {
                if x1 + dx < width {
                    canvas.put_pixel(x1 + dx, y, colour);
                }
                if x2 >= dx && x2 - dx < width {
                    canvas.put_pixel(x2 - dx, y, colour);
                }
            }
        }
    }
}
