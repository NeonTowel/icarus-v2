use crate::error::Result;
/// Image utility functions for preprocessing and postprocessing
use image::{DynamicImage, ImageBuffer, Rgb};

/// Represents a detected object with bounding box
#[derive(Debug, Clone)]
pub struct Detection {
    /// Bounding box: [x1, y1, x2, y2]
    pub bbox: [f32; 4],

    /// Confidence score
    pub confidence: f32,

    /// Class label
    pub label: String,

    /// Class ID
    pub class_id: usize,
}

/// Crop image to a bounding box
pub fn crop_image(image: &DynamicImage, bbox: [f32; 4]) -> Result<DynamicImage> {
    let [x1, y1, x2, y2] = bbox;
    let x1 = x1.max(0.0) as u32;
    let y1 = y1.max(0.0) as u32;
    let x2 = (x2.min(image.width() as f32)) as u32;
    let y2 = (y2.min(image.height() as f32)) as u32;

    if x2 <= x1 || y2 <= y1 {
        return Err(anyhow::anyhow!("Invalid bounding box"));
    }

    let width = x2 - x1;
    let height = y2 - y1;

    Ok(image.crop_imm(x1, y1, width, height))
}

/// Resize image to target dimensions
pub fn resize_image(image: &DynamicImage, width: u32, height: u32) -> DynamicImage {
    image.resize_exact(width, height, image::imageops::FilterType::Lanczos3)
}

/// Normalize image pixel values to [-1, 1] or [0, 1] range
pub fn normalize_pixels(pixels: &mut [f32], mean: &[f32], std: &[f32]) {
    for (i, pixel) in pixels.iter_mut().enumerate() {
        let channel = i % 3;
        *pixel = (*pixel - mean[channel]) / std[channel];
    }
}

/// Convert image to RGB if needed
pub fn to_rgb(image: &DynamicImage) -> DynamicImage {
    match image {
        DynamicImage::ImageRgb8(_) => image.clone(),
        DynamicImage::ImageRgba8(img) => {
            let rgb: ImageBuffer<Rgb<u8>, Vec<u8>> =
                ImageBuffer::from_fn(img.width(), img.height(), |x, y| {
                    let pixel = img.get_pixel(x, y);
                    Rgb([pixel[0], pixel[1], pixel[2]])
                });
            DynamicImage::ImageRgb8(rgb)
        }
        _ => image.to_rgb8().into(),
    }
}

/// Extract pixels from image as flat vector
pub fn extract_pixels(image: &DynamicImage) -> Vec<f32> {
    let rgb = to_rgb(image);
    let rgb8 = rgb.to_rgb8();

    rgb8.pixels()
        .flat_map(|p| vec![p[0] as f32, p[1] as f32, p[2] as f32])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_pixels() {
        let mut pixels = vec![255.0, 128.0, 64.0];
        let mean = vec![0.485, 0.456, 0.406];
        let std = vec![0.229, 0.224, 0.225];

        normalize_pixels(&mut pixels, &mean, &std);

        // (255.0 - 0.485) / 0.229 ≈ 1110.91
        assert!((pixels[0] - 1110.91).abs() < 1.0);
    }
}
