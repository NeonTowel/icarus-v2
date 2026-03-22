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
///
/// # Parameters
/// - `image`: Source image
/// - `bbox`: Bounding box as `[x1, y1, x2, y2]` in pixel coordinates
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

/// Crop an image to 21:9 ultrawide aspect ratio, centering the detected person.
///
/// # Parameters
/// - `image`: Source image
/// - `bbox`: Detected bounding box as `[x1, y1, x2, y2]` in pixel coordinates
///
/// # Returns
/// Cropped image with 21:9 aspect ratio (width ≈ 2.333 × height), or an error
/// if the resulting crop dimensions are invalid.
///
/// # Behavior
/// - Centers the person horizontally: the person's horizontal midpoint aligns
///   with the crop's horizontal midpoint.
/// - Positions crop vertically to keep person visible: crop starts at
///   `person_top - crop_height / 4` (upper-third heuristic).
/// - When the 21:9 crop height would exceed the image height, the crop is
///   scaled down to fit; width is reduced proportionally.
/// - All crop coordinates are clamped to image bounds — no transparent padding
///   is added.
///
/// # Example
/// ```rust,ignore
/// let img = image::open("photo.jpg").unwrap();
/// let bbox = [100.0, 50.0, 400.0, 800.0]; // [x1, y1, x2, y2]
/// let crop = crop_to_ultrawide_21_9(&img, bbox).unwrap();
/// assert!((crop.width() as f32 / crop.height() as f32 - 21.0 / 9.0).abs() < 1.0);
/// ```
pub fn crop_to_ultrawide_21_9(image: &DynamicImage, bbox: [f32; 4]) -> Result<DynamicImage> {
    const ASPECT_RATIO: f32 = 21.0 / 9.0; // ~2.333

    let (img_w, img_h) = (image.width() as f32, image.height() as f32);

    let [x1, y1, x2, _y2] = bbox;

    // Person's horizontal center and top edge
    let person_center_x = (x1 + x2) / 2.0;
    let person_top_y = y1;

    // Start from full image width; reduce if 21:9 height overshoots
    let mut crop_w = img_w;
    let mut crop_h = crop_w / ASPECT_RATIO;

    if crop_h > img_h {
        crop_h = img_h;
        crop_w = crop_h * ASPECT_RATIO;
    }

    // Center crop horizontally on person
    let mut crop_x = person_center_x - crop_w / 2.0;

    // Upper-third vertical positioning: leave ~1/4 of crop height above person
    let mut crop_y = person_top_y - crop_h / 4.0;

    // Clamp both axes to valid image bounds
    crop_x = crop_x.max(0.0).min((img_w - crop_w).max(0.0));
    crop_y = crop_y.max(0.0).min((img_h - crop_h).max(0.0));

    crop_image(image, [crop_x, crop_y, crop_x + crop_w, crop_y + crop_h])
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
    use image::{DynamicImage, RgbImage};

    fn make_test_image(width: u32, height: u32) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::new(width, height))
    }

    #[test]
    fn test_normalize_pixels() {
        let mut pixels = vec![255.0, 128.0, 64.0];
        let mean = vec![0.485, 0.456, 0.406];
        let std = vec![0.229, 0.224, 0.225];

        normalize_pixels(&mut pixels, &mean, &std);

        // (255.0 - 0.485) / 0.229 ≈ 1110.91
        assert!((pixels[0] - 1110.91).abs() < 1.0);
    }

    #[test]
    fn test_crop_to_ultrawide_21_9_aspect_ratio() {
        // 4000×3000 landscape image; person bbox roughly centred
        let img = make_test_image(4000, 3000);
        let bbox = [1600.0, 400.0, 2400.0, 2800.0]; // centred person

        let crop = crop_to_ultrawide_21_9(&img, bbox).expect("crop should succeed");

        let ratio = crop.width() as f32 / crop.height() as f32;
        // Allow 1-pixel rounding tolerance
        assert!(
            (ratio - 21.0 / 9.0).abs() < 0.02,
            "expected ~2.333 ratio, got {ratio:.4}"
        );
    }

    #[test]
    fn test_crop_to_ultrawide_21_9_stays_in_bounds() {
        let img = make_test_image(1920, 1080);
        // Person in the far-left corner
        let bbox = [0.0, 0.0, 200.0, 900.0];

        let crop = crop_to_ultrawide_21_9(&img, bbox).expect("crop should succeed");

        assert!(
            crop.width() <= 1920,
            "crop width must not exceed image width"
        );
        assert!(
            crop.height() <= 1080,
            "crop height must not exceed image height"
        );
    }

    #[test]
    fn test_crop_to_ultrawide_21_9_tall_image() {
        // Use a very narrow image to force the height-clamping path:
        // full-width 21:9 crop_h would be 300 / 2.333 ≈ 129, which exceeds img_h=100
        let img_narrow = make_test_image(300, 100);
        let bbox = [50.0, 10.0, 250.0, 90.0];

        let crop = crop_to_ultrawide_21_9(&img_narrow, bbox).expect("crop should succeed");

        assert!(
            crop.width() <= 300,
            "crop width must not exceed image width"
        );
        assert!(
            crop.height() <= 100,
            "crop height must not exceed image height"
        );
    }

    #[test]
    fn test_crop_to_ultrawide_21_9_person_centered_horizontally() {
        // Wide image; person centred so crop_x should be 0 (full width used)
        let img = make_test_image(2100, 900);
        let bbox = [950.0, 100.0, 1150.0, 800.0]; // person near centre

        let crop = crop_to_ultrawide_21_9(&img, bbox).expect("crop should succeed");

        // At 2100 wide, 21:9 crop_h = 2100/2.333 ≈ 900, which equals img_h → crop = full image
        assert_eq!(crop.width(), 2100);
        assert_eq!(crop.height(), 900);
    }
}
