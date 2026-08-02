//! Image I/O helpers for encoding cropped images.

use anyhow::Result;
use image::DynamicImage;
use std::path::Path;

/// Encode `image` to the format implied by `path`'s extension.
///
/// When `jpeg_quality` is `Some`, the output is always JPEG regardless of the
/// path extension. Supports `.jpg`/`.jpeg`, `.png`, and `.webp`.
///
/// # Parameters
/// - `image`: Source image to encode.
/// - `path`: Destination path; its extension selects the output format.
/// - `jpeg_quality`: JPEG quality (0–100). `None` means use the path extension's native format.
pub fn save_image(image: &DynamicImage, path: &Path, jpeg_quality: Option<u8>) -> Result<()> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("jpg")
        .to_lowercase();

    // When --jpeg is used we always output JPEG regardless of original extension
    let use_jpeg = jpeg_quality.is_some() || matches!(ext.as_str(), "jpg" | "jpeg");
    let img_format = if use_jpeg {
        image::ImageFormat::Jpeg
    } else if ext == "png" {
        image::ImageFormat::Png
    } else if ext == "webp" {
        image::ImageFormat::WebP
    } else {
        image::ImageFormat::Jpeg
    };

    let mut img_buffer: Vec<u8> = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut img_buffer);

    // Always use JPEG encoder when --jpeg flag is provided (even if path suggests PNG)
    if use_jpeg {
        let quality = jpeg_quality.unwrap_or(95);
        let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut cursor, quality);
        encoder.encode_image(image)?;
    } else {
        image.write_to(&mut cursor, img_format)?;
    }

    std::fs::write(path, img_buffer)?;
    Ok(())
}
