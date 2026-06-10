//! Image I/O helpers: encoding, EXIF injection, and XMP rating metadata.
//!
//! These functions are format-aware (JPEG/PNG/WebP) and support optional
//! EXIF rating tags and XMP packet injection for Adobe Bridge interoperability.

use anyhow::{bail, Context, Result};
use image::DynamicImage;
use little_exif::exif_tag::ExifTag;
use little_exif::filetype::FileExtension;
use little_exif::ifd::ExifTagGroup;
use little_exif::metadata::Metadata;
use std::path::Path;
use xmp_writer::XmpWriter;

/// Build a minimal XMP packet containing only an `xmp:Rating` property.
pub(crate) fn build_xmp_packet_with_rating(rating: u8) -> Result<Vec<u8>> {
    let mut writer = XmpWriter::new();
    writer.rating(i64::from(rating));
    Ok(writer.finish(None).into_bytes())
}

/// Insert an XMP APP1 segment into a JPEG byte buffer immediately after the SOI marker.
///
/// This is the minimal injection needed for Adobe Bridge to pick up the star rating.
pub(crate) fn insert_jpeg_app1_xmp_segment(
    jpeg_bytes: &mut Vec<u8>,
    xmp_packet: &[u8],
) -> Result<()> {
    const APP1_MARKER_PREFIX: [u8; 2] = [0xFF, 0xE1];
    const XMP_APP1_HEADER: &[u8] = b"http://ns.adobe.com/xap/1.0/\0";

    if jpeg_bytes.len() < 4 || jpeg_bytes[0] != 0xFF || jpeg_bytes[1] != 0xD8 {
        bail!("Output buffer is not a valid JPEG (missing SOI marker)");
    }

    let mut payload = Vec::with_capacity(XMP_APP1_HEADER.len() + xmp_packet.len());
    payload.extend_from_slice(XMP_APP1_HEADER);
    payload.extend_from_slice(xmp_packet);

    let segment_len_u16 = u16::try_from(payload.len() + 2)
        .context("XMP packet too large for a single JPEG APP1 segment")?;

    let mut segment = Vec::with_capacity(2 + 2 + payload.len());
    segment.extend_from_slice(&APP1_MARKER_PREFIX);
    segment.extend_from_slice(&segment_len_u16.to_be_bytes());
    segment.extend_from_slice(&payload);

    // Insert right after SOI to keep implementation minimal and deterministic.
    jpeg_bytes.splice(2..2, segment);

    Ok(())
}

/// Encode `image` to the format implied by `path`'s extension, optionally
/// embedding an EXIF rating tag and XMP rating packet.
///
/// When `jpeg_quality` is `Some`, the output is always JPEG regardless of the
/// path extension. Supports `.jpg`/`.jpeg`, `.png`, and `.webp`.
///
/// # Parameters
/// - `image`: Source image to encode.
/// - `path`: Destination path; its extension selects the output format.
/// - `tier`: Optional content tier (1–5). Written as EXIF `0x4746` + XMP `xmp:Rating`.
/// - `jpeg_quality`: JPEG quality (0–100). `None` means use the path extension's native format.
pub fn save_image_with_exif(
    image: &DynamicImage,
    path: &Path,
    tier: Option<u8>,
    jpeg_quality: Option<u8>,
) -> Result<()> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("jpg")
        .to_lowercase();

    // When --jpeg is used we always output JPEG regardless of original extension
    let use_jpeg = jpeg_quality.is_some() || matches!(ext.as_str(), "jpg" | "jpeg");
    let (img_format, exif_ext) = if use_jpeg {
        (image::ImageFormat::Jpeg, FileExtension::JPEG)
    } else if ext == "png" {
        (
            image::ImageFormat::Png,
            FileExtension::PNG {
                as_zTXt_chunk: true,
            },
        )
    } else if ext == "webp" {
        (image::ImageFormat::WebP, FileExtension::WEBP)
    } else {
        (image::ImageFormat::Jpeg, FileExtension::JPEG)
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

    if let Some(t) = tier {
        let mut metadata = Metadata::new();
        // 0x4746 = EXIF Rating tag
        metadata.set_tag(ExifTag::UnknownINT16U(
            vec![t as u16],
            0x4746,
            ExifTagGroup::GENERIC,
        ));

        metadata.write_to_vec(&mut img_buffer, exif_ext)?;

        // Also write XMP xmp:Rating for Adobe Bridge star interoperability.
        // XMP only makes sense for JPEG (after possible format coercion).
        let final_ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("jpg")
            .to_lowercase();
        if final_ext == "jpg" || final_ext == "jpeg" {
            let xmp_packet = build_xmp_packet_with_rating(t)?;
            insert_jpeg_app1_xmp_segment(&mut img_buffer, &xmp_packet)?;
        }
    }

    std::fs::write(path, img_buffer)?;
    Ok(())
}
