//! Pre-inference image dimension filter.
//!
//! Reads only the image file header (no pixel decode) to determine whether a
//! source photo's long side meets the configured minimum. This lets the pipeline
//! skip the dominant face-detection cost on photos that cannot produce good crops.
//!
//! ## Design invariants
//!
//! - **Header-only.** `read_long_side` calls `image::image_dimensions`, which reads
//!   only enough bytes to determine `(width, height)` without decoding pixel data.
//! - **Non-fatal.** Errors are returned as `Result`; callers must treat header-read
//!   failure as "not skipped" (fall through to normal processing). This matches the
//!   face-detection failure convention in `AGENTS.md`.
//! - **Pure.** No image writes, no side-effects beyond I/O.
//! - **Path-only input.** `directory_walker.rs` stays path-only; this module is the
//!   single place that opens a file for dimension inspection.

use std::path::Path;

/// Read image dimensions from the file header only (no pixel decode).
///
/// Returns `max(width, height)` — the long side of the image.
/// Combine with [`crate::crop::geometry::long_side_passes`] to decide whether
/// to skip the image.
///
/// # Errors
/// Returns an error if the file cannot be opened or its format is unrecognised.
/// **Callers must treat this as non-fatal and fall through to normal processing.**
///
/// # Example
/// ```rust,ignore
/// use icarus_v2::early_filter::read_long_side;
/// use icarus_v2::crop::geometry::long_side_passes;
///
/// if let Ok(long_side) = read_long_side(path) {
///     if !long_side_passes(long_side, ctx.min_long_side_pixels) {
///         // skip — no inference needed
///     }
/// }
/// // on Err: fall through to full processing
/// ```
pub fn read_long_side(path: &Path) -> anyhow::Result<u32> {
    let (w, h) = image::image_dimensions(path)
        .map_err(|e| anyhow::anyhow!("Failed to read image dimensions from {:?}: {e}", path))?;
    Ok(w.max(h))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that a nonexistent path returns an error (non-fatal failure path).
    #[test]
    fn test_read_long_side_fails_on_nonexistent_file() {
        let result = read_long_side(Path::new("/nonexistent/icarus_v2_test_image.jpg"));
        assert!(
            result.is_err(),
            "read_long_side should return Err for a nonexistent file"
        );
    }

    /// Verify that the error message includes the path (actionable error).
    #[test]
    fn test_read_long_side_error_includes_path() {
        let path = Path::new("/no/such/file.png");
        let err = read_long_side(path).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("no/such/file"),
            "error message should include the path, got: {msg}"
        );
    }

    /// Verify that a generated small image returns the correct long side (max(w,h)).
    #[test]
    fn test_read_long_side_on_generated_image() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("tiny.png");
        let img = image::RgbImage::new(3, 2);
        img.save(&path).expect("save tiny png");
        let result = read_long_side(&path).expect("read_long_side must succeed on valid image");
        assert_eq!(result, 3, "long side of 3×2 image is max(3,2)=3");
    }

    /// Verify that a non-image file returns an error (non-fatal failure path).
    #[test]
    fn test_read_long_side_fails_on_text_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("not_image.txt");
        std::fs::write(&path, b"hello world").expect("write text file");
        let result = read_long_side(&path);
        assert!(result.is_err(), "text file must not parse as image");
    }
}
