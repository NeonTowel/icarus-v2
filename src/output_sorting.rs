/// Output Sorting Module
///
/// Provides aspect-ratio-based subfolder organization for cropped output images.
/// When `--sort-output` is enabled, images are automatically placed into:
/// - `landscape/`  — 21:9 ultrawide format
/// - `portrait/`   — 9:16 vertical format
/// - `mobile/`     — 9:21 tall mobile format
///
/// Additionally, annotated/visualized images always receive an `_annotated` suffix
/// injected into the file stem. When sorting is enabled the format suffix is
/// dropped to preserve 1:1 filename parity with inputs (e.g. `photo_annotated.jpg`).
/// When sorting is disabled the format suffix is retained to prevent collisions
/// (e.g. `photo_annotated_21_9.jpg`).
///
/// # Example
/// ```rust,ignore
/// use icarus_v2::output_sorting::{ensure_output_dirs, get_sorted_output_path};
/// use std::path::Path;
///
/// ensure_output_dirs(Path::new("output/"), true)?;
/// let path = get_sorted_output_path(Path::new("output/crop.jpg"), "21:9", "crop", "jpg", true)?;
/// // Returns: output/landscape/crop.jpg
/// ```
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Aspect ratio thresholds
// ---------------------------------------------------------------------------

/// Minimum aspect ratio (width / height) to classify as landscape.
///
/// Images wider than this threshold map to the "21:9" format (ultrawide).
/// Value 1.8 was chosen as the midpoint between 16:9 (1.78) and 21:9 (2.33).
const LANDSCAPE_THRESHOLD: f32 = 1.8;

/// Minimum aspect ratio (width / height) to classify as portrait (vs. mobile/tall).
///
/// Images between 0.55 and 1.8 are considered standard portrait or square-ish
/// and map to the "9:16" format. Below 0.55 maps to "9:21" (very tall mobile).
/// Value 0.55 was chosen as slightly above 9:16 (0.5625) to be inclusive.
const PORTRAIT_THRESHOLD: f32 = 0.55;

// ---------------------------------------------------------------------------
// Format → subfolder mapping
// ---------------------------------------------------------------------------

/// Map a format string to its corresponding output subfolder name.
///
/// Returns `Some("landscape")`, `Some("portrait")`, or `Some("mobile")` for
/// known formats, and `None` for unrecognized format strings.
///
/// # Arguments
/// * `format` — The format identifier (e.g., `"21:9"`, `"9:16"`, `"9:21"`)
///
/// # Example
/// ```rust,ignore
/// assert_eq!(get_subfolder_for_format("21:9"), Some("landscape"));
/// assert_eq!(get_subfolder_for_format("9:16"), Some("portrait"));
/// assert_eq!(get_subfolder_for_format("9:21"), Some("mobile"));
/// assert_eq!(get_subfolder_for_format("unknown"), None);
/// ```
pub fn get_subfolder_for_format(format: &str) -> Option<&'static str> {
    match format {
        "21:9" => Some("landscape"),
        "9:16" => Some("portrait"),
        "9:21" => Some("mobile"),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Aspect ratio → format detection
// ---------------------------------------------------------------------------

/// Determine the best format string for a given aspect ratio (width / height).
///
/// Threshold logic:
/// - `> 1.8` (wide)                  → `"21:9"` (ultrawide landscape)
/// - `0.55 < ratio ≤ 1.8` (portrait) → `"9:16"` (standard portrait)
/// - `≤ 0.55` (tall/mobile)          → `"9:21"` (tall mobile portrait)
///
/// # Arguments
/// * `aspect_ratio` — Width divided by height (must be positive)
///
/// # Example
/// ```rust,ignore
/// assert_eq!(determine_best_format_for_aspect_ratio(2.5), "21:9");
/// assert_eq!(determine_best_format_for_aspect_ratio(1.0), "9:16");
/// assert_eq!(determine_best_format_for_aspect_ratio(0.4), "9:21");
/// ```
pub fn determine_best_format_for_aspect_ratio(aspect_ratio: f32) -> &'static str {
    if aspect_ratio > LANDSCAPE_THRESHOLD {
        "21:9"
    } else if aspect_ratio > PORTRAIT_THRESHOLD {
        "9:16"
    } else {
        "9:21"
    }
}

// ---------------------------------------------------------------------------
// Directory management
// ---------------------------------------------------------------------------

/// Ensure the aspect-ratio output subdirectories exist under `base_dir`.
///
/// Creates `landscape/`, `portrait/`, and `mobile/` subdirectories under
/// `base_dir`.
/// This operation is **idempotent**: calling it multiple times does not
/// produce errors or duplicate directories.
///
/// Does nothing if `sort_output` is `false`.
///
/// # Arguments
/// * `base_dir` — The parent output directory
/// * `sort_output` — Whether to create sorting subdirectories
///
/// # Errors
/// Returns an error if any directory cannot be created due to filesystem
/// permission or I/O issues.
///
/// # Example
/// ```rust,ignore
/// ensure_output_dirs(Path::new("output/"), true)?;
/// // Creates: output/landscape, output/portrait, output/mobile
/// ```
pub fn ensure_output_dirs(base_dir: &Path, sort_output: bool) -> Result<()> {
    if !sort_output {
        return Ok(());
    }

    let subfolders = ["landscape", "portrait", "mobile"];

    for subfolder in &subfolders {
        let dir = base_dir.join(subfolder);
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("Failed to create output subfolder: {:?}", dir))?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Path construction
// ---------------------------------------------------------------------------

/// Construct the output path for a cropped image, optionally inside a subfolder.
///
/// When `sort_output` is `true`, the image is placed in the format-appropriate
/// subfolder and the format suffix is dropped to preserve the original filename
/// (e.g. `base_dir/landscape/stem.ext`). When `sort_output` is `false`, the
/// format suffix is kept to prevent collisions (e.g. `base_dir/stem_21_9.ext`).
///
/// The `base_path` parent directory is used as `base_dir`.
///
/// # Arguments
/// * `base_path` — The user-supplied `--output` path (used to derive the parent dir)
/// * `format`    — Format string (e.g., `"21:9"`)
/// * `stem`      — File stem without extension (e.g., `"crop"`)
/// * `ext`       — File extension without leading dot (e.g., `"jpg"`)
/// * `sort_output` — Whether to place the file in a subfolder
///
/// # Errors
/// Returns an error if the format is unknown and `sort_output` is `true`.
///
/// # Example
/// ```rust,ignore
/// let path = get_sorted_output_path(Path::new("out/crop.jpg"), "21:9", "crop", "jpg", true)?;
/// // Returns: out/landscape/crop.jpg
///
/// let path = get_sorted_output_path(Path::new("out/crop.jpg"), "21:9", "crop", "jpg", false)?;
/// // Returns: out/crop_21_9.jpg
/// ```
pub fn get_sorted_output_path(
    base_path: &Path,
    format: &str,
    stem: &str,
    ext: &str,
    sort_output: bool,
) -> Result<PathBuf> {
    let dir = base_path.parent().unwrap_or(Path::new("."));
    let filename = if sort_output {
        format!("{}.{}", stem, ext) // drop suffix when sorted into unique subfolders
    } else {
        format!("{}_{}.{}", stem, format.replace(":", "_"), ext)
    };

    if sort_output {
        let subfolder = get_subfolder_for_format(format)
            .with_context(|| format!("Unknown format '{}'; cannot determine subfolder", format))?;

        Ok(dir.join(subfolder).join(filename))
    } else {
        Ok(dir.join(filename))
    }
}

/// Construct the output path for an annotated/visualized image.
///
/// Behaves like [`get_sorted_output_path`] but automatically injects `_annotated`
/// into the stem. If the `stem` already ends with `_annotated`, the suffix is not
/// duplicated. The format suffix is dropped when `sort_output` is `true` (since the
/// subfolder already disambiguates) and kept when `sort_output` is `false`.
///
/// # Arguments
/// * `base_path` — The user-supplied `--visualize` path (used to derive the parent dir)
/// * `format`    — Format string (e.g., `"21:9"`)
/// * `stem`      — File stem without extension (e.g., `"viz"`)
/// * `ext`       — File extension without leading dot (e.g., `"jpg"`)
/// * `sort_output` — Whether to place the file in a subfolder
///
/// # Example
/// ```rust,ignore
/// let path = get_annotated_output_path(Path::new("out/viz.jpg"), "21:9", "viz", "jpg", true)?;
/// // Returns: out/landscape/viz_annotated.jpg
///
/// let path = get_annotated_output_path(Path::new("out/viz.jpg"), "21:9", "viz", "jpg", false)?;
/// // Returns: out/viz_annotated_21_9.jpg
/// ```
pub fn get_annotated_output_path(
    base_path: &Path,
    format: &str,
    stem: &str,
    ext: &str,
    sort_output: bool,
) -> Result<PathBuf> {
    let dir = base_path.parent().unwrap_or(Path::new("."));
    let annotated_stem = ensure_annotated_suffix(stem);
    let filename = if sort_output {
        format!("{}.{}", annotated_stem, ext) // drop suffix when sorted into unique subfolders
    } else {
        format!("{}_{}.{}", annotated_stem, format.replace(":", "_"), ext)
    };

    if sort_output {
        let subfolder = get_subfolder_for_format(format)
            .with_context(|| format!("Unknown format '{}'; cannot determine subfolder", format))?;

        Ok(dir.join(subfolder).join(filename))
    } else {
        Ok(dir.join(filename))
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Ensure the stem ends with `_annotated`, without duplicating the suffix.
///
/// If the stem already ends with `_annotated`, it is returned unchanged.
/// Otherwise, `_annotated` is appended.
///
/// # Example
/// ```ignore
/// assert_eq!(ensure_annotated_suffix("viz"), "viz_annotated");
/// assert_eq!(ensure_annotated_suffix("viz_annotated"), "viz_annotated");
/// ```
fn ensure_annotated_suffix(stem: &str) -> String {
    if stem.ends_with("_annotated") {
        stem.to_string()
    } else {
        format!("{}_annotated", stem)
    }
}

#[cfg(test)]
mod tests {
    use super::{ensure_output_dirs, get_annotated_output_path, get_sorted_output_path};
    use std::path::Path;
    use tempfile::tempdir;

    #[test]
    fn get_sorted_output_path_uses_format_subfolder() {
        let path = get_sorted_output_path(Path::new("out/crop.jpg"), "21:9", "crop", "jpg", true)
            .expect("path generation should succeed");

        assert_eq!(path, Path::new("out/landscape/crop.jpg"));
    }

    #[test]
    fn get_annotated_output_path_uses_format_subfolder() {
        let path = get_annotated_output_path(Path::new("out/viz.jpg"), "9:21", "viz", "jpg", true)
            .expect("path generation should succeed");

        assert_eq!(path, Path::new("out/mobile/viz_annotated.jpg"));
    }

    #[test]
    fn ensure_output_dirs_creates_format_subfolders() {
        let temp_dir = tempdir().expect("tempdir should be created");
        let output_dir = temp_dir.path();

        ensure_output_dirs(output_dir, true).expect("directory creation should succeed");

        for subfolder in ["landscape", "portrait", "mobile"] {
            let expected_directory = output_dir.join(subfolder);
            assert!(
                expected_directory.is_dir(),
                "expected directory to exist: {:?}",
                expected_directory
            );
        }
    }
}
