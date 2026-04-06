//! Directory Walking Module
//!
//! Discovers supported image files in a directory, optionally traversing
//! subdirectories recursively. This module only knows about file paths.

use anyhow::{bail, Result};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Image file extensions recognised by Icarus-v2.
pub const SUPPORTED_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "bmp", "tiff", "tif", "gif"];

/// Check whether a file path has a supported image extension.
///
/// Comparison is case-insensitive.
pub fn is_supported_image(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| {
            let lower = extension.to_ascii_lowercase();
            SUPPORTED_EXTENSIONS.contains(&lower.as_str())
        })
        .unwrap_or(false)
}

fn is_visible(entry: &walkdir::DirEntry) -> bool {
    entry
        .file_name()
        .to_str()
        .map(|name| !name.starts_with('.'))
        .unwrap_or(false)
}

fn canonical_root(dir: &Path) -> Result<PathBuf> {
    if !dir.is_dir() {
        bail!("{:?} is not a directory", dir);
    }

    dir.canonicalize()
        .map_err(|error| anyhow::anyhow!("failed to canonicalize {:?}: {}", dir, error))
}

/// Discover image files in a directory.
///
/// When `recurse` is `false`, only immediate children are examined. When it is
/// `true`, all subdirectories are traversed. Hidden files and directories are
/// skipped.
pub fn discover_images(dir: &Path, recurse: bool) -> Result<Vec<PathBuf>> {
    let root = canonical_root(dir)?;
    let max_depth = if recurse { usize::MAX } else { 1 };

    let mut images: Vec<PathBuf> = WalkDir::new(&root)
        .max_depth(max_depth)
        .min_depth(1)
        .follow_links(false)
        .into_iter()
        .filter_entry(is_visible)
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().is_file())
        .map(|entry| entry.into_path())
        .filter(|path| is_supported_image(path))
        .collect();

    images.sort();
    Ok(images)
}

/// Compute the relative path of `image_path` with respect to `input_root`.
pub fn relative_to(image_path: &Path, input_root: &Path) -> Option<PathBuf> {
    image_path.strip_prefix(input_root).ok().map(PathBuf::from)
}
