use icarus_v2::directory_walker::{discover_images, relative_to};
use std::path::Path;
use tempfile::TempDir;

#[test]
fn test_batch_discover_flat_directory() {
    let temp_dir = TempDir::new().expect("temp dir");
    for name in ["a.jpg", "b.png", "c.bmp", "d.tiff", "e.gif"] {
        std::fs::write(temp_dir.path().join(name), b"fake image").expect("write image");
    }
    std::fs::write(temp_dir.path().join("readme.md"), b"not image").expect("write markdown");

    let images = discover_images(temp_dir.path(), false).expect("discover images");
    assert_eq!(images.len(), 5);
}

#[test]
fn test_batch_discover_recursive_mirrors_structure() {
    let temp_dir = TempDir::new().expect("temp dir");
    std::fs::create_dir_all(temp_dir.path().join("a/b")).expect("create nested dirs");
    std::fs::write(temp_dir.path().join("root.jpg"), b"fake").expect("write root");
    std::fs::write(temp_dir.path().join("a/mid.jpg"), b"fake").expect("write mid");
    std::fs::write(temp_dir.path().join("a/b/deep.jpg"), b"fake").expect("write deep");

    let images = discover_images(temp_dir.path(), true).expect("discover images");
    assert_eq!(images.len(), 3);

    let relative_paths: Vec<_> = images
        .iter()
        .map(|path| relative_to(path, temp_dir.path()).expect("relative path"))
        .collect();
    assert!(relative_paths.contains(&Path::new("root.jpg").to_path_buf()));
    assert!(relative_paths.contains(&Path::new("a/mid.jpg").to_path_buf()));
    assert!(relative_paths.contains(&Path::new("a/b/deep.jpg").to_path_buf()));
}

#[test]
fn test_batch_discover_empty_directory() {
    let temp_dir = TempDir::new().expect("temp dir");
    let images = discover_images(temp_dir.path(), false).expect("discover images");
    assert!(images.is_empty());
}

#[test]
fn test_batch_non_recursive_does_not_enter_subdirectories() {
    let temp_dir = TempDir::new().expect("temp dir");
    std::fs::write(temp_dir.path().join("top.jpg"), b"fake").expect("write top");
    std::fs::create_dir(temp_dir.path().join("sub")).expect("create subdir");
    std::fs::write(temp_dir.path().join("sub/nested.jpg"), b"fake").expect("write nested");

    let images = discover_images(temp_dir.path(), false).expect("discover images");
    assert_eq!(images.len(), 1);
}

#[test]
fn test_output_structure_mirroring_logic() {
    let input_root = Path::new("/data/photos");
    let image = Path::new("/data/photos/vacation/beach.jpg");
    let output_root = Path::new("/data/crops");

    let relative = relative_to(image, input_root).expect("relative path");
    let output_path = output_root.join(&relative);

    assert_eq!(output_path, Path::new("/data/crops/vacation/beach.jpg"));
}
