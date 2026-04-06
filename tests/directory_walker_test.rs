use icarus_v2::directory_walker::{discover_images, is_supported_image, relative_to};
use std::path::Path;
use tempfile::TempDir;

#[test]
fn test_is_supported_image_recognises_common_extensions() {
    assert!(is_supported_image(Path::new("photo.jpg")));
    assert!(is_supported_image(Path::new("photo.JPEG")));
    assert!(is_supported_image(Path::new("photo.PNG")));
    assert!(!is_supported_image(Path::new("readme.txt")));
    assert!(!is_supported_image(Path::new("model.onnx")));
}

#[test]
fn test_discover_images_non_recursive_only_reads_top_level() {
    let temp_dir = TempDir::new().expect("temp dir");
    std::fs::write(temp_dir.path().join("a.jpg"), b"fake").expect("write a.jpg");
    std::fs::write(temp_dir.path().join("b.png"), b"fake").expect("write b.png");
    std::fs::write(temp_dir.path().join("c.txt"), b"not image").expect("write c.txt");
    std::fs::create_dir(temp_dir.path().join("subdir")).expect("create subdir");
    std::fs::write(temp_dir.path().join("subdir/d.jpg"), b"fake").expect("write d.jpg");

    let found = discover_images(temp_dir.path(), false).expect("discover images");
    assert_eq!(found.len(), 2, "should not recurse into subdir");
}

#[test]
fn test_discover_images_recursive_includes_nested_images() {
    let temp_dir = TempDir::new().expect("temp dir");
    std::fs::write(temp_dir.path().join("a.jpg"), b"fake").expect("write a.jpg");
    std::fs::create_dir(temp_dir.path().join("subdir")).expect("create subdir");
    std::fs::write(temp_dir.path().join("subdir/b.jpg"), b"fake").expect("write b.jpg");
    std::fs::create_dir_all(temp_dir.path().join("subdir/deep")).expect("create deep dir");
    std::fs::write(temp_dir.path().join("subdir/deep/c.png"), b"fake").expect("write c.png");

    let found = discover_images(temp_dir.path(), true).expect("discover images");
    assert_eq!(found.len(), 3, "should find all nested images");
}

#[test]
fn test_discover_images_skips_hidden_entries() {
    let temp_dir = TempDir::new().expect("temp dir");
    std::fs::write(temp_dir.path().join("visible.jpg"), b"fake").expect("write visible");
    std::fs::write(temp_dir.path().join(".hidden.jpg"), b"fake").expect("write hidden");
    std::fs::create_dir(temp_dir.path().join(".hidden_dir")).expect("create hidden dir");
    std::fs::write(temp_dir.path().join(".hidden_dir/photo.jpg"), b"fake")
        .expect("write nested hidden");

    let found = discover_images(temp_dir.path(), true).expect("discover images");
    assert_eq!(found.len(), 1, "should skip hidden files and directories");
}

#[test]
fn test_relative_to_returns_relative_path() {
    let rel = relative_to(
        Path::new("/photos/vacation/beach.jpg"),
        Path::new("/photos"),
    );

    assert_eq!(rel, Some(Path::new("vacation/beach.jpg").to_path_buf()));
}

#[test]
fn test_discover_images_rejects_files() {
    let temp_dir = TempDir::new().expect("temp dir");
    let file = temp_dir.path().join("not_a_dir.jpg");
    std::fs::write(&file, b"fake").expect("write file");

    assert!(discover_images(&file, false).is_err());
}
