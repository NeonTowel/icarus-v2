# AGENTS.md: Development Guidelines for Icarus-v2

**Icarus-v2** is a production-ready Rust system for AI-powered object detection and intelligent image cropping, using YOLOv10 via ONNX Runtime and Candle-based models for multiple detection backends.

## Quick Reference

### Project Type
- **Language:** Rust (Edition 2021)
- **Build System:** Cargo
- **Architecture:** CLI application + library with multiple AI models (YOLOv10, DETR, RT-DETR, etc.)
- **Testing:** Integrated tests in `tests/` directory + inline unit tests in modules

---

## Build & Commands

### Install Dependencies
```bash
task deps
# Installs protoc (required for candle-onnx) to ~/.local/bin
# Auto-detects OS/arch (Linux x86_64/aarch64, macOS x86_64/arm64)
```

### Build
```bash
cargo build                    # Debug build
cargo build --release         # Release build (optimized)
task build                     # Using Taskfile
task build-without-warnings    # Build with warnings suppressed
```

### Run Tests
```bash
# Run all tests (unit + integration)
cargo test

# Run a specific test by name
cargo test test_crop_image_produces_correct_dimensions

# Run tests in a specific module
cargo test image_utils::

# Run integration tests only
cargo test --test integration_tests

# Run with output (don't suppress println!)
cargo test -- --nocapture

# Run a single test file
cargo test --test detr_candle_test
```

### Common Development Tasks
```bash
task test-crop          # Test multi-format cropping with sample photos
task test-yolo          # Test YOLOv10 model specifically
task test-batch         # Batch process multiple images
task test-align-issue   # Test margin/alignment edge cases
```

### Example: Run Single Test
```bash
cargo test test_crop_image_produces_correct_dimensions -- --nocapture
```

---

## Code Style Guidelines

### File Organization
```
src/
├── main.rs                    # CLI entry point (argument parsing, orchestration)
├── lib.rs                     # Library root, module exports
├── error.rs                   # Error types (Error enum, Result alias)
├── config.rs                  # Configuration structs (with Default impls)
├── image_utils.rs             # Image cropping, normalization utilities
├── multi_format_cropping.rs   # Multi-format crop logic (21:9, 9:21, 9:16)
└── models/                    # Model loading & inference
    ├── mod.rs
    ├── candle_backend.rs      # Candle-based inference (DETR, RT-DETR, etc.)
    └── yolov10_onnx.rs        # YOLOv10-specific ONNX integration

tests/
├── integration_tests.rs       # Cross-cutting image_utils tests
├── detr_candle_test.rs        # DETR model inference tests
├── rt_detr_candle_test.rs     # RT-DETR model inference tests
└── fixtures/                  # Test images/data
```

### Imports
- **Order:** `use` statements organized as:
  1. Standard library (`std::*`)
  2. External crates (alphabetical)
  3. Internal crate modules (`use crate::*`)
- **Example:**
  ```rust
  use std::path::PathBuf;
  use anyhow::Result;
  use image::DynamicImage;
  use crate::error::Error;
  use crate::models::load_model;
  ```

### Formatting & Whitespace
- **Line length:** Keep ≤100 characters (soft limit, 120 hard limit)
- **Indentation:** 4 spaces (enforced by rustfmt)
- **Blank lines:** Use between logical sections (but not excessively)
- **Run:** `rustfmt --check src/` (auto-format via `cargo fmt`)

### Types & Generics
- **Type annotations:** Explicit in public APIs, inferred in internal code
- **Generic constraints:** Placed in `where` clauses for clarity
- **Example:**
  ```rust
  pub fn crop_image(img: &DynamicImage, bbox: [f32; 4]) -> Result<DynamicImage> { }
  
  fn process_batch<T>(items: &[T]) -> Result<Vec<Output>>
  where
      T: AsRef<Path>,
  { }
  ```

### Naming Conventions
- **Modules/Crates:** `snake_case`
  - Files: `image_utils.rs`, `multi_format_cropping.rs`
- **Functions/Methods:** `snake_case`
  - Public: `load_model()`, `crop_image()`, `detect_suitable_formats()`
  - Private: `_process_inference()`, `_validate_bbox()`
- **Types/Structs/Enums:** `PascalCase`
  - `DynamicImage`, `BBox`, `CropRegion`, `DetectionConfig`
- **Constants:** `UPPER_SNAKE_CASE`
  - `VALID_MODELS`, `DEFAULT_CONFIDENCE`, `MAX_DIMENSION`
- **Abbreviations OK:** `bbox` (common), `img` (common), `onnx` (acronym)

### Documentation
- **Doc comments:** Use `///` for public items (required for public API)
- **Module docs:** `//!` at module root
- **Example:**
  ```rust
  /// Crops an image to a region defined by `[x1, y1, x2, y2]` coordinates.
  ///
  /// # Arguments
  /// * `img` - Input image
  /// * `bbox` - Bounding box as `[x1, y1, x2, y2]` in pixels
  ///
  /// # Returns
  /// A cropped `DynamicImage` or an error if bbox is invalid.
  ///
  /// # Example
  /// ```rust,ignore
  /// let bbox = [10.0, 20.0, 100.0, 150.0];
  /// let crop = crop_image(&img, bbox)?;
  /// ```
  pub fn crop_image(img: &DynamicImage, bbox: [f32; 4]) -> Result<DynamicImage> { }
  ```

### Error Handling
- **Use `Result<T>`:** Return results for fallible operations (defined in `error.rs`)
- **Error type:** `anyhow::Result<T>` for flexibility; `Error` enum for specific errors
- **Pattern:**
  ```rust
  pub type Result<T> = anyhow::Result<T>;
  
  pub fn process() -> Result<DynamicImage> {
      let img = image::open("path")?;  // ? propagates errors
      validate_image(&img)?;
      Ok(img)
  }
  ```
- **Custom errors:** Use `#[derive(Error)]` from `thiserror` crate
  ```rust
  #[derive(Error, Debug)]
  pub enum Error {
      #[error("Invalid bbox: {0}")]
      InvalidBbox(String),
  }
  ```
- **Error context:** Use `anyhow::Context` for adding context:
  ```rust
  let img = image::open(path).context("failed to load image")?;
  ```

### Struct Design
- **Derive traits:** `Debug`, `Clone` for public types
- **Builder pattern:** For complex configuration (e.g., `DetectionConfig`)
- **Example:**
  ```rust
  #[derive(Debug, Clone)]
  pub struct BBox {
      pub x1: f32,
      pub y1: f32,
      pub x2: f32,
      pub y2: f32,
  }
  
  impl BBox {
      pub fn width(&self) -> f32 { self.x2 - self.x1 }
      pub fn center_x(&self) -> f32 { (self.x1 + self.x2) / 2.0 }
  }
  ```

### Testing
- **Inline tests:** Use `#[cfg(test)] mod tests { }` in same file
- **Integration tests:** Place in `tests/` directory
- **Test naming:** Descriptive (`test_crop_image_clamps_to_image_bounds`)
- **Assertions:** Clear error messages
  ```rust
  #[test]
  fn test_crop_image_dimensions() {
      let bbox = [50.0, 60.0, 200.0, 180.0];
      let crop = crop_image(&img, bbox).expect("valid bbox");
      assert_eq!(crop.width(), 150, "width = x2 - x1");
  }
  ```

### Clippy Warnings
- Fix all clippy warnings before committing
- Common issues: unused variables, inefficient clones, missing docs
- Run: `cargo clippy --all-targets --all-features`

---

## Key Files & Responsibilities

| File | Purpose |
|------|---------|
| `src/main.rs` | CLI argument parsing via clap, orchestration |
| `src/lib.rs` | Library root, module exports |
| `src/error.rs` | Error types, Result alias |
| `src/image_utils.rs` | Image I/O, normalization, cropping |
| `src/multi_format_cropping.rs` | 21:9 / 9:21 / 9:16 intelligent cropping |
| `src/models/` | Model loading & inference (Candle + ONNX) |
| `src/config.rs` | Configuration with sensible defaults |
| `tests/` | Integration + unit tests |
| `Cargo.toml` | Dependencies, package metadata |
| `Taskfile.yaml` | Task automation (build, test, run) |

---

## Common Tasks for Agents

1. **Bug Fix:** Identify failing test → Fix implementation → Run full test suite
2. **New Feature:** Add struct/trait → Implement logic → Write tests → Doc comments
3. **Refactoring:** Extract function → Move to module → Update imports → Test
4. **Testing:** Add `#[test]` to module or create file in `tests/` → Run `cargo test`

---

## CI/Lint Standards

- **Format:** `cargo fmt` (auto-formatting)
- **Lint:** `cargo clippy` (warnings must be fixed)
- **Tests:** All tests must pass (`cargo test`)
- **Doc:** Public items must have doc comments (`///`)

---

*Last updated: 2026-03-22 | Rust Edition 2021 | Icarus-v2 v2.0.0*
