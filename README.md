# Icarus-v2

> **Production-ready Rust system for AI-powered object detection and intelligent image cropping.**

Detect objects in images with state-of-the-art neural networks (YOLOv10, DETR, RT-DETR) and automatically crop to ideal aspect ratios. Zero C++ dependencies. Pure Rust. Fast inference. Flexible.

[![Rust](https://img.shields.io/badge/Rust-2021-orange.svg)](https://www.rust-lang.org/)
[![License: Unlicense](https://img.shields.io/badge/license-Unlicense-blue.svg)](http://unlicense.org/)

---

## ✨ Key Features

- **Multiple Detection Models**: YOLOv10 (ONNX), DETR, RT-DETR, RF-DETR via Candle ML framework
- **Intelligent Multi-Format Cropping**: Automatically detect and crop to landscape (21:9), portrait (9:21), or square (9:16) formats
- **Zero C++ Dependencies**: Pure Rust implementation using Candle ML and ONNX Runtime
- **Fast Inference**: GPU-accelerated (CUDA/Metal) or CPU fallback
- **CLI + Library**: Use as a command-line tool or integrate into Rust applications
- **Automatic Model Downloads**: Models cached locally from HuggingFace Hub on first use
- **Production Ready**: Robust error handling, comprehensive testing, full documentation

---

## 🚀 Quick Start

### Prerequisites
- **Rust 1.70+** ([install rustup](https://rustup.rs/))
- **~4GB RAM** for inference
- GPU optional (NVIDIA/AMD/Intel for acceleration)

### Installation

Clone the repository:
```bash
git clone https://github.com/NeonTowel/icarus-v2.git
cd icarus-v2
```

Install build dependencies (automatic):
```bash
task deps
# Installs protoc (required for candle-onnx) to ~/.local/bin
```

Build the project:
```bash
cargo build --release
```

### Run Your First Detection

```bash
cargo run --release -- \
  --model yolov10 \
  --input input/sample.jpg \
  --output output/crop.jpg \
  --visualize output/annotated.jpg
```

**Output:**
- `crop.jpg` — Automatically cropped region
- `annotated.jpg` — Image with bounding boxes drawn

---

## 📖 Usage

### CLI Overview

```bash
icarus-v2 [OPTIONS] --input <FILE> --model <MODEL>
```

### Common Examples

**Detect with YOLOv10:**
```bash
cargo run --release -- \
  --model yolov10 \
  --input photo.jpg \
  --output crop.jpg \
  --confidence 0.5
```

**Batch Process Images:**
```bash
for img in input/*.jpg; do
  cargo run --release -- \
    --model yolov10 \
    --input "$img" \
    --output "output/$(basename $img)" \
    --quiet
done
```

**Multi-Format Cropping with Margin:**
```bash
cargo run --release -- \
  --model yolov10 \
  --input person.jpg \
  --output crop.jpg \
  --visualize annotated.jpg \
  --margin 10.0
```

**DETR Model (Transformer-based):**
```bash
cargo run --release -- \
  --model detr-resnet50 \
  --input image.jpg \
  --output crop.jpg \
  --confidence 0.25
```

### CLI Options

```
OPTIONS:
  -i, --input <FILE>              Input image path (JPEG, PNG, BMP, TIFF, GIF)
  -o, --output <FILE>             Output cropped image (optional)
  -m, --model <MODEL>             Detection model (default: yolov10)
  -c, --confidence <FLOAT>         Confidence threshold 0.0–1.0 (default: 0.5)
  --visualize <FILE>              Save annotated image with bounding boxes
  --margin <FLOAT>                Padding around detection (% of bbox, default: 0)
  --format <FORMAT>               Force crop format: landscape|portrait|square
  -q, --quiet                     Suppress stdout output
  -h, --help                      Print help
  -V, --version                   Print version
```

### Library Usage

Use Icarus-v2 as a Rust library:

```rust
use icarus_v2::{
    image_utils::crop_image,
    multi_format_cropping::{BBox, detect_suitable_formats},
};
use image::open;

fn main() -> anyhow::Result<()> {
    // Load image
    let img = open("input.jpg")?;
    
    // Define bounding box [x1, y1, x2, y2]
    let bbox = BBox {
        x1: 100.0,
        y1: 50.0,
        x2: 400.0,
        y2: 900.0,
    };
    
    // Detect suitable crop formats
    let (width, height) = (img.width(), img.height());
    let formats = detect_suitable_formats(width, height, &bbox, 5.0);
    println!("Suitable formats: {:?}", formats);
    
    // Crop to detected format
    let crop = crop_image(&img, [bbox.x1, bbox.y1, bbox.x2, bbox.y2])?;
    crop.save("output.jpg")?;
    
    Ok(())
}
```

---

## 🧠 Supported Models

| Model | Type | Input Size | Status | Backend |
|-------|------|-----------|--------|---------|
| **yolov10** | Anchor-free YOLO | 640×640 | ✅ Production | ONNX Runtime |
| **detr-resnet50** | Transformer DETR | 800×800 | ✅ Ready | Candle |
| **rt-detr-large** | Transformer RT-DETR | 640×640 | ✅ Ready | Candle |
| **rf-detr-medium** | Transformer RF-DETR | 640×640 | ✅ Ready | Candle |
| **rf-detr-large** | Transformer RF-DETR | 640×640 | ✅ Ready | Candle |

Models are **automatically downloaded** from HuggingFace Hub on first use and cached locally at `~/.cache/huggingface/hub/`.

---

## 🎯 Multi-Format Intelligent Cropping

Icarus-v2 analyzes detected objects and recommends the best crop format:

```
┌─────────────────────────────────┐
│  Landscape (21:9) Format        │  Wide aspect ratio
│  📸 Face-first positioning       │  Ideal for wallpapers
└─────────────────────────────────┘

┌─────┐
│ 9:21│  Portrait (9:21) Format
│     │  Tall aspect ratio
│ 9:16│  Portrait (9:16) Format
└─────┘  Common phone/social media
```

### How It Works

1. **Detect bounding box** around objects using your chosen model
2. **Analyze bbox geometry**: Wide → landscape only; tall → try all formats
3. **Position intelligently**:
   - **Landscape 21:9**: Face zone (top 30%) at upper-third of crop
   - **Portrait 9:21 & 9:16**: Person center at 45% from crop top
4. **Apply margin** (optional symmetric padding as % of bbox)
5. **Visibility gate**: Reject if object <50% visible in final crop

### Example: Auto-Crop a Portrait

```bash
# Input: 3000×4000 photo with person bbox [500, 300, 1500, 3500]
cargo run --release -- \
  --model yolov10 \
  --input portrait.jpg \
  --output crops/ \
  --margin 5.0
```

Icarus-v2 automatically detects:
- ✅ Portrait 9:21 (tall)
- ✅ Portrait 9:16 (common phone)
- ❌ Landscape 21:9 (too much wasted space)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│  CLI (main.rs)                      │
│  • Argument parsing via clap        │
│  • Image I/O & orchestration        │
└──────────────┬──────────────────────┘
               │
        ┌──────▼──────────┐
        │  Library (lib.rs)│
        └──────┬──────────┘
               │
    ┌──────────┼──────────┐
    │          │          │
┌───▼───┐ ┌───▼────┐ ┌──▼──────────┐
│Models │ │Image   │ │Multi-Format │
│(ONNX/ │ │Utils   │ │Cropping     │
│Candle)│ │        │ │             │
└───────┘ └────────┘ └─────────────┘
```

### Module Overview

- **`src/main.rs`**: CLI entry point, argument parsing, orchestration
- **`src/lib.rs`**: Public library API
- **`src/error.rs`**: Custom error types
- **`src/config.rs`**: Configuration with sensible defaults
- **`src/image_utils.rs`**: Image I/O, normalization, cropping utilities
- **`src/multi_format_cropping.rs`**: Intelligent aspect ratio detection and positioning
- **`src/models/`**: Model loading and inference backends
  - `candle_backend.rs` — Candle ML framework integration (DETR, RT-DETR, RF-DETR)
  - `yolov10_onnx.rs` — YOLOv10-specific ONNX Runtime integration

---

## 🧪 Testing

Run all tests:
```bash
cargo test
```

Run a specific test:
```bash
cargo test test_crop_image_produces_correct_dimensions -- --nocapture
```

Run integration tests:
```bash
cargo test --test integration_tests
```

Run real-world example tests:
```bash
task test-crop          # Multi-format cropping with sample photos
task test-yolo          # YOLOv10-specific tests
task test-batch         # Batch processing
task test-align-issue   # Edge cases and alignment
```

---

## 📊 Performance

**Inference Speed** (CPU, RTX 3090):
- **YOLOv10**: ~50 ms per image
- **DETR-ResNet50**: ~100 ms per image
- **RT-DETR**: ~80 ms per image

**Memory Usage**:
- **Model loading**: ~200–500 MB
- **Per-image inference**: ~100–300 MB

*Times vary based on image size, hardware, and model.*

---

## 🔧 Development

### Build Commands

```bash
cargo build                    # Debug build
cargo build --release         # Optimized release build
task build                     # Via Taskfile
task build-without-warnings    # Suppress clippy warnings
```

### Linting & Formatting

```bash
cargo fmt                      # Auto-format code
cargo clippy                   # Lint and suggestions
cargo test                     # Run full test suite
```

### Code Style

See [AGENTS.md](AGENTS.md) for detailed guidelines:
- Rust Edition 2021
- Max line length: 100 characters (soft), 120 (hard)
- Doc comments required for public APIs
- Error handling via `anyhow::Result<T>`
- Comprehensive unit + integration tests

---

## 📦 Dependencies

**Core ML/Image Processing:**
- `image` — Image I/O and manipulation
- `candle-core`, `candle-onnx`, `candle-nn` — Hugging Face Candle ML framework
- `ort` — ONNX Runtime bindings (auto-downloads native libraries)
- `yolov10` — YOLOv10-specific utilities

**CLI & Config:**
- `clap` — Command-line argument parsing
- `serde`, `serde_json` — Serialization

**Utilities:**
- `anyhow`, `thiserror` — Error handling
- `tokio` — Async runtime
- `log`, `env_logger` — Logging
- `nalgebra` — Linear algebra
- `uuid` — UUID generation
- `hf-hub` — HuggingFace Hub client
- `safetensors` — Model weight format

All dependencies are in [Cargo.toml](Cargo.toml) with pinned versions for reproducibility.

---

## 🐛 Troubleshooting

### "protoc not found"
```bash
task deps
# Installs protoc to ~/.local/bin automatically
```

Then add to your shell profile:
```bash
export PATH="$HOME/.local/bin:$PATH"
```

### Model Download Fails
- Ensure internet connectivity
- Check `~/.cache/huggingface/hub/` permissions
- Models automatically retry with exponential backoff

### Out of Memory
- Reduce image dimensions: `cargo run -- --input image.jpg --resize 0.5`
- Use CPU (slower but less RAM): Models default to CPU
- Process images in smaller batches

### GPU Not Detected
- Ensure CUDA/Metal drivers are installed
- Candle automatically falls back to CPU
- Set `CUDA_VISIBLE_DEVICES` to control which GPU to use

---

## 🤝 Contributing

Contributions welcome! Please:

1. **Fork** the repository
2. **Create a branch** for your feature or fix
3. **Write tests** for new functionality
4. **Run `cargo test` + `cargo clippy`** to verify
5. **Submit a pull request** with clear description

See [AGENTS.md](AGENTS.md) for detailed development guidelines.

---

## 📄 License

This project is released into the public domain under the **[Unlicense](http://unlicense.org/)**.

```
This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.
```

See [LICENSE](LICENSE) file for full details.

---

## 🔗 Resources

- **[Candle ML Framework](https://github.com/huggingface/candle)** — Pure Rust ML
- **[YOLOv10](https://github.com/THU-MIG/yolov10)** — Fast object detection
- **[DETR](https://github.com/facebookresearch/detr)** — Transformer-based detection
- **[HuggingFace Hub](https://huggingface.co/)** — Model repository
- **[Rust Book](https://doc.rust-lang.org/book/)** — Rust language guide

---

## 📞 Feedback & Questions

Have ideas or questions? Open an issue on GitHub or start a discussion. All contributions valued!

---

**Built with ❤️ in Rust • Last Updated: March 2026 • v2.0.0**
