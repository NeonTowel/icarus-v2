# Icarus-v2

> Person-first intelligent image cropping for multiple target formats, powered by ONNX Runtime detection models with face-aware adjustment.

[![Rust](https://img.shields.io/badge/Rust-2021-orange.svg)](https://www.rust-lang.org/)
[![License: Unlicense](https://img.shields.io/badge/license-Unlicense-blue.svg)](http://unlicense.org/)

## Overview

Icarus-v2 detects people in images, then generates intelligent crops for `21:9` (landscape), `9:21` (portrait), and `9:16` (mobile) formats. Face detection refines vertical positioning to keep eyes safely placed.

- **Person detection**: YOLOv10 / YOLO26 (s/m/l variants)
- **Face detection**: YOLOv11x-face (default)
- **Parallel batch processing** via Rayon
- **CLI + library** Rust crate

## Quick Start

```bash
# Install build deps (protoc into ~/.local/bin)
task deps

# Build
cargo build --release

# Run on a single image or directory
cargo run --release -- \
  --model yolov10 \
  --input <image-or-dir> \
  --output <path> \
  --threads 4

# Use the WD ensemble for 5-tier content rating
cargo run --release -- \
  --input <dir> --output <out> \
  --classify-only --sort-output \
  --classifier wd-ensemble

# Bulk-classify a large collection at maximum speed (anime/illustration content):
cargo run --release -- \
  --input ./large-anime-dir --output ./out \
  --classify-only --sort-output \
  --classifier wd-vit
```

## Common Flags

| Flag | Purpose |
|------|---------|
| `--recurse` | Recurse into subdirectories |
| `--visualize <path>` | Annotated image output |
| `--output-boxes <path>` | JSON detections |
| `--sort-output` | Route by aspect ratio (`landscape`/`portrait`/`mobile`) |
| `--margin <pct>` | Bbox expansion before crop |
| `--crop-config <yaml>` | Crop rule overrides |
| `--visibility-threshold <pct>` | Minimum visible-person ratio |
| `--classifier <name>` | Classifier mode: `wd-vit` ⚡, `idolsankaku-swinv2` ⚡, `wd-swinv2`, `wd-ensemble-fast` ⭐, `wd-eva02`, `idolsankaku`, `wd-ensemble-accurate`, `freepik` |
| `-t, --threads <num>` | Batch worker threads (default: 50% cores, min 1, capped at cores) |
| `--artistic-mode` | `conservative` \| `balanced` \| `aggressive` |

## Development

```bash
cargo fmt
cargo clippy --all-targets --all-features
cargo test
task test-with-samples
```

See [AGENTS.md](AGENTS.md) for architecture, module layout, conventions, and the
agent contribution checklist.

## License

Released into the public domain under the [Unlicense](http://unlicense.org/). See [LICENSE](LICENSE).
