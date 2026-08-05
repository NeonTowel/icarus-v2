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

```

## Common Flags

| Flag                           | Purpose                                                 |
| ------------------------------ | ------------------------------------------------------- |
| `--recurse`                    | Recurse into subdirectories                             |
| `--visualize <path>`           | Annotated image output                                  |
| `--output-boxes <path>`        | JSON detections                                         |
| `--sort-output`                | Route by aspect ratio (`landscape`/`portrait`/`mobile`) |
| `--margin <pct>`               | Bbox expansion before crop                              |
| `--crop-config <yaml>`         | Crop rule overrides                                     |
| `--visibility-threshold <pct>` | Minimum visible-person ratio                            |

| `-t, --threads <num>` | Batch worker threads (default: 50% cores, min 1, capped at cores) |
| `--artistic-mode` | `conservative` \| `balanced` \| `aggressive` |
| `--review` | Generate baseline and enhanced candidates for `editor/data` |

## Crop Review Editor

Generate a local review dataset without creating normal crop output:

```bash
cargo run --release -- \
  --review \
  --input <image-or-directory> \
  --recurse \
  --model yolo26m
```

`--review` accepts only input and detector configuration: `--input`, `--recurse`,
`--model`, `--model-path`, `--confidence`, `--threads`, and `--quiet`. It rejects
normal output and crop-tuning flags to keep candidate generation reproducible.

The command writes raw JPEG source previews and per-image baseline/enhanced candidate
coordinates under ignored `editor/data/`. Start the editor with:

```bash
npm --prefix editor run dev
```

The editor shows person/face boxes, baseline and enhanced candidates, and a constrained
manual crop. Use **Reset manual crop** or `R` to discard uncommitted drag changes and
restore the baseline candidate. Exported review JSON records accepted baseline/enhanced
candidates or manual crop coordinates with reason codes.

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
