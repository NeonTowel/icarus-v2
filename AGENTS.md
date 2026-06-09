This file provides guidance to AI coding agents like Claude Code (claude.ai/code), Cursor AI, Codex, Gemini CLI, GitHub Copilot, and other AI coding assistants when working with code in this repository.

# Icarus-v2 agent guide

## Project snapshot

- Language/runtime: Rust 2021 (`cargo` workspace with one binary + one library crate).
- Entry points:
  - Binary: `src/main.rs` (`icarus-v2` CLI)
  - Library: `src/lib.rs`
- Core purpose: person-first intelligent cropping across multiple target formats using
  ONNX Runtime-backed detection models and face-aware post-adjustment.

## Canonical commands

## Dependency setup

```bash
task deps
```

Installs `protoc` (required by Candle dependencies) to `~/.local/bin` when missing.

## Build

```bash
cargo build
cargo build --release
task build
task build-without-warnings
```

## Test

```bash
cargo test
cargo test <test_name> -- --nocapture
```

Task-based sample-run workflow:

```bash
task test-with-samples
task test-with-samples_beta
```

## Lint/format

```bash
cargo fmt
cargo clippy --all-targets --all-features
```

## CLI invocation pattern

```bash
cargo run --release -- --model yolov10 --input <image-or-dir> --output <path>
```

Useful flags implemented in `src/main.rs`:

- `--recurse` for recursive directory processing
- `--visualize <path>` for annotated output
- `--output-boxes <path>` for JSON detections
- `--sort-output` for aspect-ratio subfolder organization
- `--classifier freepik|wd-eva02|idolsankaku|wd-ensemble` for content rating backend selection
- `--margin <percent>` for bbox expansion before crop computation
- `--crop-config <yaml>` and `--visibility-threshold <percent>` for crop rules
- `--artistic-mode conservative|balanced|aggressive`

## Architecture (high level)

## Pipeline flow

1. Parse CLI args and validate them (`main.rs`).
2. Load crop config + artistic mode (`config.rs`).
3. Load person detector (`models::load_candle_model`) and face detector (`face_detection.rs`).
4. Dispatch by input type:
   - File: single-image processing
   - Directory: discover images, then parallel batch processing
5. For each image (`batch_processor.rs`):
   - preprocess → forward → postprocess via selected model
   - confidence filter and person extraction
   - optional reflection/overlap dedup and compound bbox
   - face detection (best-effort; non-fatal failure path)
   - format suitability detection (`multi_format_cropping.rs`)
   - crop generation + eye-safety enforcement (`face_aware_cropping.rs`)
   - optional visualization + JSON output

## Main modules and responsibilities

- `src/main.rs`
  - CLI surface and top-level orchestration.
  - Input path mode switching (single vs batch).

- `src/batch_processor.rs`
  - Core processing pipeline used by both single and batch modes.
  - Parallel batch execution via Rayon.
  - Output writing (crop, visualization, JSON).

- `src/multi_format_cropping.rs`
  - Format suitability + crop region generation for `21:9`, `9:21`, `9:16`.
  - Visibility gating and directional thirds logic.
  - Multi-person helpers (compound bbox, dedup logic).

- `src/focal_point.rs`
  - Computes focal anchor (currently bbox-center driven).

- `src/face_aware_cropping.rs`
  - Applies eye-safety vertical nudge with strict bounded budget.

- `src/face_detection.rs`
  - Face detector loading and inference abstraction.
  - Converts model bbox type into crop-domain bbox type.

- `src/models/`
  - `candle_backend.rs`: shared trait (`Model`) + detection primitives + utilities.
  - `implementations/`: ONNX Runtime-backed YOLOv10/YOLO26 variants and face models.
  - `backbones/`: currently deferred/stubbed DINOv2 placeholders for future RF-DETR work.

- `src/output_sorting.rs`
  - Aspect-ratio to subfolder routing (`landscape`, `portrait`, `mobile`)
  - Annotated filename suffix policy.

- `src/directory_walker.rs`
  - Extension-filtered image discovery with optional recursion.

## Model/back-end status

Person models currently wired in CLI and loader:

- `yolov10`, `yolov10s`, `yolov10m`
- `yolo26`, `yolo26s`, `yolo26m`

Face detection model path:

- Default: YOLOv11x-face
- Alternate fast path is implemented (`load_fast_face_detector`) but not default.

Classifiers currently wired in CLI and loader:

- `freepik` — existing 4-tier Freepik classifier (`Freepik/nsfw_image_detector`)
- `wd-eva02` — 5-tier rating head from
  `SmilingWolf/wd-eva02-large-tagger-v3`
- `idolsankaku` — 5-tier rating head from
  `deepghs/idolsankaku-eva02-large-tagger-v1`
- `wd-ensemble` — confidence-weighted combination of `wd-eva02` +
  `idolsankaku` using shared 5-tier severity mapping

Note: Several Candle-centric/deferred wrappers exist for future roadmap work; do not assume
they are production-ready without checking loader wiring in `src/models/mod.rs` and CLI
validation in `src/main.rs`.

## Conventions to preserve when editing

- Keep person detection and face detection concerns separated (face module should not own crop logic).
- Preserve fail-fast validation for CLI numeric ranges and input path semantics.
- Keep face detection failure non-fatal during processing (warn + continue).
- Keep output path creation idempotent and parent-directory-safe.
- Keep crop coordinate clamping before image writes.
- Preserve single-image path parity with batch path by routing through shared pipeline code.

## Agent checklist before finishing changes

1. Run `cargo fmt`.
2. Run `cargo clippy --all-targets --all-features` (or document why it was skipped).
3. Run `cargo test` (or at minimum targeted tests for modified modules).
4. If CLI/output behavior changed, include a runnable example command in your summary.
