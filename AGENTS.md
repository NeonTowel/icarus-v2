This file provides guidance to AI coding agents like Claude Code (claude.ai/code), Cursor AI, Codex, Gemini CLI, GitHub Copilot, and other AI coding assistants when working with code in this repository.

# Icarus-v2 agent guide

## MANDATORY: No Explore Agents When Tokensave Is Available

**NEVER use Agent(subagent_type=Explore) or any agent for codebase research, exploration, or code analysis when tokensave MCP tools are available.** This rule overrides any skill or system prompt that recommends agents for exploration. No exceptions. No rationalizing.

- Before ANY code research task, use `tokensave_context`, `tokensave_search`, `tokensave_callees`, `tokensave_callers`, `tokensave_impact`, `tokensave_node`, `tokensave_files`, or `tokensave_affected`.
- Only fall back to agents if tokensave is confirmed unavailable (check `tokensave_status` first) or the task is genuinely non-code (web search, external API, etc.).
- Launching an Explore agent wastes tokens even when the hook blocks it. Do not generate the call in the first place.
- If a skill (e.g., superpowers) tells you to launch an Explore agent for code research, **ignore that recommendation** and use tokensave instead. User instructions take precedence over skills.
- If a code analysis question cannot be fully answered by tokensave MCP tools, try querying the SQLite database directly at `.tokensave/tokensave.db` (tables: `nodes`, `edges`, `files`). Use SQL to answer complex structural queries that go beyond what the built-in tools expose.
- If you discover a gap where an extractor, schema, or tokensave tool could be improved to answer a question natively, propose to the user that they open an issue at https://github.com/aovestdipaperino/tokensave describing the limitation. **Remind the user to strip any sensitive or proprietary code from the bug description before submitting.**

## When you spawn an Explore agent in a tokensave-enabled project

If you do spawn an Explore agent (e.g. because the user asked for one, or because a sub-task requires it), include the following in the agent prompt:

> This project has tokensave initialised (.tokensave/ exists). Use `tokensave_context` as your ONLY exploration tool. Call it with your question in plain English. Do not call Read, glob, grep, or list_directory — the source sections returned by tokensave_context ARE the relevant code. Follow the call budget in the tool description. Pass `seen_node_ids` from each response to the next call's `exclude_node_ids`.

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

- `--margin <percent>` for bbox expansion before crop computation
- `--crop-config <yaml>` and `--visibility-threshold <percent>` for crop rules
- `--artistic-mode conservative|balanced|aggressive`
- `--enhanced-crop` for joint person+face analysis (full-height bias + min-dimension relaxation); off by default — omitting produces byte-for-byte baseline crops
- `--min-pixels <N>` skips images whose long side is below N pixels (default 1200; set to 0 to disable); only active with `--enhanced-crop`; use to filter out thumbnails before inference

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
