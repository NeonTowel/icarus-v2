/// Icarus-v2 CLI Entry Point
use anyhow::{bail, Context, Result};
use candle_core::Device;
use clap::Parser;
use icarus_v2::batch_processor::{process_single_image, run_batch, ProcessingContext};
use icarus_v2::config::{
    available_core_count, load_crop_config, resolve_thread_count, ArtisticCropConfig, ArtisticMode,
    CropConfig,
};
use icarus_v2::directory_walker::discover_images;
use icarus_v2::models::load_candle_model;
use icarus_v2::review::generate_dataset;
use std::path::{Path, PathBuf};

const VALID_MODELS: &[&str] = &[
    "yolov10", "yolov10s", "yolov10m", "yolo26", "yolo26s", "yolo26m",
];

#[derive(Parser, Debug)]
#[command(
    name = "icarus-v2",
    about = "AI Image Cropping System",
    long_about = "Detect objects in images using YOLOv10 via ONNX Runtime. \
                  Supports nano (default), small, and medium variants for accuracy/speed trade-offs. \
                  Supports saving cropped regions, annotated images, and raw detection JSON.",
    version
)]
struct Args {
    /// Input image file or directory path.
    #[arg(short, long, value_name = "PATH")]
    input: PathBuf,

    /// Output image path for cropped detection (skipped if omitted).
    #[arg(short, long, value_name = "FILE")]
    output: Option<PathBuf>,

    /// Recursively search subdirectories when --input is a directory.
    #[arg(long, default_value_t = false)]
    recurse: bool,

    /// Number of worker threads for batch processing.
    /// Defaults to 50% of available cores. Values above the core count are
    /// capped at the maximum available. Has no effect on single-file input.
    #[arg(short = 't', long, value_name = "NUM")]
    threads: Option<usize>,

    #[arg(
        long,
        default_value = "yolov10",
        value_name = "MODEL",
        help = "Person detection model. Options:\n  yolov10  — YOLOv10n nano (default, fastest)\n  yolov10s — YOLOv10s small (higher accuracy)\n  yolov10m — YOLOv10m medium (highest accuracy)\n  yolo26   — YOLO26n nano (fastest, 39ms CPU)\n  yolo26s  — YOLO26s small (balanced, 87ms CPU)\n  yolo26m  — YOLO26m medium (most accurate, 220ms CPU)"
    )]
    model: String,

    #[arg(long, value_name = "FILE")]
    model_path: Option<PathBuf>,

    #[arg(long, default_value = "0.5", value_name = "FLOAT")]
    confidence: f32,

    #[arg(long, value_name = "FILE")]
    output_boxes: Option<PathBuf>,

    #[arg(long, alias = "annotate", value_name = "FILE")]
    visualize: Option<PathBuf>,

    #[arg(long)]
    quiet: bool,

    #[arg(long, default_value_t = false)]
    keep_aspect_ratio: bool,

    #[arg(long, default_value_t = 0.0, value_name = "PERCENT")]
    margin: f32,

    #[arg(long, default_value_t = false)]
    sort_output: bool,

    #[arg(long, value_name = "FLOAT")]
    headroom_ratio: Option<f32>,

    #[arg(long, value_name = "FLOAT")]
    visibility_threshold: Option<f32>,

    #[arg(long, value_name = "FILE")]
    crop_config: Option<PathBuf>,

    #[arg(long, default_value = "balanced", value_name = "MODE")]
    artistic_mode: String,

    #[arg(long)]
    rename: bool,

    #[arg(long, num_args = 0..=1, default_missing_value = "98", value_name = "QUALITY")]
    jpeg: Option<u8>,

    #[arg(long)]
    flatten: bool,

    /// Print per-stage timing breakdown after processing (decode/person_detect/face_detect/crop/encode).
    /// Off by default; adding this flag changes no output files.
    #[arg(long, default_value_t = false)]
    metrics: bool,

    /// Use joint person+face analysis to improve full-person-height preservation and
    /// relax crops below minimum output dimensions. Off by default; when off, crops are
    /// byte-for-byte identical to the baseline. Overrides `enable_enhanced_crop` in
    /// `--crop-config`.
    #[arg(long, default_value_t = false)]
    enhanced_crop: bool,

    /// Minimum acceptable output size in pixels counted from the long side edge of the
    /// photo. Photos whose long side is below this are skipped before inference, and
    /// final crops below it are relaxed (more margin/headroom). Only active with
    /// --enhanced-crop. Applies to landscape (21:9), portrait (9:16), and mobile (9:21)
    /// categories. Set to 0 to disable the check. CLI value overrides crop_config YAML.
    #[arg(long, default_value_t = 1200, value_name = "PIXELS")]
    min_pixels: u32,

    /// Generate baseline and enhanced crop candidates for the local Icarus editor.
    /// Writes only under editor/data and rejects normal crop/output options.
    #[arg(long, default_value_t = false)]
    review: bool,
}

/// Thin entry point — always exits via [`std::process::exit`] to bypass ORT/tokio
/// teardown hang.
///
/// # Why not return from main normally?
///
/// ONNX Runtime registers global atexit handlers and spawns internal worker
/// threads for inference. When the tokio `#[tokio::main]` wrapper drops the
/// runtime it joins its blocking thread pool. Those threads' exit in turn
/// triggers ORT's global environment teardown, which blocks on ORT-internal
/// mutexes that are still held by ORT's own threads — a deadlock.
///
/// Calling `std::process::exit` skips all Rust `Drop` glue and tokio runtime
/// teardown. The OS reclaims all memory. This is safe for a batch CLI: all
/// output files are already written and closed by the time we reach exit.
fn main() {
    // env_logger must be initialised before the tokio runtime so that early
    // errors (arg parsing, model load) are visible.
    env_logger::init();

    let code = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_or_else(
            |e| {
                eprintln!("fatal: failed to build tokio runtime: {e}");
                1
            },
            |rt| match rt.block_on(run()) {
                Ok(()) => 0,
                Err(e) => {
                    eprintln!("Error: {e:#}");
                    1
                }
            },
        );

    std::process::exit(code);
}

async fn run() -> Result<()> {
    let args = Args::parse();
    validate_args(&args)?;
    if args.review {
        return run_review(&args).await;
    }

    let crop_config = build_crop_config(&args)?;
    let artistic_config = build_artistic_config(&args)?;
    // Compute thread_count before model loading so SessionPool can tune intra_op_threads (S3).
    let thread_count = resolve_thread_count(args.threads, available_core_count());
    let (model, face_detector) = load_models(&args, thread_count).await?;

    if let Some(ref model_path) = args.model_path {
        if !args.quiet {
            eprintln!(
                "Warning: --model-path {:?} is ignored; weights are loaded from HuggingFace Hub automatically.",
                model_path
            );
        }
    }

    let context = ProcessingContext {
        model: model.as_ref(),
        face_detector: &face_detector,
        crop_config: &crop_config,
        artistic_config: &artistic_config,
        confidence: args.confidence,
        margin: args.margin,
        keep_aspect_ratio: args.keep_aspect_ratio,
        sort_output: args.sort_output,
        quiet: args.quiet,
        rename: args.rename,
        jpeg_quality: args.jpeg,
        flatten: args.flatten,
        collect_metrics: args.metrics,
        enhanced_crop: args.enhanced_crop,
        // CLI --min-pixels always overrides crop_config.min_long_side_pixels (CLI wins).
        // 0 = disable the check (long_side_passes treats 0 as "always pass").
        min_long_side_pixels: args.min_pixels,
    };

    dispatch(&args, &context, thread_count).await
    // Note: model / face_detector are NOT explicitly dropped here.
    // std::process::exit in main() bypasses all Drop glue, which is intentional
    // — see the comment on main() above.
}

fn validate_args(args: &Args) -> Result<()> {
    if let Some(0) = args.threads {
        bail!("--threads must be >= 1 (omit the flag to auto-select 50% of cores)");
    }

    if args.review {
        let conflicting = review_conflicting_flags(args);
        if !conflicting.is_empty() {
            bail!(
                "--review only accepts --input, --recurse, --model, --model-path, --confidence, --threads, and --quiet; conflicting flags: {}",
                conflicting.join(", ")
            );
        }
    }

    Ok(())
}

fn review_conflicting_flags(args: &Args) -> Vec<&'static str> {
    let mut flags = Vec::new();
    if args.output.is_some() {
        flags.push("--output");
    }
    if args.output_boxes.is_some() {
        flags.push("--output-boxes");
    }
    if args.visualize.is_some() {
        flags.push("--visualize");
    }
    if args.keep_aspect_ratio {
        flags.push("--keep-aspect-ratio");
    }
    if args.margin != 0.0 {
        flags.push("--margin");
    }
    if args.sort_output {
        flags.push("--sort-output");
    }
    if args.headroom_ratio.is_some() {
        flags.push("--headroom-ratio");
    }
    if args.visibility_threshold.is_some() {
        flags.push("--visibility-threshold");
    }
    if args.crop_config.is_some() {
        flags.push("--crop-config");
    }
    if args.artistic_mode != "balanced" {
        flags.push("--artistic-mode");
    }
    if args.rename {
        flags.push("--rename");
    }
    if args.jpeg.is_some() {
        flags.push("--jpeg");
    }
    if args.flatten {
        flags.push("--flatten");
    }
    if args.metrics {
        flags.push("--metrics");
    }
    if args.enhanced_crop {
        flags.push("--enhanced-crop");
    }
    if args.min_pixels != 1200 {
        flags.push("--min-pixels");
    }
    flags
}

fn build_crop_config(args: &Args) -> Result<CropConfig> {
    if !(0.0..=1.0).contains(&args.confidence) {
        bail!(
            "--confidence must be between 0.0 and 1.0, got {}",
            args.confidence
        );
    }

    let mut crop_config = if let Some(ref config_path) = args.crop_config {
        if !config_path.exists() {
            bail!("Crop config file not found: {:?}", config_path);
        }
        load_crop_config(config_path)
            .with_context(|| format!("Failed to load crop config from {:?}", config_path))?
    } else {
        CropConfig::default()
    };

    if let Some(percent) = args.visibility_threshold {
        if !(0.0..=100.0).contains(&percent) {
            bail!(
                "--visibility-threshold must be between 0.0 and 100.0, got {}",
                percent
            );
        }
        crop_config.visibility_threshold = percent / 100.0;
    }

    Ok(crop_config)
}

fn build_artistic_config(args: &Args) -> Result<ArtisticCropConfig> {
    let artistic_mode = args
        .artistic_mode
        .parse::<ArtisticMode>()
        .with_context(|| {
            format!(
                "Invalid --artistic-mode '{}'. Options: conservative, balanced, aggressive",
                args.artistic_mode
            )
        })?;

    Ok(ArtisticCropConfig::from_mode(artistic_mode))
}

async fn load_models(
    args: &Args,
    thread_count: usize,
) -> Result<(
    Box<dyn icarus_v2::models::Model>,
    icarus_v2::face_detection::FaceDetector,
)> {
    if !VALID_MODELS.contains(&args.model.as_str()) {
        bail!(
            "Unknown model '{}'. Valid options are: {}",
            args.model,
            VALID_MODELS.join(", ")
        );
    }

    let device = Device::Cpu;
    let model = load_candle_model(&args.model, &device, thread_count)
        .await
        .with_context(|| format!("Failed to load model '{}'", args.model))?;
    let face_detector = tokio::task::spawn_blocking(move || {
        icarus_v2::face_detection::load_face_detector(thread_count)
    })
    .await
    .map_err(|error| anyhow::anyhow!("face model load task panicked: {error}"))??;

    Ok((model, face_detector))
}

async fn run_review(args: &Args) -> Result<()> {
    if !(0.0..=1.0).contains(&args.confidence) {
        bail!(
            "--confidence must be between 0.0 and 1.0, got {}",
            args.confidence
        );
    }

    let metadata = std::fs::metadata(&args.input)
        .with_context(|| format!("Cannot access input path: {:?}", args.input))?;
    if metadata.is_file() && args.recurse {
        bail!(
            "--recurse is only valid when --input is a directory, but {:?} is a file.",
            args.input
        );
    }

    let (input_root, image_paths) = if metadata.is_file() {
        let root = args
            .input
            .parent()
            .unwrap_or(Path::new("."))
            .canonicalize()
            .with_context(|| format!("Failed to canonicalize parent of {:?}", args.input))?;
        (root, vec![args.input.clone()])
    } else if metadata.is_dir() {
        let root = args
            .input
            .canonicalize()
            .with_context(|| format!("Failed to canonicalize input directory {:?}", args.input))?;
        let images = discover_images(&root, args.recurse)?;
        (root, images)
    } else {
        bail!(
            "Input path {:?} is neither a file nor a directory.",
            args.input
        );
    };

    if image_paths.is_empty() {
        if !args.quiet {
            println!("No supported images found in {:?}", args.input);
        }
        return Ok(());
    }

    let thread_count = resolve_thread_count(args.threads, available_core_count());
    let (model, face_detector) = load_models(args, thread_count).await?;
    let data_root = Path::new("editor").join("data");
    let manifest = generate_dataset(
        &image_paths,
        &input_root,
        &data_root,
        model.as_ref(),
        &face_detector,
        args.confidence,
    )?;

    if !args.quiet {
        println!(
            "Review dataset: {} sample(s) written to {:?}",
            manifest.samples.len(),
            data_root
        );
    }
    Ok(())
}

async fn dispatch(args: &Args, context: &ProcessingContext<'_>, thread_count: usize) -> Result<()> {
    let input_metadata = std::fs::metadata(&args.input)
        .with_context(|| format!("Cannot access input path: {:?}", args.input))?;

    if input_metadata.is_file() {
        if args.recurse {
            bail!(
                "--recurse is only valid when --input is a directory, but {:?} is a file.",
                args.input
            );
        }

        let result = process_single_image(
            &args.input,
            args.output.as_deref(),
            args.visualize.as_deref(),
            args.output_boxes.as_deref(),
            context,
        )?;

        if !args.quiet {
            println!(
                "Done. Found {} person(s), {} face(s).",
                result.person_count, result.face_count
            );
        }

        if args.metrics {
            let mut bm = icarus_v2::metrics::BatchMetrics::default();
            if let Some(m) = result.metrics {
                bm.record(m);
            }
            if let Some(a) = result.accuracy {
                bm.record_accuracy(a);
            }
            bm.print_summary();
        }

        return Ok(());
    }

    if !input_metadata.is_dir() {
        bail!(
            "Input path {:?} is neither a file nor a directory.",
            args.input
        );
    }

    let input_root = args
        .input
        .canonicalize()
        .with_context(|| format!("Failed to canonicalize input directory {:?}", args.input))?;

    let image_paths = discover_images(&input_root, args.recurse)?;
    if image_paths.is_empty() {
        if !args.quiet {
            println!("No supported images found in {:?}", args.input);
        }
        return Ok(());
    }

    if !args.quiet {
        println!("Found {} image(s) to process.", image_paths.len());
    }

    if !args.quiet {
        println!(
            "Using {} worker thread(s) for batch processing.",
            thread_count
        );
    }

    let summary = run_batch(
        &image_paths,
        &input_root,
        args.output.as_deref(),
        args.visualize.as_deref(),
        args.output_boxes.as_deref(),
        context,
        thread_count,
    );

    if !args.quiet {
        println!(
            "\nBatch complete: {}/{} succeeded, {} failed.",
            summary.succeeded,
            summary.total,
            summary.failed.len()
        );

        if !summary.failed.is_empty() {
            println!("Failed images:");
            for (path, error) in &summary.failed {
                println!("  {:?}: {}", path, error);
            }
        }
    }

    if !summary.failed.is_empty() {
        bail!(
            "{} of {} images failed to process.",
            summary.failed.len(),
            summary.total
        );
    }

    Ok(())
}
