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
use icarus_v2::face_detection::load_face_detector;
use icarus_v2::models::load_candle_model;
use std::path::PathBuf;

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

    #[arg(long, default_value_t = false)]
    classify_output: bool,

    #[arg(long, default_value_t = false)]
    classify_only: bool,

    #[arg(
        long,
        default_value = "freepik",
        value_name = "NAME",
        help = "Image classifier to use when --classify-output or --classify-only is set.\n\
                Options (fastest first):\n  \
                wd-vit               — WD ViT-Base v3 (~200ms CPU, F1 0.44, anime) ⚡ raw speed\n  \
                idolsankaku-swinv2   — Idolsankaku SwinV2 v1 (~300ms CPU, F1 0.62, real photos) ⚡ raw speed\n  \
                wd-swinv2            — WD SwinV2-Base v3 (~300ms CPU, F1 0.45, anime)\n  \
                wd-ensemble-fast     — SwinV2 ensemble (~500ms CPU, balanced) ⭐ recommended\n  \
                wd-eva02             — WD EVA02-Large v3 (~900ms CPU, F1 0.48, anime)\n  \
                idolsankaku          — Idolsankaku EVA02-Large v1 (~900ms CPU, F1 0.60, real photos)\n  \
                wd-ensemble-accurate — EVA02-Large ensemble (~1700ms CPU, highest F1)\n  \
                freepik              — Legacy 4-tier Freepik NSFW (current default, will change)\n  \
                wd-ensemble          — alias for wd-ensemble-accurate (backward compat)"
    )]
    classifier: String,

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
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();
    validate_args(&args)?;
    let crop_config = build_crop_config(&args)?;
    let artistic_config = build_artistic_config(&args)?;
    let (model, face_detector) = load_models(&args).await?;
    let classifier = if args.classify_output || args.classify_only {
        let kind = args
            .classifier
            .parse::<icarus_v2::models::ClassifierKind>()?;
        Some(icarus_v2::models::load_classifier(kind, &candle_core::Device::Cpu).await?)
    } else {
        None
    };

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
        classifier: classifier.as_deref(),
        crop_config: &crop_config,
        artistic_config: &artistic_config,
        confidence: args.confidence,
        margin: args.margin,
        keep_aspect_ratio: args.keep_aspect_ratio || args.classify_only,
        sort_output: args.sort_output,
        classify_output: args.classify_output || args.classify_only,
        classify_only: args.classify_only,
        quiet: args.quiet,
        rename: args.rename,
        jpeg_quality: args.jpeg,
        flatten: args.flatten,
    };

    dispatch(&args, &context).await
}

fn validate_args(args: &Args) -> Result<()> {
    if args.classify_only && args.output.is_none() {
        bail!("--classify-only requires --output");
    }

    if args.classify_only && !args.sort_output {
        bail!("--classify-only requires --sort-output");
    }

    if let Some(0) = args.threads {
        bail!("--threads must be >= 1 (omit the flag to auto-select 50% of cores)");
    }

    Ok(())
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
    let model = load_candle_model(&args.model, &device)
        .await
        .with_context(|| format!("Failed to load model '{}'", args.model))?;
    let face_detector = tokio::task::spawn_blocking(load_face_detector)
        .await
        .map_err(|error| anyhow::anyhow!("face model load task panicked: {error}"))??;

    Ok((model, face_detector))
}

async fn dispatch(args: &Args, context: &ProcessingContext<'_>) -> Result<()> {
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

    let thread_count = resolve_thread_count(args.threads, available_core_count());
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
