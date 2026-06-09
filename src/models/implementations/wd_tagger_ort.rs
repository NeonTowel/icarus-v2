//! Generic ONNX-Runtime wrapper for SmilingWolf-style WD/idolsankaku EVA02 taggers.
//!
//! Both `SmilingWolf/wd-eva02-large-tagger-v3` and
//! `deepghs/idolsankaku-eva02-large-tagger-v1` use:
//! - NHWC input shape `[1, H, W, 3]` float32 (H/W are model-specific)
//! - identical preprocessing pad-to-square with white, resize bicubic,
//!   RGB->BGR channel reorder, no [0,1] scaling, no mean/std normalization

use anyhow::{Context, Result};
use hf_hub::api::sync::Api;
use image::{DynamicImage, ImageBuffer, Rgb, RgbImage};
use ndarray::Array4;
use ort::{
    inputs,
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

#[derive(Debug, Clone, Copy)]
pub enum RatingLayoutKind {
    Wd4,
    Sankaku3,
}

#[derive(Debug, Clone, Copy)]
pub struct WdTaggerConfig {
    pub display_name: &'static str,
    pub hf_repo: &'static str,
    pub hf_onnx_filename: &'static str,
    pub hf_csv_filename: &'static str,
    pub input_size: u32,
    pub rating_layout: RatingLayoutKind,
}

pub const CONFIG_WD_EVA02: WdTaggerConfig = WdTaggerConfig {
    display_name: "wd-eva02-large-tagger-v3",
    hf_repo: "SmilingWolf/wd-eva02-large-tagger-v3",
    hf_onnx_filename: "model.onnx",
    hf_csv_filename: "selected_tags.csv",
    input_size: 448,
    rating_layout: RatingLayoutKind::Wd4,
};

pub const CONFIG_IDOLSANKAKU: WdTaggerConfig = WdTaggerConfig {
    display_name: "idolsankaku-eva02-large-tagger-v1",
    hf_repo: "deepghs/idolsankaku-eva02-large-tagger-v1",
    hf_onnx_filename: "model.onnx",
    hf_csv_filename: "selected_tags.csv",
    input_size: 480,
    rating_layout: RatingLayoutKind::Sankaku3,
};

/// Identifies a model's rating-tag taxonomy and projection strategy into the
/// canonical [general, sensitive, questionable, explicit] slots.
#[derive(Debug, Clone, Copy)]
pub enum RatingLayout {
    Wd4 { indices: [usize; 4] },
    Sankaku3 { indices: [usize; 3] },
}

impl RatingLayout {
    pub fn max_index(&self) -> usize {
        match self {
            Self::Wd4 { indices } => *indices.iter().max().unwrap_or(&0),
            Self::Sankaku3 { indices } => *indices.iter().max().unwrap_or(&0),
        }
    }

    pub fn extract_4slot(&self, logits: &[f32]) -> [f32; 4] {
        match self {
            Self::Wd4 { indices } => [
                logits[indices[0]],
                logits[indices[1]],
                logits[indices[2]],
                logits[indices[3]],
            ],
            Self::Sankaku3 { indices } => [
                logits[indices[0]],
                0.0,
                logits[indices[1]],
                logits[indices[2]],
            ],
        }
    }
}

pub struct WdTaggerOnnx {
    config: WdTaggerConfig,
    session: Mutex<Session>,
    rating_layout: RatingLayout,
    input_size: u32,
}

fn pad_to_square_white(image: &DynamicImage) -> RgbImage {
    let rgb_image = image.to_rgb8();
    let (width, height) = rgb_image.dimensions();

    if width == height {
        return rgb_image;
    }

    let side = width.max(height);
    let mut canvas: RgbImage = ImageBuffer::from_pixel(side, side, Rgb([255, 255, 255]));
    let pad_x = (side - width) / 2;
    let pad_y = (side - height) / 2;
    image::imageops::replace(&mut canvas, &rgb_image, pad_x as i64, pad_y as i64);
    canvas
}

/// Run the full WD preprocessing pipeline.
///
/// Returns an NHWC tensor with shape `[1, target_size, target_size, 3]` in BGR
/// channel order, unscaled in the raw byte range `[0.0, 255.0]`.
///
/// # Example
/// ```rust,ignore
/// let image = image::open("input.jpg")?;
/// let tensor = wd_preprocess(&image, 448);
/// assert_eq!(tensor.shape(), &[1, 448, 448, 3]);
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn wd_preprocess(image: &DynamicImage, target_size: u32) -> Array4<f32> {
    let squared = pad_to_square_white(image);
    let resized = image::imageops::resize(
        &squared,
        target_size,
        target_size,
        image::imageops::FilterType::CatmullRom,
    );

    let size = target_size as usize;
    let mut preprocessed = Array4::<f32>::zeros((1, size, size, 3));

    for (x, y, pixel) in resized.enumerate_pixels() {
        let x_index = x as usize;
        let y_index = y as usize;

        // RGB -> BGR on the last axis. No /255 scaling. No normalization.
        preprocessed[[0, y_index, x_index, 0]] = pixel[2] as f32;
        preprocessed[[0, y_index, x_index, 1]] = pixel[1] as f32;
        preprocessed[[0, y_index, x_index, 2]] = pixel[0] as f32;
    }

    preprocessed
}

/// Download a small text file from HF Hub by raw URL.
///
/// Workaround for `hf-hub` 0.3 failing on CSV redirects in sync mode.
fn download_hf_text_file_via_raw_url(repo: &str, filename: &str) -> Result<PathBuf> {
    let cache_dir = dirs::cache_dir()
        .ok_or_else(|| anyhow::anyhow!("could not determine OS cache directory"))?
        .join("icarus-v2")
        .join("hf-csvs")
        .join(repo);

    std::fs::create_dir_all(&cache_dir)
        .with_context(|| format!("create cache dir {:?}", cache_dir))?;

    let cache_path = cache_dir.join(filename);
    if cache_path.exists() {
        log::debug!("wd-tagger: using cached {filename} at {:?}", cache_path);
        return Ok(cache_path);
    }

    let url = format!("https://huggingface.co/{repo}/resolve/main/{filename}");
    log::info!("wd-tagger: downloading {filename} from {url}");

    let response = ureq::get(&url)
        .call()
        .map_err(|error| anyhow::anyhow!("GET {url}: {error}"))?;
    let mut reader = response.into_reader();
    let mut bytes = Vec::new();
    reader
        .read_to_end(&mut bytes)
        .with_context(|| format!("read response body for {url}"))?;

    std::fs::write(&cache_path, &bytes)
        .with_context(|| format!("write cached file {:?}", cache_path))?;

    Ok(cache_path)
}

impl WdTaggerOnnx {
    pub fn from_hub(config: WdTaggerConfig) -> Result<Self> {
        let api = Api::new().context("HF Hub API init")?;
        let repo = api.model(config.hf_repo.to_string());

        log::info!(
            "wd-tagger ({}): downloading ONNX from {}",
            config.display_name,
            config.hf_repo
        );
        let onnx_path: PathBuf = repo
            .get(config.hf_onnx_filename)
            .with_context(|| format!("download {}", config.hf_onnx_filename))?;

        log::info!(
            "wd-tagger ({}): downloading labels CSV via raw URL",
            config.display_name
        );
        let csv_path = download_hf_text_file_via_raw_url(config.hf_repo, config.hf_csv_filename)?;
        let rating_layout = parse_rating_layout(&csv_path, config.rating_layout)
            .with_context(|| format!("parse rating layout from {:?}", csv_path))?;

        log::info!(
            "wd-tagger ({}): rating layout = {:?}",
            config.display_name,
            rating_layout
        );

        let session = Session::builder()
            .map_err(|error| anyhow::anyhow!("ort Session::builder: {error}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|error| anyhow::anyhow!("ort optimization level: {error}"))?
            .commit_from_file(&onnx_path)
            .map_err(|error| anyhow::anyhow!("ort load model from {:?}: {error}", onnx_path))?;

        Ok(Self {
            input_size: config.input_size,
            config,
            session: Mutex::new(session),
            rating_layout,
        })
    }

    pub fn predict_ratings(&self, image: &DynamicImage) -> Result<[f32; 4]> {
        let input = wd_preprocess(image, self.input_size);
        let tensor_ref = TensorRef::from_array_view(&input)
            .map_err(|error| anyhow::anyhow!("ort TensorRef: {error}"))?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow::anyhow!("ort session mutex poisoned"))?;

        let outputs = session
            .run(inputs![tensor_ref])
            .map_err(|error| anyhow::anyhow!("ort inference: {error}"))?;

        let (_shape, all_logits) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|error| anyhow::anyhow!("ort extract output: {error}"))?;

        let max_index = self.rating_layout.max_index();
        if all_logits.len() <= max_index {
            anyhow::bail!(
                "model output length {} is too short for rating index {} (model={}, repo={})",
                all_logits.len(),
                max_index,
                self.config.display_name,
                self.config.hf_repo
            );
        }

        Ok(self.rating_layout.extract_4slot(all_logits))
    }

    pub fn display_name(&self) -> &'static str {
        self.config.display_name
    }
}

fn parse_rating_layout(csv_path: &Path, layout_kind: RatingLayoutKind) -> Result<RatingLayout> {
    let mut reader =
        csv::Reader::from_path(csv_path).with_context(|| format!("open {:?}", csv_path))?;
    let mut indices = Vec::with_capacity(4);

    for (row_index, record_result) in reader.records().enumerate() {
        let record = record_result?;
        let category = record
            .get(2)
            .ok_or_else(|| anyhow::anyhow!("CSV row {row_index} missing category column"))?
            .parse::<u32>()
            .with_context(|| format!("parse category at row {row_index}"))?;

        if category == 9 {
            indices.push(row_index);
        }
    }

    match layout_kind {
        RatingLayoutKind::Wd4 => {
            if indices.len() != 4 {
                anyhow::bail!(
                    "Wd4 layout expects 4 rating rows (category=9), CSV had {}",
                    indices.len()
                );
            }

            Ok(RatingLayout::Wd4 {
                indices: [indices[0], indices[1], indices[2], indices[3]],
            })
        }
        RatingLayoutKind::Sankaku3 => {
            if indices.len() != 3 {
                anyhow::bail!(
                    "Sankaku3 layout expects 3 rating rows (category=9), CSV had {}",
                    indices.len()
                );
            }

            Ok(RatingLayout::Sankaku3 {
                indices: [indices[0], indices[1], indices[2]],
            })
        }
    }
}

#[cfg(test)]
mod rating_layout_tests {
    use super::*;

    #[test]
    fn wd4_layout_identity_passthrough() {
        let layout = RatingLayout::Wd4 {
            indices: [0, 1, 2, 3],
        };
        let logits = [0.1, 0.2, 0.3, 0.4, 0.99];

        assert_eq!(layout.extract_4slot(&logits), [0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn sankaku3_layout_maps_safe_to_general_and_zeros_sensitive() {
        let layout = RatingLayout::Sankaku3 { indices: [0, 1, 2] };
        let logits = [0.8, 0.1, 0.05, 0.42];

        assert_eq!(layout.extract_4slot(&logits), [0.8, 0.0, 0.1, 0.05]);
    }

    #[test]
    fn sankaku3_layout_uses_non_contiguous_indices() {
        let layout = RatingLayout::Sankaku3 {
            indices: [5, 9, 11],
        };
        let mut logits = vec![0.0_f32; 12];
        logits[5] = 0.7;
        logits[9] = 0.2;
        logits[11] = 0.1;

        assert_eq!(layout.extract_4slot(&logits), [0.7, 0.0, 0.2, 0.1]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, ImageBuffer, Rgb};

    #[test]
    fn preprocess_red_image_is_nhwc_bgr_with_red_in_last_channel() {
        let image = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(448, 448, Rgb([255, 0, 0])));
        let tensor = wd_preprocess(&image, 448);

        assert_eq!(tensor.shape(), &[1, 448, 448, 3]);
        assert!(tensor
            .slice(ndarray::s![0, .., .., 0])
            .iter()
            .all(|value| *value == 0.0));
        assert!(tensor
            .slice(ndarray::s![0, .., .., 1])
            .iter()
            .all(|value| *value == 0.0));
        assert!(tensor
            .slice(ndarray::s![0, .., .., 2])
            .iter()
            .all(|value| (*value - 255.0).abs() < 1e-4));
    }

    #[test]
    fn preprocess_keeps_values_in_unscaled_byte_range() {
        let image = DynamicImage::ImageRgb8(ImageBuffer::from_fn(200, 300, |x, _| {
            Rgb([(x % 256) as u8, 128, 64])
        }));
        let tensor = wd_preprocess(&image, 448);
        let max_value = tensor.iter().cloned().fold(0.0_f32, f32::max);

        assert!(
            max_value > 200.0,
            "expected max > 200 (raw byte range); got {max_value}"
        );
        assert!(max_value <= 255.0, "expected max <= 255; got {max_value}");
    }

    #[test]
    fn preprocess_idolsankaku_size_is_480() {
        let image =
            DynamicImage::ImageRgb8(ImageBuffer::from_pixel(100, 100, Rgb([128, 128, 128])));
        let tensor = wd_preprocess(&image, 480);

        assert_eq!(tensor.shape(), &[1, 480, 480, 3]);
    }

    #[test]
    fn preprocess_non_square_pads_then_resizes_to_target() {
        let image = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(100, 400, Rgb([50, 50, 50])));
        let tensor = wd_preprocess(&image, 448);

        assert_eq!(tensor.shape(), &[1, 448, 448, 3]);
        let center_blue_channel = tensor[[0, 224, 224, 0]];
        let edge_blue_channel = tensor[[0, 224, 4, 0]];

        assert!(
            (center_blue_channel - 50.0).abs() < 5.0,
            "centre B should be ~50, got {center_blue_channel}"
        );
        assert!(
            edge_blue_channel > 240.0,
            "edge B should be near white, got {edge_blue_channel}"
        );
    }

    #[test]
    #[ignore]
    fn smoke_test_wd_eva02_returns_valid_sigmoid_ratings() {
        let model = WdTaggerOnnx::from_hub(CONFIG_WD_EVA02)
            .expect("wd-eva02 download and load should succeed");
        let image =
            DynamicImage::ImageRgb8(ImageBuffer::from_pixel(600, 400, Rgb([128, 128, 128])));

        let raw = model
            .predict_ratings(&image)
            .expect("inference should succeed");
        for rating in &raw {
            assert!(
                (0.0..=1.0).contains(rating),
                "rating out of [0,1]: {rating}"
            );
        }

        let sum: f32 = raw.iter().sum();
        assert!(sum > 0.1, "all ratings near zero: {raw:?}");

        let argmax = raw
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.partial_cmp(right.1).unwrap())
            .map(|(index, _)| index)
            .expect("raw ratings must be non-empty");
        assert_eq!(argmax, 0, "expected general as argmax, got {raw:?}");
        assert!(
            raw[0] > raw[3],
            "expected general > explicit for grey, got {raw:?}"
        );
    }

    #[test]
    #[ignore]
    fn smoke_test_idolsankaku_loads_and_runs_at_480() {
        let model = WdTaggerOnnx::from_hub(CONFIG_IDOLSANKAKU)
            .expect("idolsankaku download and load should succeed");
        assert_eq!(model.input_size, 480);

        let image =
            DynamicImage::ImageRgb8(ImageBuffer::from_pixel(512, 384, Rgb([128, 128, 128])));
        let raw = model
            .predict_ratings(&image)
            .expect("inference should succeed");

        for rating in &raw {
            assert!(
                (0.0..=1.0).contains(rating),
                "rating out of [0,1]: {rating}"
            );
        }

        let sum: f32 = raw.iter().sum();
        assert!(sum > 0.1, "all ratings near zero: {raw:?}");
        assert_eq!(
            raw[1], 0.0,
            "idolsankaku must always have sensitive slot = 0.0 (no source label)"
        );
    }
}
