//! Per-stage timing metrics and crop-accuracy counters for the image processing pipeline.
//!
//! Enabled by `--metrics` at the CLI. Default off — adding `--metrics` to a
//! run changes nothing except printing a timing table at the end.
//!
//! # Usage
//! ```rust,ignore
//! let timer = StageTimer::start();
//! // ... do work ...
//! let dur = timer.elapsed();
//!
//! let mut metrics = ImageMetrics::default();
//! metrics.person_detect = dur;
//!
//! // Accuracy counters (only meaningful when --enhanced-crop is on):
//! let acc = AccuracyMetrics {
//!     images_considered: 1,
//!     images_early_skipped: 0,
//!     crops_total: 3,
//!     crops_full_person_height: 2,
//!     crops_min_dim_relaxed: 1,
//! };
//! batch_metrics.record_accuracy(acc);
//! ```

use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// StageTimer
// ---------------------------------------------------------------------------

/// A simple wall-clock timer for one pipeline stage.
pub struct StageTimer {
    start: Instant,
}

impl StageTimer {
    /// Start the timer.
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// Return elapsed time since `start()`.
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

// ---------------------------------------------------------------------------
// Per-image timings
// ---------------------------------------------------------------------------

/// Wall-clock durations for each stage of the single-image pipeline.
#[derive(Debug, Default, Clone)]
pub struct ImageMetrics {
    /// Time to open and decode the input image.
    pub decode: Duration,
    /// Time for person detection (preprocess + inference + postprocess).
    pub person_detect: Duration,
    /// Time for face detection.
    pub face_detect: Duration,
    /// Time for crop geometry computation.
    pub crop: Duration,

    /// Time to encode and write output files.
    pub encode: Duration,
}

impl ImageMetrics {
    /// Total pipeline time for this image.
    pub fn total(&self) -> Duration {
        self.decode + self.person_detect + self.face_detect + self.crop + self.encode
    }
}

// ---------------------------------------------------------------------------
// Crop-accuracy counters
// ---------------------------------------------------------------------------

/// Crop-accuracy counters accumulated during a batch run.
///
/// These counters are only meaningful when `--enhanced-crop` is active. When
/// the flag is off, all counters remain zero and no accuracy lines are printed.
///
/// # Example
/// ```rust,ignore
/// let acc = AccuracyMetrics {
///     images_considered: 10,
///     images_early_skipped: 2,
///     crops_total: 6,
///     crops_full_person_height: 4,
///     crops_min_dim_relaxed: 1,
/// };
/// ```
#[derive(Debug, Default, Clone)]
pub struct AccuracyMetrics {
    /// Images seen by the early pre-inference long-side filter (header-read succeeded).
    ///
    /// Incremented when `--enhanced-crop` is on and `read_long_side` returns `Ok`.
    /// Denominator for the early-skip percentage.
    pub images_considered: u64,
    /// Images skipped before any inference because their long side was below `--min-pixels`.
    ///
    /// These images emit no crops. Only incremented when `images_considered` is also 1.
    pub images_early_skipped: u64,
    /// Total number of format crops attempted (counts per-format, not per-image).
    pub crops_total: u64,
    /// Crops where the joint analyzer expects the full person height to be preserved.
    pub crops_full_person_height: u64,
    /// Crops where min-dimension relaxation was triggered (extra margin applied).
    pub crops_min_dim_relaxed: u64,
}

// ---------------------------------------------------------------------------
// Batch aggregation
// ---------------------------------------------------------------------------

/// Aggregated timing statistics across all images in a batch.
#[derive(Debug, Default)]
pub struct BatchMetrics {
    samples: Vec<ImageMetrics>,
    accuracy: AccuracyMetrics,
}

impl BatchMetrics {
    /// Record one image's timing metrics into the batch aggregate.
    pub fn record(&mut self, m: ImageMetrics) {
        self.samples.push(m);
    }

    /// Accumulate crop-accuracy counters from one image into the batch total.
    ///
    /// Call once per image (after collecting per-format results).
    pub fn record_accuracy(&mut self, a: AccuracyMetrics) {
        self.accuracy.images_considered += a.images_considered;
        self.accuracy.images_early_skipped += a.images_early_skipped;
        self.accuracy.crops_total += a.crops_total;
        self.accuracy.crops_full_person_height += a.crops_full_person_height;
        self.accuracy.crops_min_dim_relaxed += a.crops_min_dim_relaxed;
    }

    /// Print a human-readable timing summary to stdout.
    ///
    /// - Always prints the per-stage timing table (unchanged from M7 baseline).
    /// - Prints the early-skip line IFF `images_considered > 0`.
    /// - Prints crop-accuracy lines IFF `crops_total > 0` (i.e. `--enhanced-crop` was active).
    /// - Prints nothing when the batch is empty and no images were considered.
    pub fn print_summary(&self) {
        if self.samples.is_empty() && self.accuracy.images_considered == 0 {
            return;
        }

        let n = self.samples.len();

        if n > 0 {
            macro_rules! stage_stats {
                ($field:ident, $label:expr) => {{
                    let mut times: Vec<f64> = self
                        .samples
                        .iter()
                        .map(|m| m.$field.as_secs_f64() * 1000.0)
                        .collect();
                    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let mean = times.iter().sum::<f64>() / n as f64;
                    let p50 = percentile(&times, 50);
                    let p95 = percentile(&times, 95);
                    println!(
                        "  {:20} mean={:7.1}ms  p50={:7.1}ms  p95={:7.1}ms",
                        $label, mean, p50, p95
                    );
                }};
            }

            println!(
                "\n── Pipeline Metrics ({} image(s)) ──────────────────────────",
                n
            );
            println!(
                "  {:20} {:>12}  {:>12}  {:>12}",
                "Stage", "mean", "p50", "p95"
            );
            println!("  {}", "─".repeat(65));
            stage_stats!(decode, "decode");
            stage_stats!(person_detect, "person_detect");
            stage_stats!(face_detect, "face_detect");
            stage_stats!(crop, "crop");
            stage_stats!(encode, "encode");

            let mut totals: Vec<f64> = self
                .samples
                .iter()
                .map(|m| m.total().as_secs_f64() * 1000.0)
                .collect();
            totals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mean_total = totals.iter().sum::<f64>() / n as f64;
            let p50_total = percentile(&totals, 50);
            let p95_total = percentile(&totals, 95);
            println!("  {}", "─".repeat(65));
            println!(
                "  {:20} mean={:7.1}ms  p50={:7.1}ms  p95={:7.1}ms",
                "TOTAL", mean_total, p50_total, p95_total
            );
            println!("──────────────────────────────────────────────────────────────\n");
        }

        // Print accuracy lines only when enhanced-crop was active.
        let acc = &self.accuracy;

        let has_early_filter = acc.images_considered > 0;
        let has_crops = acc.crops_total > 0;

        if has_early_filter || has_crops {
            println!("── Crop Accuracy ────────────────────────────────────────────");
            if has_early_filter {
                let skip_pct =
                    100.0 * acc.images_early_skipped as f64 / acc.images_considered as f64;
                println!(
                    "  early-skipped (long side < min): {} / {} ({:.1}%)",
                    acc.images_early_skipped, acc.images_considered, skip_pct
                );
            }
            if has_crops {
                let full_pct = 100.0 * acc.crops_full_person_height as f64 / acc.crops_total as f64;
                let relax_pct = 100.0 * acc.crops_min_dim_relaxed as f64 / acc.crops_total as f64;
                println!(
                    "  full-height preserved:           {} / {} ({:.1}%)",
                    acc.crops_full_person_height, acc.crops_total, full_pct
                );
                println!(
                    "  min-dimension relaxed:           {} / {} ({:.1}%)",
                    acc.crops_min_dim_relaxed, acc.crops_total, relax_pct
                );
            }
            println!("──────────────────────────────────────────────────────────────\n");
        }
    }
}

/// Linear-interpolation percentile over a sorted `f64` slice.
fn percentile(sorted: &[f64], p: usize) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let idx = (p as f64 / 100.0) * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = idx - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_timer_elapsed_is_positive() {
        let t = StageTimer::start();
        let d = t.elapsed();
        // Duration is always non-negative; just verify it doesn't panic.
        let _ = d.as_nanos();
    }

    #[test]
    fn test_image_metrics_total_is_sum_of_stages() {
        let m = ImageMetrics {
            decode: Duration::from_millis(10),
            person_detect: Duration::from_millis(100),
            face_detect: Duration::from_millis(50),
            crop: Duration::from_millis(5),
            encode: Duration::from_millis(20),
        };
        assert_eq!(m.total(), Duration::from_millis(185));
    }

    #[test]
    fn test_batch_metrics_empty_prints_nothing() {
        // Just checking it doesn't panic on empty batch.
        BatchMetrics::default().print_summary();
    }

    #[test]
    fn test_batch_metrics_record_and_aggregate() {
        let mut batch = BatchMetrics::default();
        let m1 = ImageMetrics {
            person_detect: Duration::from_millis(100),
            ..Default::default()
        };
        let m2 = ImageMetrics {
            person_detect: Duration::from_millis(200),
            ..Default::default()
        };
        batch.record(m1);
        batch.record(m2);
        assert_eq!(batch.samples.len(), 2);
        // Mean person_detect should be 150ms
        let mean: f64 = batch
            .samples
            .iter()
            .map(|m| m.person_detect.as_secs_f64() * 1000.0)
            .sum::<f64>()
            / 2.0;
        assert!((mean - 150.0).abs() < 0.01);
    }

    #[test]
    fn test_percentile_single_element() {
        let v = vec![42.0f64];
        assert_eq!(percentile(&v, 50), 42.0);
        assert_eq!(percentile(&v, 95), 42.0);
    }

    #[test]
    fn test_percentile_sorted_values() {
        let v: Vec<f64> = (1..=10).map(|i| i as f64 * 10.0).collect();
        assert!((percentile(&v, 50) - 55.0).abs() < 0.1);
        assert!((percentile(&v, 100) - 100.0).abs() < 0.1);
    }

    // ---------------------------------------------------------------------------
    // AccuracyMetrics tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_accuracy_metrics_default_is_all_zero() {
        let a = AccuracyMetrics::default();
        assert_eq!(a.images_considered, 0);
        assert_eq!(a.images_early_skipped, 0);
        assert_eq!(a.crops_total, 0);
        assert_eq!(a.crops_full_person_height, 0);
        assert_eq!(a.crops_min_dim_relaxed, 0);
    }

    #[test]
    fn test_record_accuracy_sums_fields() {
        let mut batch = BatchMetrics::default();
        batch.record_accuracy(AccuracyMetrics {
            images_considered: 5,
            images_early_skipped: 1,
            crops_total: 3,
            crops_full_person_height: 2,
            crops_min_dim_relaxed: 1,
        });
        batch.record_accuracy(AccuracyMetrics {
            images_considered: 3,
            images_early_skipped: 2,
            crops_total: 2,
            crops_full_person_height: 1,
            crops_min_dim_relaxed: 0,
        });
        assert_eq!(batch.accuracy.images_considered, 8);
        assert_eq!(batch.accuracy.images_early_skipped, 3);
        assert_eq!(batch.accuracy.crops_total, 5);
        assert_eq!(batch.accuracy.crops_full_person_height, 3);
        assert_eq!(batch.accuracy.crops_min_dim_relaxed, 1);
    }

    #[test]
    fn test_print_summary_with_accuracy_does_not_panic() {
        let mut batch = BatchMetrics::default();
        batch.record(ImageMetrics {
            person_detect: Duration::from_millis(100),
            ..Default::default()
        });
        batch.record_accuracy(AccuracyMetrics {
            images_considered: 1,
            images_early_skipped: 0,
            crops_total: 2,
            crops_full_person_height: 1,
            crops_min_dim_relaxed: 0,
        });
        // Should not panic.
        batch.print_summary();
    }

    #[test]
    fn test_print_summary_omits_accuracy_when_all_zero() {
        // When all accuracy counters are zero, accuracy block is omitted.
        let batch = BatchMetrics::default();
        batch.print_summary(); // Should be a no-op.
    }

    #[test]
    fn test_print_summary_early_filter_only_does_not_panic() {
        // images_considered > 0 but crops_total == 0 (all images skipped pre-inference).
        let mut batch = BatchMetrics::default();
        batch.record_accuracy(AccuracyMetrics {
            images_considered: 10,
            images_early_skipped: 10,
            ..Default::default()
        });
        batch.print_summary(); // Should print early-skip line without panic.
    }

    #[test]
    fn test_print_summary_early_skip_percentage_line() {
        // Verify the early-skip line is printed when images_considered > 0.
        // (behavioral: just verify no panic and field sums correctly)
        let mut batch = BatchMetrics::default();
        batch.record_accuracy(AccuracyMetrics {
            images_considered: 8,
            images_early_skipped: 3,
            crops_total: 5,
            crops_full_person_height: 4,
            crops_min_dim_relaxed: 1,
        });
        batch.print_summary();
        // Verify accumulated values match expectations.
        assert_eq!(batch.accuracy.images_considered, 8);
        assert_eq!(batch.accuracy.images_early_skipped, 3);
    }

    /// When crops_total > 0 but images_considered == 0, only crop-accuracy lines print.
    #[test]
    fn test_print_summary_crops_only_no_early_filter() {
        let mut batch = BatchMetrics::default();
        batch.record(ImageMetrics {
            person_detect: Duration::from_millis(100),
            ..Default::default()
        });
        batch.record_accuracy(AccuracyMetrics {
            images_considered: 0,
            images_early_skipped: 0,
            crops_total: 4,
            crops_full_person_height: 3,
            crops_min_dim_relaxed: 1,
        });
        // Must not panic; conceptually validates early-skip line is omitted.
        batch.print_summary();
        assert_eq!(batch.accuracy.images_considered, 0);
        assert_eq!(batch.accuracy.crops_total, 4);
    }

    /// Gate 9 — When all images are early-skipped, crops_total stays at zero.
    #[test]
    fn test_early_skip_all_images_zero_crops() {
        let mut batch = BatchMetrics::default();
        batch.record_accuracy(AccuracyMetrics {
            images_considered: 5,
            images_early_skipped: 5,
            ..Default::default()
        });
        assert_eq!(batch.accuracy.images_considered, 5);
        assert_eq!(batch.accuracy.images_early_skipped, 5);
        assert_eq!(batch.accuracy.crops_total, 0);
    }
}
