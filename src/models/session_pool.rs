//! Per-thread `SessionPool` for parallel ONNX Runtime inference (Strategy B).
//!
//! ## Problem
//!
//! `ort::Session::run` requires `&mut Session`, so a single shared session
//! must be wrapped in a `Mutex`. Under Rayon, every worker thread blocks on
//! that mutex, serialising all inference regardless of how many cores are
//! available.
//!
//! ## Solution
//!
//! `SessionPool` hands each Rayon worker its own `Session`, allocated lazily
//! on first inference. Workers reuse the same session for the rest of the
//! process lifetime (Rayon thread lifetimes are stable). A hard cap based on
//! available RAM prevents OOM on small hosts. Workers beyond the cap fall back
//! to a shared `Mutex<Session>` — the pre-pool behaviour — so degradation is
//! graceful, never fatal.
//!
//! ## Usage
//!
//! ```rust,ignore
//! let pool = SessionPool::new(
//!     PathBuf::from("model.onnx"),
//!     "my-model",
//!     500,  // footprint_mb per session
//!     1,    // models_in_group
//!     thread_count,
//! )?;
//!
//! let result = pool.with_session(|session| {
//!     let out = session.run(inputs![tensor_ref])?;
//!     Ok(decode(out))
//! })?;
//! ```
//!
//! ## Thread safety
//!
//! `SessionPool` is `Send + Sync`. The `thread_local!` storage is per-thread;
//! the shared state is limited to `AtomicUsize` (counter) and `Mutex<Session>`
//! (fallback), both of which are `Send + Sync`.

use std::cell::RefCell;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use anyhow::{Context, Result};
use ort::session::{builder::GraphOptimizationLevel, Session};

use crate::config::available_core_count;

// ---------------------------------------------------------------------------
// Thread-local session storage
// ---------------------------------------------------------------------------

thread_local! {
    /// Per-thread sessions keyed by `SessionPool` address.
    ///
    /// Multiple pools coexist via the pointer key, so each worker can hold one
    /// session per loaded pool.
    /// Entries are never removed; cleanup happens at thread death.
    static THREAD_SESSIONS: RefCell<Vec<(usize, Session)>> =
        const { RefCell::new(Vec::new()) };
}

// ---------------------------------------------------------------------------
// SessionPool
// ---------------------------------------------------------------------------

/// Per-thread ONNX Runtime session pool for parallel inference.
///
/// Eliminates `Mutex<Session>` contention by giving each Rayon worker its own
/// `Session`. Sessions are allocated lazily on first use and reused thereafter.
/// A RAM-derived cap limits total sessions; workers beyond the cap fall back to
/// a shared `Mutex<Session>` (the pre-pool behaviour).
///
/// `intra_op_threads` is tuned so that `max_sessions × intra_op_threads ≈ cores`,
/// preventing ORT from oversubscribing the CPU when multiple sessions run in
/// parallel.
///
/// # Fail-fast behaviour
///
/// A session that fails to build is never silently ignored. The atomic counter
/// is rolled back before the error is propagated (S4), so future workers will
/// retry instead of leaking a slot.
pub struct SessionPool {
    onnx_path: PathBuf,
    display_name: String,
    /// Hard cap on parallel sessions (derived from available RAM at construction).
    max_sessions: usize,
    /// Workers beyond `max_sessions` fall back to this shared session.
    fallback: Mutex<Session>,
    /// Live session count across all worker threads (fallback counts as 1).
    live_count: AtomicUsize,
    /// Per-session ORT intra-op thread count.
    intra_op_threads: usize,
    /// Number of times a worker was sent to the fallback (for `--metrics`).
    fallback_count: AtomicUsize,
}

impl SessionPool {
    /// Build a pool for `onnx_path`.
    ///
    /// Loads one initial (fallback) session synchronously. Per-worker sessions
    /// are allocated lazily by [`with_session`].
    ///
    /// # Parameters
    /// - `onnx_path`: Path to the ONNX file on disk (HF Hub cache path).
    /// - `display_name`: Human-readable name for log messages.
    /// - `footprint_mb`: Estimated per-session RAM footprint in megabytes.
    /// - `models_in_group`: Number of co-resident pools (1 = single model,
    ///   2 = ensemble). Divides the RAM-derived cap to prevent double-counting.
    /// - `thread_count`: Active Rayon thread count from `resolve_thread_count`.
    ///
    /// # Errors
    /// Returns `Err` if the ONNX model file cannot be loaded into a session.
    pub fn new(
        onnx_path: PathBuf,
        display_name: impl Into<String>,
        footprint_mb: u64,
        models_in_group: usize,
        thread_count: usize,
    ) -> Result<Self> {
        let display_name = display_name.into();
        let (max_sessions, intra_op_threads) =
            recommend_max_sessions(footprint_mb, models_in_group, thread_count);

        log::info!(
            "session-pool ({}): max_sessions={}, intra_op_threads={} \
             (footprint ~{}MB, group={}, threads={})",
            display_name,
            max_sessions,
            intra_op_threads,
            footprint_mb,
            models_in_group,
            thread_count,
        );

        let fallback = Self::build_session(&onnx_path, intra_op_threads)
            .with_context(|| format!("build fallback session for {}", display_name))?;

        Ok(Self {
            onnx_path,
            display_name,
            max_sessions,
            fallback: Mutex::new(fallback),
            live_count: AtomicUsize::new(1), // fallback counts as 1
            intra_op_threads,
            fallback_count: AtomicUsize::new(0),
        })
    }

    /// Build a single ORT session with the given intra-op thread count.
    fn build_session(onnx_path: &PathBuf, intra_op: usize) -> Result<Session> {
        Session::builder()
            .map_err(|e| anyhow::anyhow!("ort builder: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("ort opt level: {e}"))?
            .with_intra_threads(intra_op)
            .map_err(|e| anyhow::anyhow!("ort intra_threads={intra_op}: {e}"))?
            .with_inter_threads(1)
            .map_err(|e| anyhow::anyhow!("ort inter_threads=1: {e}"))?
            .commit_from_file(onnx_path)
            .map_err(|e| anyhow::anyhow!("ort load {:?}: {e}", onnx_path))
    }

    /// Execute `f` with a session for the calling thread.
    ///
    /// **Fast path**: if the thread already has a session for this pool, reuses it
    /// without any atomic operation.
    ///
    /// **Allocation path**: if under the cap, builds a new thread-local session.
    /// The counter is rolled back if the build fails (S4).
    ///
    /// **Fallback path**: if the cap is reached, locks and uses the shared
    /// fallback session (pre-pool behaviour; non-fatal degradation).
    pub fn with_session<R, F>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&mut Session) -> Result<R>,
    {
        let pool_id = self as *const Self as usize;

        // ── Fast path: existing thread-local session for this pool ──────────
        let has_existing =
            THREAD_SESSIONS.with(|cell| cell.borrow().iter().any(|(id, _)| *id == pool_id));

        if has_existing {
            return THREAD_SESSIONS.with(|cell| {
                let mut sessions = cell.borrow_mut();
                let session = sessions
                    .iter_mut()
                    .find(|(id, _)| *id == pool_id)
                    .map(|(_, s)| s)
                    .expect("thread-local session disappeared between borrow and borrow_mut");
                f(session)
            });
        }

        // ── Allocation path: try to claim a slot for a new session ──────────
        let prev = self.live_count.fetch_add(1, Ordering::SeqCst);
        if prev < self.max_sessions {
            log::debug!(
                "session-pool ({}): allocating session #{} for thread {:?}",
                self.display_name,
                prev + 1,
                std::thread::current().id()
            );

            // S4: roll back the counter before propagating a build error.
            let session =
                Self::build_session(&self.onnx_path, self.intra_op_threads).map_err(|e| {
                    self.live_count.fetch_sub(1, Ordering::SeqCst);
                    e.context(format!(
                        "session-pool ({}): build_session failed; slot released",
                        self.display_name
                    ))
                })?;

            return THREAD_SESSIONS.with(|cell| {
                let mut sessions = cell.borrow_mut();
                sessions.push((pool_id, session));
                let session = sessions
                    .last_mut()
                    .map(|(_, s)| s)
                    .expect("we just pushed a session — Vec cannot be empty");
                f(session)
            });
        }

        // ── Fallback path: cap reached — lock the shared session ────────────
        // Roll back the fetch_add we did above (we won't use the slot).
        self.live_count.fetch_sub(1, Ordering::SeqCst);
        let hits = self.fallback_count.fetch_add(1, Ordering::Relaxed) + 1;
        if hits == 1 || hits.is_multiple_of(50) {
            log::debug!(
                "session-pool ({}): cap={} reached; using fallback session (hit #{})",
                self.display_name,
                self.max_sessions,
                hits,
            );
        }

        let mut session = self.fallback.lock().map_err(|_| {
            anyhow::anyhow!(
                "session-pool ({}): fallback mutex poisoned",
                self.display_name
            )
        })?;
        f(&mut session)
    }

    /// Number of live per-thread sessions currently allocated (including the fallback).
    pub fn live_count(&self) -> usize {
        self.live_count.load(Ordering::Relaxed)
    }

    /// Number of times a worker used the fallback session (cap was reached).
    pub fn fallback_count(&self) -> usize {
        self.fallback_count.load(Ordering::Relaxed)
    }

    /// Display name of this pool (used in log messages).
    pub fn display_name(&self) -> &str {
        &self.display_name
    }
}

// ---------------------------------------------------------------------------
// recommend_max_sessions
// ---------------------------------------------------------------------------

/// Compute the session cap and per-session ORT thread count.
///
/// ## RAM budgeting
///
/// The usable RAM budget is `available_RAM × 0.70` (30% headroom).
/// `models_in_group` divides the cap for co-resident pools so they stay within
/// budget:
///
/// ```text
/// cap = floor(usable_MB / (footprint_MB × models_in_group))
/// ```
///
/// ## Thread coordination (S3)
///
/// ORT intra-op threads are sized so concurrent per-worker sessions do not
/// oversubscribe the CPU:
///
/// ```text
/// n_active        = min(thread_count, cap)
/// intra_op_threads = max(cores / n_active, 1)
/// ```
///
/// # Parameters
/// - `footprint_mb`: Estimated per-session RAM in megabytes.
/// - `models_in_group`: Number of co-resident pools.
/// - `thread_count`: Active Rayon thread count (from `resolve_thread_count`).
///
/// # Returns
/// `(max_sessions, intra_op_threads)`
pub fn recommend_max_sessions(
    footprint_mb: u64,
    models_in_group: usize,
    thread_count: usize,
) -> (usize, usize) {
    let cores = available_core_count();

    // Estimate available RAM. System::new() + refresh_memory() is lighter than
    // new_all() (which also enumerates processes and disks). available_memory()
    // returns bytes as of sysinfo 0.30+ (confirmed for 0.32).
    let available_mb: u64 = {
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        let bytes = sys.available_memory(); // bytes
        let mb = bytes / 1_048_576; // → MB
        mb.max(2048) // assume ≥ 2 GB if detection fails
    };

    // 30% headroom for I/O, image decode, crop geometry, etc.
    let usable_mb = (available_mb as f64 * 0.70) as u64;

    // Account for all co-resident pools.
    let effective_footprint_mb = footprint_mb.saturating_mul(models_in_group.max(1) as u64);
    let by_ram = (usable_mb / effective_footprint_mb.max(1)).max(1) as usize;

    // Cap at core count; never less than 1.
    let max_sessions = cores.min(by_ram).max(1);

    // S3: size intra_op_threads so concurrent sessions don't oversubscribe.
    let n_active = thread_count.min(max_sessions).max(1);
    let intra_op_threads = (cores / n_active).max(1);

    log::info!(
        "session-pool recommendation: cores={}, available={}MB, usable={}MB, \
         footprint={}MB, group={}, thread_count={} \
         → max_sessions={}, intra_op_threads={}",
        cores,
        available_mb,
        usable_mb,
        footprint_mb,
        models_in_group,
        thread_count,
        max_sessions,
        intra_op_threads,
    );

    (max_sessions, intra_op_threads)
}

// ---------------------------------------------------------------------------
// Model-specific footprint constants
// ---------------------------------------------------------------------------

/// Estimated per-session RAM for YOLOv10n/s/m (ORT arena + weights).
pub const YOLOV10_FOOTPRINT_MB: u64 = 150;

/// Estimated per-session RAM for YOLO26n/s/m.
pub const YOLO26_FOOTPRINT_MB: u64 = 100;

/// Estimated per-session RAM for YOLOv11x-Face (~60 MB ONNX + arena).
pub const YOLOV11X_FACE_FOOTPRINT_MB: u64 = 250;

/// Estimated per-session RAM for YOLOv10-Face.
pub const YOLOV10_FACE_FOOTPRINT_MB: u64 = 150;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── Compile-time Send + Sync guarantee ───────────────────────────────────

    #[allow(dead_code)]
    fn assert_send_sync<T: Send + Sync>() {}

    #[allow(dead_code)]
    fn pool_is_send_sync() {
        assert_send_sync::<SessionPool>();
    }

    // ── recommend_max_sessions unit tests ────────────────────────────────────

    #[test]
    fn recommend_always_returns_at_least_one_session() {
        // Tiny footprint, 1 thread, 1 model.
        let (max_sessions, intra_op_threads) = recommend_max_sessions(500, 1, 1);
        assert!(
            max_sessions >= 1,
            "max_sessions must be ≥ 1, got {max_sessions}"
        );
        assert!(
            intra_op_threads >= 1,
            "intra_op_threads must be ≥ 1, got {intra_op_threads}"
        );
    }

    #[test]
    fn intra_op_threads_never_exceeds_core_count() {
        let cores = available_core_count();
        let (_, intra_op_threads) = recommend_max_sessions(500, 1, 4);
        assert!(
            intra_op_threads <= cores,
            "intra_op_threads ({intra_op_threads}) must not exceed core count ({cores})"
        );
    }

    #[test]
    fn zero_thread_count_does_not_panic() {
        // thread_count=0 is clamped to 1 inside recommend_max_sessions.
        let (max_sessions, intra_op_threads) = recommend_max_sessions(500, 1, 0);
        assert!(max_sessions >= 1);
        assert!(intra_op_threads >= 1);
    }

    // ── Pool integration tests (require a real ONNX file) ────────────────────

    /// Verify that each thread in a pool gets its own session (no mutex contention).
    ///
    /// Requires a real ONNX model file; skip if unavailable.
    #[test]
    #[ignore]
    fn pool_allocates_separate_sessions_per_thread() {
        // Drive 4 Rayon threads through a small pool and verify that at most
        // max_sessions unique per-thread sessions are allocated (rest use fallback).
        // Use a small ONNX fixture or skip if unavailable.
        todo!("requires a fixture ONNX file path")
    }

    /// Verify graceful fallback when the cap is reached.
    ///
    /// With max_sessions=1, the first worker gets a thread-local session;
    /// all others must use the fallback.
    #[test]
    #[ignore]
    fn pool_falls_back_when_cap_reached() {
        // max_sessions=1, drive 4 threads, expect fallback_count >= 3.
        todo!("requires a fixture ONNX file path")
    }
}
