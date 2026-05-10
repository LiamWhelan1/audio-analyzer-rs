//! Automatic Gain Control and musical dynamics classification.
//!
//! # Pipeline
//! ```text
//! (filtered + gated slot)
//!       │
//!       ▼
//!  compute slot RMS (pre-gain)
//!       │
//!       ├──► always: update long_history (all frames, small ~12 s window)
//!       │           → p10 = noise floor estimate
//!       │
//!       ├── is_active? (SNR ≥ 20 dB above noise floor)
//!       │       │
//!       │      yes ──► update play_history (large ~8 min window)
//!       │               one sort → p50 (session median) + p95 (gain target)
//!       │      no  ──► slowly decay gain toward 0 dB (~10 s)
//!       │
//!       ▼
//!  smooth gain toward p95 target (~15 s TC when active)
//!  apply gain + peak-headroom clamp
//!       │
//!       ▼
//!  classify: !is_active → Silence
//!            rms_db − session_median_db → ppp…fff
//! ```
//!
//! ## Design rationale
//!
//! A single `play_history` ring buffer (active frames only, ~8 min at typical
//! playing density) provides both the session median and the AGC reference in
//! one sort.  Using one long window for both means:
//!
//! * The gain target changes only as the _overall_ session character changes,
//!   not as individual phrases vary—eliminating phrase-to-phrase gain pumping.
//! * The dynamics reference point (session median) is stable for the same
//!   reason: it takes many minutes of data to shift it by even a few dB.
//!
//! The `long_history` is kept small (256 frames ≈ 12 s) because the noise
//! floor in a quiet room is stationary; a short window is sufficient and keeps
//! the sort cost negligible.

use parking_lot::RwLock;
use std::sync::Arc;

// ─── Public output type ───────────────────────────────────────────────────────

/// Musical dynamic marking estimated from relative loudness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DynamicLevel {
    Silence,
    Ppp,
    Pp,
    P,
    Mp,
    Mf,
    F,
    Ff,
    Fff,
}

impl std::fmt::Display for DynamicLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            DynamicLevel::Silence => "silence",
            DynamicLevel::Ppp => "ppp",
            DynamicLevel::Pp => "pp",
            DynamicLevel::P => "p",
            DynamicLevel::Mp => "mp",
            DynamicLevel::Mf => "mf",
            DynamicLevel::F => "f",
            DynamicLevel::Ff => "ff",
            DynamicLevel::Fff => "fff",
        };
        f.write_str(s)
    }
}

#[derive(Clone, Debug)]
pub struct DynamicsOutput {
    /// Current musical dynamic level.
    pub level: DynamicLevel,
    /// Pre-gain slot RMS in dBFS.
    pub rms_db: f32,
    /// Applied AGC gain in dB (positive = boost).
    pub gain_db: f32,
    /// Median RMS of active frames in dBFS (session reference).
    pub session_median_db: f32,
    /// Estimated noise floor in dBFS.
    pub noise_floor_db: f32,
}

impl Default for DynamicsOutput {
    fn default() -> Self {
        Self {
            level: DynamicLevel::Silence,
            rms_db: -96.0,
            gain_db: 0.0,
            session_median_db: -96.0,
            noise_floor_db: -96.0,
        }
    }
}

// ─── Tracker ─────────────────────────────────────────────────────────────────

pub struct DynamicsTracker {
    // ── Long history: all frames, small window → noise floor (p10) ───────
    long_history: Vec<f32>,
    long_pos: usize,
    long_filled: bool,

    // ── Play history: active frames only, large window ────────────────────
    // Sorted once per slot to yield p50 (session median) and p95 (AGC ref).
    play_history: Vec<f32>,
    play_pos: usize,
    play_filled: bool,

    // ── Gain state ────────────────────────────────────────────────────────
    current_gain_linear: f32,
    target_db: f32,
    max_boost_db: f32,
    /// Attack smoothing coefficient (~15 s TC when active).
    smooth_alpha: f32,
    /// Decay coefficient during silence (~10 s TC toward 0 dB).
    silence_decay_alpha: f32,

    // ── Active-frame gate ─────────────────────────────────────────────────
    active_snr_db: f32,
    /// Fixed floor used before the noise estimate has warmed up.
    bootstrap_floor_db: f32,

    // ── Scratch buffer ────────────────────────────────────────────────────
    sort_buf: Vec<f32>,

    // ── Shared output ─────────────────────────────────────────────────────
    output: Arc<RwLock<DynamicsOutput>>,
}

impl DynamicsTracker {
    /// Construct a tracker.
    ///
    /// - `sample_rate`: Hz.
    /// - `slot_len`: samples per processing slot.
    /// - `target_db`: desired AGC output level in dBFS (e.g. `-18.0`),
    ///   measured against the 95th-percentile of active-frame RMS.
    /// - `max_boost_db`: ceiling on AGC gain (e.g. `36.0`).
    /// - `smooth_secs`: gain attack time constant in seconds (e.g. `15.0`).
    ///   A long TC (10–20 s) makes the gain session-stable rather than
    ///   phrase-by-phrase.
    /// - `output`: shared cell written after every slot.
    pub fn new(
        sample_rate: f32,
        slot_len: usize,
        target_db: f32,
        max_boost_db: f32,
        smooth_secs: f32,
        output: Arc<RwLock<DynamicsOutput>>,
    ) -> Self {
        let slot_rate = sample_rate / slot_len as f32;

        // Long history: 256 slots ≈ 12 s  (noise floor is stationary; no need
        // for a large window)
        let long_len = 256;

        // Play history: 5000 slots at ~10 active frames/sec ≈ 8 min.
        // Large window keeps p50 and p95 stable across the session.
        let play_len = 5000;

        let smooth_alpha = 1.0 - (-1.0 / (smooth_secs * slot_rate)).exp();
        let silence_decay_alpha = 1.0 - (-1.0 / (10.0 * slot_rate)).exp();

        Self {
            long_history: vec![0.0; long_len],
            long_pos: 0,
            long_filled: false,
            play_history: vec![0.0; play_len],
            play_pos: 0,
            play_filled: false,
            current_gain_linear: 1.0,
            target_db,
            max_boost_db,
            smooth_alpha,
            silence_decay_alpha,
            active_snr_db: 20.0,
            bootstrap_floor_db: -55.0,
            sort_buf: Vec::with_capacity(play_len),
            output,
        }
    }

    /// Process one slot in place: apply AGC gain and update dynamics state.
    /// Process one slot in place: apply AGC gain and update dynamics state.
    pub fn process_slot(&mut self, slot: &mut [f32]) {
        // ── 1. Pre-gain slot RMS ──────────────────────────────────────────
        let rms_linear = {
            let sum_sq: f32 = slot.iter().map(|s| s * s).sum();
            (sum_sq / slot.len() as f32).sqrt()
        };
        let rms_db = linear_to_db(rms_linear);

        // ── 2. Long history (quiet frames only) → noise floor ────────────────
        let long_len = self.long_history.len();
        let long_n = if self.long_filled {
            long_len
        } else {
            self.long_pos.max(1)
        };

        let noise_floor_db = if long_n >= 1 {
            self.sort_buf.clear();
            self.sort_buf
                .extend_from_slice(&self.long_history[..long_n]);
            self.sort_buf
                .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let p10_idx = ((long_n - 1) as f32 * 0.10) as usize;
            linear_to_db(self.sort_buf[p10_idx].max(1e-9))
        } else {
            self.bootstrap_floor_db
        };

        // ── 3. Active-frame gate ──────────────────────────────────────────
        let floor_db = if long_n >= 32 {
            noise_floor_db
        } else {
            self.bootstrap_floor_db
        };
        let is_active = rms_db > floor_db + self.active_snr_db;

        // ── 3b. Broadband detection (kurtosis) ───────────────────────────
        let is_broadband = if is_active {
            let n = slot.len() as f32;
            let mean_sq = rms_linear * rms_linear;
            let mean_quad = slot
                .iter()
                .map(|s| {
                    let s2 = s * s;
                    s2 * s2
                })
                .sum::<f32>()
                / n;
            let kurtosis = if mean_sq > 1e-18 {
                mean_quad / (mean_sq * mean_sq)
            } else {
                3.0
            };

            // CRITICAL FIX: True broadband noise (Gaussian) sits tightly around 3.0.
            // Hardware noise reduction removes the low-level background ("super low bins"),
            // heavily sparsifying the signal. Sparsified/gated signals have artificially
            // inflated kurtosis that easily spikes > 4.5.
            // We cap the kurtosis to prevent misclassifying gated notes as broadband noise.
            // We also add a realistic dBFS ceiling: true background room noise shouldn't be extremely loud.
            // log::trace!("kurtosis: {kurtosis}, rms_db: {rms_db}");
            kurtosis >= 2.75 && kurtosis <= 3.8 && rms_db < -45.0
        } else {
            false
        };

        // is_playing: active AND tonal (not environmental broadband noise).
        let is_playing = is_active && !is_broadband;

        // Update long_history for quiet frames *and* broadband-active frames
        // so the noise floor adapts upward when ambient noise rises.
        if !is_active || is_broadband {
            self.long_history[self.long_pos] = rms_linear;
            self.long_pos = (self.long_pos + 1) % long_len;
            if self.long_pos == 0 {
                self.long_filled = true;
            }
        }

        // ── 4. Update play history (tonal active frames only) ─────────────
        if is_playing {
            let play_len = self.play_history.len();
            self.play_history[self.play_pos] = rms_linear;
            self.play_pos = (self.play_pos + 1) % play_len;
            if self.play_pos == 0 {
                self.play_filled = true;
            }
        }

        // ── 5. Session statistics: one sort → p50 (median) + p95 (AGC) ───
        let play_n = if self.play_filled {
            self.play_history.len()
        } else {
            self.play_pos
        };

        let (raw_gain_db, session_median_db) = if play_n > 0 {
            self.sort_buf.clear();
            self.sort_buf
                .extend_from_slice(&self.play_history[..play_n]);
            self.sort_buf
                .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let p50_idx = (play_n - 1) / 2;
            let p95_idx = ((play_n - 1) as f32 * 0.95) as usize;

            let median_db = linear_to_db(self.sort_buf[p50_idx].max(1e-9));
            let p95_db = linear_to_db(self.sort_buf[p95_idx].max(1e-9));

            let gain_db = (self.target_db - p95_db).clamp(0.0, self.max_boost_db);
            (gain_db, median_db)
        } else {
            (0.0, rms_db)
        };

        // ── 6. Smooth gain ────────────────────────────────────────────────
        if is_playing {
            let target_linear = db_to_linear(raw_gain_db);
            self.current_gain_linear +=
                self.smooth_alpha * (target_linear - self.current_gain_linear);
        } else {
            self.current_gain_linear += self.silence_decay_alpha * (1.0 - self.current_gain_linear);
        }

        // ── 7. Apply gain + peak-headroom clamp ───────────────────────────
        let peak_sample = slot
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max)
            .max(1e-9);

        let headroom_limit = 0.97 / peak_sample;
        let effective_gain = self.current_gain_linear.min(headroom_limit);

        for s in slot.iter_mut() {
            *s *= effective_gain;
        }

        let applied_gain_db = linear_to_db(effective_gain);

        // ── 8. Dynamics classification ────────────────────────────────────
        let level = if !is_playing {
            DynamicLevel::Silence
        } else {
            let rel_db = rms_db - session_median_db;
            match rel_db {
                r if r < -15.0 => DynamicLevel::Ppp,
                r if r < -9.0 => DynamicLevel::Pp,
                r if r < -4.5 => DynamicLevel::P,
                r if r < -1.5 => DynamicLevel::Mp,
                r if r < 1.5 => DynamicLevel::Mf,
                r if r < 4.5 => DynamicLevel::F,
                r if r < 9.0 => DynamicLevel::Ff,
                _ => DynamicLevel::Fff,
            }
        };

        // ── 9. Publish ────────────────────────────────────────────────────
        {
            let mut out = self.output.write();
            out.level = level;
            out.rms_db = rms_db;
            out.gain_db = applied_gain_db;
            out.session_median_db = session_median_db;
            out.noise_floor_db = noise_floor_db;
        }
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

#[inline]
fn linear_to_db(linear: f32) -> f32 {
    20.0 * linear.max(1e-9).log10()
}

#[inline]
fn db_to_linear(db: f32) -> f32 {
    10.0f32.powf(db / 20.0)
}
