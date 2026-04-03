use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicI64, AtomicU32, AtomicU64, Ordering},
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default assumed round-trip latency for the visual bridge (in seconds).
/// Calibrated at runtime via `TransportSnapshot::ui_latency_compensation_s`.
const DEFAULT_UI_LATENCY_S: f64 = 0.04; // 40 ms – conservative for RN over UniFFI

// ---------------------------------------------------------------------------
// Transport Snapshot – the lightweight value shipped across the UniFFI bridge
// ---------------------------------------------------------------------------

/// A frozen, self-contained snapshot of the transport state.
///
/// This is what you send to React Native on every tick (or at a throttled
/// cadence like every ~16 ms).  It contains everything the UI needs to
/// position a cursor, flash a metronome light, or schedule an animation
/// frame – **including pre-computed compensation values** so the RN side
/// doesn't have to know about audio internals.
#[derive(Debug, Clone, Copy)]
pub struct TransportSnapshot {
    // ── core position ──────────────────────────────────────────────────
    /// Accumulated beat position (fractional beats from song start).
    pub beat_position: f64,
    /// Current BPM.
    pub bpm: f32,
    /// Whether the transport is currently playing.
    pub is_playing: bool,

    // ── frame bookkeeping ──────────────────────────────────────────────
    /// Total output frames rendered so far.
    pub output_frames: i64,
    /// Total input frames captured so far.
    pub input_frames: i64,
    /// input_frames − output_frames.  Positive ⇒ input is ahead.
    pub drift_samples: i64,

    // ── latency compensation for the UI ────────────────────────────────
    /// The beat position the *UI* should display right now, accounting for
    /// the estimated visual pipeline delay.  Use this for cursor position.
    pub display_beat_position: f64,
    /// Seconds of latency baked into `display_beat_position`.
    pub ui_latency_compensation_s: f64,

    // ── metronome helper ───────────────────────────────────────────────
    /// The integer beat number we are currently "in" (floor of beat_position).
    pub current_beat: u64,
    /// Fractional progress through the current beat (0.0 ..< 1.0).
    pub beat_phase: f64,

    // ── input‐latency compensation for onset alignment ─────────────────
    /// How many samples of input latency the system is currently
    /// compensating for.  Onset timestamps are shifted back by this amount
    /// so they land on the correct beat grid position.
    pub input_latency_samples: i64,

    // ── timestamp ──────────────────────────────────────────────────────
    /// Monotonic timestamp (in seconds) of when this snapshot was captured,
    /// using the same clock as the audio callback.  Lets the UI extrapolate
    /// position between snapshots.
    pub capture_time_s: f64,
}

// ---------------------------------------------------------------------------
// Onset Event – timestamped in *beats*
// ---------------------------------------------------------------------------

/// A detected onset, timestamped relative to the transport's beat grid.
///
/// The `beat_position` is already corrected for input latency so that it
/// aligns with what was playing at the moment the user actually struck.
#[derive(Debug, Clone, Copy)]
pub struct OnsetEvent {
    /// Beat‐grid position of the onset (latency‐compensated).
    pub beat_position: f64,
    /// Raw sample offset within the input buffer where the onset was found.
    pub raw_sample_offset: i64,
    /// Amplitude / velocity (0.0–1.0) if available.
    pub velocity: f32,
}

// ---------------------------------------------------------------------------
// MusicalTransport – the single source of truth
// ---------------------------------------------------------------------------

/// Thread-safe musical transport for tracking beats, frame counts, and
/// latency compensation across the entire audio graph.
///
/// # Architecture role
///
/// ```text
///   Input Stream ──SPMC──▸ Onset Detector ──┐
///                         Pitch Detector ──┤
///                                          ▼
///              ┌─── Central Timing System (this) ───┐
///              │                                     │
///              ▼              ▼              ▼       ▼
///         Metronome    Wav/Audio Player   Synth   UI (UniFFI → RN)
///              │              │              │
///              └──────MPSC────┴──────────────┘
///                         ▼
///                    Output Stream
/// ```
///
/// The transport is **lock-free**: every field is an atomic that can be
/// read from the audio thread without blocking.
pub struct MusicalTransport {
    // ── frame counters ─────────────────────────────────────────────────
    output_frames: AtomicI64,
    input_frames: AtomicI64,

    // ── tempo / position ───────────────────────────────────────────────
    bpm_bits: AtomicU32,
    accumulated_beats_bits: AtomicU64,

    // ── play state ─────────────────────────────────────────────────────
    is_playing: AtomicBool,

    // ── latency bookkeeping ────────────────────────────────────────────
    /// Hardware + driver output latency reported by the audio backend,
    /// stored in samples.  Updated once when the stream opens (or when
    /// the device changes).
    output_latency_samples: AtomicI64,

    /// Hardware + driver input latency, in samples.
    input_latency_samples: AtomicI64,

    /// Extra visual‐pipeline fudge factor, in f64 bits (seconds).
    /// Covers UniFFI call overhead, RN bridge, JS→native
    /// animation scheduling, and display refresh.
    ui_latency_s_bits: AtomicU64,

    // ── sample rate (written once, read often) ─────────────────────────
    sample_rate_bits: AtomicU32,

    // ── monotonic reference clock ──────────────────────────────────────
    /// A monotonic "now" in f64‐seconds, updated each audio callback so
    /// that snapshots carry a timestamp the UI can extrapolate from.
    capture_time_s_bits: AtomicU64,
}

impl MusicalTransport {
    // =====================================================================
    // Construction
    // =====================================================================

    /// Create a new transport.
    ///
    /// * `initial_bpm` – starting tempo.
    /// * `sample_rate` – audio sample rate (e.g. 48 000).
    pub fn new(initial_bpm: f32, sample_rate: f32) -> Arc<Self> {
        Arc::new(Self {
            output_frames: AtomicI64::new(0),
            input_frames: AtomicI64::new(0),
            bpm_bits: AtomicU32::new(initial_bpm.to_bits()),
            accumulated_beats_bits: AtomicU64::new(0.0f64.to_bits()),
            is_playing: AtomicBool::new(false),
            output_latency_samples: AtomicI64::new(0),
            input_latency_samples: AtomicI64::new(0),
            ui_latency_s_bits: AtomicU64::new(DEFAULT_UI_LATENCY_S.to_bits()),
            sample_rate_bits: AtomicU32::new(sample_rate.to_bits()),
            capture_time_s_bits: AtomicU64::new(0.0f64.to_bits()),
        })
    }

    // =====================================================================
    // Audio‐thread tick methods  (called from real-time callbacks)
    // =====================================================================

    /// Advance the output side by `frames` samples.
    ///
    /// Call this at the **top** of every output audio callback.
    /// `callback_time_s` should be a monotonic timestamp (e.g. from
    /// `mach_absolute_time` on macOS / `clock_gettime(CLOCK_MONOTONIC)`).
    pub fn tick_output(&self, frames: i64, callback_time_s: f64) {
        // Store the reference clock so snapshots are timestamped.
        self.capture_time_s_bits
            .store(callback_time_s.to_bits(), Ordering::Relaxed);

        self.output_frames.fetch_add(frames, Ordering::Relaxed);

        if !self.is_playing.load(Ordering::Relaxed) {
            return;
        }

        let bpm = self.get_bpm();
        let sr = self.get_sample_rate();
        let seconds = frames as f64 / sr as f64;
        let beats_delta = seconds * (bpm as f64 / 60.0);

        self.cas_add_beats(beats_delta);
    }

    /// Advance the input side by `frames` samples.
    ///
    /// Call this at the top of every input audio callback.
    pub fn tick_input(&self, frames: i64) {
        self.input_frames.fetch_add(frames, Ordering::Relaxed);
    }

    // =====================================================================
    // Onset alignment
    // =====================================================================

    /// Convert a raw onset detected at `sample_offset` within the current
    /// input buffer into a latency‐compensated `OnsetEvent`.
    ///
    /// `sample_offset` – position inside the current input buffer (0-based).
    /// `velocity`      – onset strength / amplitude.
    ///
    /// The returned `beat_position` is shifted backwards in time by the
    /// input latency so that it aligns with what was actually being played
    /// on the output side when the user struck the note.
    pub fn stamp_onset(&self, sample_offset: i64, velocity: f32) -> OnsetEvent {
        let sr = self.get_sample_rate() as f64;
        let bpm = self.get_bpm() as f64;
        let beats_per_sample = bpm / (60.0 * sr);

        let input_lat = self.input_latency_samples.load(Ordering::Relaxed);

        // The onset's "true" beat position is the current accumulated beat
        // position minus the latency‐worth‐of‐beats, plus the sub‐buffer
        // offset converted to beats.
        let current_beats = self.get_accumulated_beats();
        let latency_beats = input_lat as f64 * beats_per_sample;
        let offset_beats = sample_offset as f64 * beats_per_sample;

        let compensated = current_beats - latency_beats + offset_beats;

        OnsetEvent {
            beat_position: compensated,
            raw_sample_offset: sample_offset,
            velocity,
        }
    }

    // =====================================================================
    // Snapshot for the UI bridge
    // =====================================================================

    /// Capture a snapshot to send across UniFFI to React Native.
    ///
    /// This is cheap (a handful of atomic loads) and can safely be called
    /// from the audio thread, a dedicated timer thread, or a UniFFI
    /// callback.  Aim for ~60 Hz (every ~16 ms).
    pub fn snapshot(&self) -> TransportSnapshot {
        let bpm = self.get_bpm();
        let beat_pos = self.get_accumulated_beats();
        let playing = self.is_playing.load(Ordering::Relaxed);

        let out_f = self.output_frames.load(Ordering::Relaxed);
        let in_f = self.input_frames.load(Ordering::Relaxed);

        let ui_lat_s = f64::from_bits(self.ui_latency_s_bits.load(Ordering::Relaxed));
        let out_lat = self.output_latency_samples.load(Ordering::Relaxed);
        let in_lat = self.input_latency_samples.load(Ordering::Relaxed);
        let sr = self.get_sample_rate() as f64;

        // Total visual delay = output hardware latency + UI bridge latency.
        let output_latency_s = out_lat as f64 / sr;
        let total_visual_delay_s = output_latency_s + ui_lat_s;
        let total_visual_delay_beats = total_visual_delay_s * (bpm as f64 / 60.0);

        // The display position leads the audio position so that what the
        // user *sees* lines up with what they *hear*.
        let display_beat = beat_pos + total_visual_delay_beats;

        let current_beat = beat_pos.floor().max(0.0) as u64;
        let beat_phase = beat_pos - beat_pos.floor();

        let cap_time = f64::from_bits(self.capture_time_s_bits.load(Ordering::Relaxed));

        TransportSnapshot {
            beat_position: beat_pos,
            bpm,
            is_playing: playing,
            output_frames: out_f,
            input_frames: in_f,
            drift_samples: in_f - out_f,
            display_beat_position: display_beat,
            ui_latency_compensation_s: total_visual_delay_s,
            current_beat,
            beat_phase,
            input_latency_samples: in_lat,
            capture_time_s: cap_time,
        }
    }

    // =====================================================================
    // Metronome helper
    // =====================================================================

    /// Returns `true` if a new beat boundary was crossed during the last
    /// `frames` output samples.  Useful for firing a metronome click or
    /// a UI flash.
    ///
    /// Call this right after `tick_output`.
    pub fn did_cross_beat(&self, frames: i64) -> Option<BeatCrossing> {
        if !self.is_playing.load(Ordering::Relaxed) {
            return None;
        }

        let sr = self.get_sample_rate() as f64;
        let bpm = self.get_bpm() as f64;
        let beats_delta = (frames as f64 / sr) * (bpm / 60.0);
        let current = self.get_accumulated_beats();
        let previous = current - beats_delta;

        let prev_beat = previous.floor() as i64;
        let curr_beat = current.floor() as i64;

        if curr_beat > prev_beat && previous >= 0.0 {
            // How many samples into this buffer the crossing happened.
            let frac_before_crossing = (prev_beat + 1) as f64 - previous;
            let sample_offset = (frac_before_crossing / beats_delta * frames as f64) as i64;

            Some(BeatCrossing {
                beat_number: (prev_beat + 1) as u64,
                sample_offset_in_buffer: sample_offset,
            })
        } else {
            None
        }
    }

    // =====================================================================
    // Scheduling helpers – for synth / wav player
    // =====================================================================

    /// Convert a beat position to an output sample frame number.
    /// Useful for scheduling future synth events or wav-file start points.
    pub fn beat_to_output_frame(&self, target_beat: f64) -> i64 {
        let sr = self.get_sample_rate() as f64;
        let bpm = self.get_bpm() as f64;
        let current_beat = self.get_accumulated_beats();
        let current_frame = self.output_frames.load(Ordering::Relaxed);

        let delta_beats = target_beat - current_beat;
        let delta_seconds = delta_beats * 60.0 / bpm;
        let delta_frames = (delta_seconds * sr) as i64;

        current_frame + delta_frames
    }

    /// How many samples until the given beat position?
    /// Returns negative if the beat is in the past.
    pub fn samples_until_beat(&self, target_beat: f64) -> i64 {
        let sr = self.get_sample_rate() as f64;
        let bpm = self.get_bpm() as f64;
        let delta_beats = target_beat - self.get_accumulated_beats();
        let delta_seconds = delta_beats * 60.0 / bpm;
        (delta_seconds * sr) as i64
    }

    // =====================================================================
    // Playback transport controls
    // =====================================================================

    pub fn play(&self) {
        self.is_playing.store(true, Ordering::SeqCst);
    }

    pub fn stop(&self) {
        self.is_playing.store(false, Ordering::SeqCst);
    }

    pub fn set_playing(&self, playing: bool) {
        self.is_playing.store(playing, Ordering::SeqCst);
    }

    /// Seek to a specific beat position.  Typically called when the user
    /// scrubs the cursor.
    pub fn seek_to_beat(&self, beat: f64) {
        self.accumulated_beats_bits
            .store(beat.to_bits(), Ordering::SeqCst);
    }

    // =====================================================================
    // BPM
    // =====================================================================

    pub fn set_bpm(&self, bpm: f32) {
        self.bpm_bits.store(bpm.to_bits(), Ordering::Relaxed);
    }

    pub fn get_bpm(&self) -> f32 {
        f32::from_bits(self.bpm_bits.load(Ordering::Relaxed))
    }

    // =====================================================================
    // Latency configuration  (call when audio device is opened / changes)
    // =====================================================================

    /// Set the hardware output latency in samples.
    /// Obtain this from your audio backend (e.g. `AudioUnit` reported latency).
    pub fn set_output_latency(&self, samples: i64) {
        self.output_latency_samples
            .store(samples, Ordering::Relaxed);
    }

    /// Set the hardware input latency in samples.
    pub fn set_input_latency(&self, samples: i64) {
        self.input_latency_samples.store(samples, Ordering::Relaxed);
    }

    /// Override the estimated UI bridge latency (seconds).
    /// Call this if you measure the actual round-trip through UniFFI + RN.
    pub fn set_ui_latency(&self, seconds: f64) {
        self.ui_latency_s_bits
            .store(seconds.to_bits(), Ordering::Relaxed);
    }

    // =====================================================================
    // Getters
    // =====================================================================

    pub fn get_accumulated_beats(&self) -> f64 {
        f64::from_bits(self.accumulated_beats_bits.load(Ordering::Relaxed))
    }

    pub fn get_sample_rate(&self) -> f32 {
        f32::from_bits(self.sample_rate_bits.load(Ordering::Relaxed))
    }

    pub fn get_output_frames(&self) -> i64 {
        self.output_frames.load(Ordering::Relaxed)
    }

    pub fn get_input_frames(&self) -> i64 {
        self.input_frames.load(Ordering::Relaxed)
    }

    pub fn get_drift_samples(&self) -> i64 {
        let in_t = self.input_frames.load(Ordering::Relaxed);
        let out_t = self.output_frames.load(Ordering::Relaxed);
        in_t - out_t
    }

    pub fn is_playing(&self) -> bool {
        self.is_playing.load(Ordering::Relaxed)
    }

    // =====================================================================
    // Reset
    // =====================================================================

    /// Full reset – rewind to beat 0, clear frame counters.
    pub fn reset(&self) {
        self.accumulated_beats_bits
            .store(0.0f64.to_bits(), Ordering::SeqCst);
        self.output_frames.store(0, Ordering::SeqCst);
        self.input_frames.store(0, Ordering::SeqCst);
    }

    // =====================================================================
    // Internal helpers
    // =====================================================================

    /// Lock-free CAS loop to add `delta` to the accumulated beats.
    fn cas_add_beats(&self, delta: f64) {
        let mut current = self.accumulated_beats_bits.load(Ordering::Relaxed);
        loop {
            let new_val = f64::from_bits(current) + delta;
            match self.accumulated_beats_bits.compare_exchange_weak(
                current,
                new_val.to_bits(),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => current = x,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Beat crossing info (returned by did_cross_beat)
// ---------------------------------------------------------------------------

/// Information about a beat boundary that was crossed during an output
/// buffer.  The `sample_offset_in_buffer` tells you exactly where in the
/// current buffer to place the metronome click for sample-accurate timing.
#[derive(Debug, Clone, Copy)]
pub struct BeatCrossing {
    /// Which beat number was crossed (1-based from song start).
    pub beat_number: u64,
    /// Sample offset within the current output buffer where the beat lands.
    /// Use this to align the metronome click sample-accurately.
    pub sample_offset_in_buffer: i64,
}

// ---------------------------------------------------------------------------
// UniFFI bridge helpers
// ---------------------------------------------------------------------------
//
// The types above are all `Copy` / `Clone` and contain no heap allocations,
// so they map cleanly to UniFFI scalars and records.  Expose them like:
//
// ```
// #[uniffi::export]
// impl MusicalTransport {
//     fn snapshot(&self) -> TransportSnapshot;  // already defined
//     fn play(&self);
//     fn stop(&self);
//     fn seek_to_beat(&self, beat: f64);
//     fn set_bpm(&self, bpm: f32);
// }
// ```
//
// On the React Native side, poll `snapshot()` at ~60 Hz (or use a native
// module timer callback) and feed `display_beat_position` into your
// Animated / Reanimated value.  Example (TypeScript pseudocode):
//
// ```ts
// // In a useAnimatedFrame or setInterval(16ms):
// const snap = transport.snapshot();
// cursorX.value = snap.display_beat_position * pixelsPerBeat;
//
// // Metronome flash: compare snap.current_beat to previous
// if (snap.current_beat !== prevBeat.current) {
//   triggerFlash();
//   prevBeat.current = snap.current_beat;
// }
// ```
//
// The `display_beat_position` already accounts for output hardware latency
// plus the estimated UI bridge delay, so the cursor will visually lead the
// audio by just the right amount.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tick_and_beat_accumulation() {
        let t = MusicalTransport::new(120.0, 48000.0);
        t.play();

        // 120 BPM = 2 beats/sec.  One buffer of 480 samples at 48 kHz = 0.01 s = 0.02 beats.
        t.tick_output(480, 0.0);

        let beats = t.get_accumulated_beats();
        assert!(
            (beats - 0.02).abs() < 1e-9,
            "expected ~0.02 beats, got {beats}"
        );
    }

    #[test]
    fn test_beat_crossing_detection() {
        let t = MusicalTransport::new(120.0, 48000.0);
        t.play();

        // Advance to just before beat 1.
        // At 120 BPM, beat 1 is at 0.5 s = 24 000 samples.
        // Advance 23 520 samples (0.49 s = 0.98 beats).
        t.tick_output(23_520, 0.0);
        assert!(t.did_cross_beat(23_520).is_none());

        // Now advance 960 more samples (0.02 s = 0.04 beats → crosses 1.0).
        t.tick_output(960, 0.49);
        let crossing = t.did_cross_beat(960);
        assert!(crossing.is_some());
        let c = crossing.unwrap();
        assert_eq!(c.beat_number, 1);
        assert!(c.sample_offset_in_buffer >= 0);
        assert!(c.sample_offset_in_buffer <= 960);
    }

    #[test]
    fn test_onset_latency_compensation() {
        let t = MusicalTransport::new(120.0, 48000.0);
        t.play();
        t.set_input_latency(480); // 10 ms input latency

        // Advance to beat 2.0 exactly.
        // 120 BPM → 2 beats/sec → 1 sec = 48 000 samples for 2 beats.
        t.tick_output(48_000, 0.0);
        t.tick_input(48_000);

        let onset = t.stamp_onset(0, 0.8);
        // The onset should be placed ~0.02 beats before current position
        // (480 samples of latency at 120 BPM / 48 kHz).
        let expected = 2.0 - (480.0 / 48000.0) * (120.0 / 60.0);
        assert!(
            (onset.beat_position - expected).abs() < 1e-6,
            "expected ~{expected}, got {}",
            onset.beat_position
        );
    }

    #[test]
    fn test_snapshot_display_position_leads_audio() {
        let t = MusicalTransport::new(120.0, 48000.0);
        t.play();
        t.set_output_latency(480);
        t.set_ui_latency(0.01); // 10 ms

        t.tick_output(24_000, 0.5); // 0.5 s → 1.0 beat

        let snap = t.snapshot();
        // display_beat_position should be ahead of beat_position.
        assert!(snap.display_beat_position > snap.beat_position);
    }
}
