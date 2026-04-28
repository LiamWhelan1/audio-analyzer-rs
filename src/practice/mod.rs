use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use serde_json;

use crate::audio_io::dynamics::DynamicLevel;
use crate::audio_io::timing::MusicalTransport;
use crate::generators::{Instrument, Measure, load_midi_file};
use crate::{AudioEngineError, OnsetDetection, Tuner, lock_err};

pub mod metrics;

use metrics::{
    DynamicsEvent, ExpectedNote, INTONATION_ERR_THRESHOLD, MeasureData, Metrics, NOTE_MATCH_WINDOW,
    NoteEvent, ONSET_TIMING_ERR_THRESHOLD,
};

/// Minimum onset velocity (0.0–1.0) required to trigger an onset-driven note boundary.
/// Low-velocity onsets (bow friction, vibrato overtone shifts) are ignored for rearticulation.
const ONSET_VELOCITY_THRESHOLD: f32 = 0.3;

/// After an onset-triggered note boundary, this many beats must pass before another onset
/// can start a new note.  Prevents one physical bow-stroke from fragmenting into many
/// short `NoteEvent`s when the onset detector fires several times in quick succession.
const REFRACTORY_BEATS: f64 = 0.25;

/// Number of consecutive 5 ms polling cycles the tuner must report the same new MIDI pitch
/// before a pitch-change note boundary is confirmed. Filters vibrato overshoots (1–3 cycles)
/// while catching genuine slurred note changes (which stabilise quickly).
const PITCH_CONFIRM_POLLS: u32 = 3;

// ── Public types ──────────────────────────────────────────────────────────────

/// The ability level of the user, used to scale error tolerances.
/// Beginners get wider tolerances; experts are held to tighter standards.
#[derive(serde::Serialize, Clone, Copy, Debug, PartialEq)]
pub enum AbilityLevel {
    Beginner,
    Intermediate,
    Advanced,
    Expert,
}

impl AbilityLevel {
    /// Multiplier applied to all error thresholds.
    /// Values > 1.0 = more lenient (wider tolerance).
    /// Values < 1.0 = stricter (tighter tolerance).
    pub fn tolerance_scale(self) -> f64 {
        match self {
            AbilityLevel::Beginner => 2.0,
            AbilityLevel::Intermediate => 1.5,
            AbilityLevel::Advanced => 1.0,
            AbilityLevel::Expert => 0.7,
        }
    }
}

/// Rich per-note feedback event sent to the frontend.
///
/// Contains both the expected and received values so the frontend can display
/// "Expected D4 — received C4" or "Expected beat 3 — played at beat 3.4".
#[derive(serde::Serialize, Clone, Debug)]
pub struct SendInfo {
    pub measure: u32,
    pub note_index: usize,
    pub error_type: MusicError,
    /// Severity from 0.0 (minor) to 1.0 (maximum).
    pub intensity: f64,
    /// Human-readable description of what was expected (e.g. "D4", "beat 3", "mf").
    pub expected: String,
    /// Human-readable description of what was received (e.g. "C4", "beat 3.4", "p").
    pub received: String,
}

#[derive(serde::Serialize, Clone, PartialEq, Debug)]
pub enum MusicError {
    /// Note was played at the wrong time.
    Timing,
    /// Wrong note played when a specific pitch was expected.
    WrongNote,
    /// A note was played when silence was expected.
    UnexpectedNote,
    /// Silence was detected when a note was expected.
    MissingNote,
    /// Intonation (pitch accuracy in cents) is outside tolerance.
    Intonation,
    /// Dynamic level is outside tolerance.
    Dynamics,
    /// No error — used as a sentinel when a `SendInfo` still carries useful context.
    None,
}

// ── Internal session state ────────────────────────────────────────────────────

struct SessionState {
    /// Index into `measures` where the practice range starts.
    practice_start: usize,
    /// Index into `measures` where the practice range ends (inclusive).
    practice_end: usize,
    /// Current position within `measures`.
    current_measure_idx: usize,

    // ── Per-measure event accumulators ────────────────────────────────────
    current_onsets: Vec<crate::audio_io::timing::OnsetEvent>,
    current_notes: Vec<NoteEvent>,
    current_dynamics: Vec<DynamicsEvent>,

    // ── Note-change tracking ──────────────────────────────────────────────
    /// The note name the tuner last reported (e.g. "C#4").  `None` = silence.
    last_note_name: Option<String>,
    /// Beat position when the current note was first detected.
    last_note_beat: f64,
    /// MIDI number of the current note.
    last_note_midi: u8,
    /// Running sum of cent deviations for the current note.
    cents_sum: f64,
    /// Number of samples contributing to `cents_sum`.
    cents_count: u32,

    // ── Dynamics-change tracking ──────────────────────────────────────────
    last_dynamic: Option<DynamicLevel>,

    // ── Completed measures ────────────────────────────────────────────────
    completed_measures: Vec<MeasureData>,

    // ── Count-off state ───────────────────────────────────────────────────
    /// Beat position where the first practice measure begins.
    /// Count-off ends when transport beat reaches this value.
    first_measure_beat: f64,
    /// True while the transport beat has not yet reached `first_measure_beat`.
    in_countoff: bool,

    // ── Onset-wait mode ───────────────────────────────────────────────────
    /// When true, the measure does not advance until an onset is detected.
    wait_for_onset: bool,
    /// Beat position that the *next* note is expected at in onset-wait mode.
    next_expected_beat: f64,
    /// Set to `true` when a qualifying onset is received in the current measure.
    /// Gating measure advance on this prevents progressing without the user playing.
    onset_received_this_measure: bool,

    // ── Note-boundary refractory ──────────────────────────────────────
    /// Transport beat before which a new onset cannot trigger a note boundary.
    /// Reset each time an onset-triggered boundary fires.
    note_refractory_until: f64,

    // ── Pitch stability hysteresis ────────────────────────────────────
    /// MIDI note number of the pitch currently being evaluated for confirmation.
    /// `None` means the pitch matches the locked note and no confirmation is in progress.
    pending_midi: Option<u8>,
    /// Number of consecutive polls at `pending_midi` so far.
    pending_count: u32,
    /// Tuner beat (audio-aligned) when `pending_midi` was first observed.
    /// Used as `last_note_beat` when the boundary is eventually confirmed.
    pending_first_beat: f64,

    // ── Count-off onset calibration ───────────────────────────────────
    /// Raw fractional beat offsets (detected − nearest integer beat) collected
    /// during the count-off.  Filtered and averaged at count-off end to derive
    /// `onset_beat_offset`.
    countoff_offsets: Vec<f64>,
    /// Systematic lag in beats between detected onset positions and true beat
    /// positions, measured during the count-off.  Subtracted from every onset
    /// beat position during the performance phase.  `0.0` if no count-off data
    /// was collected (fallback: no correction applied).
    onset_beat_offset: f64,
}

impl SessionState {
    fn new(
        practice_start: usize,
        practice_end: usize,
        has_countoff: bool,
        first_measure_beat: f64,
        wait_for_onset: bool,
    ) -> Self {
        Self {
            practice_start,
            practice_end,
            current_measure_idx: practice_start,
            current_onsets: Vec::new(),
            current_notes: Vec::new(),
            current_dynamics: Vec::new(),
            last_note_name: None,
            last_note_beat: 0.0,
            last_note_midi: 0,
            cents_sum: 0.0,
            cents_count: 0,
            last_dynamic: None,
            completed_measures: Vec::new(),
            first_measure_beat,
            in_countoff: has_countoff,
            wait_for_onset,
            next_expected_beat: first_measure_beat,
            onset_received_this_measure: false,
            note_refractory_until: f64::NEG_INFINITY,
            pending_midi: None,
            pending_count: 0,
            pending_first_beat: 0.0,
            countoff_offsets: Vec::new(),
            onset_beat_offset: 0.0,
        }
    }

    /// Push the in-progress note (if any) to `current_notes` and reset tracking.
    fn finalize_pending_note(&mut self) {
        if self.last_note_name.is_some() && self.cents_count > 0 {
            self.current_notes.push(NoteEvent {
                beat_position: self.last_note_beat,
                midi_note: self.last_note_midi,
                avg_cents: self.cents_sum / self.cents_count as f64,
            });
        }
        self.last_note_name = None;
        self.cents_sum = 0.0;
        self.cents_count = 0;
    }
}

// ── PracticeSession ───────────────────────────────────────────────────────────

pub struct PracticeSession {
    /// All measures parsed from the MIDI reference file.
    measures: Arc<Vec<Measure>>,

    // Shared handles acquired from the engine at construction time.
    transport: Arc<MusicalTransport>,
    tuner: Arc<Tuner>,
    onset: Arc<OnsetDetection>,
    dynamics_output: Arc<parking_lot::RwLock<crate::audio_io::dynamics::DynamicsOutput>>,

    /// Accumulated per-measure data and note-change state.
    state: Arc<Mutex<SessionState>>,
    /// Live feedback waiting to be polled by the frontend.
    feedback: Arc<Mutex<Vec<SendInfo>>>,

    running: Arc<AtomicBool>,
    thread_handle: Mutex<Option<thread::JoinHandle<()>>>,

    /// Number of count-off beats before analysis starts.
    countoff_beats: u32,
    /// When true, measure progression waits for onset events.
    wait_for_onset: bool,
    /// User ability level — controls error tolerance scaling.
    ability_level: AbilityLevel,
}

impl PracticeSession {
    // ── Construction ─────────────────────────────────────────────────────

    /// Create a new practice session.
    ///
    /// All handles are pre-acquired by the caller (`AudioEngine::create_practice_session`).
    /// The session does not begin advancing until [`start`] is called.
    pub(crate) fn new(
        transport: Arc<MusicalTransport>,
        tuner: Arc<Tuner>,
        onset: Arc<OnsetDetection>,
        dynamics_output: Arc<parking_lot::RwLock<crate::audio_io::dynamics::DynamicsOutput>>,
        midi_path: &str,
        instrument: &str,
        countoff_beats: u32,
        wait_for_onset: bool,
        ability_level: AbilityLevel,
        bpm: f32,
    ) -> Result<Self, AudioEngineError> {
        let measures = load_midi_file(
            std::path::Path::new(midi_path),
            Instrument::from(instrument)?,
            Some(bpm),
        )
        .map_err(|e| AudioEngineError::FileError { msg: e })?;

        if measures.is_empty() {
            return Err(AudioEngineError::FileError {
                msg: "MIDI file contains no measures".into(),
            });
        }

        Ok(Self {
            measures: Arc::new(measures),
            transport,
            tuner,
            onset,
            dynamics_output,
            state: Arc::new(Mutex::new(SessionState::new(0, 0, false, 0.0, false))),
            feedback: Arc::new(Mutex::new(Vec::new())),
            running: Arc::new(AtomicBool::new(false)),
            thread_handle: Mutex::new(None),
            countoff_beats,
            wait_for_onset,
            ability_level,
        })
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────

    /// Begin the practice session over `start_measure..=end_measure` (0-indexed).
    ///
    /// Seeks the transport to the first measure's beat position, applies the BPM
    /// from the MIDI file, starts the transport, and spawns the polling thread.
    /// If `countoff_beats > 0`, the metronome plays that many beats before
    /// analysis begins.
    pub fn start(&self, start_measure: u32, end_measure: u32) -> Result<(), AudioEngineError> {
        let start = start_measure as usize;
        let end = end_measure as usize;

        if start > end {
            return Err(AudioEngineError::Internal {
                msg: format!("start_measure ({start}) > end_measure ({end})"),
            });
        }
        if end >= self.measures.len() {
            return Err(AudioEngineError::Internal {
                msg: format!(
                    "end_measure ({end}) out of range (MIDI has {} measures)",
                    self.measures.len()
                ),
            });
        }

        let first = &self.measures[start];
        let first_beat = first.global_start_beat;
        let bpm = first.bpm;

        // If there's a count-off, seek to just *before* the first count-off beat so that
        // the metronome fires a crossing event immediately and beat 1 is audible.
        // Without the -0.001, the transport starts exactly ON the integer beat and the
        // metronome only fires when it crosses the NEXT integer (beat 2).
        let seek_beat = if self.countoff_beats > 0 {
            first_beat - self.countoff_beats as f64
        } else {
            first_beat
        } - 0.001;

        // Reset state for a clean run.
        *self.state.lock().map_err(lock_err)? = SessionState::new(
            start,
            end,
            self.countoff_beats > 0,
            first_beat,
            self.wait_for_onset,
        );
        self.feedback.lock().map_err(lock_err)?.clear();

        // Stop any already-running polling thread before starting fresh.
        // This prevents two threads racing on the same state.
        if self.running.swap(false, Ordering::Relaxed) {
            if let Ok(mut guard) = self.thread_handle.lock() {
                if let Some(h) = guard.take() {
                    let _ = h.join();
                }
            }
        }

        // Configure and start the transport at the correct BPM.
        self.transport.set_bpm(bpm);
        self.transport.seek_to_beat(seek_beat);
        self.transport.play();

        // Spawn polling thread — all captures are Arc so the thread is 'static.
        self.running.store(true, Ordering::Relaxed);

        let handle = thread::spawn({
            let transport = self.transport.clone();
            let tuner_output = self.tuner.output_handle();
            let onset = self.onset.clone();
            let dynamics_output = self.dynamics_output.clone();
            let measures = self.measures.clone();
            let state = self.state.clone();
            let feedback = self.feedback.clone();
            let running = self.running.clone();
            let ability_level = self.ability_level;
            move || {
                run_session(
                    transport,
                    tuner_output,
                    onset,
                    dynamics_output,
                    measures,
                    state,
                    feedback,
                    running,
                    ability_level,
                );
            }
        });

        *self.thread_handle.lock().map_err(lock_err)? = Some(handle);
        Ok(())
    }

    /// Stop the session and join the polling thread.
    ///
    /// Tuner and onset lifecycle is managed by the lib-layer wrapper
    /// (`AudioEngine::stop_practice_session`), which calls this then
    /// clears the active tuner/onset slots.
    pub fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
        if let Ok(mut guard) = self.thread_handle.lock() {
            if let Some(h) = guard.take() {
                let _ = h.join();
            }
        }
        self.transport.stop();
    }

    /// Change the tuner mode during an active session. Reflects the FFI Tuner object's state.
    pub fn set_tuner_mode(&self, mode: String) {
        self.tuner.set_mode(mode);
    }

    /// Change the transport BPM during an active session. Reflects a Metronome BPM change.
    pub fn set_bpm(&self, bpm: f32) {
        self.transport.set_bpm(bpm);
    }

    // ── Frontend output ───────────────────────────────────────────────────

    /// Snapshot the transport enriched with practice-session position context.
    ///
    /// Combines the standard `TransportSnapshot` fields with the current measure
    /// index so the frontend can both advance the cursor and highlight the
    /// active measure without a separate call.
    ///
    /// Call at ~60 Hz alongside `poll_feedback`.  JSON shape:
    /// ```json
    /// {
    ///   "beat_position": 4.23,
    ///   "bpm": 120.0,
    ///   "is_playing": true,
    ///   "current_measure_idx": 2,
    ///   "practice_start": 0,
    ///   "practice_end": 7,
    ///   "in_countoff": false
    /// }
    /// ```
    pub fn poll_transport(&self) -> String {
        let snap = self.transport.snapshot();
        let (measure_idx, practice_start, practice_end, in_countoff) = self
            .state
            .lock()
            .map(|s| {
                (
                    s.current_measure_idx,
                    s.practice_start,
                    s.practice_end,
                    s.in_countoff,
                )
            })
            .unwrap_or((0, 0, 0, false));

        // Serialize the base snapshot then inject measure fields.
        let mut map = match serde_json::to_value(&snap) {
            Ok(serde_json::Value::Object(m)) => m,
            _ => return "{}".into(),
        };
        map.insert(
            "current_measure_idx".into(),
            serde_json::Value::Number(measure_idx.into()),
        );
        map.insert(
            "practice_start".into(),
            serde_json::Value::Number(practice_start.into()),
        );
        map.insert(
            "practice_end".into(),
            serde_json::Value::Number(practice_end.into()),
        );
        map.insert("in_countoff".into(), serde_json::Value::Bool(in_countoff));
        serde_json::Value::Object(map).to_string()
    }

    /// Drain all pending live feedback events and return them as a JSON array of `SendInfo`.
    ///
    /// Call at ~10 Hz from the frontend to receive real-time per-note feedback.
    pub fn poll_errors(&self) -> String {
        let Ok(mut guard) = self.feedback.lock() else {
            return "[]".into();
        };
        let batch = std::mem::take(&mut *guard);
        serde_json::to_string(&batch).unwrap_or_else(|_| "[]".into())
    }

    /// Compute and return performance metrics for all completed measures as JSON.
    ///
    /// Safe to call both during and after the session; returns `"{}"` if no
    /// measures have been completed yet.
    pub fn get_metrics(&self) -> String {
        let Ok(state) = self.state.lock() else {
            return "{}".into();
        };
        let completed = &state.completed_measures;
        if completed.is_empty() {
            return "{}".into();
        }

        let start_idx = completed.first().unwrap().measure_index as usize;
        let end_idx = completed.last().unwrap().measure_index as usize;
        let ref_measure = &self.measures[start_idx];

        let metrics = Metrics::compute(
            start_idx as u32,
            end_idx as u32,
            ref_measure.bpm as f64,
            completed,
        );
        serde_json::to_string(&metrics).unwrap_or_else(|_| "{}".into())
    }

    /// Returns `true` while the polling thread is still running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }
}

impl Drop for PracticeSession {
    fn drop(&mut self) {
        // Signal the polling thread to exit; it will drain itself and stop.
        // Full cleanup (tuner, onset, input stream) is handled by the
        // lib-layer wrapper's Drop or AudioEngine::stop_practice_session.
        self.running.store(false, Ordering::Relaxed);
    }
}

// ── Polling thread ────────────────────────────────────────────────────────────

fn run_session(
    transport: Arc<MusicalTransport>,
    tuner_output: Arc<parking_lot::RwLock<crate::analysis::tuner::TunerOutput>>,
    onset: Arc<OnsetDetection>,
    dynamics_output: Arc<parking_lot::RwLock<crate::audio_io::dynamics::DynamicsOutput>>,
    measures: Arc<Vec<Measure>>,
    state: Arc<Mutex<SessionState>>,
    feedback: Arc<Mutex<Vec<SendInfo>>>,
    running: Arc<AtomicBool>,
    ability_level: AbilityLevel,
) {
    let tol = ability_level.tolerance_scale();
    while running.load(Ordering::Relaxed) {
        let beat = transport.get_accumulated_beats();

        // ── 1. Drain onset ring buffer ────────────────────────────────────
        let new_onsets_raw = onset.drain_onset_events();

        // ── 2. Sample tuner (short read lock) ─────────────────────────────
        let (note_names, note_cents, tuner_beat): (Vec<String>, Vec<f64>, f64) = {
            let out = tuner_output.read();
            let names: Vec<String> = out.notes.clone();
            let cents: Vec<f64> = out.accuracies.iter().map(|&c| c as f64).collect();
            let tbeat = out.beat_position;
            (names, cents, tbeat)
        };

        // Primary note for single-pitch-style tracking (first detected pitch).
        let note_name = note_names.first().cloned();
        let note_cents_primary = note_cents.first().copied().unwrap_or(0.0);

        // ── 3. Sample dynamics (short read lock) ──────────────────────────
        let current_level = dynamics_output.read().level;

        // ── Mutate session state ──────────────────────────────────────────
        let Ok(mut s) = state.lock() else { break };

        // ── Count-off phase ───────────────────────────────────────────────
        // The transport was seeked to (first_measure_beat - countoff_beats - ε).
        // We simply wait until the transport reaches first_measure_beat.
        let mut onsets_handled = false;
        if s.in_countoff {
            if beat >= s.first_measure_beat {
                // ── Compute calibration offset from count-off detections ──────
                // During count-off we collected the fractional beat offset of each
                // detected onset (detected_beat − nearest_integer_beat).  Any
                // consistent positive offset is the system input latency.
                // Filter: accept offsets in (0.01, 0.60) beats to reject noise and
                // outliers. Require ≥ 2 samples; fall back to 0.0 if insufficient.
                let valid: Vec<f64> = s
                    .countoff_offsets
                    .iter()
                    .copied()
                    .filter(|&o| o > 0.01 && o < 0.60)
                    .collect();
                if valid.len() >= 2 {
                    s.onset_beat_offset =
                        valid.iter().sum::<f64>() / valid.len() as f64;
                    log::info!(
                        "Onset calibration: {:.4} beats from {} count-off samples",
                        s.onset_beat_offset,
                        valid.len()
                    );
                }
                // (else: onset_beat_offset stays 0.0 — no correction applied)

                s.in_countoff = false;
                // Prevent any onset from the count-off drain batch (beat < first_measure_beat)
                // from triggering a note boundary in the pitch-stability block below.
                s.note_refractory_until = s.first_measure_beat;
                // Discard all count-off onsets accumulated so far.
                s.current_onsets.clear();
                // Due to input pipeline latency, the ring buffer may deliver a count-off
                // onset in the same drain batch as the beat-0 transition.  Admit only
                // performance-period onsets (after calibration correction) so they don't
                // skew measure-0 timing metrics.
                let first_beat = s.first_measure_beat;
                let cal = s.onset_beat_offset;
                for o in &new_onsets_raw {
                    let corrected = o.beat_position - cal;
                    if corrected >= first_beat {
                        s.current_onsets.push(crate::audio_io::timing::OnsetEvent {
                            beat_position: corrected,
                            ..*o
                        });
                    }
                }
                onsets_handled = true;
            } else {
                // Still counting off — collect onset offsets for calibration but
                // skip all note/dynamic analysis.
                for o in &new_onsets_raw {
                    // offset = how many beats late the onset arrived relative to
                    // the nearest integer beat (count-off clicks are on integers).
                    let offset = o.beat_position - o.beat_position.round();
                    s.countoff_offsets.push(offset);
                }
                s.current_onsets.extend_from_slice(&new_onsets_raw);
                drop(s);
                thread::sleep(Duration::from_millis(5));
                continue;
            }
        }

        // Apply calibration offset to create beat-corrected onset events.
        // During count-off the offset is 0.0 so this is a no-op until calibrated.
        let cal = s.onset_beat_offset;
        let new_onsets: Vec<crate::audio_io::timing::OnsetEvent> = new_onsets_raw
            .into_iter()
            .map(|o| crate::audio_io::timing::OnsetEvent {
                beat_position: o.beat_position - cal,
                ..o
            })
            .collect();

        if !onsets_handled {
            s.current_onsets.extend_from_slice(&new_onsets);
        }

        // ── Onset-wait mode: detect qualifying onsets and seek ────────────
        // The transport keeps running so timing error is measurable, but
        // measure advancement is gated on onset_received_this_measure.
        if s.wait_for_onset && !new_onsets.is_empty() {
            // Find the onset nearest to next_expected_beat.
            if let Some(o) = new_onsets.iter().min_by(|a, b| {
                (a.beat_position - s.next_expected_beat)
                    .abs()
                    .partial_cmp(&(b.beat_position - s.next_expected_beat).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) {
                // Accept any onset within a generous window (2× match window).
                if (o.beat_position - s.next_expected_beat).abs() < NOTE_MATCH_WINDOW * 2.0 {
                    // Seek the transport to the actual onset beat so the beat grid
                    // reflects where the user truly played.
                    transport.seek_to_beat(o.beat_position);
                    s.onset_received_this_measure = true;
                }
            }
        }

        let prev_note_count = s.current_notes.len();

        // ── Pitch-stability hysteresis note-boundary detection ─────────────────
        //
        // A note boundary fires when:
        //   a) A qualifying onset is detected (immediate, bypasses hysteresis):
        //      same or different MIDI → rearticulation or onset-confirmed pitch change.
        //   b) The tuner reports the same NEW MIDI for PITCH_CONFIRM_POLLS consecutive
        //      polls (no onset needed — handles all slurred intervals incl. B→C, E→F).
        //   c) Silence is detected → finalise immediately.
        //
        // Brief pitch excursions (vibrato overshoot, harmonic flicker) reset the pending
        // counter because they return to the locked pitch before the threshold is reached.

        let midi_now: Option<u8> = note_name.as_deref().and_then(note_name_to_midi);
        let locked_midi: Option<u8> = s.last_note_name.as_ref().map(|_| s.last_note_midi);

        let best_onset = new_onsets
            .iter()
            .filter(|o| {
                o.velocity >= ONSET_VELOCITY_THRESHOLD
                    && o.beat_position >= s.note_refractory_until
            })
            .max_by(|a, b| {
                a.velocity
                    .partial_cmp(&b.velocity)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        match (midi_now, locked_midi) {
            (None, None) => {
                s.pending_midi = None;
                s.pending_count = 0;
            }
            (None, Some(_)) => {
                // Silence: finalise in-progress note immediately.
                s.finalize_pending_note();
                s.pending_midi = None;
                s.pending_count = 0;
            }
            (Some(_), None) | (Some(_), Some(_)) => {
                let pitch_changed = midi_now != locked_midi;

                if let Some(onset) = best_onset {
                    // Onset: immediate boundary, accurate beat from onset detector.
                    let start_beat = onset.beat_position;
                    s.finalize_pending_note();
                    let name = note_name.as_ref().unwrap();
                    s.last_note_name = Some(name.clone());
                    s.last_note_beat = start_beat;
                    s.last_note_midi = midi_now.unwrap();
                    s.cents_sum = note_cents_primary;
                    s.cents_count = 1;
                    s.pending_midi = None;
                    s.pending_count = 0;
                    s.note_refractory_until = start_beat + REFRACTORY_BEATS;
                } else if pitch_changed {
                    // No onset — use stability counter with audio-aligned beat.
                    if s.pending_midi == midi_now {
                        s.pending_count += 1;
                    } else {
                        // New candidate pitch: start fresh counter.
                        s.pending_midi = midi_now;
                        s.pending_count = 1;
                        s.pending_first_beat = tuner_beat;
                    }

                    if s.pending_count >= PITCH_CONFIRM_POLLS {
                        // Pitch has been stable long enough — confirm boundary.
                        let start_beat = s.pending_first_beat;
                        s.finalize_pending_note();
                        let name = note_name.as_ref().unwrap();
                        s.last_note_name = Some(name.clone());
                        s.last_note_beat = start_beat;
                        s.last_note_midi = midi_now.unwrap();
                        s.cents_sum = note_cents_primary;
                        s.cents_count = 1;
                        s.pending_midi = None;
                        s.pending_count = 0;
                    } else {
                        // Still confirming — accumulate on the current locked note.
                        if s.last_note_name.is_some() {
                            s.cents_sum += note_cents_primary;
                            s.cents_count += 1;
                        }
                    }
                } else {
                    // Pitch matches locked note: reset pending, accumulate.
                    s.pending_midi = None;
                    s.pending_count = 0;
                    s.cents_sum += note_cents_primary;
                    s.cents_count += 1;
                }
            }
        }

        // ── Live per-note feedback ────────────────────────────────────────────
        // Emits a SendInfo immediately each time a note is finalized, without
        // waiting for the measure boundary.  The [live] prefix in `expected`
        // distinguishes these events from the authoritative measure-end events.
        let live_item: Option<SendInfo> = if s.current_notes.len() > prev_note_count {
            let note = s.current_notes.last().unwrap();
            let expected_notes = build_expected_notes(&measures[s.current_measure_idx]);
            let closest = expected_notes.iter().enumerate().min_by(|(_, a), (_, b)| {
                (a.beat_position - note.beat_position)
                    .abs()
                    .partial_cmp(&(b.beat_position - note.beat_position).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            Some(if let Some((ei, exp)) = closest {
                let dist = (note.beat_position - exp.beat_position).abs();
                let within = dist < NOTE_MATCH_WINDOW * tol;
                let matched_midi = within && exp.midi_note == note.midi_note;
                SendInfo {
                    measure: s.current_measure_idx as u32,
                    note_index: ei,
                    error_type: if matched_midi {
                        MusicError::None
                    } else if within {
                        MusicError::WrongNote
                    } else {
                        MusicError::UnexpectedNote
                    },
                    intensity: 0.0,
                    expected: format!(
                        "[live] {} midi={} beat={:.3} (dist={:.3} window={:.3})",
                        midi_to_note_name(exp.midi_note),
                        exp.midi_note,
                        exp.beat_position,
                        dist,
                        NOTE_MATCH_WINDOW * tol,
                    ),
                    received: format!(
                        "{} midi={} beat={:.3}",
                        midi_to_note_name(note.midi_note),
                        note.midi_note,
                        note.beat_position,
                    ),
                }
            } else {
                SendInfo {
                    measure: s.current_measure_idx as u32,
                    note_index: 0,
                    error_type: MusicError::UnexpectedNote,
                    intensity: 0.0,
                    expected: "[live] silence (no expected notes)".into(),
                    received: format!(
                        "{} midi={} beat={:.3}",
                        midi_to_note_name(note.midi_note),
                        note.midi_note,
                        note.beat_position,
                    ),
                }
            })
        } else {
            None
        };

        // Dynamics-change detection (suppress Silence→Silence no-ops).
        if current_level != DynamicLevel::Silence && s.last_dynamic != Some(current_level) {
            s.current_dynamics.push(DynamicsEvent {
                beat_position: beat,
                level: current_level,
            });
            s.last_dynamic = Some(current_level);
        }

        // ── 4. Measure boundary check ─────────────────────────────────────
        let measure_end = {
            let m = &measures[s.current_measure_idx];
            m.global_start_beat + m.duration_beats()
        };

        // In onset-wait mode, block measure advance until the user has played.
        // The transport keeps running so timing error remains measurable.
        if beat >= measure_end && s.wait_for_onset && !s.onset_received_this_measure {
            drop(s);
            thread::sleep(Duration::from_millis(5));
            continue;
        }

        if beat >= measure_end {
            s.finalize_pending_note();

            // Capture the index of the measure we're closing out before mutating.
            let finished_idx = s.current_measure_idx;
            let expected = build_expected_notes(&measures[finished_idx]);

            let done = finished_idx >= s.practice_end;

            // Advance to the next measure.
            if !done {
                s.current_measure_idx += 1;
                s.onset_received_this_measure = false;

                // Update next_expected_beat for onset-wait mode.
                if s.wait_for_onset {
                    let next_idx = s.current_measure_idx;
                    if let Some(first_note) = measures[next_idx].notes.first() {
                        s.next_expected_beat = measures[next_idx].global_start_beat
                            + first_note.start_beat_in_measure as f64;
                    } else {
                        s.next_expected_beat = measures[next_idx].global_start_beat;
                    }
                }
            }

            let data = MeasureData {
                measure_index: finished_idx as u32,
                onsets: std::mem::take(&mut s.current_onsets),
                notes: std::mem::take(&mut s.current_notes),
                dynamics: std::mem::take(&mut s.current_dynamics),
                expected_notes: expected,
            };

            let new_feedback = generate_measure_feedback(&data, ability_level);
            s.completed_measures.push(data);
            s.last_dynamic = None;

            // Release state lock before locking feedback.
            drop(s);

            if let Ok(mut guard) = feedback.lock() {
                if let Some(item) = live_item {
                    guard.push(item);
                }
                guard.extend(new_feedback);
            }

            if done {
                running.store(false, Ordering::Relaxed);
                break;
            }
        } else {
            drop(s);
            if let Some(item) = live_item {
                if let Ok(mut guard) = feedback.lock() {
                    guard.push(item);
                }
            }
        }

        thread::sleep(Duration::from_millis(5));
    }
}

// ── Feedback generation ───────────────────────────────────────────────────────

fn generate_measure_feedback(data: &MeasureData, ability_level: AbilityLevel) -> Vec<SendInfo> {
    let mut feedback = Vec::new();
    let tol = ability_level.tolerance_scale();

    // Sort detected note indices by beat position for sequential lookup.
    let mut sorted_idxs: Vec<usize> = (0..data.notes.len()).collect();
    sorted_idxs.sort_by(|&a, &b| {
        data.notes[a]
            .beat_position
            .partial_cmp(&data.notes[b].beat_position)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for (ei, expected) in data.expected_notes.iter().enumerate() {
        let expected_note_label = midi_to_note_name(expected.midi_note);

        // Track whether this expected note was matched and the data.notes index of the match.
        let errors_before = feedback.len();
        let mut matched_note_idx: Option<usize> = None;

        // ── Note accuracy ─────────────────────────────────────────────────
        let matched_sp = sorted_idxs.iter().position(|&ni| {
            data.notes[ni].midi_note == expected.midi_note
                && (data.notes[ni].beat_position - expected.beat_position).abs()
                    < NOTE_MATCH_WINDOW * tol
        });

        // Check if any note was played (regardless of pitch) near the expected beat.
        let any_note_played = sorted_idxs.iter().any(|&ni| {
            (data.notes[ni].beat_position - expected.beat_position).abs() < NOTE_MATCH_WINDOW * tol
        });

        match matched_sp {
            Some(sp) => {
                let ni = sorted_idxs[sp];
                matched_note_idx = Some(ni);
                let note = &data.notes[ni];
                let received_label = if note.avg_cents.abs() < 1.0 {
                    expected_note_label.clone()
                } else {
                    format!("{} {:+.0}c", expected_note_label, note.avg_cents)
                };

                // Intonation
                let intonation_threshold = INTONATION_ERR_THRESHOLD * tol;
                if note.avg_cents.abs() > intonation_threshold {
                    feedback.push(SendInfo {
                        measure: data.measure_index,
                        note_index: ei,
                        error_type: MusicError::Intonation,
                        intensity: (note.avg_cents.abs() / 50.0).min(1.0),
                        expected: expected_note_label.clone(),
                        received: received_label.clone(),
                    });
                }
            }
            None => {
                if any_note_played {
                    // A note was played near the expected beat but with the wrong pitch.
                    let played_ni = sorted_idxs
                        .iter()
                        .find(|&&ni| {
                            (data.notes[ni].beat_position - expected.beat_position).abs()
                                < NOTE_MATCH_WINDOW * tol
                        })
                        .copied();

                    if let Some(pni) = played_ni {
                        let played_midi = data.notes[pni].midi_note;
                        let received_label = midi_to_note_name(played_midi);

                        // Check if the played pitch matches the previous or next expected note.
                        // If so, the player played the right note but at the wrong time —
                        // classify as a timing error rather than a wrong-note error.
                        let prev_midi =
                            if ei > 0 { Some(data.expected_notes[ei - 1].midi_note) } else { None };
                        let next_midi =
                            data.expected_notes.get(ei + 1).map(|e| e.midi_note);
                        let is_timing_shift =
                            Some(played_midi) == prev_midi || Some(played_midi) == next_midi;

                        if is_timing_shift {
                            feedback.push(SendInfo {
                                measure: data.measure_index,
                                note_index: ei,
                                error_type: MusicError::Timing,
                                intensity: 0.8,
                                expected: format!(
                                    "{} at beat {:.2}",
                                    expected_note_label, expected.beat_position
                                ),
                                received: format!(
                                    "{} at beat {:.2}",
                                    received_label, data.notes[pni].beat_position
                                ),
                            });
                        } else {
                            feedback.push(SendInfo {
                                measure: data.measure_index,
                                note_index: ei,
                                error_type: MusicError::WrongNote,
                                intensity: 1.0,
                                expected: expected_note_label.clone(),
                                received: received_label,
                            });
                        }
                    }
                } else {
                    // Nothing was played when a note was expected.
                    feedback.push(SendInfo {
                        measure: data.measure_index,
                        note_index: ei,
                        error_type: MusicError::MissingNote,
                        intensity: 1.0,
                        expected: expected_note_label.clone(),
                        received: "silence".into(),
                    });
                }
            }
        }

        // ── Timing ────────────────────────────────────────────────────────
        if let Some(o) = closest_onset(&data.onsets, expected.beat_position) {
            let err = o.beat_position - expected.beat_position;
            let timing_threshold = ONSET_TIMING_ERR_THRESHOLD * tol;
            if err.abs() > timing_threshold {
                feedback.push(SendInfo {
                    measure: data.measure_index,
                    note_index: ei,
                    error_type: MusicError::Timing,
                    intensity: (err.abs() / 0.5).min(1.0),
                    expected: format!("beat {:.2}", expected.beat_position),
                    received: format!("beat {:.2}", o.beat_position),
                });
            }
        }

        // ── Dynamics ─────────────────────────────────────────────────────
        if let Some(exp_dyn) = expected.dynamic {
            if let Some(act_dyn) = dynamic_at(&data.dynamics, expected.beat_position) {
                let diff = (dynamic_to_i32(act_dyn) - dynamic_to_i32(exp_dyn)).abs();
                let dynamics_threshold = if tol >= 2.0 {
                    3
                } else if tol >= 1.5 {
                    2
                } else {
                    1
                };
                if diff > dynamics_threshold {
                    feedback.push(SendInfo {
                        measure: data.measure_index,
                        note_index: ei,
                        error_type: MusicError::Dynamics,
                        intensity: (diff as f64 / 7.0).min(1.0),
                        expected: format!("{exp_dyn}"),
                        received: format!("{act_dyn}"),
                    });
                }
            }
        }

        // ── Info event for every note that passed all checks ─────────────
        // Fires whenever no errors were generated for this expected note,
        // regardless of whether the match was exact or timing-shifted.
        if feedback.len() == errors_before {
            let dyn_str = expected
                .dynamic
                .map(|d| format!(" {d}"))
                .unwrap_or_default();
            let (received_midi, received_beat, received_cents) =
                if let Some(ni) = matched_note_idx {
                    let note = &data.notes[ni];
                    (note.midi_note, note.beat_position, note.avg_cents)
                } else {
                    // No detected note — use expected values as stand-in.
                    (expected.midi_note, expected.beat_position, 0.0)
                };
            let cents_str = if received_cents.abs() > 1.0 {
                format!(" {:+.0}c", received_cents)
            } else {
                String::new()
            };
            feedback.push(SendInfo {
                measure: data.measure_index,
                note_index: ei,
                error_type: MusicError::None,
                intensity: 0.0,
                expected: format!(
                    "{} midi={} beat={:.2} dur={:.2}{}",
                    expected_note_label,
                    expected.midi_note,
                    expected.beat_position,
                    expected.duration_beats,
                    dyn_str
                ),
                received: format!(
                    "{}{} midi={} beat={:.2}",
                    midi_to_note_name(received_midi),
                    cents_str,
                    received_midi,
                    received_beat
                ),
            });
        }
    }

    // ── Unexpected notes (played when silence was expected) ───────────────
    // Any detected note that does not match any expected note within the window.
    for &ni in &sorted_idxs {
        let note = &data.notes[ni];
        let is_matched = data.expected_notes.iter().any(|expected| {
            expected.midi_note == note.midi_note
                && (note.beat_position - expected.beat_position).abs() < NOTE_MATCH_WINDOW * tol
        });
        if !is_matched {
            // Also check: is it just a wrong-note substitution (same beat, different pitch)?
            let is_substitution = data.expected_notes.iter().any(|expected| {
                (note.beat_position - expected.beat_position).abs() < NOTE_MATCH_WINDOW * tol
            });
            if !is_substitution {
                // Find the nearest expected note by beat position.
                let nearest_idx = data
                    .expected_notes
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        (note.beat_position - a.beat_position)
                            .abs()
                            .partial_cmp(&(note.beat_position - b.beat_position).abs())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i)
                    .unwrap_or(data.expected_notes.len());

                // Check if this note's MIDI matches the nearest expected note or one of
                // its immediate neighbors. If so, the player played the right note from
                // the sequence outside its expected time window — a timing error, not
                // an unexpected note.
                let is_timing_shift = if nearest_idx < data.expected_notes.len() {
                    let nei = nearest_idx;
                    let exact_midi = data.expected_notes[nei].midi_note;
                    let prev_midi =
                        if nei > 0 { Some(data.expected_notes[nei - 1].midi_note) } else { None };
                    let next_midi = data.expected_notes.get(nei + 1).map(|e| e.midi_note);
                    note.midi_note == exact_midi
                        || Some(note.midi_note) == prev_midi
                        || Some(note.midi_note) == next_midi
                } else {
                    false
                };

                if is_timing_shift {
                    let nearest_beat = data
                        .expected_notes
                        .get(nearest_idx)
                        .map(|e| e.beat_position)
                        .unwrap_or(note.beat_position);
                    feedback.push(SendInfo {
                        measure: data.measure_index,
                        note_index: nearest_idx.min(data.expected_notes.len().saturating_sub(1)),
                        error_type: MusicError::Timing,
                        intensity: 0.7,
                        expected: format!("beat {:.2}", nearest_beat),
                        received: format!(
                            "{} at beat {:.2}",
                            midi_to_note_name(note.midi_note),
                            note.beat_position
                        ),
                    });
                } else {
                    // Truly unexpected — played when silence was expected.
                    feedback.push(SendInfo {
                        measure: data.measure_index,
                        note_index: nearest_idx.min(data.expected_notes.len().saturating_sub(1)),
                        error_type: MusicError::UnexpectedNote,
                        intensity: 0.5,
                        expected: "silence".into(),
                        received: midi_to_note_name(note.midi_note),
                    });
                }
            }
        }
    }

    feedback
}

// ── MIDI reference helpers ────────────────────────────────────────────────────

/// Convert a `Measure`'s `SynthNote`s into `ExpectedNote`s with absolute beat
/// positions and dynamics derived from MIDI velocity.
fn build_expected_notes(measure: &Measure) -> Vec<ExpectedNote> {
    measure
        .notes
        .iter()
        .map(|n| ExpectedNote {
            beat_position: measure.global_start_beat + n.start_beat_in_measure as f64,
            duration_beats: n.duration_beats as f64,
            midi_note: freq_to_midi(n.freq),
            dynamic: velocity_to_dynamic(n.velocity),
        })
        .collect()
}

// ── Pure helpers ──────────────────────────────────────────────────────────────

/// Convert a frequency to the nearest MIDI note number. A4 (440 Hz) = 69.
fn freq_to_midi(freq: f32) -> u8 {
    (69.0 + 12.0 * (freq / 440.0).log2())
        .round()
        .clamp(0.0, 127.0) as u8
}

/// Parse a note name (e.g. `"C#4"`, `"Bb3"`) to a MIDI number.
/// Returns `None` on an unrecognised format.
fn note_name_to_midi(name: &str) -> Option<u8> {
    let mut chars = name.chars();
    let semitone: i16 = match chars.next()? {
        'C' => 0,
        'D' => 2,
        'E' => 4,
        'F' => 5,
        'G' => 7,
        'A' => 9,
        'B' => 11,
        _ => return None,
    };
    let next = chars.next()?;
    let (accidental, octave_str): (i16, &str) = if next == '#' {
        (1, &name[2..])
    } else if next == 'b' {
        (-1, &name[2..])
    } else {
        // `next` is already the first octave digit; slice from byte index 1.
        (0, &name[1..])
    };
    let octave: i16 = octave_str.parse().ok()?;
    // C4 = 60  →  (octave + 1) * 12 + semitone + accidental
    let midi = (octave + 1) * 12 + semitone + accidental;
    (0..=127).contains(&midi).then_some(midi as u8)
}

/// Convert a MIDI note number back to a human-readable note name (e.g. 69 → "A4").
fn midi_to_note_name(midi: u8) -> String {
    const NAMES: [&str; 12] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    let octave = (midi as i16 / 12) - 1;
    let name = NAMES[(midi % 12) as usize];
    format!("{name}{octave}")
}

/// Map a normalised MIDI velocity (0.0–1.0) to a dynamic level.
fn velocity_to_dynamic(velocity: f32) -> Option<DynamicLevel> {
    match velocity {
        v if v <= 0.0 => None,
        v if v < 0.125 => Some(DynamicLevel::Ppp),
        v if v < 0.25 => Some(DynamicLevel::Pp),
        v if v < 0.375 => Some(DynamicLevel::P),
        v if v < 0.5 => Some(DynamicLevel::Mp),
        v if v < 0.625 => Some(DynamicLevel::Mf),
        v if v < 0.75 => Some(DynamicLevel::F),
        v if v < 0.875 => Some(DynamicLevel::Ff),
        _ => Some(DynamicLevel::Fff),
    }
}

/// Return the closest onset within `NOTE_MATCH_WINDOW` beats of `target`.
fn closest_onset(
    onsets: &[crate::audio_io::timing::OnsetEvent],
    target: f64,
) -> Option<&crate::audio_io::timing::OnsetEvent> {
    onsets
        .iter()
        .min_by(|a, b| {
            (a.beat_position - target)
                .abs()
                .partial_cmp(&(b.beat_position - target).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .filter(|o| (o.beat_position - target).abs() < NOTE_MATCH_WINDOW)
}

/// Return the most recent `DynamicLevel` at or before `beat`.
fn dynamic_at(events: &[DynamicsEvent], beat: f64) -> Option<DynamicLevel> {
    events
        .iter()
        .filter(|e| e.beat_position <= beat)
        .max_by(|a, b| {
            a.beat_position
                .partial_cmp(&b.beat_position)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|e| e.level)
}

/// Map `DynamicLevel` to a signed integer for arithmetic comparison.
fn dynamic_to_i32(level: DynamicLevel) -> i32 {
    match level {
        DynamicLevel::Silence => -1,
        DynamicLevel::Ppp => 0,
        DynamicLevel::Pp => 1,
        DynamicLevel::P => 2,
        DynamicLevel::Mp => 3,
        DynamicLevel::Mf => 4,
        DynamicLevel::F => 5,
        DynamicLevel::Ff => 6,
        DynamicLevel::Fff => 7,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_io::timing::OnsetEvent;

    // ── freq_to_midi ──────────────────────────────────────────────────────────

    #[test]
    fn freq_to_midi_a4_is_69() {
        assert_eq!(freq_to_midi(440.0), 69);
    }

    #[test]
    fn freq_to_midi_c4_is_60() {
        // C4 ≈ 261.626 Hz
        assert_eq!(freq_to_midi(261.626), 60);
    }

    #[test]
    fn freq_to_midi_a5_is_81() {
        // A5 = 880 Hz (one octave above A4)
        assert_eq!(freq_to_midi(880.0), 81);
    }

    #[test]
    fn freq_to_midi_clamps_zero_to_valid_range() {
        // Very low/high freqs should not overflow u8 (clamp 0..127)
        let low = freq_to_midi(1.0);
        let high = freq_to_midi(20000.0);
        assert!(low <= 127);
        assert!(high <= 127);
    }

    // ── note_name_to_midi ─────────────────────────────────────────────────────

    #[test]
    fn note_name_to_midi_a4_is_69() {
        assert_eq!(note_name_to_midi("A4"), Some(69));
    }

    #[test]
    fn note_name_to_midi_c4_is_60() {
        assert_eq!(note_name_to_midi("C4"), Some(60));
    }

    #[test]
    fn note_name_to_midi_c_sharp_4_is_61() {
        assert_eq!(note_name_to_midi("C#4"), Some(61));
    }

    #[test]
    fn note_name_to_midi_b_flat_3_is_58() {
        // Bb3 = (3+1)*12 + 11 - 1 = 58
        assert_eq!(note_name_to_midi("Bb3"), Some(58));
    }

    #[test]
    fn note_name_to_midi_empty_string_is_none() {
        assert_eq!(note_name_to_midi(""), None);
    }

    #[test]
    fn note_name_to_midi_invalid_letter_is_none() {
        assert_eq!(note_name_to_midi("X4"), None);
    }

    #[test]
    fn note_name_to_midi_missing_octave_is_none() {
        assert_eq!(note_name_to_midi("A"), None);
    }

    #[test]
    fn note_name_to_midi_non_numeric_octave_is_none() {
        assert_eq!(note_name_to_midi("Ax"), None);
    }

    // ── midi_to_note_name ─────────────────────────────────────────────────────

    #[test]
    fn midi_to_note_name_a4_is_69() {
        assert_eq!(midi_to_note_name(69), "A4");
    }

    #[test]
    fn midi_to_note_name_c4_is_60() {
        assert_eq!(midi_to_note_name(60), "C4");
    }

    #[test]
    fn midi_to_note_name_c_sharp_4_is_61() {
        assert_eq!(midi_to_note_name(61), "C#4");
    }

    #[test]
    fn midi_to_note_name_roundtrip() {
        // Parsing and converting back should be the identity for natural notes.
        for midi in 21u8..=108 {
            let name = midi_to_note_name(midi);
            if let Some(back) = note_name_to_midi(&name) {
                assert_eq!(back, midi, "roundtrip failed for midi={midi} name={name}");
            }
        }
    }

    // ── velocity_to_dynamic ───────────────────────────────────────────────────

    #[test]
    fn velocity_zero_maps_to_none() {
        assert_eq!(velocity_to_dynamic(0.0), None);
    }

    #[test]
    fn velocity_negative_maps_to_none() {
        assert_eq!(velocity_to_dynamic(-0.1), None);
    }

    #[test]
    fn velocity_just_above_zero_maps_to_ppp() {
        assert_eq!(velocity_to_dynamic(0.05), Some(DynamicLevel::Ppp));
    }

    #[test]
    fn velocity_boundaries_map_to_correct_levels() {
        // Each boundary is the *lower* edge of a new level.
        let cases = [
            (0.125, DynamicLevel::Pp),
            (0.25, DynamicLevel::P),
            (0.375, DynamicLevel::Mp),
            (0.5, DynamicLevel::Mf),
            (0.625, DynamicLevel::F),
            (0.75, DynamicLevel::Ff),
            (0.875, DynamicLevel::Fff),
            (1.0, DynamicLevel::Fff),
        ];
        for (vel, expected) in cases {
            assert_eq!(
                velocity_to_dynamic(vel),
                Some(expected),
                "velocity {} should map to {:?}",
                vel,
                expected
            );
        }
    }

    // ── closest_onset ─────────────────────────────────────────────────────────

    fn onset(beat: f64) -> OnsetEvent {
        OnsetEvent {
            beat_position: beat,
            raw_sample_offset: 0,
            velocity: 0.8,
        }
    }

    #[test]
    fn closest_onset_returns_nearest_within_window() {
        let onsets = [onset(1.0), onset(2.0), onset(3.0)];
        let result = closest_onset(&onsets, 2.05);
        assert!(result.is_some());
        assert!((result.unwrap().beat_position - 2.0).abs() < 1e-9);
    }

    #[test]
    fn closest_onset_returns_none_when_outside_window() {
        let onsets = [onset(1.0)];
        // NOTE_MATCH_WINDOW is 0.25 beats; 0.5 beats away is outside.
        let result = closest_onset(&onsets, 1.5);
        assert!(result.is_none());
    }

    #[test]
    fn closest_onset_returns_none_on_empty_slice() {
        let result = closest_onset(&[], 1.0);
        assert!(result.is_none());
    }

    // ── dynamic_at ────────────────────────────────────────────────────────────

    fn dyn_event(beat: f64, level: DynamicLevel) -> DynamicsEvent {
        DynamicsEvent {
            beat_position: beat,
            level,
        }
    }

    #[test]
    fn dynamic_at_returns_most_recent_level_at_or_before_beat() {
        let events = [
            dyn_event(0.0, DynamicLevel::P),
            dyn_event(2.0, DynamicLevel::F),
            dyn_event(4.0, DynamicLevel::Ppp),
        ];
        assert_eq!(dynamic_at(&events, 3.0), Some(DynamicLevel::F));
    }

    #[test]
    fn dynamic_at_returns_none_before_any_event() {
        let events = [dyn_event(2.0, DynamicLevel::Mf)];
        assert_eq!(dynamic_at(&events, 1.0), None);
    }

    #[test]
    fn dynamic_at_returns_event_at_exact_beat() {
        let events = [dyn_event(2.0, DynamicLevel::Ff)];
        assert_eq!(dynamic_at(&events, 2.0), Some(DynamicLevel::Ff));
    }

    #[test]
    fn dynamic_at_returns_none_on_empty_slice() {
        assert_eq!(dynamic_at(&[], 1.0), None);
    }

    // ── dynamic_to_i32 ────────────────────────────────────────────────────────

    #[test]
    fn dynamic_to_i32_ordering_is_monotone() {
        use DynamicLevel::*;
        let levels = [Silence, Ppp, Pp, P, Mp, Mf, F, Ff, Fff];
        let ints: Vec<i32> = levels.iter().map(|&l| dynamic_to_i32(l)).collect();
        for window in ints.windows(2) {
            assert!(
                window[0] < window[1],
                "dynamic_to_i32 ordering broken at {:?}: {} vs {}",
                levels,
                window[0],
                window[1]
            );
        }
    }

    // ── AbilityLevel tolerance_scale ─────────────────────────────────────────

    #[test]
    fn ability_level_tolerance_scale_ordering() {
        // Beginner should have the widest tolerance, expert the narrowest.
        assert!(
            AbilityLevel::Beginner.tolerance_scale() > AbilityLevel::Intermediate.tolerance_scale()
        );
        assert!(
            AbilityLevel::Intermediate.tolerance_scale() > AbilityLevel::Advanced.tolerance_scale()
        );
        assert!(AbilityLevel::Advanced.tolerance_scale() > AbilityLevel::Expert.tolerance_scale());
    }

    // ── generate_measure_feedback ─────────────────────────────────────────────

    fn make_measure(
        notes: Vec<NoteEvent>,
        onsets: Vec<OnsetEvent>,
        expected: Vec<ExpectedNote>,
    ) -> MeasureData {
        MeasureData {
            measure_index: 0,
            onsets,
            notes,
            dynamics: vec![],
            expected_notes: expected,
        }
    }

    #[test]
    fn feedback_no_errors_on_perfect_performance() {
        let data = make_measure(
            vec![NoteEvent {
                beat_position: 0.0,
                midi_note: 60,
                avg_cents: 0.0,
            }],
            vec![onset(0.0)],
            vec![ExpectedNote {
                beat_position: 0.0,
                duration_beats: 1.0,
                midi_note: 60,
                dynamic: None,
            }],
        );
        let fb = generate_measure_feedback(&data, AbilityLevel::Advanced);
        assert!(
            fb.is_empty(),
            "perfect performance should produce no feedback: {fb:?}"
        );
    }

    #[test]
    fn feedback_missing_note_when_nothing_played() {
        let data = make_measure(
            vec![],
            vec![],
            vec![ExpectedNote {
                beat_position: 0.0,
                duration_beats: 1.0,
                midi_note: 60,
                dynamic: None,
            }],
        );
        let fb = generate_measure_feedback(&data, AbilityLevel::Advanced);
        assert!(
            fb.iter().any(|e| e.error_type == MusicError::MissingNote),
            "should flag MissingNote when nothing is played"
        );
    }

    #[test]
    fn feedback_wrong_note_when_different_pitch_at_right_beat() {
        let data = make_measure(
            vec![NoteEvent {
                beat_position: 0.0,
                midi_note: 62, /* D4 */
                avg_cents: 0.0,
            }],
            vec![onset(0.0)],
            vec![ExpectedNote {
                beat_position: 0.0,
                duration_beats: 1.0,
                midi_note: 60, /* C4 */
                dynamic: None,
            }],
        );
        let fb = generate_measure_feedback(&data, AbilityLevel::Advanced);
        assert!(
            fb.iter().any(|e| e.error_type == MusicError::WrongNote),
            "should flag WrongNote when wrong pitch played at correct beat"
        );
        let wrong = fb
            .iter()
            .find(|e| e.error_type == MusicError::WrongNote)
            .unwrap();
        assert_eq!(wrong.expected, "C4");
        assert_eq!(wrong.received, "D4");
    }

    #[test]
    fn feedback_timing_error_carries_beat_positions() {
        // Onset 0.2 beats late — inside NOTE_MATCH_WINDOW (0.25) so it is found,
        // and over ONSET_TIMING_ERR_THRESHOLD (0.15) so a Timing error is raised.
        let data = make_measure(
            vec![NoteEvent {
                beat_position: 0.2,
                midi_note: 60,
                avg_cents: 0.0,
            }],
            vec![onset(0.2)],
            vec![ExpectedNote {
                beat_position: 0.0,
                duration_beats: 1.0,
                midi_note: 60,
                dynamic: None,
            }],
        );
        let fb = generate_measure_feedback(&data, AbilityLevel::Advanced);
        let timing = fb.iter().find(|e| e.error_type == MusicError::Timing);
        assert!(
            timing.is_some(),
            "should flag Timing when onset is late: {fb:?}"
        );
        let t = timing.unwrap();
        assert!(
            t.expected.contains("beat 0.00"),
            "expected field should mention beat 0.00, got: {}",
            t.expected
        );
        assert!(
            t.received.contains("beat 0.20"),
            "received field should mention beat 0.20, got: {}",
            t.received
        );
    }

    #[test]
    fn feedback_intonation_error_carries_cent_info() {
        let data = make_measure(
            vec![NoteEvent {
                beat_position: 0.0,
                midi_note: 60,
                avg_cents: 35.0,
            }],
            vec![onset(0.0)],
            vec![ExpectedNote {
                beat_position: 0.0,
                duration_beats: 1.0,
                midi_note: 60,
                dynamic: None,
            }],
        );
        let fb = generate_measure_feedback(&data, AbilityLevel::Advanced);
        let inton = fb.iter().find(|e| e.error_type == MusicError::Intonation);
        assert!(
            inton.is_some(),
            "should flag Intonation for 35 cent deviation"
        );
        let i = inton.unwrap();
        assert!(
            i.received.contains("+35"),
            "received should show cent deviation, got: {}",
            i.received
        );
    }

    #[test]
    fn feedback_beginner_tolerates_wider_timing() {
        // 0.2 beat late — over Advanced threshold (0.15) but under Beginner (0.15 * 2.0 = 0.30).
        let data = make_measure(
            vec![NoteEvent {
                beat_position: 0.2,
                midi_note: 60,
                avg_cents: 0.0,
            }],
            vec![onset(0.2)],
            vec![ExpectedNote {
                beat_position: 0.0,
                duration_beats: 1.0,
                midi_note: 60,
                dynamic: None,
            }],
        );
        let fb_advanced = generate_measure_feedback(&data, AbilityLevel::Advanced);
        let fb_beginner = generate_measure_feedback(&data, AbilityLevel::Beginner);
        let advanced_has_timing = fb_advanced
            .iter()
            .any(|e| e.error_type == MusicError::Timing);
        let beginner_has_timing = fb_beginner
            .iter()
            .any(|e| e.error_type == MusicError::Timing);
        assert!(
            advanced_has_timing,
            "Advanced should flag 0.2 beat timing error"
        );
        assert!(
            !beginner_has_timing,
            "Beginner should not flag 0.2 beat timing error"
        );
    }
}
