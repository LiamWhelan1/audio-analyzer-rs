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
    ARTICULATION_TOLERANCE, DynamicsEvent, ExpectedNote, INTONATION_ERR_THRESHOLD, MeasureData,
    Metrics, NOTE_MATCH_WINDOW, NoteEvent, ONSET_TIMING_ERR_THRESHOLD,
};

// ── Public types ──────────────────────────────────────────────────────────────

#[derive(serde::Serialize, Clone)]
pub struct SendError {
    pub measure: u32,
    pub note_index: usize,
    pub error_type: MusicError,
    /// Severity from 0.0 (minor) to 1.0 (maximum).
    pub intensity: f64,
}

#[derive(serde::Serialize, Clone)]
pub enum MusicError {
    Timing,
    Note,
    Intonation,
    Dynamics,
    Articulation,
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
}

impl SessionState {
    fn new(practice_start: usize, practice_end: usize) -> Self {
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
    /// Live errors waiting to be polled by the frontend.
    errors: Arc<Mutex<Vec<SendError>>>,

    running: Arc<AtomicBool>,
    thread_handle: Mutex<Option<thread::JoinHandle<()>>>,
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
    ) -> Result<Self, AudioEngineError> {
        let measures = load_midi_file(
            std::path::Path::new(midi_path),
            Instrument::from(instrument)?,
        )
        .map_err(|e| AudioEngineError::FileError { msg: e })?;

        if measures.is_empty() {
            return Err(AudioEngineError::FileError {
                msg: "MIDI file contains no measures".into(),
            });
        }

        // Single-pitch mode gives cleaner per-note cent data during practice.
        tuner.set_mode("SinglePitch".into());

        Ok(Self {
            measures: Arc::new(measures),
            transport,
            tuner,
            onset,
            dynamics_output,
            state: Arc::new(Mutex::new(SessionState::new(0, 0))),
            errors: Arc::new(Mutex::new(Vec::new())),
            running: Arc::new(AtomicBool::new(false)),
            thread_handle: Mutex::new(None),
        })
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────

    /// Begin the practice session over `start_measure..=end_measure` (0-indexed).
    ///
    /// Seeks the transport to the first measure's beat position, applies the BPM
    /// from the MIDI file, starts the transport, and spawns the polling thread.
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

        // Reset state for a clean run.
        *self.state.lock().map_err(lock_err)? = SessionState::new(start, end);
        self.errors.lock().map_err(lock_err)?.clear();

        // Seek and configure the transport.
        let first = &self.measures[start];
        self.transport.set_bpm(first.bpm);
        self.transport.seek_to_beat(first.global_start_beat);
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
            let errors = self.errors.clone();
            let running = self.running.clone();
            move || {
                run_session(
                    transport,
                    tuner_output,
                    onset,
                    dynamics_output,
                    measures,
                    state,
                    errors,
                    running,
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

    // ── Frontend output ───────────────────────────────────────────────────

    /// Snapshot the transport enriched with practice-session position context.
    ///
    /// Combines the standard `TransportSnapshot` fields with the current measure
    /// index so the frontend can both advance the cursor and highlight the
    /// active measure without a separate call.
    ///
    /// Call at ~60 Hz alongside `poll_errors`.  JSON shape:
    /// ```json
    /// {
    ///   "beat_position": 4.23,
    ///   "display_beat_position": 4.31,
    ///   "bpm": 120.0,
    ///   "is_playing": true,
    ///   "current_beat": 4,
    ///   "beat_phase": 0.23,
    ///   "capture_time_s": 12.34,
    ///   "drift_samples": 0,
    ///   "output_frames": 576000,
    ///   "input_frames": 576000,
    ///   "input_latency_samples": 2048,
    ///   "ui_latency_compensation_s": 0.05,
    ///   "current_measure_idx": 2,
    ///   "practice_start": 0,
    ///   "practice_end": 7
    /// }
    /// ```
    pub fn poll_transport(&self) -> String {
        let snap = self.transport.snapshot();
        let (measure_idx, practice_start, practice_end) = self
            .state
            .lock()
            .map(|s| (s.current_measure_idx, s.practice_start, s.practice_end))
            .unwrap_or((0, 0, 0));

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
        serde_json::Value::Object(map).to_string()
    }

    /// Drain all pending live errors and return them as a JSON array of `SendError`.
    ///
    /// Call at ~10 Hz from the frontend to receive real-time per-note feedback.
    pub fn poll_errors(&self) -> String {
        let Ok(mut guard) = self.errors.lock() else {
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
            ref_measure.time_signature.0 as u32,
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
    errors: Arc<Mutex<Vec<SendError>>>,
    running: Arc<AtomicBool>,
) {
    while running.load(Ordering::Relaxed) {
        let beat = transport.get_accumulated_beats();

        // ── 1. Drain onset ring buffer ────────────────────────────────────
        let new_onsets = onset.drain_onset_events();

        // ── 2. Sample tuner (short read lock) ─────────────────────────────
        let (note_name, note_cents): (Option<String>, f64) = {
            let out = tuner_output.read();
            let name = out.notes.first().cloned();
            let cents = out.accuracies.first().copied().unwrap_or(0.0) as f64;
            (name, cents)
        };

        // ── 3. Sample dynamics (short read lock) ──────────────────────────
        let current_level = dynamics_output.read().level;

        // ── Mutate session state ──────────────────────────────────────────
        let Ok(mut s) = state.lock() else { break };

        s.current_onsets.extend_from_slice(&new_onsets);

        // Note-change detection
        match note_name {
            Some(ref name) if s.last_note_name.as_deref() != Some(name.as_str()) => {
                // A different note has appeared: finalise the previous one.
                s.finalize_pending_note();
                s.last_note_name = Some(name.clone());
                s.last_note_beat = beat;
                s.last_note_midi = note_name_to_midi(name).unwrap_or(0);
                s.cents_sum = note_cents;
                s.cents_count = 1;
            }
            Some(_) => {
                // Same note: accumulate cents for the running average.
                s.cents_sum += note_cents;
                s.cents_count += 1;
            }
            None => {
                // Silence: finalise any note in progress.
                if s.last_note_name.is_some() {
                    s.finalize_pending_note();
                }
            }
        }

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

        if beat >= measure_end {
            s.finalize_pending_note();

            let expected = build_expected_notes(&measures[s.current_measure_idx]);
            let data = MeasureData {
                measure_index: s.current_measure_idx as u32,
                onsets: std::mem::take(&mut s.current_onsets),
                notes: std::mem::take(&mut s.current_notes),
                dynamics: std::mem::take(&mut s.current_dynamics),
                expected_notes: expected,
            };

            let new_errors = generate_measure_errors(&data);
            s.completed_measures.push(data);
            s.last_dynamic = None;

            let done = s.current_measure_idx >= s.practice_end;
            if !done {
                s.current_measure_idx += 1;
            }

            // Release state lock before locking errors.
            drop(s);

            if let Ok(mut guard) = errors.lock() {
                guard.extend(new_errors);
            }

            if done {
                running.store(false, Ordering::Relaxed);
                break;
            }
        } else {
            drop(s);
        }

        thread::sleep(Duration::from_millis(5));
    }
}

// ── Error generation ──────────────────────────────────────────────────────────

fn generate_measure_errors(data: &MeasureData) -> Vec<SendError> {
    let mut errors = Vec::new();

    // Pre-sort detected note indices by beat position for articulation duration.
    let mut sorted_idxs: Vec<usize> = (0..data.notes.len()).collect();
    sorted_idxs.sort_by(|&a, &b| {
        data.notes[a]
            .beat_position
            .partial_cmp(&data.notes[b].beat_position)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for (ei, expected) in data.expected_notes.iter().enumerate() {
        // ── Note accuracy ─────────────────────────────────────────────────
        let matched_sp = sorted_idxs.iter().position(|&ni| {
            data.notes[ni].midi_note == expected.midi_note
                && (data.notes[ni].beat_position - expected.beat_position).abs() < NOTE_MATCH_WINDOW
        });

        match matched_sp {
            Some(sp) => {
                let note = &data.notes[sorted_idxs[sp]];

                // Intonation
                if note.avg_cents.abs() > INTONATION_ERR_THRESHOLD {
                    errors.push(SendError {
                        measure: data.measure_index,
                        note_index: ei,
                        error_type: MusicError::Intonation,
                        intensity: (note.avg_cents.abs() / 50.0).min(1.0),
                    });
                }

                // Articulation (requires a subsequent detected note)
                if sp + 1 < sorted_idxs.len() {
                    let next = &data.notes[sorted_idxs[sp + 1]];
                    let actual_dur = next.beat_position - note.beat_position;
                    let ratio = (actual_dur - expected.duration_beats).abs()
                        / expected.duration_beats.max(1e-6);
                    if ratio > ARTICULATION_TOLERANCE {
                        errors.push(SendError {
                            measure: data.measure_index,
                            note_index: ei,
                            error_type: MusicError::Articulation,
                            intensity: ratio.min(1.0),
                        });
                    }
                }
            }
            None => {
                errors.push(SendError {
                    measure: data.measure_index,
                    note_index: ei,
                    error_type: MusicError::Note,
                    intensity: 1.0,
                });
            }
        }

        // ── Timing ────────────────────────────────────────────────────────
        if let Some(o) = closest_onset(&data.onsets, expected.beat_position) {
            let err = (o.beat_position - expected.beat_position).abs();
            if err > ONSET_TIMING_ERR_THRESHOLD {
                errors.push(SendError {
                    measure: data.measure_index,
                    note_index: ei,
                    error_type: MusicError::Timing,
                    intensity: (err / 0.5).min(1.0),
                });
            }
        }

        // ── Dynamics ─────────────────────────────────────────────────────
        if let Some(exp_dyn) = expected.dynamic {
            if let Some(act_dyn) = dynamic_at(&data.dynamics, expected.beat_position) {
                let diff = (dynamic_to_i32(act_dyn) - dynamic_to_i32(exp_dyn)).abs();
                if diff > 1 {
                    errors.push(SendError {
                        measure: data.measure_index,
                        note_index: ei,
                        error_type: MusicError::Dynamics,
                        intensity: (diff as f64 / 7.0).min(1.0),
                    });
                }
            }
        }
    }

    errors
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
