use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use serde_json;

use crate::audio_io::timing::MusicalTransport;
use crate::generators::{Instrument, Measure, load_midi_file};
use crate::{AudioEngineError, OnsetDetection, Tuner, lock_err};

pub mod buffer;
pub mod clock;
pub mod conditioner;
pub mod matcher;
pub mod metrics;
pub mod mode;
pub mod types;

use metrics::{MeasureData, Metrics};

// ── Public types ──────────────────────────────────────────────────────────────

/// The ability level of the user, used to scale error tolerances.
/// Beginners get wider tolerances; experts are held to tighter standards.
#[derive(serde::Serialize, Clone, Copy, Debug, PartialEq)]
pub enum AbilityLevel {
    Beginner,
    Intermediate,
    Advanced,
    Pro,
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
            AbilityLevel::Pro => 0.7,
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
    /// Sustained tempo deviation (rushing/dragging across notes).
    Tempo,
    /// Note held longer than expected duration.
    HeldTooLong,
    /// Note released earlier than expected duration.
    HeldTooShort,
    /// No error — used as a sentinel when a `SendInfo` still carries useful context.
    None,
}

// ── Internal session state ────────────────────────────────────────────────────

struct SessionState {
    /// Index into `measures` where the practice range starts.
    practice_start: usize,
    /// Index into `measures` where the practice range ends (inclusive).
    practice_end: usize,
    /// Current position within `measures` (mirrors ModeController buffer state).
    current_measure_idx: usize,
    /// Completed measure data drained from ModeController.
    completed_measures: Vec<MeasureData>,
    /// Beat position where the first practice measure begins.
    /// Count-off ends when transport beat reaches this value.
    first_measure_beat: f64,
    /// True while the transport beat has not yet reached `first_measure_beat`.
    in_countoff: bool,
}

impl SessionState {
    fn new(
        practice_start: usize,
        practice_end: usize,
        has_countoff: bool,
        first_measure_beat: f64,
    ) -> Self {
        Self {
            practice_start,
            practice_end,
            current_measure_idx: practice_start,
            completed_measures: Vec::new(),
            first_measure_beat,
            in_countoff: has_countoff,
        }
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
    /// Practice mode — controls clock action filtering in the run loop.
    mode: crate::practice::types::PracticeMode,
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
        mode: crate::practice::types::PracticeMode,
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
            state: Arc::new(Mutex::new(SessionState::new(0, 0, false, 0.0))),
            feedback: Arc::new(Mutex::new(Vec::new())),
            running: Arc::new(AtomicBool::new(false)),
            thread_handle: Mutex::new(None),
            countoff_beats,
            mode,
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
        *self.state.lock().map_err(lock_err)? =
            SessionState::new(start, end, self.countoff_beats > 0, first_beat);
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
            let mode = self.mode;
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
                    mode,
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
    mode: crate::practice::types::PracticeMode,
) {
    use crate::practice::buffer::MeasureBuffer;
    use crate::practice::clock::{ClockConfig, ClockManager};
    use crate::practice::conditioner::InputConditioner;
    use crate::practice::mode::{ModeController, TickInputs};
    use crate::practice::types::TunerFrame;

    let (practice_start, practice_end, in_countoff_initial, first_measure_beat) = {
        let s = state.lock().unwrap();
        (
            s.practice_start,
            s.practice_end,
            s.in_countoff,
            s.first_measure_beat,
        )
    };

    let buffer = MeasureBuffer::new(measures.clone(), practice_start, practice_end);
    let conditioner = InputConditioner::new(transport.clone());
    let clock = ClockManager::new(
        transport.clone(),
        ClockConfig::default(),
        transport.get_bpm(),
    );
    let mut mc = ModeController::new(
        mode,
        ability_level,
        transport.clone(),
        conditioner,
        buffer,
        clock,
        practice_start,
    );

    let mut last_tuner_beat: Option<f64> = None;
    let mut in_countoff = in_countoff_initial;

    while running.load(Ordering::Relaxed) {
        let beat = transport.get_accumulated_beats();

        if in_countoff {
            if beat >= first_measure_beat {
                in_countoff = false;
                if let Ok(mut s) = state.lock() {
                    s.in_countoff = false;
                }
            } else {
                thread::sleep(Duration::from_millis(10));
                continue;
            }
        }

        let new_onsets: Vec<crate::audio_io::timing::OnsetEvent> = onset.drain_onset_events();
        let (note_names, note_cents, raw_tuner_beat) = {
            let out = tuner_output.read();
            (
                out.notes.clone(),
                out.accuracies.iter().map(|&c| c as f64).collect::<Vec<_>>(),
                out.beat_position,
            )
        };
        let calibrated_tuner_beat = transport.calibrated_beat(raw_tuner_beat);
        let tuner_frame: Option<TunerFrame> = if last_tuner_beat == Some(calibrated_tuner_beat) {
            None
        } else {
            last_tuner_beat = Some(calibrated_tuner_beat);
            let pairs: Vec<(u8, f64)> = note_names
                .iter()
                .zip(note_cents.iter())
                .filter_map(|(n, c)| note_name_to_midi(n).map(|m| (m, *c)))
                .collect();
            Some(TunerFrame {
                notes: pairs,
                tuner_beat: calibrated_tuner_beat,
            })
        };
        let dynamic_level = dynamics_output.read().level;

        let outputs = mc.tick(TickInputs {
            transport_beat: beat,
            tuner_frame: tuner_frame.as_ref(),
            new_onsets: &new_onsets,
            dynamic_level,
        });

        // Drain feedback to the shared queue.
        if !mc.feedback.is_empty() {
            if let Ok(mut g) = feedback.lock() {
                g.extend(std::mem::take(&mut mc.feedback));
            }
        }

        // Push aged-out measures into completed_measures for metrics.
        if !outputs.aged_measures.is_empty() {
            if let Ok(mut s) = state.lock() {
                s.current_measure_idx = mc.buffer.current_idx();
                s.completed_measures.extend(outputs.aged_measures);
            }
        }

        // Done condition. The session ends when the buffer has aged out the
        // last measure in the practice range (covers silent / Performance-mode
        // runs where the frontier never advances) OR when the frontier itself
        // has stepped past the last note (covers normal runs that match through
        // to the end before transport crosses the final boundary).
        if mc.buffer.is_done() || mc.frontier.0 > practice_end {
            // Drain any final feedback emissions to the shared queue so the
            // last measure's events aren't lost when we break.
            if !mc.feedback.is_empty() {
                if let Ok(mut g) = feedback.lock() {
                    g.extend(std::mem::take(&mut mc.feedback));
                }
            }
            running.store(false, Ordering::Relaxed);
            break;
        }

        thread::sleep(Duration::from_millis(10));
    }
}

// ── Pure helpers ──────────────────────────────────────────────────────────────

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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_io::timing::OnsetEvent;

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
        assert!(AbilityLevel::Advanced.tolerance_scale() > AbilityLevel::Pro.tolerance_scale());
    }
}
