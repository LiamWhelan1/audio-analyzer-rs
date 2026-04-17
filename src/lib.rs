pub mod analysis;
pub mod audio_io;
pub mod dsp;
pub mod generators;
pub mod practice;
pub mod testing;
pub mod traits;

use crate::analysis::tuner::{TunerCommand, TunerMode, TunerOutput, TuningSystem};
use crate::audio_io::AudioPipeline;
use crate::audio_io::dynamics::DynamicsOutput;
use crate::audio_io::timing::{MusicalTransport, OnsetEvent};
use crate::generators::player::PlayerController;
use crate::generators::{
    metronome::{BeatStrength, MetronomeCommand},
    synth::SynthCommand,
};
use crate::practice as practice_mod;
use rtrb::Producer;
use std::sync::{Arc, Mutex};

uniffi::setup_scaffolding!();

// --- Error Type ---

#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum AudioEngineError {
    /// Audio hardware could not be found or configured.
    #[error("Audio device unavailable: {msg}")]
    DeviceUnavailable { msg: String },

    /// An audio stream (input or output) failed to start.
    #[error("Audio stream failed: {msg}")]
    StreamFailed { msg: String },

    /// A processing component (tuner, metronome, etc.) failed to initialize.
    #[error("Failed to start {component}: {msg}")]
    SpawnFailed { component: String, msg: String },

    /// A file operation (load or save) failed.
    #[error("File error: {msg}")]
    FileError { msg: String },

    /// Internal engine state error (e.g. poisoned mutex).
    #[error("Internal engine error: {msg}")]
    Internal { msg: String },
}

pub(crate) fn lock_err<T>(_: std::sync::PoisonError<T>) -> AudioEngineError {
    AudioEngineError::Internal {
        msg: "Mutex was poisoned by a previous panic".into(),
    }
}

fn spawn_err(component: &'static str) -> impl Fn(anyhow::Error) -> AudioEngineError {
    move |e| AudioEngineError::SpawnFailed {
        component: component.into(),
        msg: e.to_string(),
    }
}

// --- Exported Objects ---

#[derive(uniffi::Object)]
pub struct Tuner {
    producer: Mutex<Producer<TunerCommand>>,
    output: Arc<parking_lot::RwLock<TunerOutput>>,
}

impl Tuner {
    /// Clone the shared tuner output handle for internal subscribers.
    pub(crate) fn output_handle(&self) -> Arc<parking_lot::RwLock<TunerOutput>> {
        self.output.clone()
    }
}

#[uniffi::export]
impl Tuner {
    pub fn poll_output(&self) -> String {
        serde_json::to_string(&*self.output.read()).unwrap_or_else(|_| "{}".into())
    }
    pub fn set_base_freq(&self, freq: f32) {
        let Ok(mut guard) = self.producer.lock() else {
            return;
        };
        let _ = guard.push(TunerCommand::SetBaseFreq(freq));
    }
    pub fn set_key(&self, key: String) {
        let Ok(mut guard) = self.producer.lock() else {
            return;
        };
        let _ = guard.push(TunerCommand::SetKey(key));
    }
    pub fn set_mode(&self, mode: String) {
        let m = match mode.as_str() {
            "SinglePitch" => TunerMode::SinglePitch,
            _ => TunerMode::MultiPitch,
        };
        let Ok(mut guard) = self.producer.lock() else {
            return;
        };
        let _ = guard.push(TunerCommand::SetMode(m));
    }
    pub fn set_system(&self, system: String) {
        let s = match system.as_str() {
            "JustIntonation" => TuningSystem::JustIntonation,
            _ => TuningSystem::EqualTemperament,
        };
        let Ok(mut guard) = self.producer.lock() else {
            return;
        };
        let _ = guard.push(TunerCommand::SetSystem(s));
    }
}

#[derive(uniffi::Object)]
pub struct Metronome {
    producer: Mutex<rtrb::Producer<MetronomeCommand>>,
}

#[uniffi::export]
impl Metronome {
    pub fn set_bpm(&self, bpm: f32) -> bool {
        let Ok(mut guard) = self.producer.lock() else {
            return false;
        };
        guard.push(MetronomeCommand::SetBpm(bpm)).is_ok()
    }
    pub fn set_volume(&self, volume: f32) -> bool {
        let Ok(mut guard) = self.producer.lock() else {
            return false;
        };
        guard.push(MetronomeCommand::SetVolume(volume)).is_ok()
    }
    pub fn set_pattern(&self, pattern: Vec<i32>) -> bool {
        let strengths = pattern
            .into_iter()
            .map(|p| match p {
                3 => BeatStrength::Strong,
                2 => BeatStrength::Medium,
                1 => BeatStrength::Weak,
                _ => BeatStrength::None,
            })
            .collect();
        let Ok(mut guard) = self.producer.lock() else {
            return false;
        };
        guard.push(MetronomeCommand::SetPattern(strengths)).is_ok()
    }
    pub fn set_muted(&self, muted: bool) -> bool {
        let Ok(mut guard) = self.producer.lock() else {
            return false;
        };
        guard.push(MetronomeCommand::SetMuted(muted)).is_ok()
    }
    pub fn set_polyrhythm(&self, subdivisions: Vec<u64>, beat_index: u64) -> bool {
        let subdivisions: Vec<usize> = subdivisions.into_iter().map(|s| s as usize).collect();
        let Ok(mut guard) = self.producer.lock() else {
            return false;
        };
        guard
            .push(MetronomeCommand::SetPolyrhythm((
                subdivisions,
                beat_index as usize,
            )))
            .is_ok()
    }
}

#[derive(uniffi::Object)]
pub struct Synth {
    producer: Mutex<rtrb::Producer<SynthCommand>>,
}

#[uniffi::export]
impl Synth {
    pub fn load_file(&self, path: String, instrument: String) -> bool {
        let inst = match instrument.as_str() {
            "Piano" => crate::generators::Instrument::Piano,
            _ => crate::generators::Instrument::Violin,
        };
        let Ok(mut guard) = self.producer.lock() else {
            return false;
        };
        guard.push(SynthCommand::LoadFile(path, inst)).is_ok()
    }
    pub fn play(&self, start_measure_idx: u64) -> bool {
        let Ok(mut guard) = self.producer.lock() else {
            return false;
        };
        guard
            .push(SynthCommand::Play {
                start_measure_idx: start_measure_idx as usize,
            })
            .is_ok()
    }
    pub fn play_note(&self, freq: f32, velocity: f32 /* 0-127 */, instrument: String) -> bool {
        let cmd = if velocity > 0.0 {
            let inst = match instrument.as_str() {
                "Piano" => crate::generators::Instrument::Piano,
                _ => crate::generators::Instrument::Violin,
            };
            SynthCommand::NoteOn {
                freq,
                velocity,
                instrument: inst,
            }
        } else {
            SynthCommand::NoteOff { freq }
        };
        let Ok(mut guard) = self.producer.lock() else {
            return false;
        };
        guard.push(cmd).is_ok()
    }
    pub fn pause(&self) {
        let Ok(mut guard) = self.producer.lock() else {
            return;
        };
        let _ = guard.push(SynthCommand::Pause);
    }
    pub fn resume(&self) {
        let Ok(mut guard) = self.producer.lock() else {
            return;
        };
        let _ = guard.push(SynthCommand::Resume);
    }
    pub fn clear(&self) -> bool {
        let Ok(mut guard) = self.producer.lock() else {
            return false;
        };
        guard.push(SynthCommand::Clear).is_ok()
    }
    pub fn set_volume(&self, volume: f32) {
        let Ok(mut guard) = self.producer.lock() else {
            return;
        };
        let _ = guard.push(SynthCommand::SetVolume(volume));
    }
    pub fn set_muted(&self, muted: bool) -> bool {
        let Ok(mut guard) = self.producer.lock() else {
            return false;
        };
        guard.push(SynthCommand::SetMuted(muted)).is_ok()
    }
}

#[derive(uniffi::Object)]
pub struct Player {
    controller: Mutex<PlayerController>,
}

#[uniffi::export]
impl Player {
    pub fn load_track(&self, path: String) -> Result<(), AudioEngineError> {
        self.controller
            .lock()
            .map_err(lock_err)?
            .load_file(&path)
            .map_err(|e| AudioEngineError::FileError { msg: e.to_string() })
    }
    pub fn play(&self) {
        let Ok(mut guard) = self.controller.lock() else {
            return;
        };
        guard.play();
    }
    pub fn pause(&self) {
        let Ok(mut guard) = self.controller.lock() else {
            return;
        };
        guard.pause();
    }
    pub fn seek(&self, seconds: f64) {
        let Ok(mut guard) = self.controller.lock() else {
            return;
        };
        guard.seek(seconds);
    }
}

#[derive(uniffi::Object)]
pub struct Recording {
    recorder: Mutex<crate::audio_io::recorder::Recorder>,
}

#[uniffi::export]
impl Recording {
    pub fn pause(&self) {
        let Ok(mut guard) = self.recorder.lock() else {
            return;
        };
        guard.pause();
    }
    pub fn resume(&self) {
        let Ok(mut guard) = self.recorder.lock() else {
            return;
        };
        guard.resume();
    }
}

#[derive(uniffi::Object)]
pub struct OnsetDetection {
    detector: Mutex<crate::analysis::onset::OnsetDetector>,
    onset_rx: Mutex<rtrb::Consumer<OnsetEvent>>,
}

impl OnsetDetection {
    /// Drain all pending onset events for internal subscribers (e.g. PracticeSession).
    pub(crate) fn drain_onset_events(&self) -> Vec<OnsetEvent> {
        let Ok(mut rx) = self.onset_rx.lock() else {
            return Vec::new();
        };
        let mut events = Vec::new();
        while let Ok(e) = rx.pop() {
            events.push(e);
        }
        events
    }
}

#[uniffi::export]
impl OnsetDetection {
    pub fn poll_onsets(&self) -> String {
        let Ok(mut rx) = self.onset_rx.lock() else {
            return "[]".into();
        };
        let mut items = Vec::new();
        while let Ok(e) = rx.pop() {
            items.push(format!(
                r#"{{"beat_position":{:.6},"raw_sample_offset":{},"velocity":{:.4}}}"#,
                e.beat_position, e.raw_sample_offset, e.velocity
            ));
        }
        format!("[{}]", items.join(","))
    }
    pub fn pause(&self) {
        let Ok(mut guard) = self.detector.lock() else {
            return;
        };
        guard.pause();
    }
    pub fn resume(&self) {
        let Ok(mut guard) = self.detector.lock() else {
            return;
        };
        guard.resume();
    }
}

// --- Practice Session ---

#[derive(uniffi::Object)]
pub struct PracticeSession {
    inner: Mutex<practice_mod::PracticeSession>,
}

#[uniffi::export]
impl PracticeSession {
    /// Begin (or restart) the session over `start_measure..=end_measure` (0-indexed).
    pub fn start(&self, start_measure: u32, end_measure: u32) -> Result<(), AudioEngineError> {
        self.inner
            .lock()
            .map_err(lock_err)?
            .start(start_measure, end_measure)
    }

    /// Stop the session and join the polling thread.
    ///
    /// Prefer calling `AudioEngine::stop_practice_session` so that the tuner
    /// and onset detector are also torn down.  This method is a convenience for
    /// pausing mid-session while keeping the engine running.
    pub fn stop(&self) {
        if let Ok(inner) = self.inner.lock() {
            inner.stop();
        }
    }

    /// Transport snapshot enriched with practice position context.
    ///
    /// Call at ~60 Hz.  Returns the `TransportSnapshot` fields plus
    /// `current_measure_idx`, `practice_start`, `practice_end`, and `in_countoff`.
    pub fn poll_transport(&self) -> String {
        self.inner
            .lock()
            .map(|s| s.poll_transport())
            .unwrap_or_else(|_| "{}".into())
    }

    /// Drain pending live feedback events as a JSON array of `SendInfo`.  Call at ~10 Hz.
    pub fn poll_errors(&self) -> String {
        self.inner
            .lock()
            .map(|s| s.poll_errors())
            .unwrap_or_else(|_| "[]".into())
    }

    /// Performance metrics for all completed measures as JSON.
    pub fn get_metrics(&self) -> String {
        self.inner
            .lock()
            .map(|s| s.get_metrics())
            .unwrap_or_else(|_| "{}".into())
    }

    /// `true` while the polling thread is running (i.e. the session is active).
    pub fn is_running(&self) -> bool {
        self.inner.lock().map(|s| s.is_running()).unwrap_or(false)
    }

    /// Change the tuner mode for the active session.
    ///
    /// Mirrors calling `Tuner::set_mode` — use this when you want a single call
    /// that keeps the practice session in sync with the standalone tuner object.
    pub fn set_tuner_mode(&self, mode: String) {
        if let Ok(inner) = self.inner.lock() {
            inner.set_tuner_mode(mode);
        }
    }

    /// Change the transport BPM for the active session.
    ///
    /// Mirrors `Metronome::set_bpm` — use this to keep the practice session tempo
    /// in sync when the user adjusts the metronome BPM.
    pub fn set_bpm(&self, bpm: f32) {
        if let Ok(inner) = self.inner.lock() {
            inner.set_bpm(bpm);
        }
    }
}

// --- The Main Engine ---

#[derive(uniffi::Object)]
pub struct AudioEngine {
    pipeline: Mutex<AudioPipeline>,
    active_tuner: Mutex<Option<Arc<Tuner>>>,
    active_metronome: Mutex<Option<Arc<Metronome>>>,
    active_synth: Mutex<Option<Arc<Synth>>>,
    active_player: Mutex<Option<Arc<Player>>>,
    active_recording: Mutex<Option<Arc<Recording>>>,
    active_onset: Mutex<Option<Arc<OnsetDetection>>>,
    active_practice_session: Mutex<Option<Arc<PracticeSession>>>,
}

#[uniffi::export]
impl AudioEngine {
    #[uniffi::constructor]
    pub fn new() -> Result<Arc<Self>, AudioEngineError> {
        let pipeline = AudioPipeline::new()
            .map_err(|e| AudioEngineError::DeviceUnavailable { msg: e.to_string() })?;
        Ok(Arc::new(AudioEngine {
            pipeline: Mutex::new(pipeline),
            active_tuner: Mutex::new(None),
            active_metronome: Mutex::new(None),
            active_synth: Mutex::new(None),
            active_player: Mutex::new(None),
            active_recording: Mutex::new(None),
            active_onset: Mutex::new(None),
            active_practice_session: Mutex::new(None),
        }))
    }

    pub fn start_input(&self) -> Result<(), AudioEngineError> {
        self.pipeline
            .lock()
            .map_err(lock_err)?
            .start_input()
            .map_err(|e| AudioEngineError::StreamFailed { msg: e.to_string() })
    }

    pub fn start_output(&self) -> Result<(), AudioEngineError> {
        self.pipeline
            .lock()
            .map_err(lock_err)?
            .start_output()
            .map_err(|e| AudioEngineError::StreamFailed { msg: e.to_string() })
    }

    // --- Creators ---

    pub fn create_metronome(&self, bpm: f32) -> Result<Arc<Metronome>, AudioEngineError> {
        if self.active_metronome.lock().map_err(lock_err)?.is_some() {
            return Err(AudioEngineError::SpawnFailed {
                component: "metronome".into(),
                msg: "Already active".into(),
            });
        }
        let producer = self
            .pipeline
            .lock()
            .map_err(lock_err)?
            .spawn_metronome(Some(bpm), None, None, true)
            .map_err(spawn_err("metronome"))?;
        let metronome = Arc::new(Metronome {
            producer: Mutex::new(producer),
        });
        *self.active_metronome.lock().map_err(lock_err)? = Some(metronome.clone());
        Ok(metronome)
    }

    pub fn create_synth(&self) -> Result<Arc<Synth>, AudioEngineError> {
        if self.active_synth.lock().map_err(lock_err)?.is_some() {
            return Err(AudioEngineError::SpawnFailed {
                component: "synth".into(),
                msg: "Already active".into(),
            });
        }
        let producer = self
            .pipeline
            .lock()
            .map_err(lock_err)?
            .spawn_synthesizer()
            .map_err(spawn_err("synthesizer"))?;
        let synth = Arc::new(Synth {
            producer: Mutex::new(producer),
        });
        *self.active_synth.lock().map_err(lock_err)? = Some(synth.clone());
        Ok(synth)
    }

    pub fn create_player(&self) -> Result<Arc<Player>, AudioEngineError> {
        if self.active_player.lock().map_err(lock_err)?.is_some() {
            return Err(AudioEngineError::SpawnFailed {
                component: "player".into(),
                msg: "Already active".into(),
            });
        }
        let controller = self
            .pipeline
            .lock()
            .map_err(lock_err)?
            .spawn_player()
            .map_err(spawn_err("player"))?;
        let player = Arc::new(Player {
            controller: Mutex::new(controller),
        });
        *self.active_player.lock().map_err(lock_err)? = Some(player.clone());
        Ok(player)
    }

    pub fn start_recording(&self, path: String) -> Result<Arc<Recording>, AudioEngineError> {
        if self.active_recording.lock().map_err(lock_err)?.is_some() {
            return Err(AudioEngineError::SpawnFailed {
                component: "recorder".into(),
                msg: "Already active".into(),
            });
        }
        let recorder = self
            .pipeline
            .lock()
            .map_err(lock_err)?
            .spawn_recorder(&path)
            .map_err(spawn_err("recorder"))?;
        let rec = Arc::new(Recording {
            recorder: Mutex::new(recorder),
        });
        *self.active_recording.lock().map_err(lock_err)? = Some(rec.clone());
        Ok(rec)
    }

    pub fn start_onset_detection(&self) -> Result<Arc<OnsetDetection>, AudioEngineError> {
        if self.active_onset.lock().map_err(lock_err)?.is_some() {
            return Err(AudioEngineError::SpawnFailed {
                component: "onset detector".into(),
                msg: "Already active".into(),
            });
        }
        let (detector, onset_rx) = self
            .pipeline
            .lock()
            .map_err(lock_err)?
            .spawn_onset()
            .map_err(spawn_err("onset detector"))?;
        let onset = Arc::new(OnsetDetection {
            detector: Mutex::new(detector),
            onset_rx: Mutex::new(onset_rx),
        });
        *self.active_onset.lock().map_err(lock_err)? = Some(onset.clone());
        Ok(onset)
    }

    pub fn start_tuner(&self) -> Result<Arc<Tuner>, AudioEngineError> {
        if self.active_tuner.lock().map_err(lock_err)?.is_some() {
            return Err(AudioEngineError::SpawnFailed {
                component: "tuner".into(),
                msg: "Already active".into(),
            });
        }
        let (producer, output) = self
            .pipeline
            .lock()
            .map_err(lock_err)?
            .spawn_tuner()
            .map_err(spawn_err("tuner"))?;
        let tuner = Arc::new(Tuner {
            producer: Mutex::new(producer),
            output,
        });
        *self.active_tuner.lock().map_err(lock_err)? = Some(tuner.clone());
        Ok(tuner)
    }

    // --- Stop Functions (Clears Mutex and Drops Engine's Arc) ---

    pub fn stop_metronome(&self) {
        if let Ok(mut guard) = self.active_metronome.lock() {
            if let Some(m) = guard.take() {
                if let Ok(mut prod) = m.producer.lock() {
                    let _ = prod.push(MetronomeCommand::Stop);
                }
            }
        }
        self.clean_output();
    }

    pub fn stop_synth(&self) {
        if let Ok(mut guard) = self.active_synth.lock() {
            if let Some(s) = guard.take() {
                if let Ok(mut prod) = s.producer.lock() {
                    let _ = prod.push(SynthCommand::Stop);
                    let _ = prod.push(SynthCommand::End);
                }
            }
        }
        self.clean_output();
    }

    pub fn stop_player(&self) {
        if let Ok(mut guard) = self.active_player.lock() {
            if let Some(p) = guard.take() {
                if let Ok(mut ctrl) = p.controller.lock() {
                    ctrl.stop();
                }
            }
        }
        self.clean_output();
    }

    pub fn stop_recording(&self) {
        if let Ok(mut guard) = self.active_recording.lock() {
            if let Some(r) = guard.take() {
                if let Ok(mut rec) = r.recorder.lock() {
                    rec.stop();
                }
            }
        }
        self.clean_input();
    }

    pub fn stop_onset_detection(&self) {
        if let Ok(mut guard) = self.active_onset.lock() {
            if let Some(o) = guard.take() {
                if let Ok(mut det) = o.detector.lock() {
                    det.stop();
                }
            }
        }
        self.clean_input();
    }

    /// Create a new practice session.
    ///
    /// * `midi_path`        — path to the MIDI reference file.
    /// * `instrument`       — instrument name (e.g. `"Piano"`, `"Violin"`).
    /// * `countoff_beats`   — number of metronome beats to play before analysis begins (0 = no count-off).
    /// * `wait_for_onset`   — when `true`, measure progression waits for the user to play before advancing.
    /// * `ability_level`    — `"Beginner"`, `"Intermediate"`, `"Advanced"`, or `"Expert"`.
    ///                         Controls how wide error tolerances are.
    pub fn create_practice_session(
        &self,
        midi_path: String,
        instrument: String,
        countoff_beats: u32,
        wait_for_onset: bool,
        ability_level: String,
        bpm: f32,
    ) -> Result<Arc<PracticeSession>, AudioEngineError> {
        if self
            .active_practice_session
            .lock()
            .map_err(lock_err)?
            .is_some()
        {
            return Err(AudioEngineError::SpawnFailed {
                component: "practice session".into(),
                msg: "Already active".into(),
            });
        }

        let level = match ability_level.to_ascii_lowercase().as_str() {
            "beginner" => practice_mod::AbilityLevel::Beginner,
            "intermediate" => practice_mod::AbilityLevel::Intermediate,
            "advanced" => practice_mod::AbilityLevel::Advanced,
            "expert" => practice_mod::AbilityLevel::Expert,
            other => {
                return Err(AudioEngineError::Internal {
                    msg: format!(
                        "Unknown ability level '{other}'. Expected one of: Beginner, Intermediate, Advanced, Expert"
                    ),
                });
            }
        };

        let tuner = self.start_tuner()?;
        let onset = self.start_onset_detection()?;
        let transport = self.transport()?;
        let dynamics = self.dynamics_output()?;

        let inner = match practice_mod::PracticeSession::new(
            transport,
            tuner,
            onset,
            dynamics,
            &midi_path,
            &instrument,
            countoff_beats,
            wait_for_onset,
            level,
            bpm,
        ) {
            Ok(i) => i,
            Err(e) => {
                self.stop_tuner();
                self.stop_onset_detection();
                return Err(e);
            }
        };
        let session = Arc::new(PracticeSession {
            inner: Mutex::new(inner),
        });
        *self.active_practice_session.lock().map_err(lock_err)? = Some(session.clone());
        Ok(session)
    }

    pub fn stop_practice_session(&self) {
        if let Ok(mut guard) = self.active_practice_session.lock() {
            if let Some(ps) = guard.take() {
                if let Ok(inner) = ps.inner.lock() {
                    inner.stop();
                }
            }
        }
        self.stop_tuner();
        self.stop_onset_detection();
    }

    pub fn stop_tuner(&self) {
        if let Ok(mut guard) = self.active_tuner.lock() {
            if let Some(t) = guard.take() {
                if let Ok(mut prod) = t.producer.lock() {
                    let _ = prod.push(TunerCommand::End);
                }
            }
        }
        self.clean_input();
    }

    pub fn poll_dynamics(&self) -> String {
        let Ok(pipe) = self.pipeline.lock() else {
            return "{}".into();
        };
        let out = pipe.dynamics_output.read();
        format!(
            r#"{{"level":"{}","rms_db":{:.1},"gain_db":{:.1},"session_median_db":{:.1},"noise_floor_db":{:.1}}}"#,
            out.level, out.rms_db, out.gain_db, out.session_median_db, out.noise_floor_db
        )
    }

    /// Snapshot the musical transport and return it as JSON.
    ///
    /// Call at ~60 Hz (every ~16 ms) from the frontend.  The returned object
    /// contains everything needed to:
    ///   - Flash a metronome beat indicator (`current_beat` / `beat_phase`)
    ///   - Advance an animated cursor (`display_beat_position`)
    ///   - Extrapolate position between polls (`capture_time_s`)
    ///
    /// Fields match `TransportSnapshot` (snake_case JSON keys).
    pub fn poll_transport(&self) -> String {
        let Ok(pipe) = self.pipeline.lock() else {
            return "{}".into();
        };
        let snap = pipe.transport.snapshot();
        serde_json::to_string(&snap).unwrap_or_else(|_| "{}".into())
    }

    pub fn clean_input(&self) {
        if let Ok(mut pipe) = self.pipeline.lock() {
            pipe.try_auto_stop_input();
        }
    }
    pub fn clean_output(&self) {
        if let Ok(mut pipe) = self.pipeline.lock() {
            pipe.try_auto_stop_output();
        }
    }
}

// ── Internal accessors (not exported through UniFFI) ──────────────────────────

impl AudioEngine {
    /// Clone the transport handle for internal subscribers (e.g. `PracticeSession`).
    pub(crate) fn transport(&self) -> Result<Arc<MusicalTransport>, AudioEngineError> {
        Ok(self.pipeline.lock().map_err(lock_err)?.transport.clone())
    }

    /// Clone the shared dynamics output for internal subscribers.
    pub(crate) fn dynamics_output(
        &self,
    ) -> Result<Arc<parking_lot::RwLock<DynamicsOutput>>, AudioEngineError> {
        Ok(self
            .pipeline
            .lock()
            .map_err(lock_err)?
            .dynamics_output
            .clone())
    }
}
