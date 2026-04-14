pub mod analysis;
pub mod audio_io;
pub mod dsp;
pub mod generators;
pub mod practice;
pub mod testing;
pub mod traits;

use crate::analysis::tuner::{TunerCommand, TunerMode, TunerOutput, TuningSystem};
use crate::audio_io::AudioPipeline;
use crate::audio_io::timing::OnsetEvent;
use crate::generators::player::PlayerController;
use crate::generators::{
    metronome::{BeatStrength, MetronomeCommand},
    synth::SynthCommand,
};
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

fn lock_err<T>(_: std::sync::PoisonError<T>) -> AudioEngineError {
    AudioEngineError::Internal { msg: "Mutex was poisoned by a previous panic".into() }
}

fn spawn_err(component: &'static str) -> impl Fn(anyhow::Error) -> AudioEngineError {
    move |e| AudioEngineError::SpawnFailed { component: component.into(), msg: e.to_string() }
}

// --- Exported Objects ---

#[derive(uniffi::Object)]
pub struct Tuner {
    producer: Mutex<Producer<TunerCommand>>,
    output: Arc<parking_lot::RwLock<TunerOutput>>,
}

#[uniffi::export]
impl Tuner {
    pub fn poll_output(&self) -> String {
        serde_json::to_string(&*self.output.read()).unwrap_or_else(|_| "{}".into())
    }
    pub fn set_base_freq(&self, freq: f32) {
        let Ok(mut guard) = self.producer.lock() else { return };
        let _ = guard.push(TunerCommand::SetBaseFreq(freq));
    }
    pub fn set_key(&self, key: String) {
        let Ok(mut guard) = self.producer.lock() else { return };
        let _ = guard.push(TunerCommand::SetKey(key));
    }
    pub fn set_mode(&self, mode: String) {
        let m = match mode.as_str() {
            "SinglePitch" => TunerMode::SinglePitch,
            _ => TunerMode::MultiPitch,
        };
        let Ok(mut guard) = self.producer.lock() else { return };
        let _ = guard.push(TunerCommand::SetMode(m));
    }
    pub fn set_system(&self, system: String) {
        let s = match system.as_str() {
            "JustIntonation" => TuningSystem::JustIntonation,
            _ => TuningSystem::EqualTemperament,
        };
        let Ok(mut guard) = self.producer.lock() else { return };
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
        let Ok(mut guard) = self.producer.lock() else { return false };
        guard.push(MetronomeCommand::SetBpm(bpm)).is_ok()
    }
    pub fn set_volume(&self, volume: f32) -> bool {
        let Ok(mut guard) = self.producer.lock() else { return false };
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
        let Ok(mut guard) = self.producer.lock() else { return false };
        guard.push(MetronomeCommand::SetPattern(strengths)).is_ok()
    }
    pub fn set_muted(&self, muted: bool) -> bool {
        let Ok(mut guard) = self.producer.lock() else { return false };
        guard.push(MetronomeCommand::SetMuted(muted)).is_ok()
    }
    pub fn set_polyrhythm(&self, subdivisions: Vec<u64>, beat_index: u64) -> bool {
        let subdivisions: Vec<usize> = subdivisions.into_iter().map(|s| s as usize).collect();
        let Ok(mut guard) = self.producer.lock() else { return false };
        guard.push(MetronomeCommand::SetPolyrhythm((subdivisions, beat_index as usize))).is_ok()
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
        let Ok(mut guard) = self.producer.lock() else { return false };
        guard.push(SynthCommand::LoadFile(path, inst)).is_ok()
    }
    pub fn play(&self, start_measure_idx: u64) -> bool {
        let Ok(mut guard) = self.producer.lock() else { return false };
        guard.push(SynthCommand::Play { start_measure_idx: start_measure_idx as usize }).is_ok()
    }
    pub fn play_note(&self, freq: f32, velocity: f32 /* 0-127 */, instrument: String) -> bool {
        let cmd = if velocity > 0.0 {
            let inst = match instrument.as_str() {
                "Piano" => crate::generators::Instrument::Piano,
                _ => crate::generators::Instrument::Violin,
            };
            SynthCommand::NoteOn { freq, velocity, instrument: inst }
        } else {
            SynthCommand::NoteOff { freq }
        };
        let Ok(mut guard) = self.producer.lock() else { return false };
        guard.push(cmd).is_ok()
    }
    pub fn pause(&self) {
        let Ok(mut guard) = self.producer.lock() else { return };
        let _ = guard.push(SynthCommand::Pause);
    }
    pub fn resume(&self) {
        let Ok(mut guard) = self.producer.lock() else { return };
        let _ = guard.push(SynthCommand::Resume);
    }
    pub fn clear(&self) -> bool {
        let Ok(mut guard) = self.producer.lock() else { return false };
        guard.push(SynthCommand::Clear).is_ok()
    }
    pub fn set_volume(&self, volume: f32) {
        let Ok(mut guard) = self.producer.lock() else { return };
        let _ = guard.push(SynthCommand::SetVolume(volume));
    }
    pub fn set_muted(&self, muted: bool) -> bool {
        let Ok(mut guard) = self.producer.lock() else { return false };
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
        let Ok(mut guard) = self.controller.lock() else { return };
        guard.play();
    }
    pub fn pause(&self) {
        let Ok(mut guard) = self.controller.lock() else { return };
        guard.pause();
    }
    pub fn seek(&self, seconds: f64) {
        let Ok(mut guard) = self.controller.lock() else { return };
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
        let Ok(mut guard) = self.recorder.lock() else { return };
        guard.pause();
    }
    pub fn resume(&self) {
        let Ok(mut guard) = self.recorder.lock() else { return };
        guard.resume();
    }
}

#[derive(uniffi::Object)]
pub struct OnsetDetection {
    detector: Mutex<crate::analysis::onset::OnsetDetector>,
    onset_rx: Mutex<rtrb::Consumer<OnsetEvent>>,
}

#[uniffi::export]
impl OnsetDetection {
    pub fn poll_onsets(&self) -> String {
        let Ok(mut rx) = self.onset_rx.lock() else { return "[]".into() };
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
        let Ok(mut guard) = self.detector.lock() else { return };
        guard.pause();
    }
    pub fn resume(&self) {
        let Ok(mut guard) = self.detector.lock() else { return };
        guard.resume();
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
        let producer = self
            .pipeline
            .lock()
            .map_err(lock_err)?
            .spawn_metronome(Some(bpm), None, None, false)
            .map_err(spawn_err("metronome"))?;
        let metronome = Arc::new(Metronome {
            producer: Mutex::new(producer),
        });
        *self.active_metronome.lock().map_err(lock_err)? = Some(metronome.clone());
        Ok(metronome)
    }

    pub fn create_synth(&self) -> Result<Arc<Synth>, AudioEngineError> {
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
    }

    pub fn stop_player(&self) {
        if let Ok(mut guard) = self.active_player.lock() {
            if let Some(p) = guard.take() {
                if let Ok(mut ctrl) = p.controller.lock() {
                    ctrl.stop();
                }
            }
        }
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
        let Ok(pipe) = self.pipeline.lock() else { return "{}".into() };
        let out = pipe.dynamics_output.read();
        format!(
            r#"{{"level":"{}","rms_db":{:.1},"gain_db":{:.1},"session_median_db":{:.1},"noise_floor_db":{:.1}}}"#,
            out.level, out.rms_db, out.gain_db, out.session_median_db, out.noise_floor_db
        )
    }

    pub fn clean_input(&self) {
        if let Ok(mut pipe) = self.pipeline.lock() {
            pipe.try_auto_stop_input();
        }
    }
}
