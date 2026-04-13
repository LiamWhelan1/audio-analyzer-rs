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
    #[error("{msg}")]
    Error { msg: String },
}

impl From<anyhow::Error> for AudioEngineError {
    fn from(e: anyhow::Error) -> Self {
        Self::Error { msg: e.to_string() }
    }
}

fn lock_err<T>(_: std::sync::PoisonError<T>) -> AudioEngineError {
    AudioEngineError::Error { msg: "Internal lock error".into() }
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
}

#[derive(uniffi::Object)]
pub struct Synth {
    producer: Mutex<rtrb::Producer<SynthCommand>>,
}

#[uniffi::export]
impl Synth {
    pub fn play_note(&self, freq: f32, velocity: f32 /* 0-127 */) -> bool {
        let cmd = if velocity > 0.0 {
            SynthCommand::NoteOn {
                freq,
                velocity,
                instrument: crate::generators::Instrument::Violin,
            }
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
    pub fn set_volume(&self, volume: f32) {
        let Ok(mut guard) = self.producer.lock() else { return };
        let _ = guard.push(SynthCommand::SetVolume(volume));
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
            .map_err(|e| AudioEngineError::Error { msg: e.to_string() })
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
        let pipeline = AudioPipeline::new()?;
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
        self.pipeline.lock().map_err(lock_err)?.start_input().map_err(Into::into)
    }

    pub fn start_output(&self) -> Result<(), AudioEngineError> {
        self.pipeline.lock().map_err(lock_err)?.start_output().map_err(Into::into)
    }

    // --- Creators ---

    pub fn create_metronome(&self, bpm: f32) -> Result<Arc<Metronome>, AudioEngineError> {
        let producer = self
            .pipeline
            .lock()
            .map_err(lock_err)?
            .spawn_metronome(Some(bpm), None, None, false)?;
        let metronome = Arc::new(Metronome {
            producer: Mutex::new(producer),
        });
        *self.active_metronome.lock().map_err(lock_err)? = Some(metronome.clone());
        Ok(metronome)
    }

    pub fn create_synth(&self) -> Result<Arc<Synth>, AudioEngineError> {
        let producer = self.pipeline.lock().map_err(lock_err)?.spawn_synthesizer()?;
        let synth = Arc::new(Synth {
            producer: Mutex::new(producer),
        });
        *self.active_synth.lock().map_err(lock_err)? = Some(synth.clone());
        Ok(synth)
    }

    pub fn create_player(&self) -> Result<Arc<Player>, AudioEngineError> {
        let controller = self.pipeline.lock().map_err(lock_err)?.spawn_player()?;
        let player = Arc::new(Player {
            controller: Mutex::new(controller),
        });
        *self.active_player.lock().map_err(lock_err)? = Some(player.clone());
        Ok(player)
    }

    pub fn start_recording(&self, path: String) -> Result<Arc<Recording>, AudioEngineError> {
        let recorder = self.pipeline.lock().map_err(lock_err)?.spawn_recorder(&path)?;
        let rec = Arc::new(Recording {
            recorder: Mutex::new(recorder),
        });
        *self.active_recording.lock().map_err(lock_err)? = Some(rec.clone());
        Ok(rec)
    }

    pub fn start_onset_detection(&self) -> Result<Arc<OnsetDetection>, AudioEngineError> {
        let (detector, onset_rx) = self.pipeline.lock().map_err(lock_err)?.spawn_onset()?;
        let onset = Arc::new(OnsetDetection {
            detector: Mutex::new(detector),
            onset_rx: Mutex::new(onset_rx),
        });
        *self.active_onset.lock().map_err(lock_err)? = Some(onset.clone());
        Ok(onset)
    }

    pub fn start_tuner(&self) -> Result<Arc<Tuner>, AudioEngineError> {
        let (producer, output) = self.pipeline.lock().map_err(lock_err)?.spawn_tuner()?;
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
