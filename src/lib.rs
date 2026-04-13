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
        let _ = self
            .producer
            .lock()
            .unwrap()
            .push(TunerCommand::SetBaseFreq(freq));
    }
    pub fn set_key(&self, key: String) {
        let _ = self
            .producer
            .lock()
            .unwrap()
            .push(TunerCommand::SetKey(key));
    }
    pub fn set_mode(&self, mode: String) {
        let m = match mode.as_str() {
            "SinglePitch" => TunerMode::SinglePitch,
            _ => TunerMode::MultiPitch,
        };
        let _ = self.producer.lock().unwrap().push(TunerCommand::SetMode(m));
    }
    pub fn set_system(&self, system: String) {
        let s = match system.as_str() {
            "JustIntonation" => TuningSystem::JustIntonation,
            _ => TuningSystem::EqualTemperament,
        };
        let _ = self
            .producer
            .lock()
            .unwrap()
            .push(TunerCommand::SetSystem(s));
    }
}

#[derive(uniffi::Object)]
pub struct Metronome {
    producer: Mutex<rtrb::Producer<MetronomeCommand>>,
}

#[uniffi::export]
impl Metronome {
    pub fn set_bpm(&self, bpm: f32) -> bool {
        self.producer
            .lock()
            .unwrap()
            .push(MetronomeCommand::SetBpm(bpm))
            .is_ok()
    }
    pub fn set_volume(&self, volume: f32) -> bool {
        self.producer
            .lock()
            .unwrap()
            .push(MetronomeCommand::SetVolume(volume))
            .is_ok()
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
        self.producer
            .lock()
            .unwrap()
            .push(MetronomeCommand::SetPattern(strengths))
            .is_ok()
    }
    pub fn set_muted(&self, muted: bool) -> bool {
        self.producer
            .lock()
            .unwrap()
            .push(MetronomeCommand::SetMuted(muted))
            .is_ok()
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
        self.producer.lock().unwrap().push(cmd).is_ok()
    }
    pub fn pause(&self) {
        let _ = self.producer.lock().unwrap().push(SynthCommand::Pause);
    }
    pub fn resume(&self) {
        let _ = self.producer.lock().unwrap().push(SynthCommand::Resume);
    }
    pub fn set_volume(&self, volume: f32) {
        let _ = self
            .producer
            .lock()
            .unwrap()
            .push(SynthCommand::SetVolume(volume));
    }
}

#[derive(uniffi::Object)]
pub struct Player {
    controller: Mutex<PlayerController>,
}

#[uniffi::export]
impl Player {
    pub fn load_track(&self, path: String) -> bool {
        self.controller.lock().unwrap().load_file(&path).is_ok()
    }
    pub fn play(&self) {
        self.controller.lock().unwrap().play();
    }
    pub fn pause(&self) {
        self.controller.lock().unwrap().pause();
    }
    pub fn seek(&self, seconds: f64) {
        self.controller.lock().unwrap().seek(seconds);
    }
}

#[derive(uniffi::Object)]
pub struct Recording {
    recorder: Mutex<crate::audio_io::recorder::Recorder>,
}

#[uniffi::export]
impl Recording {
    pub fn pause(&self) {
        self.recorder.lock().unwrap().pause();
    }
    pub fn resume(&self) {
        self.recorder.lock().unwrap().resume();
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
        let mut rx = self.onset_rx.lock().unwrap();
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
        self.detector.lock().unwrap().pause();
    }
    pub fn resume(&self) {
        self.detector.lock().unwrap().resume();
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
    pub fn new() -> Arc<Self> {
        let pipeline = AudioPipeline::new().expect("Failed to initialize audio pipeline");
        Arc::new(AudioEngine {
            pipeline: Mutex::new(pipeline),
            active_tuner: Mutex::new(None),
            active_metronome: Mutex::new(None),
            active_synth: Mutex::new(None),
            active_player: Mutex::new(None),
            active_recording: Mutex::new(None),
            active_onset: Mutex::new(None),
        })
    }

    pub fn start_input(&self) -> bool {
        self.pipeline.lock().unwrap().start_input().is_ok()
    }
    pub fn start_output(&self) -> bool {
        self.pipeline.lock().unwrap().start_output().is_ok()
    }

    // --- Creators ---

    pub fn create_metronome(&self, bpm: f32) -> Arc<Metronome> {
        let mut pipe = self.pipeline.lock().unwrap();
        let producer = pipe
            .spawn_metronome(Some(bpm), None, None, false)
            .expect("metronome failed");
        let metronome = Arc::new(Metronome {
            producer: Mutex::new(producer),
        });
        *self.active_metronome.lock().unwrap() = Some(metronome.clone());
        metronome
    }

    pub fn create_synth(&self) -> Arc<Synth> {
        let mut pipe = self.pipeline.lock().unwrap();
        let producer = pipe.spawn_synthesizer().expect("synth failed");
        let synth = Arc::new(Synth {
            producer: Mutex::new(producer),
        });
        *self.active_synth.lock().unwrap() = Some(synth.clone());
        synth
    }

    pub fn create_player(&self) -> Arc<Player> {
        let mut pipe = self.pipeline.lock().unwrap();
        let controller = pipe.spawn_player().expect("player failed");
        let player = Arc::new(Player {
            controller: Mutex::new(controller),
        });
        *self.active_player.lock().unwrap() = Some(player.clone());
        player
    }

    pub fn start_recording(&self, path: String) -> Option<Arc<Recording>> {
        let mut pipe = self.pipeline.lock().unwrap();
        if let Ok(recorder) = pipe.spawn_recorder(&path) {
            let rec = Arc::new(Recording {
                recorder: Mutex::new(recorder),
            });
            *self.active_recording.lock().unwrap() = Some(rec.clone());
            return Some(rec);
        }
        None
    }

    pub fn start_onset_detection(&self) -> Option<Arc<OnsetDetection>> {
        let mut pipe = self.pipeline.lock().unwrap();
        if let Ok((detector, onset_rx)) = pipe.spawn_onset() {
            let onset = Arc::new(OnsetDetection {
                detector: Mutex::new(detector),
                onset_rx: Mutex::new(onset_rx),
            });
            *self.active_onset.lock().unwrap() = Some(onset.clone());
            return Some(onset);
        }
        None
    }

    pub fn start_tuner(&self) -> Option<Arc<Tuner>> {
        let mut pipe = self.pipeline.lock().unwrap();
        let (producer, output) = pipe.spawn_tuner().expect("tuner failed");
        let tuner = Arc::new(Tuner {
            producer: Mutex::new(producer),
            output,
        });
        *self.active_tuner.lock().unwrap() = Some(tuner.clone());
        Some(tuner)
    }

    // --- Stop Functions (Clears Mutex and Drops Engine's Arc) ---

    pub fn stop_metronome(&self) {
        if let Some(m) = self.active_metronome.lock().unwrap().take() {
            let _ = m.producer.lock().unwrap().push(MetronomeCommand::Stop);
        }
    }

    pub fn stop_synth(&self) {
        if let Some(s) = self.active_synth.lock().unwrap().take() {
            let _ = s.producer.lock().unwrap().push(SynthCommand::Stop);
            let _ = s.producer.lock().unwrap().push(SynthCommand::End);
        }
    }

    pub fn stop_player(&self) {
        if let Some(p) = self.active_player.lock().unwrap().take() {
            p.controller.lock().unwrap().stop();
        }
    }

    pub fn stop_recording(&self) {
        if let Some(r) = self.active_recording.lock().unwrap().take() {
            r.recorder.lock().unwrap().stop();
        }
        self.clean_input();
    }

    pub fn stop_onset_detection(&self) {
        if let Some(o) = self.active_onset.lock().unwrap().take() {
            o.detector.lock().unwrap().stop();
        }
        self.clean_input();
    }

    pub fn stop_tuner(&self) {
        if let Some(t) = self.active_tuner.lock().unwrap().take() {
            let _ = t.producer.lock().unwrap().push(TunerCommand::End);
        }
        self.clean_input();
    }

    pub fn poll_dynamics(&self) -> String {
        let pipe = self.pipeline.lock().unwrap();
        let out = pipe.dynamics_output.read();
        format!(
            r#"{{"level":"{}","rms_db":{:.1},"gain_db":{:.1},"session_median_db":{:.1},"noise_floor_db":{:.1}}}"#,
            out.level, out.rms_db, out.gain_db, out.session_median_db, out.noise_floor_db
        )
    }

    pub fn clean_input(&self) {
        self.pipeline.lock().unwrap().try_auto_stop_input();
    }
}
