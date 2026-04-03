pub mod analysis;
pub mod audio_io;
pub mod dsp;
pub mod generators;
pub mod practice;
pub mod testing;
pub mod traits;

use crate::analysis::tuner::{TunerCommand, TunerMode, TunerOutput, TuningSystem};
use crate::audio_io::AudioPipeline;
use crate::generators::player::PlayerController;
use crate::generators::{
    metronome::{BeatStrength, MetronomeCommand},
    synth::SynthCommand,
};
use crate::traits::Worker;
use rtrb::Producer;
use std::sync::{Arc, Mutex};

uniffi::setup_scaffolding!();

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
        push_command(self, TunerCommand::SetBaseFreq(freq));
    }
    pub fn set_key(&self, key: String) {
        push_command(self, TunerCommand::SetKey(key));
    }
    pub fn set_mode(&self, mode: String) {
        let m = match mode.as_str() {
            "SinglePitch" => TunerMode::SinglePitch,
            _ => TunerMode::MultiPitch,
        };
        push_command(self, TunerCommand::SetMode(m));
    }
    pub fn set_system(&self, system: String) {
        let s = match system.as_str() {
            "JustIntonation" => TuningSystem::JustIntonation,
            _ => TuningSystem::EqualTemperament,
        };
        push_command(self, TunerCommand::SetSystem(s));
    }
    pub fn end(&self) {
        push_command(self, TunerCommand::End);
    }
}

fn push_command(tuner: &Tuner, cmd: TunerCommand) {
    let _ = tuner.producer.lock().unwrap().push(cmd);
}

#[derive(uniffi::Object)]
pub struct Metronome {
    // Wrapped in Mutex to satisfy Sync requirement for UniFFI/Arc
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

    pub fn set_subdivisions(&self, subdivisions: Vec<u32>, beat_index: u32) -> bool {
        let divs = subdivisions.into_iter().map(|s| s as usize).collect();
        self.producer
            .lock()
            .unwrap()
            .push(MetronomeCommand::SetPolyrhythm((divs, beat_index as usize)))
            .is_ok()
    }

    pub fn set_muted(&self, muted: bool) -> bool {
        self.producer
            .lock()
            .unwrap()
            .push(MetronomeCommand::SetMuted(muted))
            .is_ok()
    }

    pub fn stop(&self) {
        let _ = self.producer.lock().unwrap().push(MetronomeCommand::Stop);
    }
}

#[derive(uniffi::Object)]
pub struct Synth {
    producer: Mutex<rtrb::Producer<SynthCommand>>,
}

#[uniffi::export]
impl Synth {
    pub fn play_note(&self, freq: f32, velocity: f32) -> bool {
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

    pub fn load_midi(&self, path: String) -> bool {
        self.producer
            .lock()
            .unwrap()
            .push(SynthCommand::LoadFile(
                path,
                crate::generators::Instrument::Piano,
            ))
            .is_ok()
    }

    pub fn play_from_measure(&self, measure: u32) -> bool {
        self.producer
            .lock()
            .unwrap()
            .push(SynthCommand::Play {
                start_measure_idx: measure as usize,
            })
            .is_ok()
    }

    pub fn pause(&self) {
        let _ = self.producer.lock().unwrap().push(SynthCommand::Pause);
    }
    pub fn resume(&self) {
        let _ = self.producer.lock().unwrap().push(SynthCommand::Resume);
    }
    pub fn clear(&self) {
        let _ = self.producer.lock().unwrap().push(SynthCommand::Clear);
    }
    pub fn set_volume(&self, volume: f32) {
        let _ = self
            .producer
            .lock()
            .unwrap()
            .push(SynthCommand::SetVolume(volume));
    }
    pub fn stop(&self) {
        let _ = self.producer.lock().unwrap().push(SynthCommand::Stop);
    }
}

#[derive(uniffi::Object)]
pub struct Player {
    // PlayerController likely has internal state, Mutex protects it
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
    pub fn stop(&self) {
        self.controller.lock().unwrap().stop();
    }
    pub fn seek(&self, seconds: f64) {
        self.controller.lock().unwrap().seek(seconds);
    }
}

// --- The Main Engine ---

#[derive(uniffi::Object)]
pub struct AudioEngine {
    // Wrap the pipeline as well to ensure the entire Engine is Sync
    pipeline: Mutex<AudioPipeline>,
}

#[uniffi::export]
impl AudioEngine {
    #[uniffi::constructor]
    pub fn new() -> Arc<Self> {
        let pipeline = AudioPipeline::new().expect("Failed to initialize audio pipeline");
        Arc::new(AudioEngine {
            pipeline: Mutex::new(pipeline),
        })
    }

    pub fn start_input(&self) -> bool {
        self.pipeline.lock().unwrap().start_input().is_ok()
    }

    pub fn start_output(&self) -> bool {
        self.pipeline.lock().unwrap().start_output().is_ok()
    }

    pub fn create_metronome(&self, bpm: f32) -> Arc<Metronome> {
        let mut pipe = self.pipeline.lock().unwrap();
        let _ = pipe.start_output();
        let producer = pipe.spawn_metronome(Some(bpm), None, None, false);
        Arc::new(Metronome {
            producer: Mutex::new(producer),
        })
    }

    pub fn create_synth(&self) -> Arc<Synth> {
        let mut pipe = self.pipeline.lock().unwrap();
        let _ = pipe.start_output();
        let producer = pipe.spawn_synthesizer();
        Arc::new(Synth {
            producer: Mutex::new(producer),
        })
    }

    pub fn create_player(&self) -> Arc<Player> {
        let mut pipe = self.pipeline.lock().unwrap();
        let _ = pipe.start_output();
        let controller = pipe.spawn_player();
        Arc::new(Player {
            controller: Mutex::new(controller),
        })
    }

    pub fn start_recording(&self, path: String) -> bool {
        let mut pipe = self.pipeline.lock().unwrap();
        let _ = pipe.start_input();
        if let Ok(mut recorder) = pipe.spawn_recorder(&path) {
            recorder.start();
            return true;
        }
        false
    }

    pub fn start_onset_detection(&self) -> bool {
        let mut pipe = self.pipeline.lock().unwrap();
        let _ = pipe.start_input();
        if let Ok(mut onset) = pipe.spawn_onset() {
            onset.start();
            return true;
        }
        false
    }

    pub fn start_tuner(&self) -> Arc<Tuner> {
        let mut pipe = self.pipeline.lock().unwrap();
        let _ = pipe.start_input();
        let (producer, output) = pipe.spawn_tuner();
        Arc::new(Tuner {
            producer: Mutex::new(producer),
            output,
        })
    }
}
