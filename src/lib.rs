pub mod analysis;
pub mod audio_io;
pub mod dsp;
pub mod generators;
// pub mod pipeline;
pub mod resample;
pub mod testing;
pub mod traits;

use anyhow::Result;
use crossbeam_channel::{Sender, bounded};
use std::thread;

use crate::audio_io::AudioPipeline;
use crate::audio_io::stft::STFT;
use crate::generators::Note;
use crate::traits::Worker;

pub fn play(builder: &mut AudioPipeline) -> Result<()> {
    println!("▶️  Play command received.");
    builder.start_input()?;
    Ok(())
}

pub fn pause() -> Result<()> {
    println!("⏸️  Pause command received.");
    // TODO: Implement pause logic
    Ok(())
}

pub fn start_worker(worker: &mut Box<dyn Worker>) {
    worker.start();
}

pub fn stop_worker(worker: &mut Box<dyn Worker>, builder: &mut AudioPipeline) {
    let handle = worker.stop();
    builder.remove_consumer(handle);
}

pub fn pause_worker(worker: &mut Box<dyn Worker>) {
    worker.pause();
}

pub fn record(builder: &mut AudioPipeline, path: &str) -> Result<audio_io::recorder::Recorder> {
    println!("🎙️  Record command received.");
    builder.start_input()?;
    let rec = builder.spawn_recorder(path);
    Ok(rec)
}

pub fn tune(
    mode: analysis::TuningSystem,
    base: Option<f32>,
    builder: &mut AudioPipeline,
) -> Result<STFT> {
    println!("🎵  Tune command received.");
    let (s, r) = bounded(1);
    // TODO: Implement tuning (pitch analysis, reference tone, etc.)
    // Should have multiple modes to select from:
    // Regular notes, chord, orchestra, ...?
    // Different tuning systems: equal temperament, just intonation, ...?
    builder.start_input()?;
    let tuner = builder.spawn_transformer(s);
    thread::spawn(move || {
        loop {
            let notes = if let Ok(n) = r.recv() {
                n
            } else {
                break;
            };
            for note in notes {
                print!("{}, ", Note::from_freq(note, base));
            }
            println!("");
        }
    });
    Ok(tuner)
}

pub fn metronome(
    bpm: f32,
    beats: usize,
    division: usize,
    subdivision: Option<Vec<usize>>,
    pattern: Vec<generators::metronome::BeatStrength>,
) -> Result<Sender<bool>> {
    println!("🕒  Metronome command received.");
    let (s, r) = bounded(1);
    let metronome =
        generators::metronome::Metronome::new(bpm, beats, division, subdivision, pattern);
    metronome.play(r);
    Ok(s)
}

pub fn upload<T>(pic: T) -> Result<()> {
    println!("Upload command received.");
    // TODO: Implement piece scanning
    Ok(())
}

pub fn drone(pitch: &str) -> Result<()> {
    println!("Drone at {pitch}");
    // TODO: Implement drone pitch
    Ok(())
}

#[cfg(target_os = "android")]
pub mod android {
    use crate::*;
    use jni::JNIEnv;
    use jni::objects::JClass;
    use jni::sys::jint;
    #[no_mangle]
    pub unsafe extern "C" fn Java_com_aariyauna_cadenza_RustAnalyzerModule_play(
        _env: JNIEnv,
        _class: JClass,
    ) {
    }
    #[no_mangle]
    pub unsafe extern "C" fn Java_com_aariyauna_cadenza_RustAnalyzerModule_pause() {}
    #[no_mangle]
    pub unsafe extern "C" fn Java_com_aariyauna_cadenza_RustAnalyzerModule_start_worker() {}
    #[no_mangle]
    pub unsafe extern "C" fn Java_com_aariyauna_cadenza_RustAnalyzerModule_stop_worker() {}
    #[no_mangle]
    pub unsafe extern "C" fn Java_com_aariyauna_cadenza_RustAnalyzerModule_pause_worker() {}
    #[no_mangle]
    pub unsafe extern "C" fn Java_com_aariyauna_cadenza_RustAnalyzerModule_record() {}
    #[no_mangle]
    pub unsafe extern "C" fn Java_com_aariyauna_cadenza_RustAnalyzerModule_tune() {}
    #[no_mangle]
    pub unsafe extern "C" fn Java_com_aariyauna_cadenza_RustAnalyzerModule_metronome() {}
    #[no_mangle]
    pub unsafe extern "C" fn Java_com_aariyauna_cadenza_RustAnalyzerModule_upload() {}
    #[no_mangle]
    pub unsafe extern "C" fn Java_com_aariyauna_cadenza_RustAnalyzerModule_drone() {}
}
