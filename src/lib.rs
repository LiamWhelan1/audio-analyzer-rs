pub mod analysis;
pub mod audio_io;
pub mod dsp;
pub mod generators;
// pub mod pipeline;
pub mod resample;
pub mod testing;
pub mod traits;
pub mod types;

use anyhow::Result;
use crossbeam_channel::{Receiver, Sender, bounded};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::audio_io::AudioPipeline;
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

pub fn record(builder: &mut AudioPipeline, path: &str) -> Result<audio_io::Recorder> {
    println!("🎙️  Record command received.");
    builder.start_input()?;
    let rec = builder.spawn_recorder(path);
    Ok(rec)
}

pub fn tune(
    mode: analysis::TuningSystem,
    builder: &mut AudioPipeline,
) -> Result<Receiver<(Note, i16)>> {
    println!("🎵  Tune command received.");
    let (s, r) = bounded(1);
    // TODO: Implement tuning (pitch analysis, reference tone, etc.)
    // Should have multiple modes to select from:
    // Regular notes, chord, orchestra, ...?
    // Different tuning systems: equal temperament, just intonation, ...?
    builder.start_input()?;
    thread::spawn(move || {
        loop {
            thread::sleep(Duration::from_secs_f32(0.5));
        }
    });
    Ok(r)
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
