use crossbeam_channel::Receiver;
use std::io::{self, Write};
use std::thread;
use std::time::Duration;

pub enum BeatStrength {
    Strong,
    Medium,
    Weak,
    None,
    Subdivision(usize),
}

pub struct Metronome {
    pub bpm: f32,
    pub beats: usize,
    pub division: usize,
    pub subdivision: Vec<usize>,
    pub pattern: Vec<BeatStrength>,
}

impl Metronome {
    pub fn new(
        bpm: f32,
        beats: usize,
        division: usize,
        subdivision: Option<Vec<usize>>,
        pattern: Vec<BeatStrength>,
    ) -> Self {
        Metronome {
            bpm,
            beats,
            division,
            subdivision: subdivision.unwrap_or(vec![1]),
            pattern,
        }
    }

    pub fn play(&self, r: Receiver<bool>) {
        let interval = 60.0 / self.bpm; // milliseconds per beat
        thread::spawn(move || {
            while let Err(_) = r.try_recv() {
                print!("🟢 Beat\n");
                io::stdout().flush().unwrap();
                thread::sleep(Duration::from_secs_f32(interval));
            }
        });
    }
}
