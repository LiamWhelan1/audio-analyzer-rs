use std::sync::Arc;

use rtrb::{Consumer, Producer, RingBuffer};

use crate::{audio_io::output::MusicalTransport, traits::AudioSource};

use super::{MIN_ENVELOPE, TWO_PI};
// ============================================================================
//  MODULE: METRONOME
// ============================================================================

#[derive(Clone, Debug, PartialEq)]
pub enum BeatStrength {
    Strong,
    Medium,
    Weak,
    Subdivision(usize),
    None,
}

pub enum MetronomeCommand {
    SetBpm(f32),
    SetVolume(f32),
    SetPattern(Vec<BeatStrength>),
    SetSubdivisions(Vec<usize>),
    SetMuted(bool),
    Stop,
}

struct TickGenerator {
    phase: f32,
    freq: f32,
    volume: f32,
    envelope: f32,
    decay_rate: f32,
    is_noise: bool,
    noise_seed: u32,
}

impl TickGenerator {
    fn process(&mut self, sample_rate: f32) -> f32 {
        let output;

        if self.is_noise {
            self.noise_seed =
                (self.noise_seed.wrapping_mul(1103515245).wrapping_add(12345)) & 0x7FFFFFFF;
            let noise_float = (self.noise_seed as f32 / 2147483648.0) - 1.0;
            output = noise_float * self.volume * self.envelope;
        } else {
            let phase_inc = (self.freq * TWO_PI) / sample_rate;
            let sample = (self.phase * phase_inc).sin();
            self.phase += 1.0;
            output = sample * self.volume * self.envelope;
        }

        self.envelope *= self.decay_rate;
        output
    }
}

pub struct Metronome {
    sample_rate: f32,
    transport: Arc<MusicalTransport>,
    volume: f32,
    muted: bool,

    pattern: Vec<BeatStrength>,
    subdivisions: Vec<usize>,
    current_beat_index: usize,

    samples_per_beat: f32,
    main_beat_counter: f32,
    subdivision_counters: Vec<f32>,

    active_ticks: Vec<TickGenerator>,
    command_rx: Consumer<MetronomeCommand>,
    finished: bool,
}

impl Metronome {
    pub fn new(
        sample_rate: u32,
        initial_bpm: f32,
        transport: Arc<MusicalTransport>,
    ) -> (Self, Producer<MetronomeCommand>) {
        let (prod, cons) = RingBuffer::new(128);
        transport.set_bpm(initial_bpm);
        transport.reset_beats();

        let mut met = Self {
            sample_rate: sample_rate as f32,
            transport,
            volume: 0.8,
            muted: false,
            pattern: vec![
                BeatStrength::Strong,
                BeatStrength::Weak,
                BeatStrength::Weak,
                BeatStrength::Weak,
            ],
            subdivisions: vec![],
            samples_per_beat: 0.0,
            main_beat_counter: 0.0,
            current_beat_index: 0,
            subdivision_counters: Vec::new(),
            active_ticks: Vec::with_capacity(16),
            command_rx: cons,
            finished: false,
        };
        met.update_bpm(initial_bpm);
        (met, prod)
    }

    fn update_bpm(&mut self, new_bpm: f32) {
        let bpm = new_bpm.max(1.0);
        self.samples_per_beat = (self.sample_rate * 60.0) / bpm;
        self.transport.set_bpm(bpm);
        self.transport.reset_beats();
    }

    fn spawn_tick(&mut self, strength: &BeatStrength) {
        if self.muted {
            return;
        }

        let (freq, vol_mod, decay_ms) = match strength {
            BeatStrength::Strong => (2500.0, 1.0, 100.0),
            BeatStrength::Medium => (2000.0, 0.7, 100.0),
            BeatStrength::Weak => (1500.0, 0.5, 100.0),
            BeatStrength::Subdivision(n) => (2000.0 / (*n as f32).max(1.0), 0.4, 80.0),
            BeatStrength::None => return,
        };

        let decay_samples = self.sample_rate * (decay_ms / 1000.0);
        let decay_rate = (MIN_ENVELOPE).powf(1.0 / decay_samples);

        self.active_ticks.push(TickGenerator {
            phase: 0.0,
            freq,
            volume: vol_mod,
            envelope: 1.0,
            decay_rate,
            is_noise: false,
            noise_seed: 0,
        });

        if matches!(strength, BeatStrength::Strong | BeatStrength::Medium) {
            let click_decay_rate = (MIN_ENVELOPE).powf(1.0 / (self.sample_rate * 0.015));
            self.active_ticks.push(TickGenerator {
                phase: 0.0,
                freq: 0.0,
                volume: vol_mod * 0.5,
                envelope: 1.0,
                decay_rate: click_decay_rate,
                is_noise: true,
                noise_seed: 12345,
            });
        }
    }

    fn handle_commands(&mut self) {
        while let Ok(cmd) = self.command_rx.pop() {
            match cmd {
                MetronomeCommand::SetBpm(b) => self.update_bpm(b),
                MetronomeCommand::SetVolume(v) => self.volume = v.clamp(0.0, 2.0),
                MetronomeCommand::SetPattern(p) => self.pattern = p,
                MetronomeCommand::SetSubdivisions(s) => {
                    self.subdivisions = s;
                    self.subdivision_counters =
                        vec![self.main_beat_counter; self.subdivisions.len()];
                }
                MetronomeCommand::SetMuted(m) => self.muted = m,
                MetronomeCommand::Stop => self.finished = true,
            }
        }
    }
}

impl AudioSource for Metronome {
    fn is_finished(&self) -> bool {
        self.finished
    }

    fn process(&mut self, buffer: &mut [f32], channels: usize) {
        self.handle_commands();
        if self.finished {
            return;
        }

        for frame_idx in (0..buffer.len()).step_by(channels) {
            self.main_beat_counter += 1.0;
            if self.main_beat_counter >= self.samples_per_beat {
                self.main_beat_counter -= self.samples_per_beat;
                if !self.pattern.is_empty() {
                    let s = self.pattern[self.current_beat_index].clone();
                    self.spawn_tick(&s);
                    self.current_beat_index = (self.current_beat_index + 1) % self.pattern.len();
                }
            }

            let subdivisions = self.subdivisions.clone();
            for (i, &div) in subdivisions.iter().enumerate() {
                if div == 0 {
                    continue;
                }
                let samples_per_sub = self.samples_per_beat / (div as f32);
                self.subdivision_counters[i] += 1.0;
                if self.subdivision_counters[i] >= samples_per_sub {
                    self.subdivision_counters[i] -= samples_per_sub;
                    if self.main_beat_counter > 100.0 {
                        self.spawn_tick(&BeatStrength::Subdivision(div));
                    }
                }
            }

            let mut sample = 0.0;
            self.active_ticks.retain(|t| t.envelope > MIN_ENVELOPE);
            for tick in &mut self.active_ticks {
                sample += tick.process(self.sample_rate);
            }

            let final_sample = sample * self.volume;
            for ch in 0..channels {
                buffer[frame_idx + ch] += final_sample;
            }
        }
    }
}
