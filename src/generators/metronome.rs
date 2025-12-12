use super::{MIN_ENVELOPE, TWO_PI};
use crate::{audio_io::output::MusicalTransport, traits::AudioSource};
use log;
use rtrb::{Consumer, Producer, RingBuffer};
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq)]
/// Strength or role of a beat within the metronome pattern.
pub enum BeatStrength {
    Strong,
    Medium,
    Weak,
    Subdivision(usize),
    None,
}

/// Commands used to control a running `Metronome` instance.
pub enum MetronomeCommand {
    SetBpm(f32),
    SetVolume(f32),
    SetPattern(Vec<BeatStrength>),
    /// (List of subdivisions, beat index)
    SetPolyrhythm((Vec<usize>, usize)),
    SetMuted(bool),
    Stop,
}

/// Internal generator for short click/noise ticks.
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
    /// Produce the next sample and advance the internal envelope.
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

/// Stateful metronome audio source that emits ticks according to a pattern and BPM.
pub struct Metronome {
    sample_rate: f32,
    transport: Arc<MusicalTransport>,
    volume: f32,
    muted: bool,

    pattern: Vec<BeatStrength>,
    beat_polyrhythms: Vec<Vec<usize>>,

    current_beat_index: usize,
    samples_per_beat: u64,
    main_beat_counter: u64,

    /// (division_amount, current_counter_value) for the current beat
    active_subdivision_counters: Vec<(usize, u64)>,

    active_ticks: Vec<TickGenerator>,
    command_rx: Consumer<MetronomeCommand>,
    finished: bool,
}

impl Metronome {
    /// Create a new `Metronome` and return it with a command `Producer`.
    pub fn new(
        sample_rate: u32,
        bpm: Option<f32>,
        pattern: Option<Vec<BeatStrength>>,
        mut polys: Option<Vec<Vec<usize>>>,
        restart: bool,
        transport: Arc<MusicalTransport>,
    ) -> (Self, Producer<MetronomeCommand>) {
        let (prod, cons) = RingBuffer::new(128);
        let bpm = if let Some(b) = bpm {
            b
        } else {
            transport.get_bpm()
        };

        let initial_pattern = if let Some(p) = pattern {
            p
        } else {
            vec![
                BeatStrength::Strong,
                BeatStrength::Weak,
                BeatStrength::Weak,
                BeatStrength::Weak,
            ]
        };
        let patt_len = initial_pattern.len();
        let beat_polyrhythms = if let Some(p) = polys.as_mut() {
            let poly_len = p.len();
            if poly_len < patt_len {
                p.append(&mut vec![Vec::new(); patt_len - poly_len]);
            } else if poly_len > patt_len {
                while poly_len > patt_len {
                    p.pop();
                }
            }
            p.to_owned()
        } else {
            vec![Vec::new(); initial_pattern.len()]
        };
        let samples_per_beat = ((sample_rate as f32 * 60.0) / bpm) as u64;
        let beats = transport.get_accumulated_beats();
        let current_beat_index = (beats % patt_len as f64).floor() as usize;
        let main_beat_counter = ((beats % 1.0) * samples_per_beat as f64) as u64;

        let mut met = Self {
            sample_rate: sample_rate as f32,
            transport,
            volume: 0.8,
            muted: false,
            pattern: initial_pattern,
            beat_polyrhythms,
            samples_per_beat,
            main_beat_counter,
            current_beat_index,
            active_subdivision_counters: Vec::with_capacity(patt_len),
            active_ticks: Vec::with_capacity(16),
            command_rx: cons,
            finished: false,
        };
        met.update_bpm(bpm);
        if restart {
            met.reset_beat();
        }
        (met, prod)
    }

    /// Reset transport position and internal counters to the start.
    pub fn reset_beat(&mut self) {
        self.transport.reset_to_beats(0.0);
        self.active_subdivision_counters.clear();
        self.active_ticks.clear();
        self.current_beat_index = 0;
        self.main_beat_counter = 0;
    }

    /// Update BPM and recompute samples-per-beat.
    fn update_bpm(&mut self, new_bpm: f32) {
        let bpm = new_bpm.max(1.0);
        self.samples_per_beat = ((self.sample_rate * 60.0) / bpm) as u64;
        self.transport.set_bpm(bpm);
    }

    /// Spawn one or more tick generators for the given `BeatStrength`.
    fn spawn_tick(&mut self, strength: &BeatStrength) {
        log::trace!("Tick at {}", self.transport.get_accumulated_beats());
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

    /// Process any pending metronome commands from the command channel.
    fn handle_commands(&mut self) {
        while let Ok(cmd) = self.command_rx.pop() {
            match cmd {
                MetronomeCommand::SetBpm(b) => self.update_bpm(b),
                MetronomeCommand::SetVolume(v) => self.volume = v.clamp(0.0, 2.0),
                MetronomeCommand::SetPattern(p) => {
                    self.pattern = p;
                    self.beat_polyrhythms.resize(self.pattern.len(), Vec::new());
                    if self.current_beat_index >= self.pattern.len() {
                        self.current_beat_index = 0;
                    }
                }
                MetronomeCommand::SetPolyrhythm((divs, index)) => {
                    if index < self.beat_polyrhythms.len() {
                        self.beat_polyrhythms[index] = divs;
                    }
                }
                MetronomeCommand::SetMuted(m) => self.muted = m,
                MetronomeCommand::Stop => self.finished = true,
            }
        }
    }

    /// Load subdivisions for the current beat into the active counters.
    fn load_active_subdivisions(&mut self) {
        self.active_subdivision_counters.clear();
        if self.current_beat_index < self.beat_polyrhythms.len() {
            let divs = &self.beat_polyrhythms[self.current_beat_index];
            for &div in divs {
                if div > 0 {
                    self.active_subdivision_counters.push((div, 0));
                }
            }
        }
    }
}

impl AudioSource for Metronome {
    /// Whether the metronome has been stopped.
    fn is_finished(&self) -> bool {
        self.finished
    }

    /// Fill `buffer` with generated tick samples, advancing internal state.
    fn process(&mut self, buffer: &mut [f32], channels: usize) {
        self.handle_commands();
        if self.finished {
            return;
        }

        for frame_idx in (0..buffer.len()).step_by(channels) {
            self.main_beat_counter += 1;

            if self.main_beat_counter >= self.samples_per_beat {
                self.main_beat_counter -= self.samples_per_beat;

                if !self.pattern.is_empty() {
                    let s = self.pattern[self.current_beat_index].clone();
                    self.spawn_tick(&s);
                    self.load_active_subdivisions();
                    self.current_beat_index = (self.current_beat_index + 1) % self.pattern.len();
                }
            }

            for i in 0..self.active_subdivision_counters.len() {
                let (div, ref mut counter) = self.active_subdivision_counters[i];

                let samples_per_sub = self.samples_per_beat / (div as u64);
                *counter += 1;

                if *counter >= samples_per_sub {
                    *counter -= samples_per_sub;

                    if self.main_beat_counter > self.sample_rate as u64 / 100 {
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
