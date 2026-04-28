use super::{MIN_ENVELOPE, TWO_PI};
use crate::{audio_io::timing::MusicalTransport, traits::AudioSource};
// use log;
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
    /// Sample offset within the current buffer where this tick should start.
    /// Used for sample-accurate placement when a beat crossing happens
    /// partway through a buffer.
    pending_delay_samples: i64,
}

impl TickGenerator {
    /// Produce the next sample and advance the internal envelope.
    fn process(&mut self, sample_rate: f32) -> f32 {
        // If this tick was spawned with a delay, stay silent until the
        // delay is exhausted.
        if self.pending_delay_samples > 0 {
            self.pending_delay_samples -= 1;
            return 0.0;
        }

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

/// Stateful metronome audio source that emits ticks according to a pattern
/// and BPM, using the central `MusicalTransport` for sample-accurate beat
/// detection.
pub struct Metronome {
    sample_rate: f32,
    transport: Arc<MusicalTransport>,
    volume: f32,
    muted: bool,

    pattern: Vec<BeatStrength>,
    beat_polyrhythms: Vec<Vec<usize>>,

    /// Tracks which pattern index the *next* beat crossing should trigger.
    current_beat_index: usize,

    /// Subdivision state: (division_amount, phase_counter) for the current beat.
    active_subdivision_counters: Vec<(usize, u64)>,
    samples_per_beat: u64,

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
        let bpm = bpm.unwrap_or_else(|| transport.get_bpm());

        let initial_pattern = pattern.unwrap_or_else(|| {
            vec![
                BeatStrength::Strong,
                BeatStrength::Weak,
                BeatStrength::Weak,
                BeatStrength::Weak,
            ]
        });
        let patt_len = initial_pattern.len();
        let beat_polyrhythms = if let Some(p) = polys.as_mut() {
            let poly_len = p.len();
            if poly_len < patt_len {
                p.append(&mut vec![Vec::new(); patt_len - poly_len]);
            } else if poly_len > patt_len {
                while p.len() > patt_len {
                    p.pop();
                }
            }
            p.to_owned()
        } else {
            vec![Vec::new(); initial_pattern.len()]
        };

        let samples_per_beat = ((sample_rate as f32 * 60.0) / bpm) as u64;

        // Sync the starting beat index to the transport's current position
        // so we pick up mid-bar if the transport is already running.
        let beats = transport.get_accumulated_beats();
        let current_beat_index = if patt_len > 0 {
            (beats.max(0.0) as usize) % patt_len
        } else {
            0
        };

        let mut met = Self {
            sample_rate: sample_rate as f32,
            transport,
            volume: 0.8,
            muted: false,
            pattern: initial_pattern,
            beat_polyrhythms,
            samples_per_beat,
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
        self.transport.seek_to_beat(0.0001);
        self.active_subdivision_counters.clear();
        self.active_ticks.clear();

        // FORCE the very first tick to spawn instantly and guarantee Beat 1 audio
        if !self.pattern.is_empty() {
            let strength = self.pattern[0].clone();
            if strength != BeatStrength::None {
                // At reset time output_frames is at or near 0; use current value
                // as the click frame (echo window is opened from frame 0 onward).
                self.transport.notify_tick_at_frame(self.transport.get_output_frames());
                self.spawn_tick(&strength, 0);
                self.current_beat_index = 0;
                self.load_active_subdivisions();
            }
            // Advance to beat 1 for the normal transport crossing logic
            self.current_beat_index = 1 % self.pattern.len();
        }
    }

    /// Update BPM and recompute samples-per-beat.
    fn update_bpm(&mut self, new_bpm: f32) {
        let bpm = new_bpm.max(1.0);
        self.samples_per_beat = ((self.sample_rate * 60.0) / bpm) as u64;
        self.transport.set_bpm(bpm);
    }

    /// Spawn one or more tick generators for the given `BeatStrength`,
    /// with an optional sample-offset delay for sub-buffer accuracy.
    ///
    /// Callers must call `transport.notify_tick_at_frame(click_frame)` before
    /// this function so the suppression window is anchored to the correct frame.
    fn spawn_tick(&mut self, strength: &BeatStrength, delay_samples: i64) {
        // log::trace!("Tick at {:.4}", self.transport.get_accumulated_beats());
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

        let decay_samples_count = self.sample_rate * (decay_ms / 1000.0);
        let decay_rate = (MIN_ENVELOPE).powf(1.0 / decay_samples_count);

        self.active_ticks.push(TickGenerator {
            phase: 0.0,
            freq,
            volume: vol_mod,
            envelope: 1.0,
            decay_rate,
            is_noise: false,
            noise_seed: 0,
            pending_delay_samples: delay_samples,
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
                pending_delay_samples: delay_samples,
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
    ///
    /// Beat crossings are detected via `transport.did_cross_beat()` which
    /// returns the exact sample offset within the buffer, giving us
    /// sample-accurate tick placement rather than per-buffer jitter.
    fn process(&mut self, buffer: &mut [f32], channels: usize) {
        self.handle_commands();
        if self.finished {
            return;
        }

        let total_frames = buffer.len() / channels;

        // Output frames were already incremented by tick_output before process()
        // is called.  The start of this buffer is therefore at:
        //   output_frames_now − total_frames
        // Any click at sample offset S from the buffer start is at absolute frame:
        //   buffer_start_frame + S
        let buffer_start_frame = self.transport.get_output_frames() - total_frames as i64;

        // ── Detect beat crossing for this buffer ───────────────────────
        // did_cross_beat looks at the accumulated beats position (which
        // was already advanced by tick_output in the audio callback) and
        // tells us if a beat boundary fell inside this buffer, and where.
        if let Some(crossing) = self.transport.did_cross_beat(total_frames as i64) {
            if !self.pattern.is_empty() {
                let patt_len = self.pattern.len() as i64;
                // Derive pattern index from the absolute beat number so that
                // non-integer-multiple count-offs land on the correct beat
                // strength when the performance begins.
                let beat_idx = ((crossing.beat_number % patt_len + patt_len) % patt_len) as usize;
                let strength = self.pattern[beat_idx].clone();

                if strength != BeatStrength::None {
                    // Record the actual click frame so the onset detector's
                    // suppression window opens at the right time.
                    let click_frame = buffer_start_frame + crossing.sample_offset_in_buffer;
                    self.transport.notify_tick_at_frame(click_frame);
                    self.spawn_tick(&strength, crossing.sample_offset_in_buffer);
                    self.current_beat_index = beat_idx;
                    self.load_active_subdivisions();
                } else {
                    // FIX: Muted beats must immediately kill active polyrhythm counters!
                    self.active_subdivision_counters.clear();
                }

                self.current_beat_index = (beat_idx + 1) % self.pattern.len();
            }
        }

        // ── Per-sample rendering ───────────────────────────────────────
        for frame_idx in (0..buffer.len()).step_by(channels) {
            let sample_in_buffer = (frame_idx / channels) as i64;

            // Subdivision counters still run per-sample since they are
            // sub-divisions of the beat, not aligned to transport beats.
            for i in 0..self.active_subdivision_counters.len() {
                let (div, ref mut counter) = self.active_subdivision_counters[i];
                let samples_per_sub = self.samples_per_beat / (div as u64);
                *counter += 1;
                if *counter >= samples_per_sub {
                    *counter -= samples_per_sub;
                    // Avoid firing a subdivision tick right on top of the
                    // main beat tick (first ~10ms of the beat).
                    let guard_samples = (self.sample_rate / 100.0) as u64;
                    let beat_phase_samples = {
                        let phase = self.transport.get_accumulated_beats().fract();
                        (phase * self.samples_per_beat as f64) as u64
                    };
                    if beat_phase_samples > guard_samples {
                        self.transport.notify_tick_at_frame(buffer_start_frame + sample_in_buffer);
                        self.spawn_tick(&BeatStrength::Subdivision(div), 0);
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
