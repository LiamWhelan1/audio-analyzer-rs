use super::{MAX_MIDI_VELOCITY, SynthNote, TWO_PI, Waveform};
use crate::audio_io::output::MusicalTransport;
use crate::traits::AudioSource;
use rtrb::{Consumer, Producer, RingBuffer};
use std::{f32::consts::PI, sync::Arc};

// ============================================================================
//  MODULE: SYNTHESIS (Normalized)
// ============================================================================

pub enum SynthCommand {
    LoadSequence(Vec<SynthNote>),
    NoteOn {
        freq: f32,
        velocity: f32,
        waveform: Waveform,
    },
    NoteOff {
        freq: f32,
    },
    SetVolume(f32),
    Stop,
    Pause,
    Resume,
}

enum EnvState {
    Attack,
    Decay,
    Sustain,
    Release,
    Finished,
}

struct Voice {
    freq: f32,
    phase: f32,
    velocity: f32,
    remaining_beats: Option<f32>,
    envelope: f32,
    state: EnvState,
    waveform: Waveform,
}

impl Voice {
    /// Creates a voice.
    /// IMPORTANT: Takes `velocity` as 0.0-1.0 float.
    /// If you pass 80.0 here, it will blow out the speakers.
    fn new(freq: f32, velocity: f32, duration_beats: Option<f32>, waveform: Waveform) -> Self {
        Self {
            freq,
            velocity,
            waveform,
            phase: 0.0,
            remaining_beats: duration_beats,
            envelope: 0.0,
            state: EnvState::Attack,
        }
    }

    fn process(&mut self, sample_rate: f32, beats_delta: f32) -> f32 {
        if matches!(self.state, EnvState::Finished) {
            return 0.0;
        }

        let phase_inc = (self.freq * TWO_PI) / sample_rate;
        self.phase = (self.phase + phase_inc) % TWO_PI;

        let raw = match self.waveform {
            Waveform::Sine => self.phase.sin(),
            Waveform::Square => {
                if self.phase < PI {
                    1.0
                } else {
                    -1.0
                }
            }
            Waveform::Sawtooth => (self.phase / PI) - 1.0,
            Waveform::Triangle => {
                let t = self.phase / TWO_PI;
                4.0 * (t - 0.5).abs() - 1.0
            }
        };

        match self.state {
            EnvState::Attack => {
                self.envelope += 0.05;
                if self.envelope >= 1.0 {
                    self.state = EnvState::Decay;
                }
            }
            EnvState::Decay => self.state = EnvState::Sustain,
            EnvState::Sustain => {
                if let Some(rem) = &mut self.remaining_beats {
                    *rem -= beats_delta;
                    if *rem <= 0.0 {
                        self.state = EnvState::Release;
                    }
                }
            }
            EnvState::Release => {
                self.envelope -= 0.005;
                if self.envelope <= 0.0 {
                    self.state = EnvState::Finished;
                }
            }
            _ => {}
        }
        raw * self.envelope * self.velocity
    }
}

pub struct Synthesizer {
    sample_rate: f32,
    transport: Arc<MusicalTransport>,
    volume: f32,
    voices: Vec<Voice>,
    sequence: Vec<SynthNote>,
    sequence_cursor: usize,
    pending_start: bool,
    is_playing_seq: bool,
    start_beat_offset: f64,
    command_rx: Consumer<SynthCommand>,
    finished: bool,
}

impl Synthesizer {
    pub fn new(
        sample_rate: u32,
        transport: Arc<MusicalTransport>,
    ) -> (Self, Producer<SynthCommand>) {
        let (prod, cons) = RingBuffer::new(128);
        (
            Self {
                sample_rate: sample_rate as f32,
                transport,
                volume: 0.5,
                voices: Vec::with_capacity(32),
                sequence: Vec::new(),
                sequence_cursor: 0,
                pending_start: false,
                is_playing_seq: false,
                start_beat_offset: 0.0,
                command_rx: cons,
                finished: false,
            },
            prod,
        )
    }

    fn handle_commands(&mut self) {
        while let Ok(cmd) = self.command_rx.pop() {
            match cmd {
                SynthCommand::LoadSequence(mut notes) => {
                    notes.sort_by(|a, b| {
                        a.onset_beats
                            .partial_cmp(&b.onset_beats)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    self.sequence = notes;
                    self.sequence_cursor = 0;
                    self.is_playing_seq = false;
                    self.pending_start = true;
                }
                SynthCommand::SetVolume(v) => self.volume = v.clamp(0.0, 2.0),
                SynthCommand::NoteOn {
                    freq,
                    velocity,
                    waveform,
                } => {
                    // FIX: Normalize manual note velocity (0-127 -> 0.0-1.0)
                    let norm_vel = velocity / MAX_MIDI_VELOCITY;
                    self.voices.push(Voice::new(freq, norm_vel, None, waveform));
                }
                SynthCommand::NoteOff { freq } => {
                    for v in &mut self.voices {
                        if (v.freq - freq).abs() < 0.1 {
                            v.state = EnvState::Release;
                        }
                    }
                }
                SynthCommand::Stop => self.finished = true,
                SynthCommand::Pause => self.is_playing_seq = false,
                SynthCommand::Resume => self.is_playing_seq = true,
            }
        }
    }
}

impl AudioSource for Synthesizer {
    fn is_finished(&self) -> bool {
        self.finished
    }

    fn process(&mut self, buffer: &mut [f32], channels: usize) {
        self.handle_commands();
        if self.finished {
            return;
        }

        let bpm = self.transport.get_bpm();
        let beats_per_sample = (bpm / 60.0) / self.sample_rate;
        let mut transport_beat = self.transport.get_accumulated_beats() as f32;

        for frame_idx in (0..buffer.len()).step_by(channels) {
            // 1. Quantization
            if self.pending_start {
                let next_beat_boundary = transport_beat.ceil() - 0.05 * bpm / 50.0;
                if next_beat_boundary - transport_beat <= beats_per_sample {
                    self.start_beat_offset = next_beat_boundary as f64;
                    self.pending_start = false;
                    self.is_playing_seq = true;
                    self.sequence_cursor = 0;
                }
            }

            // 2. Sequencer
            if self.is_playing_seq {
                let seq_time_f32 = (transport_beat as f64 - self.start_beat_offset) as f32;
                while self.sequence_cursor < self.sequence.len() {
                    let note = &self.sequence[self.sequence_cursor];
                    if note.onset_beats > seq_time_f32 {
                        break;
                    }
                    if note.onset_beats >= (seq_time_f32 - 1.0) {
                        // FIX: Normalize sequencer note velocity (0-127 -> 0.0-1.0)
                        let norm_vel = note.velocity / MAX_MIDI_VELOCITY;
                        self.voices.push(Voice::new(
                            note.freq,
                            norm_vel,
                            Some(note.duration_beats),
                            note.waveform,
                        ));
                    }
                    self.sequence_cursor += 1;
                }
            }

            // 3. Voice Processing
            let mut sum = 0.0;
            let mut active_count = 0.0;

            for v in &mut self.voices {
                sum += v.process(self.sample_rate, beats_per_sample);
                if !matches!(v.state, EnvState::Finished) {
                    active_count += 1.0;
                }
            }
            self.voices
                .retain(|v| !matches!(v.state, EnvState::Finished));

            // --- POLYPHONY NORMALIZATION ---
            // Even with normalized velocity (0-1.0), 3 voices = 3.0 amplitude.
            // We scale by 1 / active_count so the total output never exceeds 1.0
            // regardless of how many notes are played.
            let norm_factor = if active_count > 1.0 {
                1.0 / active_count
            } else {
                1.0
            };

            let final_val = sum * norm_factor * self.volume;

            for ch in 0..channels {
                buffer[frame_idx + ch] += final_val;
            }

            transport_beat += beats_per_sample;
        }
    }
}
