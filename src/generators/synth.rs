use super::{MAX_MIDI_VELOCITY, TWO_PI};
use crate::generators::{EnvState, Instrument, InstrumentParams, Measure, load_midi_file};
use crate::traits::AudioSource;
use crate::{audio_io::output::MusicalTransport, generators::metronome::MetronomeCommand};
use rtrb::{Consumer, Producer, RingBuffer};
use std::path::Path;
use std::{f32::consts::PI, sync::Arc};

/// Commands accepted by the `Synthesizer`.
pub enum SynthCommand {
    LoadFile(String, Instrument),
    Play {
        start_measure_idx: usize,
    },
    Pause,
    Resume,
    Stop,
    Clear,
    LinkMetronome(Producer<MetronomeCommand>),
    NoteOn {
        freq: f32,
        velocity: f32,
        instrument: Instrument,
    },
    NoteOff {
        freq: f32,
    },
    SetVolume(f32),
    SetMuted(bool),
}

/// Single-voice state for synthesis (oscillator + ADSR).
struct Voice {
    freq: f32,
    phase: f32,
    velocity: f32,
    /// If Some, auto-release after X beats (sequencer); None means manual sustain.
    remaining_beats: Option<f32>,
    envelope: f32,
    state: EnvState,
    params: InstrumentParams,
    instrument: Instrument,
}

impl Voice {
    /// Return instrument-specific synthesis parameters.
    fn get_instrument_params(instrument: &Instrument) -> InstrumentParams {
        match instrument {
            Instrument::Piano => InstrumentParams {
                attack_sec: 0.005,
                decay_sec: 0.15,
                sustain_level: 0.6,
                release_sec: 0.7,
                timbre_mix: 0.8,
            },
            Instrument::Violin => InstrumentParams {
                attack_sec: 0.3,
                decay_sec: 0.1,
                sustain_level: 0.9,
                release_sec: 0.5,
                timbre_mix: 0.4,
            },
        }
    }

    /// Create a new voice instance.
    fn new(freq: f32, velocity: f32, duration_beats: Option<f32>, instrument: Instrument) -> Self {
        let params = Self::get_instrument_params(&instrument);
        Self {
            freq,
            velocity,
            instrument,
            params,
            phase: 0.0,
            remaining_beats: duration_beats,
            envelope: 0.0,
            state: EnvState::Attack,
        }
    }

    /// Synthesize one sample for this voice and advance its envelope.
    fn process(&mut self, sample_rate: f32, beats_delta: f32) -> f32 {
        if matches!(self.state, EnvState::Finished) {
            return 0.0;
        }

        let sample_rate_inv = 1.0 / sample_rate;
        let phase_inc = (self.freq * TWO_PI) * sample_rate_inv;
        self.phase = (self.phase + phase_inc) % TWO_PI;

        let fund = self.phase.sin();
        let raw_sig;

        match self.instrument {
            Instrument::Piano => {
                let bright_comp = ((self.phase * 2.0).sin() + (self.phase / PI) - 1.0) * 0.5;
                raw_sig =
                    fund * (1.0 - self.params.timbre_mix) + bright_comp * self.params.timbre_mix;
            }
            Instrument::Violin => {
                let t = self.phase / TWO_PI;
                let tri = 4.0 * (t - 0.5).abs() - 1.0;
                raw_sig = fund * (1.0 - self.params.timbre_mix) + tri * self.params.timbre_mix;
            }
        }

        match self.state {
            EnvState::Attack => {
                let attack_rate = sample_rate_inv / self.params.attack_sec.max(0.001);
                self.envelope += attack_rate;
                if self.envelope >= 1.0 {
                    self.envelope = 1.0;
                    self.state = EnvState::Decay;
                }
            }
            EnvState::Decay => {
                let decay_rate = (1.0 - self.params.sustain_level)
                    * (sample_rate_inv / self.params.decay_sec.max(0.001));
                self.envelope -= decay_rate;
                if self.envelope <= self.params.sustain_level {
                    self.envelope = self.params.sustain_level;
                    self.state = EnvState::Sustain;
                }
            }
            EnvState::Sustain => {
                if let Some(rem) = &mut self.remaining_beats {
                    *rem -= beats_delta;
                    if *rem <= 0.0 {
                        self.state = EnvState::Release;
                    }
                }
            }
            EnvState::Release => {
                let release_rate = self.params.sustain_level
                    * (sample_rate_inv / self.params.release_sec.max(0.001));
                self.envelope -= release_rate;
                if self.envelope <= 0.0 {
                    self.envelope = 0.0;
                    self.state = EnvState::Finished;
                }
            }
            _ => {}
        }

        raw_sig * self.envelope * self.velocity * 3.0
    }
}

/// Polyphonic synthesizer and sequencer.
pub struct Synthesizer {
    sample_rate: f32,
    transport: Arc<MusicalTransport>,

    volume: f32,
    voices: Vec<Voice>,
    muted: bool,

    measures: Vec<Measure>,

    is_playing_seq: bool,
    current_measure_index: usize,
    playback_cursor_global_beats: f64,
    start_measure_global_offset: f64,
    count_in_duration: f64,

    metro_cmd_tx: Option<Producer<MetronomeCommand>>,
    command_rx: Consumer<SynthCommand>,
    finished: bool,
}

impl Synthesizer {
    /// Create a new `Synthesizer` and its command `Producer`.
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
                muted: false,
                measures: Vec::new(),
                is_playing_seq: false,
                current_measure_index: 0,
                playback_cursor_global_beats: 0.0,
                start_measure_global_offset: 0.0,
                count_in_duration: 0.0,
                metro_cmd_tx: None,
                command_rx: cons,
                finished: false,
            },
            prod,
        )
    }

    /// Sync the linked metronome to the given measure index.
    fn sync_metronome(&mut self, measure_idx: usize) {
        if let Some(tx) = &mut self.metro_cmd_tx {
            if measure_idx < self.measures.len() {
                let m = &self.measures[measure_idx];
                let _ = tx.push(MetronomeCommand::SetBpm(m.bpm));
                let _ = tx.push(MetronomeCommand::SetPattern(m.get_pattern()));
                self.transport.set_bpm(m.bpm);
            }
        }
    }

    /// Poll and handle synthesizer commands.
    fn handle_commands(&mut self) {
        while let Ok(cmd) = self.command_rx.pop() {
            match cmd {
                SynthCommand::LinkMetronome(tx) => self.metro_cmd_tx = Some(tx),
                SynthCommand::LoadFile(path_str, instrument) => {
                    let path = Path::new(&path_str);
                    if let Ok(m) = load_midi_file(path, instrument) {
                        self.measures = m;
                        self.is_playing_seq = false;
                        self.voices.clear();
                    }
                }
                SynthCommand::Clear => {
                    self.measures.clear();
                    self.voices.clear();
                    self.is_playing_seq = false;
                }
                SynthCommand::SetVolume(v) => self.volume = v.clamp(0.0, 2.0),
                SynthCommand::NoteOn {
                    freq,
                    velocity,
                    instrument,
                } => {
                    let norm_vel = velocity / MAX_MIDI_VELOCITY;
                    self.voices
                        .push(Voice::new(freq, norm_vel, None, instrument));
                }
                SynthCommand::NoteOff { freq } => {
                    for v in &mut self.voices {
                        if (v.freq - freq).abs() < 0.1 {
                            v.state = EnvState::Release;
                        }
                    }
                }
                SynthCommand::Play { start_measure_idx } => {
                    if start_measure_idx < self.measures.len() {
                        let start_measure = &self.measures[start_measure_idx];
                        self.start_measure_global_offset = start_measure.global_start_beat;
                        self.count_in_duration = start_measure.duration_beats();
                        self.playback_cursor_global_beats = -self.count_in_duration;
                        self.transport.reset_to_beats(-self.count_in_duration);
                        self.sync_metronome(start_measure_idx);

                        self.current_measure_index = start_measure_idx;
                        self.is_playing_seq = true;
                    }
                }
                SynthCommand::Pause => self.is_playing_seq = false,
                SynthCommand::Resume => self.is_playing_seq = true,
                SynthCommand::Stop => {
                    self.is_playing_seq = false;
                    self.voices.clear();
                    self.playback_cursor_global_beats = 0.0;
                    self.transport.reset_to_beats(0.0);
                }
                SynthCommand::SetMuted(m) => self.muted = m,
            }
        }
    }
}

impl AudioSource for Synthesizer {
    fn is_finished(&self) -> bool {
        self.finished
    }

    /// Process synthesizer audio: sequencer, voices, and mixing.
    fn process(&mut self, buffer: &mut [f32], channels: usize) {
        self.handle_commands();
        if self.finished {
            return;
        }

        let bpm = self.transport.get_bpm();
        let beats_per_sample = (bpm / 60.0) / self.sample_rate;

        for frame_idx in (0..buffer.len()).step_by(channels) {
            if self.is_playing_seq {
                let prev_cursor = self.playback_cursor_global_beats;
                self.playback_cursor_global_beats += beats_per_sample as f64;
                let curr_cursor = self.playback_cursor_global_beats;

                if curr_cursor < 0.0 {
                    if let Some(tx) = &mut self.metro_cmd_tx {
                        let _ = tx.push(MetronomeCommand::SetMuted(false));
                    }
                } else {
                    if self.current_measure_index < self.measures.len() {
                        let m = &self.measures[self.current_measure_index];
                        let measure_end = m.global_start_beat + m.duration_beats();
                        let abs_time = curr_cursor + self.start_measure_global_offset;

                        if abs_time >= measure_end {
                            self.current_measure_index += 1;
                            self.sync_metronome(self.current_measure_index);
                        }
                    }
                }

                if curr_cursor >= 0.0 && self.current_measure_index < self.measures.len() {
                    let m = &self.measures[self.current_measure_index];
                    let beat_in_measure =
                        (curr_cursor + self.start_measure_global_offset) - m.global_start_beat;
                    let prev_beat_in_measure =
                        (prev_cursor + self.start_measure_global_offset) - m.global_start_beat;

                    for note in &m.notes {
                        if note.start_beat_in_measure as f64 > prev_beat_in_measure
                            && note.start_beat_in_measure as f64 <= beat_in_measure
                        {
                            let velocity = if self.muted { 0.0 } else { note.velocity };
                            self.voices.push(Voice::new(
                                note.freq,
                                velocity,
                                Some(note.duration_beats),
                                note.instrument,
                            ));
                        }
                    }
                }
            } else {
                if let Some(tx) = &mut self.metro_cmd_tx {
                    let _ = tx.push(MetronomeCommand::SetMuted(true));
                }
            }

            let mut sum = 0.0;
            let mut active_count: f32 = 0.0;

            for v in &mut self.voices {
                sum += v.process(self.sample_rate, beats_per_sample);
                if !matches!(v.state, EnvState::Finished) {
                    active_count += 1.0;
                }
            }
            self.voices
                .retain(|v| !matches!(v.state, EnvState::Finished));

            let norm_factor = if active_count > 1.0 {
                1.0 / active_count.sqrt()
            } else {
                1.0
            };
            let final_val = sum * norm_factor * self.volume;

            for ch in 0..channels {
                buffer[frame_idx + ch] += final_val;
            }
        }
    }
}
