use super::{MAX_MIDI_VELOCITY, TWO_PI, Waveform};
use crate::generators::{EnvState, Instrument, InstrumentParams, Measure, load_midi_file};
use crate::traits::AudioSource;
use crate::{audio_io::output::MusicalTransport, generators::metronome::MetronomeCommand};
use rtrb::{Consumer, Producer, RingBuffer};
use std::path::Path;
use std::{f32::consts::PI, sync::Arc};

// ============================================================================
//  MODULE: SYNTHESIS (Normalized)
// ============================================================================
pub enum SynthCommand {
    // --- File Playback Commands ---
    LoadFile(String),
    Play {
        start_measure_idx: usize,
    },
    Pause,
    Resume,
    Stop,
    Clear,
    LinkMetronome(Producer<MetronomeCommand>),

    // --- Manual/Live Commands ---
    NoteOn {
        freq: f32,
        velocity: f32,
        instrument: Instrument,
    },
    NoteOff {
        freq: f32,
    },

    // --- Global ---
    SetVolume(f32),
}

struct Voice {
    freq: f32,
    phase: f32,
    velocity: f32,
    /// If Some, the note auto-releases after X beats (Sequencer).
    /// If None, the note sustains indefinitely until NoteOff (Manual).
    remaining_beats: Option<f32>,
    envelope: f32,
    state: EnvState,
    params: InstrumentParams, // New: Holds instrument configuration
    instrument: Instrument,   // New: Stores the instrument type
}

impl Voice {
    fn get_instrument_params(instrument: &Instrument) -> InstrumentParams {
        match instrument {
            Instrument::Piano => InstrumentParams {
                // Quick attack, fast decay to lower sustain (mimics hammer impact)
                attack_sec: 0.005,
                decay_sec: 0.15,
                sustain_level: 0.2,
                release_sec: 0.7, // Long release for damper pedal effect
                timbre_mix: 0.8,  // Higher mix for harmonics/percussive sound
            },
            Instrument::Violin => InstrumentParams {
                // Slow attack, steady sustain (mimics bow drawing across strings)
                attack_sec: 0.3,
                decay_sec: 0.1,
                sustain_level: 0.9,
                release_sec: 0.5,
                timbre_mix: 0.4, // Lower mix for a smoother, purer tone
            },
        }
    }

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

    fn process(&mut self, sample_rate: f32, beats_delta: f32) -> f32 {
        if matches!(self.state, EnvState::Finished) {
            return 0.0;
        }

        let sample_rate_inv = 1.0 / sample_rate;
        let phase_inc = (self.freq * TWO_PI) * sample_rate_inv;
        self.phase = (self.phase + phase_inc) % TWO_PI;

        // --- Timbre Generation ---
        // Basic oscillator mix (Violin uses purer sine/tri, Piano uses richer saw/noise)
        let fund = self.phase.sin(); // Fundamental frequency (Sine)
        let mut raw_sig = fund;

        match self.instrument {
            Instrument::Piano => {
                // Mix in a bit of Sawtooth for brightness and a harmonic
                let bright_comp = ((self.phase * 2.0).sin() + (self.phase / PI) - 1.0) * 0.5;
                raw_sig =
                    fund * (1.0 - self.params.timbre_mix) + bright_comp * self.params.timbre_mix;
            }
            Instrument::Violin => {
                // Mix in a Triangle wave for a richer, less buzzy sound than Sawtooth
                let t = self.phase / TWO_PI;
                let tri = 4.0 * (t - 0.5).abs() - 1.0;
                raw_sig = fund * (1.0 - self.params.timbre_mix) + tri * self.params.timbre_mix;
            }
        }

        // --- ADSR Envelope Logic (Time-based for better quality) ---
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
                // Check if the note is finished (only for sequenced notes)
                if let Some(rem) = &mut self.remaining_beats {
                    *rem -= beats_delta;
                    if *rem <= 0.0 {
                        self.state = EnvState::Release;
                    }
                }
                // Sustain phase holds steady at sustain_level
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

        raw_sig * self.envelope * self.velocity
    }
}

pub struct Synthesizer {
    sample_rate: f32,
    transport: Arc<MusicalTransport>,

    // Audio State
    volume: f32,
    voices: Vec<Voice>,

    // Sequence Data
    measures: Vec<Measure>,

    // Playback State
    is_playing_seq: bool, // Renamed for clarity: tracks if the SEQUENCER is running
    current_measure_index: usize,
    playback_cursor_global_beats: f64,
    start_measure_global_offset: f64,
    count_in_duration: f64,

    metro_cmd_tx: Option<Producer<MetronomeCommand>>,
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

    fn handle_commands(&mut self) {
        while let Ok(cmd) = self.command_rx.pop() {
            match cmd {
                SynthCommand::LinkMetronome(tx) => {
                    self.metro_cmd_tx = Some(tx);
                }
                SynthCommand::LoadFile(path_str) => {
                    let path = Path::new(&path_str);
                    if let Ok(m) = load_midi_file(path) {
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

                // --- MANUAL NOTE LOGIC ---
                SynthCommand::NoteOn {
                    freq,
                    velocity,
                    instrument,
                } => {
                    // Normalize Manual Velocity
                    let norm_vel = velocity / MAX_MIDI_VELOCITY;
                    // Pass 'None' for duration so it sustains indefinitely
                    self.voices
                        .push(Voice::new(freq, norm_vel, None, instrument));
                }
                SynthCommand::NoteOff { freq } => {
                    for v in &mut self.voices {
                        // Match frequency loosely to account for floats
                        if (v.freq - freq).abs() < 0.1 {
                            v.state = EnvState::Release;
                        }
                    }
                }

                // --- SEQUENCER COMMANDS ---
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
                        // Note: We do NOT clear voices here, allowing you to play over the start
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

        for frame_idx in (0..buffer.len()).step_by(channels) {
            // --- SEQUENCER LOGIC ---
            if self.is_playing_seq {
                let prev_cursor = self.playback_cursor_global_beats;
                self.playback_cursor_global_beats += beats_per_sample as f64;
                let curr_cursor = self.playback_cursor_global_beats;

                // self.transport.advance(beats_per_sample as f64); // I don't think this code is necessary

                // Metronome Handling
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

                // Note Triggering
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
                            // SEQUENCER NOTES: Pass Some(duration) so they auto-release
                            self.voices.push(Voice::new(
                                note.freq,
                                note.velocity, // Already normalized in Load
                                Some(note.duration_beats),
                                note.instrument,
                            ));
                        }
                    }
                }
            } else {
                // Mute metronome if sequencer isn't running
                if let Some(tx) = &mut self.metro_cmd_tx {
                    let _ = tx.push(MetronomeCommand::SetMuted(true));
                }
            }

            // --- PROCESS VOICES (Both Manual and Sequencer) ---
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
            // As requested, prevents clipping when many notes play at once
            let norm_factor = if active_count > 1.0 {
                1.0 / active_count
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
