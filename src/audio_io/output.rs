use rtrb::{Consumer, Producer, RingBuffer};
use spin; // For real-time safe mutex
use std::f32::consts::PI; // For sine wave
use std::sync::Arc;

/// A trait for any object that can generate audio.
/// This is the core abstraction for your output system.
pub trait AudioSource {
    /// Generates audio and *adds* it to the buffer.
    /// Does not clear the buffer first.
    fn process(&mut self, buffer: &mut [f32], channels: usize);

    /// Returns true if this source is done playing and can be removed from the mixer.
    fn is_finished(&self) -> bool;
}

/// The Mixer combines all active `AudioSource`s.
pub struct Mixer {
    sources: Vec<Box<dyn AudioSource + Send>>,
    scratch_buf: Vec<f32>,
}

impl Mixer {
    pub fn new(channels: usize) -> Self {
        Self {
            sources: Vec::new(),
            scratch_buf: vec![0.0; 2048 * channels],
        }
    }

    pub fn process(&mut self, out_buffer: &mut [f32], channels: usize) {
        // 0. Garbage Collect: Remove finished sources
        self.sources.retain(|s| !s.is_finished());

        // 1. Clear the main output buffer
        out_buffer.fill(0.0);

        // 2. Process each source
        for source in self.sources.iter_mut() {
            if self.scratch_buf.len() != out_buffer.len() {
                self.scratch_buf.resize(out_buffer.len(), 0.0);
            }
            self.scratch_buf.fill(0.0);

            source.process(&mut self.scratch_buf, channels);

            for (out_sample, src_sample) in out_buffer.iter_mut().zip(self.scratch_buf.iter()) {
                *out_sample += *src_sample;
            }
        }

        // 3. Optional: Master limiter
        for sample in out_buffer.iter_mut() {
            *sample = sample.clamp(-1.0, 1.0);
        }
    }

    fn add_source(&mut self, source: Box<dyn AudioSource + Send>) {
        self.sources.push(source);
    }
}

// ... OutputController remains the same ...
#[derive(Clone)]
pub struct OutputController {
    mixer: Arc<spin::Mutex<Mixer>>,
}

impl OutputController {
    pub fn new(mixer: Arc<spin::Mutex<Mixer>>) -> Self {
        Self { mixer }
    }
    pub fn add_source(&self, source: Box<dyn AudioSource + Send>) {
        self.mixer.lock().add_source(source);
    }
}

// ... BeatStrength remains the same ...
#[derive(Clone, Debug, PartialEq)]
pub enum BeatStrength {
    Strong,
    Medium,
    Weak,
    None,
    Subdivision(usize),
}

pub enum MetronomeCommand {
    SetBpm(f32),
    SetVolume(f32),
    SetPattern(Vec<BeatStrength>),
    SetSubdivisions(Vec<usize>),
    Stop, // <-- NEW: Explicit stop command
}

// ... SubdivisionTracker, TickGenerator remain the same ...
struct SubdivisionTracker {
    divisions: usize,
    samples_per_sub: f32,
    sample_counter: f32,
    current_sub_index: usize,
}

struct TickGenerator {
    phase: f32,
    freq: f32,
    volume: f32,
    envelope: f32,
    decay_rate: f32,
}

impl TickGenerator {
    fn process(&mut self, sample_rate: f32) -> f32 {
        let phase_inc = (self.freq * 2.0 * PI) / sample_rate;
        let sample = (self.phase * phase_inc).sin();
        self.phase += 1.0;
        let output = sample * self.volume * self.envelope;
        self.envelope *= self.decay_rate;
        output
    }
    fn is_finished(&self) -> bool {
        self.envelope < 0.001
    }
}

pub struct Metronome {
    sample_rate: f32,
    volume: f32,
    bpm: f32,
    pattern: Vec<BeatStrength>,
    subdivisions: Vec<usize>,

    samples_per_beat: f32,
    main_beat_counter: f32,
    current_beat_index: usize,
    subdivision_trackers: Vec<SubdivisionTracker>,
    active_ticks: Vec<TickGenerator>,

    command_rx: Consumer<MetronomeCommand>,
    finished: bool, // <-- NEW: Track if we should die
}

impl Metronome {
    pub fn new(
        sample_rate: u32,
        initial_bpm: f32,
        initial_pattern: Vec<BeatStrength>,
        initial_subdivisions: Option<Vec<usize>>,
    ) -> (Self, Producer<MetronomeCommand>) {
        let (prod, cons) = RingBuffer::new(128);
        let sr_f32 = sample_rate as f32;
        let samples_per_beat = (sr_f32 * 60.0) / initial_bpm;
        let subdivisions = initial_subdivisions.unwrap_or(vec![]);

        let mut metronome = Self {
            sample_rate: sr_f32,
            volume: 0.5,
            bpm: initial_bpm,
            pattern: initial_pattern,
            subdivisions: vec![],
            samples_per_beat,
            main_beat_counter: 0.0,
            current_beat_index: 0,
            subdivision_trackers: Vec::new(),
            active_ticks: Vec::new(),
            command_rx: cons,
            finished: false,
        };

        metronome.update_subdivision_trackers(subdivisions);
        (metronome, prod)
    }

    fn update_bpm(&mut self, new_bpm: f32) {
        self.bpm = new_bpm.max(1.0);
        self.samples_per_beat = (self.sample_rate * 60.0) / self.bpm;
        self.update_subdivision_trackers(self.subdivisions.clone());
    }

    fn update_subdivision_trackers(&mut self, new_subdivisions: Vec<usize>) {
        self.subdivisions = new_subdivisions;
        self.subdivision_trackers.clear();
        for &div in &self.subdivisions {
            if div > 0 {
                let samples_per_sub = self.samples_per_beat / (div as f32);
                self.subdivision_trackers.push(SubdivisionTracker {
                    divisions: div,
                    samples_per_sub,
                    sample_counter: self.main_beat_counter,
                    current_sub_index: self.current_beat_index * div,
                });
            }
        }
    }

    fn handle_commands(&mut self) {
        // If the main thread dropped the handle, we should stop eventually.
        if self.command_rx.is_abandoned() {
            self.finished = true;
            return;
        }

        while let Ok(cmd) = self.command_rx.pop() {
            match cmd {
                MetronomeCommand::SetBpm(bpm) => self.update_bpm(bpm),
                MetronomeCommand::SetVolume(vol) => self.volume = vol.clamp(0.0, 2.0),
                MetronomeCommand::SetPattern(pat) => self.pattern = pat,
                MetronomeCommand::SetSubdivisions(subs) => self.update_subdivision_trackers(subs),
                MetronomeCommand::Stop => self.finished = true,
            }
        }
    }

    fn spawn_tick(&mut self, strength: &BeatStrength) {
        let (freq, volume, decay_time_ms) = match strength {
            BeatStrength::Strong => (2000.0, 1.0, 150.0),
            BeatStrength::Medium => (1500.0, 0.7, 150.0),
            BeatStrength::Weak => (1250.0, 0.4, 150.0),
            BeatStrength::Subdivision(n) => (1500.0 / (*n as f32).max(1.0), 0.5, 150.0),
            BeatStrength::None => return,
        };

        let decay_samples = self.sample_rate * (decay_time_ms / 1000.0);
        let decay_rate = (0.001_f32).powf(1.0 / decay_samples);

        self.active_ticks.push(TickGenerator {
            phase: 0.0,
            freq,
            volume,
            envelope: 1.0,
            decay_rate,
        });
    }
}

impl AudioSource for Metronome {
    fn is_finished(&self) -> bool {
        self.finished
    }

    fn process(&mut self, buffer: &mut [f32], channels: usize) {
        self.handle_commands();

        if self.finished {
            return; // Don't generate audio if finished
        }

        if self.pattern.is_empty() {
            return;
        }

        for frame_idx in (0..buffer.len()).step_by(channels) {
            let mut pending_subdivisions = None;

            // Check main beat
            self.main_beat_counter += 1.0;
            let mut main_beat_triggered = false;

            if self.main_beat_counter >= self.samples_per_beat {
                main_beat_triggered = true;
                self.main_beat_counter -= self.samples_per_beat;

                let strength = self
                    .pattern
                    .get(self.current_beat_index)
                    .unwrap_or(&BeatStrength::Weak)
                    .clone();

                self.spawn_tick(&strength);
                self.current_beat_index = (self.current_beat_index + 1) % self.pattern.len();
            }

            // Check subdivisions
            for tracker in &mut self.subdivision_trackers {
                tracker.sample_counter += 1.0;
                if tracker.sample_counter >= tracker.samples_per_sub {
                    tracker.sample_counter -= tracker.samples_per_sub;
                    let is_main_beat_position = tracker.current_sub_index == 0;

                    if !(main_beat_triggered && is_main_beat_position) {
                        pending_subdivisions = Some(tracker.divisions);
                    }
                    tracker.current_sub_index = (tracker.current_sub_index + 1) % tracker.divisions;
                }
            }

            if let Some(divisions) = pending_subdivisions {
                self.spawn_tick(&BeatStrength::Subdivision(divisions));
            }

            // Process ticks
            let mut sample = 0.0;
            for tick in &mut self.active_ticks {
                sample += tick.process(self.sample_rate);
            }
            self.active_ticks.retain(|tick| !tick.is_finished());

            // Output
            let final_sample = (sample * self.volume).clamp(-1.0, 1.0);
            for ch in 0..channels {
                buffer[frame_idx + ch] += final_sample;
            }
        }
    }
}

// ============================================================================
//  POLYPHONIC SYNTHESIZER
// ============================================================================

#[derive(Clone, Debug)]
pub struct SynthNote {
    pub freq: f32,
    pub duration_ms: f32, // If 0.0, it's indefinite (live mode)
    pub onset_ms: f32,    // Relative to sequence start. Ignored for live mode.
    pub velocity: f32,
}

pub enum SynthCommand {
    /// Clears current state and loads a new sequence of notes to play.
    LoadSequence(Vec<SynthNote>),
    /// Plays a single note immediately (Live mode).
    NoteOn { freq: f32, velocity: f32 },
    /// Stops a specific frequency (Live mode).
    NoteOff { freq: f32 },
    /// Pauses the sequence cursor. Active notes decay naturally.
    Pause,
    /// Resumes the sequence cursor.
    Resume,
    /// Stops all sound and kills the synthesizer source.
    Stop,
}

struct Voice {
    freq: f32,
    phase: f32,
    velocity: f32,
    /// How much time is left for this note (if fixed duration)
    remaining_duration_samples: Option<f32>,
    /// ADSR Envelope State
    envelope_val: f32,
    state: EnvelopeState,
}

enum EnvelopeState {
    Attack,
    Decay,
    Sustain,
    Release,
    Finished,
}

impl Voice {
    fn new(freq: f32, velocity: f32, duration_ms: f32, sample_rate: f32) -> Self {
        let remaining = if duration_ms > 0.0 {
            Some(duration_ms / 1000.0 * sample_rate)
        } else {
            None
        };

        Self {
            freq,
            phase: 0.0,
            velocity,
            remaining_duration_samples: remaining,
            envelope_val: 0.0,
            state: EnvelopeState::Attack,
        }
    }

    fn release(&mut self) {
        if !matches!(self.state, EnvelopeState::Finished) {
            self.state = EnvelopeState::Release;
        }
    }

    fn process(&mut self, sample_rate: f32) -> f32 {
        if matches!(self.state, EnvelopeState::Finished) {
            return 0.0;
        }

        // 1. Oscillator (Simple Triangle Wave)
        let phase_inc = (self.freq * 2.0 * PI) / sample_rate;
        self.phase += phase_inc;
        if self.phase > 2.0 * PI {
            self.phase -= 2.0 * PI;
        }

        // Triangle wave math
        let raw_sample = (2.0 / PI) * (self.phase.sin().asin());

        // 2. Envelope Logic (Simple ADSR)
        let attack_rate = 0.05;
        let release_rate = 0.005;

        match self.state {
            EnvelopeState::Attack => {
                self.envelope_val += attack_rate;
                if self.envelope_val >= 1.0 {
                    self.envelope_val = 1.0;
                    self.state = EnvelopeState::Decay;
                }
            }
            EnvelopeState::Decay => {
                // Simple sustain at 1.0 for now, or decay to 0.8
                self.state = EnvelopeState::Sustain;
            }
            EnvelopeState::Sustain => {
                // Hold value
                if let Some(rem) = &mut self.remaining_duration_samples {
                    *rem -= 1.0;
                    if *rem <= 0.0 {
                        self.state = EnvelopeState::Release;
                    }
                }
            }
            EnvelopeState::Release => {
                self.envelope_val -= release_rate;
                if self.envelope_val <= 0.0 {
                    self.envelope_val = 0.0;
                    self.state = EnvelopeState::Finished;
                }
            }
            _ => {}
        }

        raw_sample * self.envelope_val * self.velocity
    }
}

pub struct Synthesizer {
    sample_rate: f32,
    voices: Vec<Voice>,

    // Sequence State
    sequence: Vec<SynthNote>,
    sequence_cursor: usize,
    playback_samples: f32,
    is_playing_sequence: bool,

    command_rx: Consumer<SynthCommand>,
    finished: bool,
}

impl Synthesizer {
    pub fn new(sample_rate: u32) -> (Self, Producer<SynthCommand>) {
        let (prod, cons) = RingBuffer::new(128);
        (
            Self {
                sample_rate: sample_rate as f32,
                voices: Vec::with_capacity(16),
                sequence: Vec::new(),
                sequence_cursor: 0,
                playback_samples: 0.0,
                is_playing_sequence: false,
                command_rx: cons,
                finished: false,
            },
            prod,
        )
    }

    fn handle_commands(&mut self) {
        if self.command_rx.is_abandoned() {
            self.finished = true;
            return;
        }

        while let Ok(cmd) = self.command_rx.pop() {
            match cmd {
                SynthCommand::LoadSequence(notes) => {
                    self.sequence = notes;
                    // Sort by onset to ensure correct playback order
                    self.sequence
                        .sort_by(|a, b| a.onset_ms.partial_cmp(&b.onset_ms).unwrap());
                    self.sequence_cursor = 0;
                    self.playback_samples = 0.0;
                    self.is_playing_sequence = true;
                    // Note: We do NOT clear active voices, allowing tails to ring out
                }
                SynthCommand::Pause => {
                    self.is_playing_sequence = false;
                    // Optional: Release all voices on pause?
                    // For now, let's keep them sustaining if they are indefinite,
                    // but usually "Pause" stops the timeline.
                }
                SynthCommand::Resume => {
                    self.is_playing_sequence = true;
                }
                SynthCommand::Stop => {
                    self.finished = true;
                }
                SynthCommand::NoteOn { freq, velocity } => {
                    // Live Mode: Trigger immediately (indefinite duration)
                    self.voices
                        .push(Voice::new(freq, velocity, 0.0, self.sample_rate));
                }
                SynthCommand::NoteOff { freq } => {
                    // Find voices with this freq and release them
                    for voice in self.voices.iter_mut() {
                        if (voice.freq - freq).abs() < 0.1 {
                            voice.release();
                        }
                    }
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

        for frame_idx in (0..buffer.len()).step_by(channels) {
            // 1. Sequencer Logic
            if self.is_playing_sequence {
                let current_ms = (self.playback_samples / self.sample_rate) * 1000.0;

                // Check if we need to trigger new notes
                while self.sequence_cursor < self.sequence.len() {
                    let note = &self.sequence[self.sequence_cursor];
                    if current_ms >= note.onset_ms {
                        // Trigger Note
                        self.voices.push(Voice::new(
                            note.freq,
                            note.velocity,
                            note.duration_ms,
                            self.sample_rate,
                        ));
                        self.sequence_cursor += 1;
                    } else {
                        // Next note is in the future
                        break;
                    }
                }
                self.playback_samples += 1.0;
            }

            // 2. Voice Processing & Mixing
            let mut sample_sum = 0.0;

            // Process all voices
            for voice in &mut self.voices {
                sample_sum += voice.process(self.sample_rate);
            }

            // Remove finished voices
            self.voices
                .retain(|v| !matches!(v.state, EnvelopeState::Finished));

            // 3. Write to Output
            // Simple hard clip to prevent massive polyphony distortion
            sample_sum = sample_sum.clamp(-1.0, 1.0);

            for ch in 0..channels {
                buffer[frame_idx + ch] += sample_sum * 0.5; // Master synth volume
            }
        }
    }
}
