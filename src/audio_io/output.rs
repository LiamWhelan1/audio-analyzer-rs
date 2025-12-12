use crate::traits::AudioSource;
use spin;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, AtomicU32, AtomicU64, Ordering};

/// Thread-safe musical transport for tracking beats and frame counts.
pub struct MusicalTransport {
    output_frames: AtomicI64,
    input_frames: AtomicI64,
    bpm_bits: AtomicU32,
    accumulated_beats_bits: AtomicU64,
}

impl MusicalTransport {
    /// Create a new transport with an initial BPM.
    pub fn new(initial_bpm: f32) -> Arc<Self> {
        Arc::new(Self {
            output_frames: AtomicI64::new(0),
            input_frames: AtomicI64::new(0),
            bpm_bits: AtomicU32::new(initial_bpm.to_bits()),
            accumulated_beats_bits: AtomicU64::new(0.0f64.to_bits()),
        })
    }

    /// Advance output frame counters and update accumulated beat time.
    pub fn tick_output(&self, frames: i64, sample_rate: f32) {
        self.output_frames.fetch_add(frames, Ordering::Relaxed);

        let bpm = self.get_bpm();
        let seconds = frames as f64 / sample_rate as f64;
        let beats_delta = seconds * (bpm as f64 / 60.0);

        let mut current = self.accumulated_beats_bits.load(Ordering::Relaxed);
        loop {
            let current_val = f64::from_bits(current);
            let new_val = current_val + beats_delta;
            match self.accumulated_beats_bits.compare_exchange_weak(
                current,
                new_val.to_bits(),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => current = x,
            }
        }
    }

    /// Advance input frame counters.
    pub fn tick_input(&self, frames: i64) {
        self.input_frames.fetch_add(frames, Ordering::Relaxed);
    }

    /// Return difference between input and output frame counters (drift in samples).
    pub fn get_drift_samples(&self) -> i64 {
        let in_t = self.input_frames.load(Ordering::Relaxed);
        let out_t = self.output_frames.load(Ordering::Relaxed);
        in_t - out_t
    }

    /// Set the current BPM.
    pub fn set_bpm(&self, bpm: f32) {
        self.bpm_bits.store(bpm.to_bits(), Ordering::Relaxed);
    }

    /// Get the current BPM.
    pub fn get_bpm(&self) -> f32 {
        f32::from_bits(self.bpm_bits.load(Ordering::Relaxed))
    }

    /// Read accumulated beat time as floating-point beats.
    pub fn get_accumulated_beats(&self) -> f64 {
        f64::from_bits(self.accumulated_beats_bits.load(Ordering::Relaxed))
    }

    /// Reset accumulated beats to a specific value.
    pub fn reset_to_beats(&self, beats: f64) {
        self.accumulated_beats_bits
            .store(beats.to_bits(), Ordering::SeqCst);
    }
}

/// Mixer that aggregates `AudioSource` inputs into a single output buffer.
pub struct Mixer {
    sources: Vec<Box<dyn AudioSource + Send>>,
    scratch_buf: Vec<f32>,
}

impl Mixer {
    /// Create a new mixer configured for `channels` channels.
    pub fn new(channels: usize) -> Self {
        Self {
            sources: Vec::new(),
            scratch_buf: vec![0.0; 2048 * channels],
        }
    }

    /// Add a new audio source to the mixer.
    pub fn add_source(&mut self, source: Box<dyn AudioSource + Send>) {
        self.sources.push(source);
    }

    /// Mix active sources into `out_buffer`. `channels` is the number of output channels.
    pub fn process(&mut self, out_buffer: &mut [f32], channels: usize) {
        self.sources.retain(|s| !s.is_finished());
        out_buffer.fill(0.0);

        if self.scratch_buf.len() != out_buffer.len() {
            self.scratch_buf.resize(out_buffer.len(), 0.0);
        }

        for source in self.sources.iter_mut() {
            self.scratch_buf.fill(0.0);
            source.process(&mut self.scratch_buf, channels);

            for (out, src) in out_buffer.iter_mut().zip(self.scratch_buf.iter()) {
                *out += *src;
            }
        }

        for sample in out_buffer.iter_mut() {
            *sample = sample.clamp(-1.0, 1.0);
        }
    }
}

#[derive(Clone)]
/// Lightweight controller exposing mixer operations to other modules.
pub struct OutputController {
    mixer: Arc<spin::Mutex<Mixer>>,
}

impl OutputController {
    /// Construct an `OutputController` that operates on the provided `mixer`.
    pub fn new(mixer: Arc<spin::Mutex<Mixer>>) -> Self {
        Self { mixer }
    }

    /// Add an `AudioSource` into the controlled mixer.
    pub fn add_source(&self, source: Box<dyn AudioSource + Send>) {
        self.mixer.lock().add_source(source);
    }
}
