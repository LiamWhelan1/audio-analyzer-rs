use crate::traits::AudioSource;
use spin;
use std::sync::Arc;

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

    pub fn has_sources(&self) -> bool {
        !self.mixer.lock().sources.is_empty()
    }
}
