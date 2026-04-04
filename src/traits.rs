/// Trait for all audio producers to be processed by the Mixer
pub trait AudioSource {
    /// Writes audio data to the buffer
    fn process(&mut self, buffer: &mut [f32], channels: usize);
    /// Signals if Self is finished
    fn is_finished(&self) -> bool;
}
