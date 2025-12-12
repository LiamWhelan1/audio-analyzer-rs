pub trait Worker: Send + Sync {
    /// Resumes or starts the worker logic
    fn start(&mut self) -> u8;

    /// Pauses the worker logic
    fn pause(&mut self) -> u8;

    /// Stops the worker logic and prepares for cleanup
    fn stop(&mut self) -> u8;
}

/// Trait for all audio producers to be processed by the Mixer
pub trait AudioSource {
    /// Writes audio data to the buffer
    fn process(&mut self, buffer: &mut [f32], channels: usize);
    /// Signals if Self is finished
    fn is_finished(&self) -> bool;
}
