pub trait Worker: Send + Sync {
    /// Resumes or starts the worker logic.
    fn start(&mut self) -> u8;

    /// Pauses the worker logic (thread sleeps or waits).
    fn pause(&mut self) -> u8;

    /// Stops the worker logic and prepares for cleanup.
    fn stop(&mut self) -> u8;
}
