pub trait Worker {
    fn stop(&mut self) -> u8;
    fn pause(&mut self) -> u8;
    fn start(&mut self) -> u8;
}
