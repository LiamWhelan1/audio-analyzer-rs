pub mod metrics;

#[derive(serde::Serialize)]
pub struct SendError {
    pub measure: u32,
    pub note_index: usize,
    pub error_type: MusicError,
    pub intensity: f64, // 0.0-1.0
}

#[derive(serde::Serialize)]
pub enum MusicError {
    Timing,
    Note,
    Intonation,
    Dynamics,
    Articulation,
}
