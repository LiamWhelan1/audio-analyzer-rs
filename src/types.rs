pub type Sample = f32;

#[derive(Debug, Clone)]
pub struct AudioChunk {
    /// interleaved samples (for mono apps use single channel)
    pub samples: Vec<Sample>,
    pub sample_rate: u32,
    pub channels: usize,
}
