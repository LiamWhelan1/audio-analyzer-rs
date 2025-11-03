use rubato::{
    SincFixedIn, SincInterpolationParameters, SincInterpolationType, VecResampler, WindowFunction,
};

pub struct Resampler {
    inner: SincFixedIn<f32>,
    in_channels: usize,
    out_channels: usize,
}

impl Resampler {
    pub fn new(in_sr: usize, out_sr: usize, channels: usize, chunk_size: usize) -> Self {
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 160,
            window: WindowFunction::Hann2,
        };
        let inner = SincFixedIn::<f32>::new(
            out_sr as f64 / in_sr as f64,
            10.0,
            params,
            chunk_size,
            channels,
        )
        .expect("resampler init");
        Self {
            inner,
            in_channels: channels,
            out_channels: channels,
        }
    }

    /// resample expects frames as Vec<Vec<Sample>> per channel
    pub fn process(&mut self, in_frames: Vec<Vec<f32>>) -> anyhow::Result<Vec<Vec<f32>>> {
        let out = self.inner.process(&in_frames, None)?;
        Ok(out)
    }
}
