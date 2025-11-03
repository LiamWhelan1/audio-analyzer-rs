use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex;
use std::sync::Arc;

pub struct FftProcessor {
    _planner: RealFftPlanner<f32>,
    forward: FftForward,
    inverse: FftInverse,
}

impl FftProcessor {
    pub fn new(len: usize) -> Self {
        let mut planner = RealFftPlanner::new();
        let fft = planner.plan_fft_forward(len);
        let inv_fft = planner.plan_fft_inverse(len);
        let output = inv_fft.make_output_vec();
        let scratch_inv = inv_fft.make_scratch_vec();
        let spectrum = fft.make_output_vec();
        let scratch = fft.make_scratch_vec();
        let forward = FftForward::new(fft, scratch, spectrum);
        let inverse = FftInverse::new(inv_fft, scratch_inv, output);
        Self {
            _planner: planner,
            forward,
            inverse,
        }
    }

    pub fn process_forward(&mut self, windowed: &mut [f32]) -> &[Complex<f32>] {
        self.forward.process(windowed)
    }
    pub fn process_inverse(&mut self, windowed: &mut [Complex<f32>]) -> &[f32] {
        self.inverse.process(windowed)
    }
}

pub struct FftForward {
    fft: Arc<dyn RealToComplex<f32>>,
    scratch: Vec<Complex<f32>>,
    spectrum: Vec<Complex<f32>>,
}

impl FftForward {
    pub fn new(
        fft: Arc<dyn RealToComplex<f32>>,
        scratch: Vec<Complex<f32>>,
        spectrum: Vec<Complex<f32>>,
    ) -> Self {
        FftForward {
            fft,
            scratch,
            spectrum,
        }
    }
    pub fn process(&mut self, windowed: &mut [f32]) -> &[Complex<f32>] {
        self.fft
            .process_with_scratch(windowed, &mut self.spectrum, &mut self.scratch)
            .unwrap();
        &self.spectrum
    }
}

pub struct FftInverse {
    inv_fft: Arc<dyn ComplexToReal<f32>>,
    scratch: Vec<Complex<f32>>,
    output: Vec<f32>,
}

impl FftInverse {
    pub fn new(
        inv_fft: Arc<dyn ComplexToReal<f32>>,
        scratch: Vec<Complex<f32>>,
        output: Vec<f32>,
    ) -> Self {
        FftInverse {
            inv_fft,
            scratch,
            output,
        }
    }
    pub fn process(&mut self, windowed: &mut [Complex<f32>]) -> &[f32] {
        self.inv_fft
            .process_with_scratch(windowed, &mut self.output, &mut self.scratch)
            .unwrap();
        &self.output
    }
}
