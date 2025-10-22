use crate::types::Sample;
use realfft::{RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex;
use std::sync::Arc;

pub struct FftProcessor {
    planner: RealFftPlanner<Sample>,
    len: usize,
    fft: Arc<dyn RealToComplex<Sample>>,
    scratch: Vec<Complex<Sample>>,
    spectrum: Vec<Complex<Sample>>,
}

impl FftProcessor {
    pub fn new(len: usize) -> Self {
        let mut planner = RealFftPlanner::new();
        let fft = planner.plan_fft_forward(len);
        let spectrum = fft.make_output_vec();
        let scratch = fft.make_scratch_vec();
        Self {
            planner,
            len,
            fft,
            scratch,
            spectrum,
        }
    }

    pub fn process(&mut self, windowed: &mut [Sample]) -> &[Complex<Sample>] {
        self.fft
            .process_with_scratch(windowed, &mut self.spectrum, &mut self.scratch)
            .unwrap();
        &self.spectrum
    }
}
