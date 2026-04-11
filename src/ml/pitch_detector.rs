//! Polyphonic pitch detection via a tiny ONNX model (tract runtime).
//!
//! [`PitchDetector::new`] loads the embedded model and pre-computes a mel
//! filterbank that mirrors `src/dsp/spectrogram.rs` and the Python training
//! pipeline.  [`PitchDetector::detect`] accepts the FFT power spectrum
//! produced by `stft.rs` and returns `(frequency_hz, confidence)` pairs for
//! every note the model believes is present.
//!
//! # Normalization constants
//!
//! `NORM_MEAN` and `NORM_STD` must match the values printed by
//! `training/export.py` after running the full training pipeline.
//! Update them here before shipping the trained model.

use tract_onnx::prelude::*;

use crate::dsp::spectrogram::{build_mel_filterbank, spect_to_mel};

// ── model constants ──────────────────────────────────────────────────────────

/// MIDI note range covered by the model output (inclusive).
const MIDI_MIN: u8 = 24; // C1 ≈ 32.7 Hz
const MIDI_MAX: u8 = 95; // B6 ≈ 1975.5 Hz
const NUM_SEMITONES: usize = (MIDI_MAX - MIDI_MIN + 1) as usize; // 72
/// 20-cent resolution bins per semitone — must match training/model.py.
const BINS_PER_SEMITONE: usize = 5;
const NUM_BINS: usize = NUM_SEMITONES * BINS_PER_SEMITONE; // 360

const N_MELS: usize = 128;
const FMIN_HZ: f32 = 30.0; // covers C1 fundamental (32.7 Hz); must match prepare_dataset.py
const FMAX_HZ: f32 = 8_000.0;

// Normalization constants computed over the training set by prepare_dataset.py.
// Replace these with the values printed by training/export.py once training is
// complete. The placeholder values (0 mean, 1 std) leave features unscaled and
// will give poor detections — they exist only so the code compiles before
// training.
const NORM_MEAN: f32 = -5.064;
const NORM_STD: f32 = 5.48986;

// Embedded ONNX model bytes.  The path is relative to this source file.
// `include_bytes!` is checked at compile time; place the file at
// `models/pitch_detector.onnx` (repo root) before building.
//
// During development (before the model is trained) comment this line out and
// use the `PitchDetector::unavailable()` fallback so the rest of the pipeline
// still works.
#[cfg(feature = "ml_model")]
const MODEL_BYTES: &[u8] = include_bytes!("../../models/pitch_detector.onnx");

// ── public struct ────────────────────────────────────────────────────────────

type OnnxPlan = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

pub(crate) struct PitchDetector {
    plan: OnnxPlan,
    mel_filters: Vec<Vec<f32>>,
}

impl PitchDetector {
    /// Load the embedded ONNX model and pre-compute the mel filterbank.
    ///
    /// Returns `Err` if the model bytes are absent or fail to parse (e.g.
    /// before training is complete).  Callers should wrap the result in
    /// `Option` and fall back to the DSP pipeline when `None`.
    pub(crate) fn new(sample_rate: u32, fft_size: usize) -> anyhow::Result<Self> {
        let model_bytes = Self::model_bytes()?;

        let input_shape: TVec<TDim> =
            tvec![TDim::from(1usize), TDim::from(1usize), TDim::from(N_MELS),];

        let plan = tract_onnx::onnx()
            .model_for_read(&mut std::io::Cursor::new(model_bytes))?
            .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), input_shape))?
            .into_optimized()?
            .into_runnable()?;

        let mel_filters =
            build_mel_filterbank(sample_rate as usize, fft_size, N_MELS, FMIN_HZ, FMAX_HZ);

        Ok(Self { plan, mel_filters })
    }

    /// Try to load the model from disk at `models/pitch_detector.onnx`
    /// relative to the workspace root.  Falls back to embedded bytes when the
    /// `ml_model` feature is enabled.
    ///
    /// Returns `Err` when neither source is available (pre-training).
    fn model_bytes() -> anyhow::Result<Vec<u8>> {
        // Runtime file load first — useful during development.
        let candidates = [
            "models/pitch_detector.onnx",
            "../models/pitch_detector.onnx",
        ];
        for path in &candidates {
            if let Ok(bytes) = std::fs::read(path) {
                return Ok(bytes);
            }
        }

        // Compile-time embedded fallback (production / mobile).
        #[cfg(feature = "ml_model")]
        return Ok(MODEL_BYTES.to_vec());

        #[cfg(not(feature = "ml_model"))]
        anyhow::bail!(
            "pitch_detector.onnx not found at models/pitch_detector.onnx. \
             Train the model first (see models/README.md)."
        )
    }

    /// Given the FFT power spectrum from `stft.rs` (half-complex output,
    /// length `fft_size / 2 + 1`), return `(frequency_hz, confidence)` pairs
    /// for every note the model detects above [`DETECTION_THRESHOLD`].
    ///
    /// Returns an empty `Vec` on inference failure so the caller can fall back
    /// to DSP-only peak picking.
    pub(crate) fn detect(&self, power_spectrum: &[f32]) -> Vec<(f32, f32)> {
        match self.run_inference(power_spectrum) {
            Ok(v) => v,
            Err(e) => {
                log::warn!("ML pitch inference failed: {e}");
                Vec::new()
            }
        }
    }

    fn run_inference(&self, power_spectrum: &[f32]) -> anyhow::Result<Vec<(f32, f32)>> {
        // 1. Mel filterbank
        let mel = spect_to_mel(power_spectrum, &self.mel_filters);

        // 2. Log compress + z-score normalise.
        // Must match prepare_dataset.py exactly: log(v + 1e-6), NOT ln(1+v).
        // ln_1p(v) ≈ log(v+1e-6) only near zero; they diverge at larger values.
        let features: Vec<f32> = mel
            .iter()
            .map(|&v| ((v + 1e-6_f32).ln() - NORM_MEAN) / NORM_STD)
            .collect();

        // 3. Build tract tensor [1, 1, 128]
        let input = tract_ndarray::Array3::from_shape_fn((1, 1, N_MELS), |(_, _, i)| {
            features.get(i).copied().unwrap_or(0.0)
        });
        let input_tensor: Tensor = input.into();

        // 4. Inference — model returns two outputs:
        //      result[0] = count_logits  [1, 5]    (how many notes: 0–4)
        //      result[1] = note_logits   [1, 360]  (20-cent bins: 72 semitones × 5)
        let result = self.plan.run(tvec![input_tensor.into()])?;
        let count_logits = result[0].to_array_view::<f32>()?;
        let note_logits  = result[1].to_array_view::<f32>()?;

        // 5. Predicted count — argmax over the 5 count classes, clamp to ≥ 1.
        let predicted_count = count_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(1)
            .max(1);

        // 6. Max-pool 360 bins → 72 semitones: take the peak probability within
        //    each group of BINS_PER_SEMITONE consecutive bins.  The 5 bins of one
        //    semitone vote together rather than competing for top-K slots.
        let mut semitone_probs: Vec<(usize, f32)> = (0..NUM_SEMITONES)
            .map(|s| {
                let start = s * BINS_PER_SEMITONE;
                let peak = note_logits
                    .iter()
                    .skip(start)
                    .take(BINS_PER_SEMITONE)
                    .map(|&l| sigmoid(l))
                    .fold(f32::NEG_INFINITY, f32::max);
                (s, peak)
            })
            .collect();

        // 7. Top-K selection at semitone level — matches training metric.
        semitone_probs.sort_unstable_by(|(_, a), (_, b)| {
            b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
        });

        let detections = semitone_probs
            .into_iter()
            .take(predicted_count)
            .map(|(semitone_idx, confidence)| {
                let midi = MIDI_MIN as usize + semitone_idx;
                (midi_to_hz(midi as u8), confidence)
            })
            .collect();

        Ok(detections)
    }
}

// ── helpers ──────────────────────────────────────────────────────────────────

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Convert a MIDI note number to its equal-temperament frequency in Hz
/// using A4 = 440 Hz as the reference.
#[inline(always)]
pub(crate) fn midi_to_hz(midi: u8) -> f32 {
    440.0 * 2.0_f32.powf((midi as f32 - 69.0) / 12.0)
}

// ── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn midi_to_hz_a4() {
        let hz = midi_to_hz(69);
        assert!((hz - 440.0).abs() < 0.01, "A4 should be 440 Hz, got {hz}");
    }

    #[test]
    fn midi_to_hz_middle_c() {
        let hz = midi_to_hz(60);
        assert!(
            (hz - 261.626).abs() < 0.1,
            "C4 should be ~261.6 Hz, got {hz}"
        );
    }

    #[test]
    fn sigmoid_bounds() {
        assert!(sigmoid(0.0) > 0.49 && sigmoid(0.0) < 0.51);
        assert!(sigmoid(100.0) > 0.99);
        assert!(sigmoid(-100.0) < 0.01);
    }

    /// Verifies that the mel filterbank dimensions are consistent with the
    /// power spectrum size expected by stft.rs (fft_size=2048, sr=44100).
    #[test]
    fn filterbank_dimensions() {
        let filters = build_mel_filterbank(44_100, 2048, N_MELS, FMIN_HZ, FMAX_HZ);
        assert_eq!(filters.len(), N_MELS);
        // Each filter row spans half_size = fft_size/2 + 1 = 1025 bins
        for row in &filters {
            assert_eq!(row.len(), 1025);
        }
    }
}
