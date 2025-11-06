use std::{
    collections::VecDeque,
    sync::{
        Arc,
        atomic::{AtomicI8, Ordering},
    },
    thread,
    time::Duration,
};

use crossbeam_channel::Sender;
use rustfft::num_complex::Complex;

use crate::{audio_io::SlotPool, dsp::fft::FftProcessor, generators::Note, traits::Worker};

pub struct STFT {
    sender: Sender<Note>,
    state: Arc<AtomicI8>,
    handle: u8,
}
impl Worker for STFT {
    fn stop(&mut self) -> u8 {
        self.state.store(-1, Ordering::Relaxed);
        self.handle
    }
    fn pause(&mut self) -> u8 {
        self.state.store(0, Ordering::Relaxed);
        self.handle
    }
    fn start(&mut self) -> u8 {
        self.state.store(1, Ordering::Relaxed);
        self.handle
    }
}
impl STFT {
    pub fn new(handle: u8, sender: Sender<Note>) -> Self {
        STFT {
            sender,
            state: Arc::new(AtomicI8::new(0)),
            handle,
        }
    }
    pub fn pitch(
        &mut self,
        slots: Arc<SlotPool>,
        slot_len: usize,
        mut cons: rtrb::Consumer<usize>,
        reclaim: Sender<usize>,
        sr: u32,
        base_freq: Option<f32>,
    ) {
        self.state.store(1, Ordering::Relaxed);
        let state = self.state.clone();

        // Window + buffers reused across frames (no per-frame alloc).
        let hann = Self::hann_window(slot_len);
        let mut windowed: Vec<f32> = vec![0.0; slot_len];
        // output length from your FftProcessor: slot_len/2 + 1
        let half_len = slot_len / 2 + 1;
        let mut output: Vec<Complex<f32>> = vec![Complex { re: 0.0, im: 0.0 }; half_len];
        let mut fft = FftProcessor::new(slot_len);

        // frequency per bin
        let freqs: Vec<f32> = (0..half_len)
            .map(|i| (i as f32) * (sr as f32) / (slot_len as f32))
            .collect();

        let sender = self.sender.clone();
        let mut prev_freq = 0.0_f32;

        // --- parameters you can tune ---
        let hop_size = slot_len / 4; // 75% overlap by default
        let hps_levels = 3usize; // HPS depth (2..4 typical)
        let snr_db_threshold = 60.0; // require peak to be X dB above median noise floor
        let min_power = 1e-10f32; // numerical floor for log
        // --------------------------------
        let mut freq_buffer: VecDeque<f32> = VecDeque::with_capacity(5);
        // circular_frame_buffer collects samples from consecutive slots; we will produce frames with step=hop_size
        let mut circular_frame_buffer: Vec<f32> = Vec::with_capacity(slot_len * 4);

        thread::spawn(move || {
            while state.load(Ordering::Relaxed) != -1 || !cons.is_empty() {
                // pause handling: we still pop but release quickly (matching your prior logic)
                if state.load(Ordering::Relaxed) == 0 {
                    match cons.pop() {
                        Ok(idx) => {
                            if slots.release(idx) {
                                let _ = reclaim.send(idx);
                            }
                        }
                        Err(_) => thread::sleep(Duration::from_millis(5)),
                    }
                    continue;
                }

                match cons.pop() {
                    Ok(idx) => {
                        debug_assert!(idx < slots.slots.len());
                        unsafe {
                            // append the incoming slot samples to the circular buffer
                            let slot_slice: &mut [f32] = &mut *slots.slots[idx].get();
                            circular_frame_buffer.extend_from_slice(slot_slice);
                        }

                        // release the slot ASAP (we've copied data into our circular buffer)
                        if slots.release(idx) {
                            let _ = reclaim.send(idx);
                        }

                        // while enough samples for one FFT frame, process (overlap via hop_size)
                        while circular_frame_buffer.len() >= slot_len {
                            // copy next frame (reuse windowed buffer)
                            for i in 0..slot_len {
                                windowed[i] = circular_frame_buffer[i] * hann[i];
                            }

                            // compute FFT (your FftProcessor API)
                            output.copy_from_slice(fft.process_forward(&mut windowed));

                            // Compute magnitude (power) for half spectrum:
                            // use magnitude^2 (power) for HPS and SNR calculations
                            let mut mag_power: Vec<f32> = vec![0.0f32; half_len];
                            for (k, c) in output.iter().enumerate().take(half_len) {
                                mag_power[k] = c.re * c.re + c.im * c.im + min_power; // add floor
                            }

                            // Basic noise floor estimate: median power across bins (robust to outliers)
                            // Simple median implementation: clone & sort small vector (half_len typically small, e.g., 512)
                            let mut mags_for_med = mag_power.clone();
                            mags_for_med.sort_by(|a, b| a.partial_cmp(b).unwrap());
                            let median_power = mags_for_med[mags_for_med.len() / 2].max(min_power);
                            let peak_bin = {
                                // find index of max power (ignore DC bin 0)
                                let mut max_i = 1usize;
                                let mut max_v = mag_power[1];
                                for i in 2..half_len {
                                    if mag_power[i] > max_v {
                                        max_v = mag_power[i];
                                        max_i = i;
                                    }
                                }
                                max_i
                            };

                            // SNR-like check: require peak to be sufficiently above median
                            let peak_power = mag_power[peak_bin].max(min_power);
                            let peak_db = 10.0 * (peak_power / min_power).log10();
                            let median_db = 10.0 * (median_power / min_power).log10();
                            if (peak_db - median_db) < snr_db_threshold || peak_bin <= 1 {
                                // not a reliable spectral peak; slide hop and continue
                                // drop hop_size samples from buffer to move window forward
                                circular_frame_buffer.drain(0..hop_size);
                                continue;
                            }

                            // --- Harmonic Product Spectrum (HPS) to favor fundamentals over harmonics ---
                            // Build HPS by multiplying downsampled spectral magnitudes (use power)
                            let mut hps = mag_power.clone(); // start with original power
                            for h in 2..=hps_levels {
                                for i in 0..(half_len / h) {
                                    // multiply by downsampled bin (product)
                                    hps[i] *= mag_power[i * h];
                                }
                            }
                            // find peak in HPS (ignore DC)
                            let mut hps_peak = 1usize;
                            let mut hps_max = hps[1];
                            for i in 2..(half_len / hps_levels) {
                                if hps[i] > hps_max {
                                    hps_max = hps[i];
                                    hps_peak = i;
                                }
                            }

                            // Use HPS peak as candidate for refinement (prefer this to raw peak)
                            let candidate_bin = hps_peak;

                            // Quadratic interpolation (parabolic) on log-magnitude around candidate_bin to get sub-bin peak
                            // we'll build `y` values from log(power) to better match gaussian-like shape
                            let y_minus =
                                (mag_power[candidate_bin.saturating_sub(1)].max(min_power)).ln();
                            let y0 = (mag_power[candidate_bin].max(min_power)).ln();
                            let y_plus = (mag_power[candidate_bin + 1.min(half_len - 1)]
                                .max(min_power))
                            .ln();

                            // handle edge conditions - if we're at edges, fall back to simple bin center
                            let delta = {
                                let denom = (y_minus - 2.0 * y0 + y_plus);
                                if denom.abs() < 1e-12 {
                                    0.0f32
                                } else {
                                    0.5 * (y_minus - y_plus) / denom
                                }
                            };

                            let refined_bin = (candidate_bin as f32) + delta;
                            let freq = refined_bin * (sr as f32) / (slot_len as f32);

                            // Optional: extra heuristic: if HPS suggested a lower fundamental but very weak, we can compare
                            // the HPS peak energy vs the original peak energy (not necessary in many cases)

                            if freq_buffer.len() >= 5 {
                                freq_buffer.pop_front();
                            }
                            freq_buffer.push_back(freq);
                            let freq =
                                freq_buffer.iter().copied().sum::<f32>() / freq_buffer.len() as f32;
                            // Send note if frequency changed
                            if (freq - prev_freq).abs() >= 0.5 {
                                prev_freq = freq;
                                let _ = sender.send(Note::from_freq(freq, base_freq));
                            }

                            // advance by hop_size (overlap)
                            circular_frame_buffer.drain(0..hop_size);
                        } // end while enough samples
                    } // Ok(idx)
                    Err(_) => {
                        // nothing available
                        thread::sleep(Duration::from_millis(3));
                    }
                } // match cons.pop
            } // main while
        });
    }
    fn hann_window(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = (i as f32) / (n as f32);
                0.5 - 0.5 * (2.0 * std::f32::consts::PI * x).cos()
            })
            .collect()
    }
}
