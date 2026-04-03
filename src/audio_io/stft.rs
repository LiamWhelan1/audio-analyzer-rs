use std::{
    sync::{
        Arc,
        atomic::{AtomicI8, Ordering},
    },
    thread,
    time::Duration,
};

use crossbeam_channel::Sender;
use rtrb::Producer;
use rustfft::num_complex::Complex;

use crate::{audio_io::SlotPool, dsp::fft::FftProcessor, traits::Worker};

/// Short-time Fourier transform worker for pitch detection.
///
/// The STFT no longer converts frequencies to note names or invokes an
/// FFI callback.  Instead it pushes raw `Vec<(f32, i32)>` (frequency,
/// hit count) frames into an `rtrb` ring buffer that the [`Tuner`]
/// consumes on its own thread.
pub struct STFT {
    state: Arc<AtomicI8>,
    handle: u8,
}

impl Worker for STFT {
    /// Stop the worker and return its handle.
    fn stop(&mut self) -> u8 {
        self.state.store(-1, Ordering::Relaxed);
        self.handle
    }

    /// Pause the worker and return its handle.
    fn pause(&mut self) -> u8 {
        self.state.store(0, Ordering::Relaxed);
        self.handle
    }

    /// Start or resume the worker and return its handle.
    fn start(&mut self) -> u8 {
        self.state.store(1, Ordering::Relaxed);
        self.handle
    }
}

/// Tracking struct for note hysteresis.
#[derive(Clone, Debug)]
struct NoteTrack {
    freq: f32,
    hits: i32,
    misses: i32,
}

impl STFT {
    /// Create a new STFT worker with the given handle.
    pub fn new(handle: u8) -> Self {
        STFT {
            state: Arc::new(AtomicI8::new(0)),
            handle,
        }
    }

    /// Spawn pitch detection processing on a background thread.
    ///
    /// Instead of the old `AnalysisCallback` that pushed C-strings over
    /// FFI, the STFT now writes each analysis frame as a
    /// `Vec<(f32, i32)>` — raw frequency + hysteresis hit count — into
    /// `note_tx`.  The [`Tuner`] reads the other end of this channel,
    /// applies its own key/mode/system settings, and publishes a
    /// [`TunerOutput`] for the React frontend to poll.
    ///
    /// # Arguments
    ///
    /// * `slots`    – shared sample buffer pool
    /// * `slot_len` – number of samples per slot
    /// * `cons`     – receives filled slot indices from the audio thread
    /// * `reclaim`  – returns freed slot indices to the audio thread
    /// * `sr`       – sampling rate in Hz
    /// * `note_tx`  – producer end of the STFT → Tuner channel
    pub fn detect_pitches(
        &mut self,
        slots: Arc<SlotPool>,
        slot_len: usize,
        mut cons: rtrb::Consumer<usize>,
        reclaim: Sender<usize>,
        sr: u32,
        note_tx: Producer<Vec<(f32, i32)>>,
    ) {
        self.state.store(1, Ordering::Relaxed);
        let state = self.state.clone();

        let hop_size = slot_len / 4;
        let window_size = slot_len;
        let ring_buffer_len = 8192.max(window_size * 4);

        const MIN_FREQ: f32 = 25.0;
        const MAX_FREQ: f32 = 4200.0;
        const MIN_SNR_DB: f32 = 10.0;
        const ABSOLUTE_THRESHOLD: f32 = 5e-7;
        const CONFIDENCE_THRESH: f32 = 0.60;

        thread::spawn(move || {
            // We need note_tx inside the closure but must also let the
            // compiler know when the producer is dropped (on thread exit).
            let mut note_tx = note_tx;

            let hann = Self::hann_window(window_size);
            let mut fft_processor = FftProcessor::new(window_size);

            let mut ring_buffer: Vec<f32> = vec![0.0; ring_buffer_len];
            let mut ring_write_pos = 0;
            let mut ring_read_pos = 0;
            let mut available_samples = 0;

            let mut time_domain_window: Vec<f32> = vec![0.0; window_size];
            let mut complex_output: Vec<Complex<f32>> = vec![Complex::default(); window_size];

            let mut active_tracks: Vec<NoteTrack> = Vec::with_capacity(12);

            while state.load(Ordering::Relaxed) != -1 || !cons.is_empty() {
                if state.load(Ordering::Relaxed) == 0 || state.load(Ordering::Relaxed) == -1 {
                    while let Ok(idx) = cons.pop() {
                        if idx < slots.slots.len() {
                            slots.release(idx);
                        }
                        let _ = reclaim.send(idx);
                    }
                    thread::sleep(Duration::from_millis(10));
                    continue;
                }

                let mut new_data = false;
                while let Ok(idx) = cons.pop() {
                    if idx >= slots.slots.len() {
                        let _ = reclaim.send(idx);
                        continue;
                    }

                    unsafe {
                        let slot_slice = &*slots.slots[idx].get();
                        for &sample in slot_slice.iter() {
                            ring_buffer[ring_write_pos] = sample;
                            ring_write_pos = (ring_write_pos + 1) % ring_buffer_len;
                            available_samples += 1;
                        }
                    }
                    slots.release(idx);
                    let _ = reclaim.send(idx);
                    new_data = true;
                }

                if !new_data && available_samples < window_size {
                    thread::sleep(Duration::from_millis(1));
                    continue;
                }

                while available_samples >= window_size {
                    for i in 0..window_size {
                        let idx = (ring_read_pos + i) % ring_buffer_len;
                        time_domain_window[i] = ring_buffer[idx] * hann[i];
                    }

                    let fft_res = fft_processor.process_forward(&mut time_domain_window);
                    let half_size = window_size / 2 + 1;
                    let process_len = fft_res.len().min(half_size);
                    for (i, val) in fft_res.iter().enumerate().take(process_len) {
                        complex_output[i] = *val;
                    }

                    let power_spectrum: Vec<f32> = complex_output
                        .iter()
                        .take(half_size)
                        .map(|c| c.norm_sqr())
                        .collect();

                    let mut sorted_power = power_spectrum.clone();
                    sorted_power
                        .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let noise_floor = sorted_power
                        .get(half_size / 2)
                        .copied()
                        .unwrap_or(ABSOLUTE_THRESHOLD)
                        .max(ABSOLUTE_THRESHOLD);
                    let min_peak_power = noise_floor * 10.0_f32.powf(MIN_SNR_DB / 10.0);

                    let mut peaks: Vec<(usize, f32)> = Vec::new();
                    for i in 2..(half_size.saturating_sub(2)) {
                        let p = power_spectrum[i];
                        if p > min_peak_power
                            && p > power_spectrum[i - 1]
                            && p > power_spectrum[i + 1]
                        {
                            peaks.push((i, p));
                        }
                    }
                    peaks
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    peaks.truncate(10);

                    let bin_width = sr as f32 / window_size as f32;
                    let mut confirmed_candidates: Vec<f32> = Vec::new();
                    let mut is_harmonic = vec![false; peaks.len()];

                    for i in 0..peaks.len() {
                        if is_harmonic[i] {
                            continue;
                        }
                        let (bin, _) = peaks[i];

                        let y_l = power_spectrum[bin - 1].ln();
                        let y_c = power_spectrum[bin].ln();
                        let y_r = power_spectrum[bin + 1].ln();
                        let denom = y_l - 2.0 * y_c + y_r;
                        let delta = if denom.abs() < 1e-5 {
                            0.0
                        } else {
                            0.5 * (y_l - y_r) / denom
                        };

                        let mut cand_freq = (bin as f32 + delta) * bin_width;

                        if cand_freq < MIN_FREQ || cand_freq > MAX_FREQ {
                            continue;
                        }

                        let sub_harmonic = cand_freq * 0.5;
                        let sub_bin = (sub_harmonic / bin_width).round() as usize;
                        if sub_bin > 1 && sub_bin < half_size - 1 {
                            let sub_p = power_spectrum[sub_bin];
                            if sub_p > power_spectrum[bin] * 0.2 {
                                let period = sr as f32 / sub_harmonic;
                                let conf = Self::compute_autocorr(
                                    &ring_buffer,
                                    ring_read_pos,
                                    ring_buffer_len,
                                    window_size,
                                    period,
                                );
                                if conf > CONFIDENCE_THRESH {
                                    cand_freq = sub_harmonic;
                                }
                            }
                        }

                        let period_samples = sr as f32 / cand_freq;
                        let confidence = Self::compute_autocorr(
                            &ring_buffer,
                            ring_read_pos,
                            ring_buffer_len,
                            window_size,
                            period_samples,
                        );

                        if confidence > CONFIDENCE_THRESH {
                            confirmed_candidates.push(cand_freq);
                            for j in (i + 1)..peaks.len() {
                                let (bin_j, _) = peaks[j];
                                let freq_j = bin_j as f32 * bin_width;
                                let ratio = freq_j / cand_freq;
                                if (ratio - ratio.round()).abs() < 0.06 && ratio.round() >= 2.0 {
                                    is_harmonic[j] = true;
                                }
                            }
                        }
                    }

                    let num_existing_tracks = active_tracks.len();

                    let mut matched = vec![false; num_existing_tracks];

                    for freq in confirmed_candidates {
                        let mut best_idx = None;
                        let mut min_dist = f32::MAX;

                        for (k, track) in active_tracks.iter().enumerate().take(num_existing_tracks)
                        {
                            let dist = (track.freq - freq).abs();
                            if dist < track.freq * 0.04 && dist < min_dist {
                                min_dist = dist;
                                best_idx = Some(k);
                            }
                        }

                        if let Some(k) = best_idx {
                            active_tracks[k].freq = active_tracks[k].freq * 0.6 + freq * 0.4;
                            active_tracks[k].hits += 1;
                            active_tracks[k].misses = 0;
                            matched[k] = true;
                        } else {
                            if active_tracks.len() < 12 {
                                active_tracks.push(NoteTrack {
                                    freq,
                                    hits: 1,
                                    misses: 0,
                                });
                            }
                        }
                    }

                    for (k, was_matched) in matched.iter().enumerate() {
                        if !was_matched {
                            active_tracks[k].misses += 1;
                        }
                    }

                    active_tracks.retain(|t| t.misses < 4);

                    // ── NEW: send raw (freq, hits) to the Tuner ─────
                    //
                    // Previously this is where the STFT would convert
                    // frequencies to Note names via Note::from_freq,
                    // build CStrings, and fire the AnalysisCallback.
                    //
                    // Now we just forward the tracked frequencies and
                    // their confidence (hit count) as-is.  The Tuner
                    // applies key, mode, system, and base pitch on its
                    // own thread before publishing to the UI.

                    let output: Vec<(f32, i32)> = active_tracks
                        .iter()
                        .filter(|t| t.hits >= 3)
                        .map(|t| (t.freq, t.hits))
                        .collect();

                    if !output.is_empty() {
                        // try_push: if the Tuner is behind we silently
                        // drop the oldest frame — acceptable for a
                        // real-time display that only needs the latest.
                        let _ = note_tx.push(output);
                    }

                    ring_read_pos = (ring_read_pos + hop_size) % ring_buffer_len;
                    available_samples -= hop_size;
                }
            }
        });
    }

    /// Compute a normalized autocorrelation for the given lag (`period`).
    fn compute_autocorr(
        buffer: &[f32],
        start: usize,
        len: usize,
        window: usize,
        period: f32,
    ) -> f32 {
        if period < 2.0 {
            return 0.0;
        }

        let lag = period.round() as usize;

        let mut sum_prod = 0.0;
        let mut sum_sq_x = 0.0;
        let mut sum_sq_y = 0.0;

        let n = window / 2;
        let offset = window / 4;

        for i in 0..n {
            let idx_x = (start + offset + i) % len;
            let idx_y = (start + offset + i + lag) % len;

            let x = buffer[idx_x];
            let y = buffer[idx_y];

            sum_prod += x * y;
            sum_sq_x += x * x;
            sum_sq_y += y * y;
        }

        if sum_sq_x < 1e-9 || sum_sq_y < 1e-9 {
            return 0.0;
        }

        sum_prod / (sum_sq_x.sqrt() * sum_sq_y.sqrt())
    }

    /// Generate a Hann window of length `n`.
    fn hann_window(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = (i as f32) / (n as f32);
                0.5 - 0.5 * (2.0 * std::f32::consts::PI * x).cos()
            })
            .collect()
    }
}
