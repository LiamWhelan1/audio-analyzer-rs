use std::{
    collections::VecDeque,
    ffi::CString,
    sync::{
        Arc,
        atomic::{AtomicI8, Ordering},
    },
    thread,
    time::Duration,
};

use crossbeam_channel::Sender;
use rustfft::num_complex::Complex;

use crate::{AnalysisCallback, generators::Note};
use crate::{audio_io::SlotPool, dsp::fft::FftProcessor, traits::Worker};

pub struct STFT {
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
    pub fn new(handle: u8) -> Self {
        STFT {
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
        callback: AnalysisCallback,
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
        let mut prev_freq = 0.0_f32;

        // --- parameters you can tune ---
        let hop_size = slot_len / 4; // 75% overlap by default
        let hps_levels = 2usize; // HPS depth (2..4 typical)
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
                                let note = Note::from_freq(freq, None);
                                let name = CString::new(note.get_name()).unwrap();
                                let cents = note.get_cents();
                                let _ = callback(name.as_ptr(), cents);
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
    pub fn pitches(
        &mut self,
        slots: Arc<SlotPool>,
        slot_len: usize,
        mut cons: rtrb::Consumer<usize>,
        reclaim: Sender<usize>,
        sr: u32,
        sender: Sender<Vec<f32>>,
    ) {
        self.state.store(1, Ordering::Relaxed);
        let state = self.state.clone();

        // --- Tunable parameters ---
        let hop = slot_len / 4; // hop size (75% overlap)
        let ring_len = slot_len * 4; // length of circular buffer in samples
        const MAX_PEAKS: usize = 10; // how many local peaks to consider
        const MAX_FUNDAMENTALS: usize = 6; // how many fundamentals to keep
        const HARM_TOL: f32 = 0.2; // harmonic tolerance (fractional) e.g. 10%
        const MIN_PEAK_DB: f32 = -80.0; // dynamic range floor (dB) for peaks
        const MIN_POWER: f32 = 1e-12;
        const TRACK_ALPHA: f32 = 0.8; // EMA smoothing factor for tracked freq
        const TRACK_DECAY: i32 = 3; // frames before dropping an unmatched track
        const STABLE_HITS: i32 = 3; // hits required before considering "stable"
        // -----------------------------

        // precompute hann window (reuse)
        let hann = Self::hann_window(slot_len);

        // FFT processor and buffers
        let mut fft = FftProcessor::new(slot_len);
        let half_len = slot_len / 2 + 1;
        let freqs: Vec<f32> = (0..half_len)
            .map(|i| (i as f32) * (sr as f32) / (slot_len as f32))
            .collect();

        // ring buffer for incoming samples
        let mut ring = vec![0.0_f32; ring_len];
        let mut ring_idx: usize = 0;

        // temp arrays reused
        let mut windowed: Vec<f32> = vec![0.0_f32; slot_len];
        let mut spectrum: Vec<rustfft::num_complex::Complex<f32>> =
            vec![rustfft::num_complex::Complex { re: 0.0, im: 0.0 }; half_len];
        let mut temp_output_full: Vec<rustfft::num_complex::Complex<f32>> =
            vec![rustfft::num_complex::Complex { re: 0.0, im: 0.0 }; half_len];

        // tracking structure for stable notes (keeps frequency, hit_count, decay counter)
        #[derive(Clone, Debug)]
        struct Track {
            freq: f32,
            hits: i32,
            life: i32, // counts down when unmatched
        }
        let mut tracked_notes: Vec<Track> = Vec::new();

        let sender = sender.clone();
        let mut prev_sent_freqs: Vec<f32> = Vec::new(); // optionally used to reduce duplicate sends

        thread::spawn(move || {
            while state.load(Ordering::Relaxed) != -1 || !cons.is_empty() {
                if state.load(Ordering::Relaxed) == 0 {
                    // paused: quickly release slots
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
                            let slot_slice: &mut [f32] = &mut *slots.slots[idx].get();
                            // push samples into ring buffer (mono assumed)
                            for &s in slot_slice.iter() {
                                ring[ring_idx] = s;
                                ring_idx += 1;
                                if ring_idx >= ring_len {
                                    ring_idx = 0;
                                }
                            }
                        }
                        // release the slot ASAP
                        if slots.release(idx) {
                            let _ = reclaim.send(idx);
                        }

                        // decide whether to run an analysis frame: if we've advanced by hop from previous analysis point
                        // We'll do analysis whenever ring contains at least slot_len samples and ring_idx % hop == 0
                        // (this gives roughly aligned hop timing)
                        // compute available samples (we assume we always keep ring filled enough); check simple condition:
                        if (ring_len >= slot_len) && (ring_idx % hop == 0) {
                            // copy slot_len previous samples ending at ring_idx into windowed, applying hann
                            let start = if ring_idx >= slot_len {
                                ring_idx - slot_len
                            } else {
                                ring_len + ring_idx - slot_len
                            };
                            for i in 0..slot_len {
                                let pos = (start + i) % ring_len;
                                windowed[i] = ring[pos] * hann[i];
                            }

                            // run FFT (your FftProcessor likely expects full-length input and returns half+1 complex bins)
                            // if FftProcessor returns full-length complex output we convert to half_len slice. Adjust if needed.
                            temp_output_full.copy_from_slice(fft.process_forward(&mut windowed));
                            // build half-spectrum (0..half_len-1)
                            for k in 0..half_len {
                                spectrum[k] = temp_output_full[k];
                            }

                            // build power spectrum (magnitude^2) and compute log-power for interpolation
                            let mut power: Vec<f32> = vec![0.0f32; half_len];
                            for k in 0..half_len {
                                let c = spectrum[k];
                                power[k] = c.re * c.re + c.im * c.im + MIN_POWER;
                            }

                            // Simple dynamic threshold: compute median power to estimate noise floor
                            let mut med = power.clone();
                            med.sort_by(|a, b| a.partial_cmp(b).unwrap());
                            let med_val = med[med.len() / 2].max(MIN_POWER);
                            // we'll use dB relative to median
                            let med_db = 10.0 * med_val.log10();

                            // collect local peaks (ignore DC bin 0)
                            let mut peaks: Vec<(usize, f32)> = Vec::new(); // (bin_idx, power)
                            for i in 2..(half_len - 2) {
                                let p = power[i];
                                // local max test (simple)
                                if p > power[i - 1] && p > power[i + 1] {
                                    // dB relative to median
                                    let db_rel = 10.0 * p.log10() - med_db;
                                    if db_rel > 60.0 {
                                        // only consider peaks > 60 dB above median (tunable)
                                        peaks.push((i, p));
                                    }
                                }
                            }

                            if peaks.is_empty() {
                                // no meaningful peaks: decay tracked notes
                                for t in tracked_notes.iter_mut() {
                                    t.life -= 1;
                                }
                                tracked_notes.retain(|t| t.life > -TRACK_DECAY * 2);
                                // possibly send no-note messages here
                                continue;
                            }

                            // sort peaks by power desc and keep top MAX_PEAKS
                            peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                            if peaks.len() > MAX_PEAKS {
                                peaks.truncate(MAX_PEAKS);
                            }

                            // refine peaks with parabolic interpolation on log-power (gives sub-bin)
                            // We'll store (refined_freq, power, bin_idx)
                            let mut refined: Vec<(f32, f32, usize)> =
                                Vec::with_capacity(peaks.len());
                            for &(bin, p) in peaks.iter() {
                                // ensure neighbors exist
                                let left = power[bin.saturating_sub(1)].max(MIN_POWER);
                                let center = power[bin].max(MIN_POWER);
                                let right = power[(bin + 1).min(half_len - 1)].max(MIN_POWER);

                                // interpolate on log(power)
                                let l = left.log10();
                                let c = center.log10();
                                let r = right.log10();
                                let denom = l - 2.0 * c + r;
                                let delta = if denom.abs() < 1e-12 {
                                    0.0
                                } else {
                                    0.5 * (l - r) / denom
                                };
                                let refined_bin = (bin as f32) + delta;
                                let refined_freq = refined_bin * (sr as f32) / (slot_len as f32);
                                refined.push((refined_freq, p, bin));
                            }

                            // Now: harmonic filtering.
                            // We'll select fundamentals by iterating refined peaks in order of power descending,
                            // and skipping peaks that are near an integer multiple of an already-chosen fundamental.
                            refined.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()); // power desc
                            let mut fundamentals: Vec<(f32, f32)> = Vec::new(); // (freq, power)
                            'outer: for &(freq_cand, p, _bin) in refined.iter() {
                                // if we already have enough fundamentals, break
                                if fundamentals.len() >= MAX_FUNDAMENTALS {
                                    break;
                                }
                                // skip if cand is too low or beyond Nyquist
                                if freq_cand < 30.0 || freq_cand > (sr as f32) / 2.0 {
                                    continue;
                                }
                                // check if candidate is likely a harmonic of an already chosen fundamental
                                for &(f_base, _pb) in fundamentals.iter() {
                                    // ratio
                                    let ratio = freq_cand / f_base;
                                    let nearest_k = ratio.round();
                                    if nearest_k >= 2.0 && nearest_k <= 8.0 {
                                        let err = (ratio - nearest_k).abs();
                                        if err < HARM_TOL * nearest_k {
                                            // it's close to an integer multiple -> treat as harmonic -> skip
                                            continue 'outer;
                                        }
                                    }
                                }
                                // otherwise, accept as a (new) fundamental
                                fundamentals.push((freq_cand, p));
                            }

                            // At this point fundamentals contains candidate fundamentals (freq, power).
                            // Now update tracked_notes with these candidates via nearest matching & EMA smoothing.
                            // For matching, use relative threshold (e.g., 4% or fixed cents).
                            const MATCH_REL: f32 = 0.04; // 4% matching tolerance
                            let mut matched_this_frame = vec![false; fundamentals.len()];

                            // First, attempt to match existing tracks to incoming fundamentals
                            for track in tracked_notes.iter_mut() {
                                let mut best_idx: Option<usize> = None;
                                let mut best_err = f32::INFINITY;
                                for (j, &(f_freq, _p)) in fundamentals.iter().enumerate() {
                                    if matched_this_frame[j] {
                                        continue;
                                    }
                                    let rel_err = ((track.freq - f_freq) / f_freq).abs();
                                    if rel_err < MATCH_REL && rel_err < best_err {
                                        best_err = rel_err;
                                        best_idx = Some(j);
                                    }
                                }
                                if let Some(j) = best_idx {
                                    // match found -> update track with EMA
                                    let (f_freq, _p) = fundamentals[j];
                                    track.freq =
                                        TRACK_ALPHA * f_freq + (1.0 - TRACK_ALPHA) * track.freq;
                                    track.hits += 1;
                                    track.life = TRACK_DECAY;
                                    matched_this_frame[j] = true;
                                } else {
                                    // no match -> decay life
                                    track.life -= 1;
                                }
                            }

                            // Add any unmatched fundamentals as new tracks
                            for (j, &(f_freq, _p)) in fundamentals.iter().enumerate() {
                                if matched_this_frame[j] {
                                    continue;
                                }
                                if tracked_notes.len() < 12 {
                                    tracked_notes.push(Track {
                                        freq: f_freq,
                                        hits: 1,
                                        life: TRACK_DECAY,
                                    });
                                } else {
                                    // if too many tracks, try to replace a weak/expired one
                                    if let Some((idx_min, _)) = tracked_notes
                                        .iter()
                                        .enumerate()
                                        .min_by(|a, b| a.1.hits.cmp(&b.1.hits))
                                    {
                                        tracked_notes[idx_min] = Track {
                                            freq: f_freq,
                                            hits: 1,
                                            life: TRACK_DECAY,
                                        };
                                    }
                                }
                            }

                            // remove tracks whose life expired
                            tracked_notes.retain(|t| t.life > -TRACK_DECAY * 2);

                            // Build list of stable tracks to send out:
                            // define "stable" as hits >= STABLE_HITS OR life > 0 and hits > 0
                            tracked_notes.sort_by(|a, b| b.hits.cmp(&a.hits)); // highest-hits first
                            let mut send_freqs: Vec<f32> = tracked_notes
                                .iter()
                                .filter(|t| t.hits >= STABLE_HITS || (t.life > 0 && t.hits > 0))
                                .take(MAX_FUNDAMENTALS)
                                .map(|t| t.freq)
                                .collect();

                            // Optionally dedupe near-equal tracks and sort ascending
                            send_freqs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                            send_freqs.dedup_by(|a, b| ((*a - *b) / *b).abs() < 0.008); // 0.8% dedupe

                            // Send Notes only if different from previous sent set (coalesce small changes)
                            let mut changed = false;
                            if send_freqs.len() != prev_sent_freqs.len() {
                                changed = true;
                            } else {
                                for (i, &f) in send_freqs.iter().enumerate() {
                                    if ((f - prev_sent_freqs[i]) / f).abs() > 0.004 {
                                        // >0.4% change
                                        changed = true;
                                        break;
                                    }
                                }
                            }

                            if changed {
                                prev_sent_freqs = send_freqs.clone();
                                // send each as Note::from_freq (you may prefer to send a Vec<Note> instead)
                                let _ = sender.send(send_freqs);
                            }
                        } // end if ready to analyze
                    } // Ok(idx)
                    Err(_) => {
                        thread::sleep(Duration::from_millis(3));
                    }
                }
            } // main loop
        }); // thread spawn
    } // end fn pitch

    fn hann_window(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = (i as f32) / (n as f32);
                0.5 - 0.5 * (2.0 * std::f32::consts::PI * x).cos()
            })
            .collect()
    }
}
