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

use crate::{
    audio_io::{
        SlotPool,
        dynamics::{DynamicLevel, DynamicsOutput},
    },
    dsp::fft::FftProcessor,
};

/// Tracking struct for note hysteresis.
#[derive(Clone, Debug)]
struct NoteTrack {
    freq: f32,
    score: f32,
    life: i32,
}

/// Manages the lifecycle of detected pitches across consecutive frames.
struct PitchTracker {
    tracks: Vec<NoteTrack>,
    display_threshold: i32,
    max_life: i32,
    tolerance: f32,
}

impl PitchTracker {
    fn new() -> Self {
        Self {
            tracks: Vec::new(),
            display_threshold: 2, // "once the value hits n_hits to display"
            max_life: 4,          // "max count value is n_misses to go away"
            tolerance: 0.03,      // ~3% frequency tolerance
        }
    }

    fn process(&mut self, raw_pitches: Vec<(f32, f32)>) -> Vec<(f32, f32)> {
        let mut matched = vec![false; self.tracks.len()];
        let mut active_pitches = Vec::new();

        // 1. Attempt to match incoming pitches with existing tracks
        for (raw_freq, raw_score) in raw_pitches {
            let mut found = false;
            for (i, track) in self.tracks.iter_mut().enumerate() {
                if matched[i] {
                    continue;
                }

                if (track.freq - raw_freq).abs() / track.freq < self.tolerance {
                    track.freq = track.freq * 0.6 + raw_freq * 0.4;
                    track.score = raw_score;
                    // Hit: Increase life, clamped to max_life
                    track.life = (track.life + 1).min(self.max_life);
                    matched[i] = true;
                    found = true;
                    break;
                }
            }

            // 2. If it didn't match anything, spawn a new track
            if !found {
                self.tracks.push(NoteTrack {
                    freq: raw_freq,
                    score: raw_score,
                    life: 1, // Start with 1 hit
                });
                matched.push(true);
            }
        }

        // 3. Process misses, reap dead tracks, and emit stable ones
        let mut i = 0;
        while i < self.tracks.len() {
            if !matched[i] {
                // Miss: Decrease life
                self.tracks[i].life -= 1;
            }

            if self.tracks[i].life <= 0 {
                // Decay to 0 -> Destroy track
                self.tracks.remove(i);
                if matched.len() > i {
                    matched.remove(i);
                }
            } else {
                // Only output if the track has reached the display threshold
                if self.tracks[i].life >= self.display_threshold {
                    active_pitches.push((self.tracks[i].freq, self.tracks[i].score));
                }
                i += 1;
            }
        }

        active_pitches
    }
}

/// Short-time Fourier transform worker for pitch detection.
pub struct STFT {
    state: Arc<AtomicI8>,
    handle: u8,
    reducer_remove_tx: Sender<u8>,
}

impl STFT {
    pub fn stop(&mut self) {
        self.state.store(-1, Ordering::Relaxed);
    }
    pub fn pause(&mut self) {
        self.state.store(0, Ordering::Relaxed);
    }
    pub fn resume(&mut self) {
        self.state.store(1, Ordering::Relaxed);
    }
}

impl Drop for STFT {
    fn drop(&mut self) {
        println!("Dropped stft");
        let _ = self.reducer_remove_tx.send(self.handle);
        self.stop();
    }
}

impl STFT {
    pub fn new(handle: u8, reducer_remove_tx: Sender<u8>) -> Self {
        STFT {
            handle,
            reducer_remove_tx,
            state: Arc::new(AtomicI8::new(0)),
        }
    }

    pub fn detect_pitches(
        &mut self,
        slots: Arc<SlotPool>,
        slot_len: usize,
        mut cons: rtrb::Consumer<usize>,
        reclaim: Sender<usize>,
        sr: u32,
        note_tx: Producer<Vec<(f32, f32)>>,
        dynamics_output: Arc<parking_lot::RwLock<DynamicsOutput>>,
    ) {
        // Prevent panics caused by 0-length window initialization
        if slot_len < 4 {
            println!("STFT Error: slot_len too small");
            return;
        }

        self.state.store(1, Ordering::Relaxed);
        let state = self.state.clone();

        let hop_size = slot_len / 4;
        let window_size = slot_len;
        let ring_buffer_len = 8192.max(window_size * 4);

        const MIN_FREQ: f32 = 30.0;
        const MAX_FREQ: f32 = 10_000.0;

        thread::spawn(move || {
            let mut note_tx = note_tx;

            let hann = Self::hann_window(window_size);
            let mut fft_processor = FftProcessor::new(window_size);

            let mut pitch_tracker = PitchTracker::new();

            #[cfg(feature = "dev-tools")]
            let mut rerun_frame: usize = 0;
            #[cfg(feature = "dev-tools")]
            let mut png_frame: usize = 0usize;

            #[cfg(feature = "dev-tools")]
            let rec = rerun::RecordingStreamBuilder::new("audio_analyzer")
                .spawn()
                .expect("failed to spawn Rerun viewer");

            #[cfg(feature = "dev-tools")]
            std::fs::create_dir_all("debug_frames")
                .expect("failed to create debug_frames/ directory");

            let mut ring_buffer: Vec<f32> = vec![0.0; ring_buffer_len];
            let mut ring_write_pos = 0usize;
            let mut ring_read_pos = 0usize;
            let mut available_samples = 0usize;

            let half_size = window_size / 2 + 1;
            let mut time_domain_window: Vec<f32> = vec![0.0; window_size];

            // Allocate safely using window_size to prevent bounds mismatch
            let mut complex_output: Vec<Complex<f32>> = vec![Complex::default(); window_size];

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
                let is_silence = dynamics_output.read().level == DynamicLevel::Silence;

                while let Ok(idx) = cons.pop() {
                    // Bounds checking prevents out-of-bounds on slot slices
                    if idx >= slots.slots.len() {
                        let _ = reclaim.send(idx);
                        continue;
                    }

                    unsafe {
                        let slot_slice = &*slots.slots[idx].get();
                        for &sample in slot_slice.iter() {
                            // If silence, feed zeroes to keep the clock aligned and decay seamlessly
                            ring_buffer[ring_write_pos] = if is_silence { 0.0 } else { sample };
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
                    // Debug trigger fires on every 1,000th FFT frame (dev-tools builds only).
                    #[cfg(feature = "dev-tools")]
                    let dbg_trigger = {
                        let t = rerun_frame % 1_000 == 0;
                        rerun_frame += 1;
                        t
                    };

                    // Holds captured buffers for the debug dump; only populated when triggered.
                    #[cfg(feature = "dev-tools")]
                    let mut dbg_capture: Option<(
                        Vec<f32>,
                        Vec<f32>,
                        Vec<f32>,
                        f32,
                    )> = None;

                    #[cfg(feature = "dev-tools")]
                    let mut dev_rerun_data: Option<(Vec<f32>, f32, f32)> = None;
                    #[cfg(feature = "dev-tools")]
                    let mut dev_png_data: Option<(Vec<f32>, Vec<f32>)> = None;

                    // Bypass the FFT entirely during silence
                    let raw_pitches = if is_silence {
                        Vec::new() // Feed an empty list so tracker counts it as a miss for all
                    } else {
                        // Capture pre-hann raw signal from the ring buffer.
                        #[cfg(feature = "dev-tools")]
                        let dbg_raw: Vec<f32> = if dbg_trigger {
                            (0..window_size)
                                .map(|i| ring_buffer[(ring_read_pos + i) % ring_buffer_len])
                                .collect()
                        } else {
                            Vec::new()
                        };

                        #[cfg(feature = "dev-tools")]
                        let png_raw: Option<Vec<f32>> = if png_frame % 2500 == 0 {
                            Some(
                                (0..window_size)
                                    .map(|i| ring_buffer[(ring_read_pos + i) % ring_buffer_len])
                                    .collect(),
                            )
                        } else {
                            None
                        };

                        for i in 0..window_size {
                            let idx = (ring_read_pos + i) % ring_buffer_len;
                            time_domain_window[i] = ring_buffer[idx] * hann[i];
                        }

                        // Capture hann-windowed signal.
                        #[cfg(feature = "dev-tools")]
                        let dbg_windowed: Vec<f32> = if dbg_trigger {
                            time_domain_window.clone()
                        } else {
                            Vec::new()
                        };

                        #[cfg(feature = "dev-tools")]
                        if let Some(raw_captured) = png_raw {
                            dev_png_data = Some((raw_captured, time_domain_window.clone()));
                        }

                        let fft_res = fft_processor.process_forward(&mut time_domain_window);

                        // process_len will safely never exceed complex_output's capacity
                        let process_len = fft_res.len().min(half_size).min(complex_output.len());
                        for (i, val) in fft_res.iter().enumerate().take(process_len) {
                            complex_output[i] = *val;
                        }

                        let magnitudes: Vec<f32> = complex_output
                            .iter()
                            .take(half_size)
                            .map(|c| c.norm())
                            .collect();

                        let bin_width = sr as f32 / window_size as f32;

                        // Store all captured buffers so they can be read after this block.
                        #[cfg(feature = "dev-tools")]
                        if dbg_trigger {
                            dbg_capture =
                                Some((dbg_raw, dbg_windowed, magnitudes.clone(), bin_width));
                        }

                        let noise_floor = 10.0f32
                            .powf(dynamics_output.read().noise_floor_db / 20.0)
                            * half_size as f32
                            / 2.0;
                        #[cfg(feature = "dev-tools")]
                        {
                            dev_rerun_data = Some((magnitudes.clone(), bin_width, noise_floor));
                        }
                        Self::extract_pitches(
                            &magnitudes,
                            half_size,
                            bin_width,
                            MIN_FREQ,
                            MAX_FREQ,
                            noise_floor,
                        )
                    };

                    // Send raw array to tracker (empty array decays notes, array with hits feeds notes)
                    let stable_pitches = pitch_tracker.process(raw_pitches);

                    // Rerun live viewer — log every frame.
                    #[cfg(feature = "dev-tools")]
                    if let Some((ref mags, bw, nf)) = dev_rerun_data {
                        Self::dbg_log_rerun(
                            &rec,
                            rerun_frame - 1,
                            mags,
                            bw,
                            MIN_FREQ,
                            MAX_FREQ,
                            nf,
                            &stable_pitches,
                        );
                    }

                    // PNG export — fires every 2500 frames.
                    #[cfg(feature = "dev-tools")]
                    {
                        if let Some((ref raw, ref windowed)) = dev_png_data {
                            if let Some((ref mags, bw, nf)) = dev_rerun_data {
                                if let Err(e) = Self::dbg_export_png(
                                    png_frame,
                                    raw,
                                    windowed,
                                    mags,
                                    bw,
                                    MIN_FREQ,
                                    MAX_FREQ,
                                    nf,
                                    &stable_pitches,
                                ) {
                                    eprintln!("PNG export error (frame {png_frame}): {e}");
                                }
                            }
                        }
                        png_frame += 1;
                    }

                    // Emit debug frame; borrows stable_pitches before the move below.
                    #[cfg(feature = "dev-tools")]
                    if dbg_trigger {
                        let frame_idx = rerun_frame - 1;
                        if let Some((raw, windowed, mags, bw)) = dbg_capture {
                            let min_bin = ((MIN_FREQ / bw).ceil() as usize).max(1);
                            let max_bin = ((MAX_FREQ / bw).floor() as usize)
                                .min(mags.len().saturating_sub(2));
                            let noise_floor = 10.0f32
                                .powf(dynamics_output.read().noise_floor_db / 20.0)
                                * half_size as f32
                                / 2.0;
                            // Detect all spectral peaks above noise_floor.
                            let half_sz = mags.len();
                            let mut is_peak_dbg = vec![false; half_sz];
                            let mut peak_bins_dbg: Vec<usize> = Vec::new();
                            if min_bin + 1 < max_bin {
                                for k in (min_bin + 1)..max_bin {
                                    let m = mags[k];
                                    if m > noise_floor && m >= mags[k - 1] && m >= mags[k + 1] {
                                        is_peak_dbg[k] = true;
                                        peak_bins_dbg.push(k);
                                    }
                                }
                            }

                            // Pre-scoring list: sorted by raw magnitude.
                            let mut top_peaks: Vec<(usize, f32)> =
                                peak_bins_dbg.iter().map(|&k| (k, mags[k])).collect();
                            top_peaks.sort_unstable_by(|a, b| {
                                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            top_peaks.truncate(15);

                            // Harmonic scores: replicate extract_pitches scoring.
                            // Compute all scores first so we can apply a relative cutoff.
                            // Each entry: (fundamental_bin, total_score, per-bin contributions).
                            let all_scores: Vec<(usize, f32, Vec<(usize, f32)>)> = peak_bins_dbg
                                .iter()
                                .map(|&k| {
                                    let fund_mag = mags[k];
                                    if fund_mag < noise_floor * 5.0 {
                                        return (k, 0.0, Vec::new());
                                    }
                                    // Parabolic interpolation of the fundamental.
                                    let frac_bin = if k >= 1 && k + 1 < half_sz {
                                        let y_l = mags[k - 1].ln();
                                        let y_c = mags[k].ln();
                                        let y_r = mags[k + 1].ln();
                                        let denom = y_l - 2.0 * y_c + y_r;
                                        let delta = if denom.abs() < 1e-30 {
                                            0.0
                                        } else {
                                            (0.5 * (y_l - y_r) / denom).clamp(-1.0, 1.0)
                                        };
                                        k as f32 + delta
                                    } else {
                                        k as f32
                                    };
                                    let mut score = fund_mag;
                                    // Seed contributions with the fundamental (normalised later).
                                    let mut contribs: Vec<(usize, f32)> = vec![(k, fund_mag)];
                                    let mut last = k;
                                    let mut longest_run = 0;
                                    let mut current_run = 0;
                                    for n in 2..=12usize {
                                        let expected_f = frac_bin * n as f32;
                                        if expected_f >= half_sz as f32 {
                                            break;
                                        }
                                        let lo =
                                            ((expected_f - 1.0).floor() as usize).max(last + 1);
                                        let hi =
                                            ((expected_f + 1.0).ceil() as usize).min(half_sz - 1);
                                        let mut best_mag = 0.0f32;
                                        let mut best_bin = 0usize;
                                        for h in lo..=hi {
                                            if is_peak_dbg[h] && mags[h] > best_mag {
                                                best_mag = mags[h];
                                                best_bin = h;
                                            }
                                        }
                                        // if best_mag / 5.0 > score / (n - 1) as f32 {
                                        //     contribs.push((n, 0.0));
                                        //     continue;
                                        // }
                                        if best_bin != 0 {
                                            last = best_bin;
                                            let c = best_mag;
                                            score += c;
                                            contribs.push((best_bin, c));
                                            current_run += 1;
                                        } else {
                                            contribs.push((n, 0.0));
                                            if current_run > longest_run {
                                                longest_run = current_run;
                                            }
                                            current_run = 0;
                                        }
                                    }
                                    if current_run > longest_run {
                                        longest_run = current_run;
                                    }
                                    let final_score = if longest_run < 3 {
                                        0.0
                                    } else {
                                        let log_score = (1.0 + score).log2();
                                        let struct_mult = (1.0 + longest_run as f32) / (1.0 + 12.0);
                                        log_score * struct_mult
                                    };
                                    (k, final_score, contribs)
                                })
                                .collect();
                            let max_dbg_score =
                                all_scores.iter().map(|&(_, s, _)| s).fold(0.0f32, f32::max);
                            let dbg_cutoff = max_dbg_score * 0.05;
                            let mut top_scores: Vec<(usize, f32, Vec<(usize, f32)>)> = all_scores
                                .into_iter()
                                .filter(|(_, s, _)| *s >= dbg_cutoff)
                                .collect();
                            top_scores.sort_unstable_by(|a, b| {
                                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                            });

                            Self::dbg_emit_frame(
                                frame_idx,
                                &raw,
                                &windowed,
                                &mags,
                                bw,
                                MIN_FREQ,
                                MAX_FREQ,
                                noise_floor,
                                &top_peaks,
                                &top_scores,
                                &stable_pitches,
                            );
                        } else {
                            eprintln!("\n╔══ STFT DEBUG FRAME #{frame_idx} | SILENCE ══\n");
                        }
                    }

                    // Note emission will still happen during silence as long as stable_pitches exists
                    if !stable_pitches.is_empty() {
                        let _ = note_tx.push(stable_pitches);
                    }

                    ring_read_pos = (ring_read_pos + hop_size) % ring_buffer_len;
                    available_samples -= hop_size;
                }
            }
        });
    }

    fn extract_pitches(
        magnitudes: &[f32],
        half_size: usize,
        bin_width: f32,
        min_freq: f32,
        max_freq: f32,
        noise_floor: f32,
    ) -> Vec<(f32, f32)> {
        const MAX_HARMONICS: usize = 12;
        const MAX_NOTES: usize = 8;

        let min_bin = ((min_freq / bin_width).ceil() as usize).max(1);
        let max_bin = ((max_freq / bin_width).floor() as usize).min(half_size.saturating_sub(2));

        if min_bin >= max_bin {
            return Vec::new();
        }

        let mut is_peak = vec![false; half_size];
        let mut peak_bins: Vec<usize> = Vec::new();
        for k in (min_bin + 1)..max_bin {
            let m = magnitudes[k];
            if m > noise_floor && m >= magnitudes[k - 1] && m >= magnitudes[k + 1] {
                is_peak[k] = true;
                peak_bins.push(k);
            }
        }

        if peak_bins.is_empty() {
            return Vec::new();
        }

        let mut scores = vec![0.0f32; half_size];
        let mut frac_bins = vec![0.0f32; half_size];
        for &k in &peak_bins {
            let fund_mag = magnitudes[k];
            if fund_mag < noise_floor * 5.0 {
                scores[k] = 0.0;
                continue;
            }
            // Parabolic interpolation of the fundamental to get sub-bin frequency.
            let frac_bin = if k >= 1 && k + 1 < half_size {
                let y_l = magnitudes[k - 1].ln();
                let y_c = magnitudes[k].ln();
                let y_r = magnitudes[k + 1].ln();
                let denom = y_l - 2.0 * y_c + y_r;
                let delta = if denom.abs() < 1e-30 {
                    0.0
                } else {
                    (0.5 * (y_l - y_r) / denom).clamp(-1.0, 1.0)
                };
                k as f32 + delta
            } else {
                k as f32
            };
            frac_bins[k] = frac_bin;
            let mut score = fund_mag;
            let mut last = k;
            let mut longest_run = 0;
            let mut current_run = 0;
            for n in 2..=MAX_HARMONICS {
                let expected_f = frac_bin * n as f32;
                if expected_f >= half_size as f32 {
                    break;
                }
                let search_start = ((expected_f - 1.0).floor() as usize).max(last + 1);
                let search_end = ((expected_f + 1.0).ceil() as usize).min(half_size - 1);

                let mut best_hbin = 0;
                let mut best_mag = 0.0;

                for h in search_start..=search_end {
                    if is_peak[h] && magnitudes[h] > best_mag {
                        best_mag = magnitudes[h];
                        best_hbin = h;
                    }
                }
                // if best_mag / 5.0 > score / (n - 1) as f32 {
                //     best_mag *= 0.2;
                // }
                if best_hbin != 0 {
                    score += best_mag;
                    last = best_hbin;
                    current_run += 1;
                } else {
                    if current_run > longest_run {
                        longest_run = current_run;
                    }
                    current_run = 0;
                }
            }
            if current_run > longest_run {
                longest_run = current_run;
            }
            if longest_run < 3 {
                scores[k] = 0.0;
            } else {
                const STRUCT_BASE: f32 = 1.0;
                let log_score = (1.0 + score).log2();
                let struct_mult =
                    (STRUCT_BASE + longest_run as f32) / (STRUCT_BASE + MAX_HARMONICS as f32);
                scores[k] = log_score * struct_mult;
            }
        }

        let max_score = peak_bins.iter().map(|&k| scores[k]).fold(0.0f32, f32::max);
        if max_score == 0.0 {
            return Vec::new();
        }
        let cutoff = max_score * 0.5;

        let mut candidates: Vec<(usize, f32)> = peak_bins
            .iter()
            .filter_map(|&k| {
                if scores[k] >= cutoff {
                    Some((k, scores[k]))
                } else {
                    None
                }
            })
            .collect();

        // Suppress harmonic ghosts: if freq_i ≈ N × freq_j (N=2,3,4) and
        // candidate i scores < 80% of candidate j, i is likely a ghost.
        let suppressed: Vec<bool> = (0..candidates.len())
            .map(|i| {
                let (bin_i, score_i) = candidates[i];
                let freq_i = frac_bins[bin_i] * bin_width;
                candidates.iter().enumerate().any(|(j, &(bin_j, score_j))| {
                    if i == j {
                        return false;
                    }
                    let freq_j = frac_bins[bin_j] * bin_width;
                    let ratio = freq_i / freq_j;
                    let nearest = ratio.round();
                    nearest >= 2.0
                        && nearest <= 4.0
                        && (ratio / nearest - 1.0).abs() < 0.03
                        && score_i < score_j * 0.8
                })
            })
            .collect();
        candidates = candidates
            .into_iter()
            .enumerate()
            .filter(|(idx, _)| !suppressed[*idx])
            .map(|(_, c)| c)
            .collect();

        candidates
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(MAX_NOTES);

        candidates
            .iter()
            .filter_map(|&(bin, score)| {
                // Reuse the fractional bin already computed during harmonic scoring.
                let freq = frac_bins[bin] * bin_width;
                if freq >= min_freq && freq <= max_freq {
                    Some((freq, score))
                } else {
                    None
                }
            })
            .collect()
    }

    fn hann_window(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = (i as f32) / (n as f32);
                0.5 - 0.5 * (2.0 * std::f32::consts::PI * x).cos()
            })
            .collect()
    }

    /// Convert a frequency in Hz to the nearest MIDI note name + cents deviation.
    fn freq_to_note_label(freq: f32) -> String {
        if freq <= 0.0 {
            return "?".to_string();
        }
        let midi = 69.0 + 12.0 * (freq / 440.0).log2();
        let midi_round = midi.round() as i32;
        let names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
        let name = names[((midi_round % 12 + 12) % 12) as usize];
        let octave = midi_round / 12 - 1;
        let cents = ((midi - midi.round()) * 100.0) as i32;
        if cents == 0 {
            format!("{name}{octave}")
        } else {
            format!("{name}{octave} {cents:+}¢")
        }
    }
}

// ─── Debug visualisation helpers (compiled only in debug builds) ───────────────

#[cfg(debug_assertions)]
impl STFT {
    /// ASCII waveform plot printed to stderr.
    fn dbg_plot_signal(label: &str, samples: &[f32]) {
        const W: usize = 120;
        const H: usize = 20;
        if samples.is_empty() {
            return;
        }

        let step = (samples.len() as f32 / W as f32).max(1.0);
        let pts: Vec<f32> = (0..W)
            .map(|col| {
                let s = (col as f32 * step) as usize;
                let e = (((col + 1) as f32 * step) as usize)
                    .max(s + 1)
                    .min(samples.len());
                samples[s..e].iter().copied().sum::<f32>() / (e - s) as f32
            })
            .collect();

        let min_v = pts.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_v = pts.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = (max_v - min_v).max(1e-10);

        let mut grid = vec![vec![' '; W]; H];

        // Zero line (only if zero falls within the visible range)
        if min_v <= 0.0 && max_v >= 0.0 {
            let zr =
                (H - 1).saturating_sub(((0.0 - min_v) / range * (H - 1) as f32).round() as usize);
            for c in 0..W {
                grid[zr][c] = '·';
            }
        }

        for (col, &v) in pts.iter().enumerate() {
            let row =
                (H - 1).saturating_sub(((v - min_v) / range * (H - 1) as f32).round() as usize);
            grid[row][col] = '●';
        }

        eprintln!("  ── {} ({} samples) ──", label, samples.len());
        for (i, row) in grid.iter().enumerate() {
            let v = max_v - (i as f32 / (H - 1).max(1) as f32) * range;
            eprint!("  {:+10.5} │", v);
            for &c in row {
                eprint!("{}", c);
            }
            eprintln!();
        }
        eprint!("             └");
        for _ in 0..W {
            eprint!("─");
        }
        eprintln!();
        eprintln!("              0{:>w$}", samples.len(), w = W - 1);
    }

    /// ASCII log-scale frequency bar chart with labelled tick marks, printed to stderr.
    ///
    /// The x-axis is logarithmic so low frequencies get proportionally more columns
    /// instead of being crammed into the left ~10% of a linear axis.
    fn dbg_plot_spectrum(mags: &[f32], bin_width: f32, min_freq: f32, max_freq: f32) {
        const W: usize = 120;
        const H: usize = 20;
        if mags.is_empty() {
            return;
        }

        let min_bin = ((min_freq / bin_width).ceil() as usize).max(1);
        let max_bin = ((max_freq / bin_width).floor() as usize).min(mags.len().saturating_sub(1));

        if min_bin >= max_bin {
            return;
        }

        let log_min = min_freq.ln();
        let log_max = max_freq.ln();
        let log_range = log_max - log_min;

        // Map a frequency to its column on the log scale.
        let freq_to_col = |f: f32| -> usize {
            ((f.ln() - log_min) / log_range * (W - 1) as f32)
                .round()
                .clamp(0.0, (W - 1) as f32) as usize
        };

        // Each column holds the max magnitude of all bins whose centre frequency
        // falls in that column's [f_lo, f_hi) slice of the log scale.
        let pts: Vec<f32> = (0..W)
            .map(|col| {
                let f_lo = (log_min + col as f32 / W as f32 * log_range).exp();
                let f_hi = (log_min + (col + 1) as f32 / W as f32 * log_range).exp();
                let b_lo = ((f_lo / bin_width).floor() as usize).clamp(min_bin, max_bin);
                let b_hi = ((f_hi / bin_width).ceil() as usize).clamp(min_bin, max_bin);
                mags[b_lo..=b_hi].iter().cloned().fold(0.0f32, f32::max)
            })
            .collect();

        let max_v = pts.iter().cloned().fold(0.0f32, f32::max).max(1e-10);

        let mut grid = vec![vec![' '; W]; H];
        for (col, &v) in pts.iter().enumerate() {
            let bar_h = ((v / max_v) * H as f32).round() as usize;
            for row in H.saturating_sub(bar_h)..H {
                grid[row][col] = '█';
            }
        }

        let tick_freqs: &[f32] = &[50.0, 100.0, 200.0, 500.0, 1_000.0, 2_000.0, 4_000.0];

        eprintln!(
            "  ── Frequency Spectrum (log scale, {:.0}–{:.0} Hz) ──",
            min_freq, max_freq
        );
        for (i, row) in grid.iter().enumerate() {
            let v = max_v * (1.0 - i as f32 / (H - 1).max(1) as f32);
            eprint!("  {:10.3e} │", v);
            for &c in row {
                eprint!("{}", c);
            }
            eprintln!();
        }

        // Bottom axis with tick marks at labelled frequencies.
        eprint!("             └");
        for col in 0..W {
            let is_tick = tick_freqs
                .iter()
                .filter(|&&f| f >= min_freq && f <= max_freq)
                .any(|&f| freq_to_col(f) == col);
            eprint!("{}", if is_tick { '┬' } else { '─' });
        }
        eprintln!();

        // Label row: centre each label on its tick column.
        let mut label_buf = vec![b' '; W];
        for &f in tick_freqs
            .iter()
            .filter(|&&f| f >= min_freq && f <= max_freq)
        {
            let label = if f >= 1_000.0 {
                format!("{:.0}k", f / 1_000.0)
            } else {
                format!("{:.0}", f)
            };
            let col = freq_to_col(f);
            let start = col.saturating_sub(label.len() / 2);
            for (j, b) in label.bytes().enumerate() {
                if start + j < W {
                    label_buf[start + j] = b;
                }
            }
        }
        eprintln!("              {}", String::from_utf8_lossy(&label_buf));
    }

    /// Convert a frequency in Hz to the nearest note name + cents deviation.
    fn dbg_freq_to_note(freq: f32) -> String {
        if freq <= 0.0 {
            return "?".to_string();
        }
        let midi = 69.0 + 12.0 * (freq / 440.0).log2();
        let midi_round = midi.round() as i32;
        let names = [
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
        ];
        let name = names[((midi_round % 12 + 12) % 12) as usize];
        let octave = midi_round / 12 - 1;
        let cents = ((midi - midi.round()) * 100.0) as i32;
        if cents == 0 {
            format!("{}{}", name, octave)
        } else {
            format!("{}{} {:+}¢", name, octave, cents)
        }
    }

    /// Print a full debug frame to stderr: raw signal, windowed signal,
    /// frequency spectrum, and key scalar/vector stats.
    fn dbg_emit_frame(
        frame: usize,
        raw: &[f32],
        windowed: &[f32],
        mags: &[f32],
        bin_width: f32,
        min_freq: f32,
        max_freq: f32,
        noise_floor: f32,
        top_peaks: &[(usize, f32)],
        top_scores: &[(usize, f32, Vec<(usize, f32)>)],
        stable: &[(f32, f32)],
    ) {
        eprintln!();
        eprintln!(
            "╔══════════════════════════════════════════════════════════════════════════════"
        );
        eprintln!("║  STFT DEBUG FRAME #{frame:>6}  (debug_assertions)");
        eprintln!(
            "╠══════════════════════════════════════════════════════════════════════════════"
        );
        // eprintln!("║  [1/4] Raw Signal");
        // Self::dbg_plot_signal("Raw signal", raw);
        // eprintln!(
        //     "╠══════════════════════════════════════════════════════════════════════════════"
        // );
        // eprintln!("║  [2/4] Hann-Windowed Signal");
        // Self::dbg_plot_signal("Hann-windowed signal", windowed);
        // eprintln!(
        //     "╠══════════════════════════════════════════════════════════════════════════════"
        // );
        eprintln!("║  [3/4] Frequency Spectrum");
        Self::dbg_plot_spectrum(mags, bin_width, min_freq, max_freq);
        eprintln!(
            "╠══════════════════════════════════════════════════════════════════════════════"
        );
        eprintln!("║  [4/4] Analysis Stats");
        eprintln!("║  noise_floor   : {noise_floor:.6e}");
        eprintln!("║  bin_width : {bin_width:.4} Hz/bin");
        eprintln!("║");
        eprintln!("║  Top spectral peaks (magnitude-sorted, pre-scoring):");
        if top_peaks.is_empty() {
            eprintln!("║    (none above noise_floor threshold)");
        } else {
            for &(bin, mag) in top_peaks.iter().take(10) {
                let freq = bin as f32 * bin_width;
                eprintln!(
                    "║    bin {:5}  →  {:8.2} Hz  ({:>9})  mag {:.4e}",
                    bin,
                    freq,
                    Self::dbg_freq_to_note(freq),
                    mag
                );
            }
        }
        eprintln!("║");
        eprintln!("║  Top harmonic scores (5% of max, score-sorted):");
        if top_scores.is_empty() {
            eprintln!("║    (none)");
        } else {
            for (bin, score, contribs) in top_scores {
                let freq = *bin as f32 * bin_width;
                let contrib_str = contribs
                    .iter()
                    .map(|&(b, v)| format!("({b}, {v:.2e})"))
                    .collect::<Vec<_>>()
                    .join(", ");
                eprintln!(
                    "║    bin {:5}  →  {:8.2} Hz  ({:>9})  score {:.3}  [{}]",
                    bin,
                    freq,
                    Self::dbg_freq_to_note(freq),
                    score,
                    contrib_str
                );
            }
        }
        eprintln!("║");
        eprintln!("║  Stable pitches (after hysteresis tracker):");
        if stable.is_empty() {
            eprintln!("║    (none)");
        } else {
            for &(freq, score) in stable {
                eprintln!(
                    "║    {:8.2} Hz  ({:>9})  score {:.3}",
                    freq,
                    Self::dbg_freq_to_note(freq),
                    score
                );
            }
        }
        eprintln!(
            "╚══════════════════════════════════════════════════════════════════════════════"
        );
        eprintln!();
    }
}

// ─── dev-tools visualization helpers ─────────────────────────────────────────

#[cfg(feature = "dev-tools")]
impl STFT {
    /// Log one FFT frame to the Rerun live viewer.
    ///
    /// Three entities are logged:
    /// - `spectrum/line`        — full magnitude spectrum as a line strip (log10-freq x-axis)
    /// - `spectrum/noise_floor` — horizontal line at the noise floor magnitude
    /// - `spectrum/pitches`     — labeled points at each hysteresis-stable pitch
    fn dbg_log_rerun(
        rec: &rerun::RecordingStream,
        frame: usize,
        mags: &[f32],
        bin_width: f32,
        min_freq: f32,
        max_freq: f32,
        noise_floor: f32,
        stable: &[(f32, f32)],
    ) {
        rec.set_time_sequence("frame", frame as i64);

        let min_bin = ((min_freq / bin_width).ceil() as usize).max(1);
        let max_bin =
            ((max_freq / bin_width).floor() as usize).min(mags.len().saturating_sub(1));

        // Spectrum line — (log10(freq), magnitude) per bin.
        let spectrum_strip: Vec<[f32; 2]> = (min_bin..=max_bin)
            .map(|b| [(b as f32 * bin_width).log10(), mags[b]])
            .collect();
        let _ = rec.log(
            "spectrum/line",
            &rerun::LineStrips2D::new([spectrum_strip])
                .with_colors([rerun::Color::from_rgb(100, 200, 255)]),
        );

        // Noise floor — horizontal line across the full frequency range.
        let x_lo = min_freq.log10();
        let x_hi = max_freq.log10();
        let _ = rec.log(
            "spectrum/noise_floor",
            &rerun::LineStrips2D::new([vec![[x_lo, noise_floor], [x_hi, noise_floor]]])
                .with_colors([rerun::Color::from_rgb(255, 80, 80)]),
        );

        // Stable pitches — labeled points on the spectrum line.
        if !stable.is_empty() {
            let positions: Vec<[f32; 2]> = stable
                .iter()
                .map(|&(freq, _)| {
                    let bin = (freq / bin_width).round() as usize;
                    let mag = mags[bin.min(mags.len().saturating_sub(1))];
                    [freq.log10(), mag]
                })
                .collect();
            let labels: Vec<String> = stable
                .iter()
                .map(|&(freq, score)| {
                    format!("{} {:.1}", Self::freq_to_note_label(freq), score)
                })
                .collect();
            let _ = rec.log(
                "spectrum/pitches",
                &rerun::Points2D::new(positions)
                    .with_labels(labels)
                    .with_colors([rerun::Color::from_rgb(255, 220, 50)])
                    .with_radii([rerun::Radius::new_scene_units(0.003)]),
            );
        }
    }

    /// Export a three-panel PNG to `debug_frames/frame_{n}.png`.
    ///
    /// Panel 1 — Raw Signal (time domain)
    /// Panel 2 — Hann-Windowed Signal (same y-scale as panel 1)
    /// Panel 3 — Log-scale FFT spectrum with noise floor + pitch labels
    fn dbg_export_png(
        frame: usize,
        raw: &[f32],
        windowed: &[f32],
        mags: &[f32],
        bin_width: f32,
        min_freq: f32,
        max_freq: f32,
        noise_floor: f32,
        stable: &[(f32, f32)],
    ) -> Result<(), Box<dyn std::error::Error>> {
        use plotters::prelude::*;

        let path = format!("debug_frames/frame_{frame}.png");
        let root = BitMapBackend::new(&path, (1200, 1800)).into_drawing_area();
        root.fill(&RGBColor(20, 20, 30))?;

        let areas = root.split_evenly((3, 1));
        let (area1, area2, area3) = (&areas[0], &areas[1], &areas[2]);

        // ── Panel 1: Raw Signal ──────────────────────────────────────────────────
        let raw_min = raw.iter().cloned().fold(f32::INFINITY, f32::min);
        let raw_max = raw.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let y_pad = (raw_max - raw_min).abs() * 0.05;
        let y_lo = raw_min - y_pad;
        let y_hi = raw_max + y_pad;
        let (y_lo, y_hi) = if (y_hi - y_lo).abs() < 1e-10 {
            (y_lo - 1.0, y_hi + 1.0)
        } else {
            (y_lo, y_hi)
        };

        {
            let mut chart = ChartBuilder::on(area1)
                .caption(
                    format!("Raw Signal — Frame {frame}"),
                    ("sans-serif", 18).into_font().color(&WHITE),
                )
                .margin(15)
                .x_label_area_size(30)
                .y_label_area_size(70)
                .build_cartesian_2d(0usize..raw.len(), y_lo..y_hi)?;

            chart
                .configure_mesh()
                .x_labels(5)
                .y_labels(5)
                .label_style(("sans-serif", 11).into_font().color(&WHITE))
                .axis_style(RGBColor(100, 100, 120))
                .draw()?;

            chart.draw_series(LineSeries::new(
                raw.iter().enumerate().map(|(i, &v)| (i, v)),
                ShapeStyle::from(&WHITE).stroke_width(1),
            ))?;
        }

        // ── Panel 2: Hann-Windowed Signal (shared y-range with Panel 1) ─────────
        {
            let mut chart = ChartBuilder::on(area2)
                .caption(
                    "Hann-Windowed Signal",
                    ("sans-serif", 18).into_font().color(&WHITE),
                )
                .margin(15)
                .x_label_area_size(30)
                .y_label_area_size(70)
                .build_cartesian_2d(0usize..windowed.len(), y_lo..y_hi)?;

            chart
                .configure_mesh()
                .x_labels(5)
                .y_labels(5)
                .label_style(("sans-serif", 11).into_font().color(&WHITE))
                .axis_style(RGBColor(100, 100, 120))
                .draw()?;

            chart.draw_series(LineSeries::new(
                windowed.iter().enumerate().map(|(i, &v)| (i, v)),
                ShapeStyle::from(&RGBColor(100, 200, 255)).stroke_width(1),
            ))?;
        }

        // ── Panel 3: Log-scale FFT Spectrum ──────────────────────────────────────
        let min_bin = ((min_freq / bin_width).ceil() as usize).max(1);
        let max_bin =
            ((max_freq / bin_width).floor() as usize).min(mags.len().saturating_sub(1));
        let x_lo = min_freq.log10();
        let x_hi = max_freq.log10();
        let spec_max = mags[min_bin..=max_bin]
            .iter()
            .cloned()
            .fold(0.0f32, f32::max)
            .max(noise_floor * 2.0);

        {
            let mut chart = ChartBuilder::on(area3)
                .caption(
                    "FFT Spectrum — Detected Pitches",
                    ("sans-serif", 18).into_font().color(&WHITE),
                )
                .margin(15)
                .x_label_area_size(40)
                .y_label_area_size(70)
                .build_cartesian_2d(x_lo..x_hi, 0f32..spec_max)?;

            chart
                .configure_mesh()
                .x_labels(8)
                .x_label_formatter(&|x| {
                    let hz = 10f32.powf(*x);
                    if hz >= 1_000.0 {
                        format!("{:.0}k", hz / 1_000.0)
                    } else {
                        format!("{hz:.0}")
                    }
                })
                .y_labels(5)
                .label_style(("sans-serif", 11).into_font().color(&WHITE))
                .axis_style(RGBColor(100, 100, 120))
                .draw()?;

            // Spectrum line.
            chart.draw_series(LineSeries::new(
                (min_bin..=max_bin)
                    .map(|b| ((b as f32 * bin_width).log10(), mags[b])),
                ShapeStyle::from(&RGBColor(100, 200, 255)).stroke_width(1),
            ))?;

            // Noise floor — solid red line.
            chart
                .draw_series(LineSeries::new(
                    vec![(x_lo, noise_floor), (x_hi, noise_floor)],
                    ShapeStyle::from(&RED).stroke_width(1),
                ))?
                .label("noise floor")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 15, y)], RED));

            // Pitch labels and circles.
            for &(freq, score) in stable {
                if freq < min_freq || freq > max_freq {
                    continue;
                }
                let x = freq.log10();
                let bin = (freq / bin_width).round() as usize;
                let y = mags[bin.min(mags.len().saturating_sub(1))];
                let y_label = y + spec_max * 0.04;

                chart.draw_series(std::iter::once(Circle::new(
                    (x, y),
                    5i32,
                    YELLOW.filled(),
                )))?;
                chart.draw_series(std::iter::once(Text::new(
                    format!("{} {score:.1}", Self::freq_to_note_label(freq)),
                    (x, y_label),
                    ("sans-serif", 11).into_font().color(&YELLOW),
                )))?;
            }

            chart
                .configure_series_labels()
                .label_font(("sans-serif", 11).into_font().color(&WHITE))
                .draw()?;
        }

        root.present()?;
        Ok(())
    }
}
