use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicI8, AtomicI64, Ordering},
    },
    thread,
    time::Duration,
};

use crossbeam_channel::Sender;
use rtrb::{Consumer, Producer};

use crate::{
    audio_io::SlotPool,
    audio_io::dynamics::{DynamicLevel, DynamicsOutput},
    audio_io::timing::{MusicalTransport, OnsetEvent},
    dsp::fft::FftProcessor,
};

/// Detects onsets from incoming audio buffers and reports beat timestamps.
pub struct OnsetDetector {
    state: Arc<AtomicI8>,
    handle: u8,
    reducer_remove_tx: Sender<u8>,
}

impl OnsetDetector {
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

impl Drop for OnsetDetector {
    fn drop(&mut self) {
        self.state.store(-1, Ordering::Relaxed);
        let _ = self.reducer_remove_tx.send(self.handle);
        println!("Dropped onset detector");
    }
}

/// Tracks a dynamic spectral-flux threshold for onset detection.
struct FluxTracker {
    threshold: f32,
    multiplier: f32,
    rise_memory: f32,
    decay_memory: f32,
}

impl FluxTracker {
    /// Create a new tracker with multiplier and memory settings.
    fn new(multiplier: f32, rise_memory: f32, decay_memory: f32) -> Self {
        Self {
            threshold: 0.0,
            multiplier,
            rise_memory,
            decay_memory,
        }
    }

    /// Update the tracker with the latest flux value and return true if an onset occurred.
    fn update(&mut self, current_flux: f32) -> bool {
        let memory = if current_flux > self.threshold {
            self.rise_memory
        } else {
            self.decay_memory
        };

        let is_onset = current_flux > self.threshold;

        self.threshold = self.threshold * memory + current_flux * (1.0 - memory);

        if self.threshold < 0.5 {
            self.threshold = 0.5;
        }

        is_onset && current_flux > (self.threshold * self.multiplier)
    }
}

impl OnsetDetector {
    /// Construct a new `OnsetDetector` with the provided worker handle.
    pub fn new(handle: u8, reducer_remove_tx: Sender<u8>) -> Self {
        OnsetDetector {
            state: Arc::new(AtomicI8::new(0)),
            handle,
            reducer_remove_tx,
        }
    }

    /// Spawn the detection thread which reads audio slots and pushes
    /// latency-compensated `OnsetEvent`s.
    ///
    /// - `transport` provides timing information and latency compensation.
    /// - `slots` is the shared buffer pool.
    /// - `cons` receives available slot indices.
    /// - `reclaim` is used to return slots to the free pool.
    /// - `onset_tx` receives detected onsets as `OnsetEvent`.
    pub fn detect_onsets(
        &mut self,
        transport: Arc<MusicalTransport>,
        slots: Arc<SlotPool>,
        mut cons: Consumer<usize>,
        reclaim: Sender<usize>,
        mut onset_tx: Producer<OnsetEvent>,
        onset_pending: Arc<AtomicBool>,
        dynamics_output: Arc<parking_lot::RwLock<DynamicsOutput>>,
        calibration_target: Arc<AtomicI64>,
    ) {
        self.state.store(1, Ordering::Relaxed);
        let state = self.state.clone();

        // Smaller window and hop than a pitch-tracking STFT — we don't need
        // frequency resolution, only when the spectrum changed.  256/64 at
        // 48 kHz gives ~5.3 ms windows and ~1.3 ms detection granularity,
        // halving the centre-of-window reporting delay vs the prior 512/128.
        let window_size = 256;
        let hop_size = 64;
        let ring_buffer_len = 4096;
        let half_size = window_size / 2 + 1;

        // ── Calibration state ─────────────────────────────────────────────
        // If the transport hasn't been calibrated this session, the first
        // onset that follows a published `calibration_target` frame is used
        // to compute the residual round-trip latency the OS doesn't report.
        // After that the detector behaves normally.
        let calibration_initially_done = transport.is_calibrated();
        let calibration_start_frame = transport.get_output_frames();
        // 2 seconds of slack — the click is scheduled ~200 ms ahead and the
        // round-trip can add up to ~200 ms more on Windows.
        let calibration_timeout_samples = transport.get_sample_rate() as i64 * 2;

        thread::spawn(move || {
            let mut calibration_done = calibration_initially_done;
            let hann = Self::hann_window(window_size);
            let mut fft_processor = FftProcessor::new(window_size);

            let mut ring_buffer: Vec<f32> = vec![0.0; ring_buffer_len];
            let mut ring_write_pos = 0;
            let mut ring_read_pos = 0;
            let mut available_samples = 0;

            let mut time_domain_window: Vec<f32> = vec![0.0; window_size];
            let mut prev_magnitude: Vec<f32> = vec![0.0; half_size];

            // Tracker memory tuned for hop=64: memory_new = memory_old^(64/128)
            // preserves the time constants of the prior 512/128 setup.
            let mut tracker = FluxTracker::new(1.5, 0.84, 0.89);

            let min_inter_onset_samples = 1024;
            let mut samples_since_onset = min_inter_onset_samples;

            // ── Energy EMA for harmonic-variation gate ────────────────────
            // Tracks the recent background energy level.  A genuine onset
            // must show a rapid energy increase relative to this baseline;
            // sustained notes with shifting harmonics do not.
            // Memories rescaled for the smaller hop (same TC as before).
            let mut energy_ema: f32 = 0.0;
            const ENERGY_EMA_RISE: f32 = 0.84;
            const ENERGY_EMA_DECAY: f32 = 0.95;

            // ── Per-bin delayed-prior noise floor ─────────────────────────
            // Mirrors the STFT pipeline's per-bin floor, except on non-silent
            // frames the floor is allowed to rise (slowly) but never fall.
            // It therefore lags the active spectrum, acting as a delayed
            // prior frame: any bin whose magnitude jumps far above its floor
            // is evidence of a fresh onset, even mid-phrase.
            let mut noise_floor_per_bin: Vec<f32> = vec![0.0; half_size];
            let mut floor_initialized = false;
            let mut floor_silent_count: u32 = 0;
            const FLOOR_ATTACK: f32 = 0.2; // silent: mag > floor → rise fast
            const FLOOR_RELEASE: f32 = 0.1; // silent: mag < floor → fall slower
            const FLOOR_RISE_ACTIVE: f32 = 0.1; // active: rise only, slower

            // ── Goal 1: static parameters for output-event gate ───────────
            // Suppress onsets whose input-frame position falls inside the
            // acoustic-echo window of the most recently rendered device tick.
            let sr = transport.get_sample_rate() as i64;
            // 100 ms of acoustic travel margin (generous for typical room sizes).
            let acoustic_margin_samples = sr / 10;
            // Metronome clicks decay over ~100 ms; allow 150 ms to be safe.
            let click_duration_samples = sr * 15 / 100;

            while state.load(Ordering::Relaxed) != -1 || !cons.is_empty() {
                if state.load(Ordering::Relaxed) == 0 || state.load(Ordering::Relaxed) == -1 {
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

                let mut new_data = false;
                while let Ok(idx) = cons.pop() {
                    debug_assert!(idx < slots.slots.len());

                    unsafe {
                        let slot_slice = &*slots.slots[idx].get();
                        for &sample in slot_slice.iter() {
                            ring_buffer[ring_write_pos] = sample;
                            ring_write_pos = (ring_write_pos + 1) % ring_buffer_len;
                            available_samples += 1;
                        }
                    }
                    if slots.release(idx) {
                        let _ = reclaim.send(idx);
                    }
                    new_data = true;
                }

                if available_samples > ring_buffer_len {
                    let excess = available_samples - ring_buffer_len;
                    ring_read_pos = (ring_read_pos + excess) % ring_buffer_len;
                    available_samples = ring_buffer_len;
                }

                if !new_data && available_samples < window_size {
                    thread::sleep(Duration::from_millis(1));
                    continue;
                }

                // ── Goal 1: output-event suppression window ───────────────
                // Compute once per batch; transport values only change at
                // buffer boundaries so this is accurate enough.
                let out_lat = transport.get_output_latency_samples();
                let in_lat = transport.get_input_latency_samples();
                let last_tick = transport.get_last_tick_output_frame();
                let current_input = transport.get_input_frames();
                // Earliest frame at which the click echo can arrive at the mic:
                // last_tick + output hardware buffer + input hardware buffer.
                // The echo can arrive any time in [echo_start, echo_start +
                // acoustic_margin_samples], then decays over click_duration_samples.
                let echo_start = last_tick + out_lat + in_lat;
                let echo_end = echo_start + acoustic_margin_samples + click_duration_samples;
                let suppressed_by_output = current_input >= echo_start && current_input < echo_end;

                while available_samples >= window_size {
                    for i in 0..window_size {
                        let idx = (ring_read_pos + i) % ring_buffer_len;
                        time_domain_window[i] = ring_buffer[idx] * hann[i];
                    }

                    let fft_res = fft_processor.process_forward(&mut time_domain_window);

                    let mut current_flux = 0.0;
                    let mut frame_energy = 0.0;

                    let get_smoothed_mag = |idx: usize, mags: &[f32]| -> f32 {
                        if idx == 0 || idx >= mags.len() - 1 {
                            return mags[idx];
                        }
                        (mags[idx - 1] + mags[idx] + mags[idx + 1]) / 3.0
                    };

                    let current_mags: Vec<f32> =
                        fft_res.iter().take(half_size).map(|c| c.norm()).collect();

                    for i in 0..half_size {
                        let mag = current_mags[i];
                        frame_energy += mag;

                        let smoothed_mag = get_smoothed_mag(i, &current_mags);

                        let weight = 1.0 - (i as f32 / half_size as f32);

                        let diff = smoothed_mag - prev_magnitude[i];

                        if diff > 0.0 {
                            current_flux += diff * weight;
                        }

                        prev_magnitude[i] = mag;
                    }

                    // ── Update per-bin delayed-prior floor ────────────────
                    let (noise_floor_db, dyn_level) = {
                        let d = dynamics_output.read();
                        (d.noise_floor_db, d.level)
                    };
                    let global_floor = 10.0f32.powf(noise_floor_db / 20.0) * half_size as f32 / 2.0;
                    let is_silent = dyn_level == DynamicLevel::Silence;

                    if !floor_initialized {
                        for k in 0..half_size {
                            noise_floor_per_bin[k] = current_mags[k].max(global_floor);
                        }
                        floor_initialized = true;
                    } else if is_silent {
                        floor_silent_count += 1;
                        if floor_silent_count > 60 {
                            for k in 0..half_size {
                                let mag = current_mags[k];
                                let alpha = if mag > noise_floor_per_bin[k] {
                                    FLOOR_ATTACK
                                } else {
                                    FLOOR_RELEASE
                                };
                                noise_floor_per_bin[k] +=
                                    alpha * (mag - noise_floor_per_bin[k] + 0.07);
                            }
                        }
                    } else {
                        floor_silent_count = 0;
                        // Active: rise only, never fall — delayed prior.
                        for k in 0..half_size {
                            let mag = current_mags[k];
                            if mag > noise_floor_per_bin[k] {
                                noise_floor_per_bin[k] +=
                                    FLOOR_RISE_ACTIVE * (mag - noise_floor_per_bin[k]);
                            }
                        }
                    }

                    // Per-bin burst metric vs the delayed prior.  When silent
                    // the floor has converged on the spectrum so all ratios
                    // collapse to ~1.0; during sustained playing the floor
                    // catches up only slowly, so a sudden bin spike pops out.
                    let mut max_bin_excess = 0.0f32;
                    let mut bin_burst_count: u32 = 0;
                    let floor_eps = global_floor.max(0.01);
                    for k in 0..half_size {
                        let floor_k = noise_floor_per_bin[k].max(floor_eps);
                        let r = current_mags[k] / floor_k;
                        if r > 2.5 {
                            bin_burst_count += 1;
                        }
                        if r > max_bin_excess {
                            max_bin_excess = r;
                        }
                    }

                    // Silence gate: replaces the old static frame_energy<10
                    // check.  No bins exceeding their delayed prior means the
                    // current spectrum looks like ambient noise.
                    if bin_burst_count < 2 {
                        current_flux = 0.0;
                    }

                    // ── Update energy EMA ─────────────────────────────────
                    // Asymmetric memory: rise fast so genuine attacks are
                    // compared against a baseline that hasn't caught up yet,
                    // decay slower so sustained notes keep the EMA elevated.
                    let ema_memory = if frame_energy > energy_ema {
                        ENERGY_EMA_RISE
                    } else {
                        ENERGY_EMA_DECAY
                    };
                    energy_ema = energy_ema * ema_memory + frame_energy * (1.0 - ema_memory);

                    // Combined onset trigger:
                    //  - flux-based detector for spectrum-wide rises
                    //  - bin-burst for fast attacks even mid-phrase
                    let flux_onset = tracker.update(current_flux);
                    let bin_burst_onset = max_bin_excess > 3.0 && bin_burst_count >= 4;
                    let onset_detected = flux_onset || bin_burst_onset;

                    // Calibration timeout — fall back to offset 0 if no onset
                    // arrived in time.
                    if !calibration_done {
                        let elapsed = transport.get_output_frames() - calibration_start_frame;
                        if elapsed > calibration_timeout_samples {
                            log::warn!(
                                "onset calibration timed out after {} samples — using offset 0",
                                elapsed
                            );
                            transport.set_calibration_offset(0);
                            calibration_done = true;
                        }
                    }

                    if samples_since_onset >= min_inter_onset_samples && onset_detected {
                        // Require a rapid energy increase (1.5× the baseline)
                        // — harmonic shifts in sustained notes keep the ratio
                        // near 1.0 and are therefore suppressed.
                        let energy_rising = frame_energy > energy_ema * 1.5;

                        if !suppressed_by_output && energy_rising {
                            // The onset centre is window_size/2 samples behind the
                            // current ring-buffer write position.  Negate so
                            // stamp_onset shifts the beat position backwards in time.
                            let window_centre_offset =
                                -(available_samples as i64 - window_size as i64 / 2);
                            let velocity =
                                (current_flux.max(max_bin_excess * 5.0) / 50.0).clamp(0.0, 1.0);
                            let event = transport.stamp_onset(window_centre_offset, velocity);

                            if !calibration_done {
                                let target = calibration_target.load(Ordering::Relaxed);
                                if target == 0 {
                                    // Click hasn't fired yet — pre-cal noise, ignore.
                                    log::trace!("pre-calibration onset ignored (target not set)");
                                } else {
                                    let bpm = transport.get_bpm() as f64;
                                    let sr_f64 = transport.get_sample_rate() as f64;
                                    let beats_per_sample = bpm / (60.0 * sr_f64);
                                    let target_beats = target as f64 * beats_per_sample;
                                    let residual_beats = event.beat_position - target_beats;
                                    let calibration_samples =
                                        (residual_beats / beats_per_sample).round() as i64;
                                    let residual_ms = calibration_samples as f64 * 1000.0 / sr_f64;
                                    log::info!(
                                        "onset calibration: residual={:.1}ms ({} samples) at target frame {}",
                                        residual_ms,
                                        calibration_samples,
                                        target,
                                    );
                                    transport.set_calibration_offset(calibration_samples);
                                    calibration_done = true;
                                    onset_pending.store(false, Ordering::Relaxed);
                                    samples_since_onset = 0;
                                }
                            } else {
                                log::trace!(
                                    "onset @ beat {:.4} (raw offset {}, flux={:.1}, burst={:.2}/{})",
                                    event.beat_position,
                                    event.raw_sample_offset,
                                    current_flux,
                                    max_bin_excess,
                                    bin_burst_count,
                                );

                                let _ = onset_tx.push(event);
                                onset_pending.store(true, Ordering::Relaxed);
                                samples_since_onset = 0;
                            }
                        }
                    } else {
                        samples_since_onset += hop_size;
                    }

                    ring_read_pos = (ring_read_pos + hop_size) % ring_buffer_len;
                    available_samples -= hop_size;
                }
            }
        });
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
