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
    audio_io::dynamics::DynamicsOutput,
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

        if self.threshold < 0.9 {
            self.threshold = 0.9;
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

            // ── Energy EMA for harmonic-variation gate ────────────────────
            // Tracks the recent background energy level.  A genuine onset
            // must show a rapid energy increase relative to this baseline;
            // sustained notes with shifting harmonics do not.
            // Memories rescaled for the smaller hop (same TC as before).
            let mut energy_ema: f32 = 0.0;
            const ENERGY_EMA_RISE: f32 = 0.84;
            const ENERGY_EMA_DECAY: f32 = 0.95;

            // ── Per-bin "rise-once, suppress" noise floor ─────────────────
            // Any bin whose magnitude exceeds its floor by BIN_BURST_RATIO
            // is counted as an onset contribution AND has its floor jumped
            // to mag × FLOOR_OVERCOMPENSATE — lifting the floor above the
            // burst so the sustained tail of the same onset can't re-fire.
            // Other bins keep their floors and track via an asymmetric IIR,
            // so a fresh onset on different partials (e.g. a real note
            // right after a metronome click) can still trigger immediately.
            // Same-bin retrigger is therefore *level-gated*: a louder hit
            // breaks through, an equally-loud one waits for FLOOR_DECAY to
            // bring the floor back down.
            let mut noise_floor_per_bin: Vec<f32> = vec![0.0; half_size];
            let mut floor_initialized = false;
            const BIN_BURST_RATIO: f32 = 2.5;
            const FLOOR_OVERCOMPENSATE: f32 = 1.3;
            const FLOOR_RISE: f32 = 0.1;
            const FLOOR_DECAY: f32 = 0.04;

            // ── Tick-based suppression window ─────────────────────────────
            // Any detected onset whose calibrated beat lands within
            // ±TICK_GUARD_S of a metronome tick (recorded in transport)
            // is rejected.
            const TICK_GUARD_S: f64 = 0.015;

            #[cfg(feature = "dev-tools")]
            let bin_width = transport.get_sample_rate() / window_size as f32;

            #[cfg(feature = "dev-tools")]
            let mut rerun_frame: usize = 0;
            #[cfg(feature = "dev-tools")]
            let rec = rerun::RecordingStreamBuilder::new("audio_analyzer")
                .spawn()
                .expect("failed to spawn Rerun viewer");

            // Block emission on the immediately following frame after an onset.
            // Initialized to 4 so the first frame is always "allowed".
            let mut frames_since_onset: usize = 4;

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

                while available_samples >= window_size {
                    #[cfg(feature = "dev-tools")]
                    {
                        rerun_frame += 1;
                    }
                    let mut onset_fired = false;
                    #[cfg(feature = "dev-tools")]
                    let mut dev_flux_bins: Vec<(f32, f32)> = Vec::new();
                    #[cfg(feature = "dev-tools")]
                    let mut dev_burst_bins: Vec<(f32, f32)> = Vec::new();
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
                            #[cfg(feature = "dev-tools")]
                            dev_flux_bins.push((i as f32 * bin_width, smoothed_mag));
                        }

                        prev_magnitude[i] = mag;
                    }

                    // ── Combined per-bin burst + floor update ─────────────
                    // Compute burst against the pre-update floor, then for
                    // each bin: bursting bins jump above their current
                    // magnitude (killing the sustained tail of this onset);
                    // non-bursting bins use a slow asymmetric IIR. Other
                    // bins retain their floors so a fresh onset on
                    // different partials still bursts this same frame.
                    let noise_floor_db = dynamics_output.read().noise_floor_db;
                    let global_floor = 10.0f32.powf(noise_floor_db / 20.0) * half_size as f32 / 2.0;
                    let floor_eps = global_floor.max(0.01);

                    if !floor_initialized {
                        for k in 0..half_size {
                            noise_floor_per_bin[k] = current_mags[k].max(global_floor);
                        }
                        floor_initialized = true;
                    }

                    let mut max_bin_excess = 0.0f32;
                    let mut bin_burst_count: u32 = 0;
                    for k in 0..half_size {
                        let mag = current_mags[k];
                        let floor_k = noise_floor_per_bin[k].max(floor_eps);
                        let r = mag / floor_k;

                        if r > BIN_BURST_RATIO {
                            bin_burst_count += 1;
                            #[cfg(feature = "dev-tools")]
                            dev_burst_bins.push((k as f32 * bin_width, mag));
                            noise_floor_per_bin[k] = mag * FLOOR_OVERCOMPENSATE;
                        } else if mag > noise_floor_per_bin[k] {
                            noise_floor_per_bin[k] += FLOOR_RISE * (mag - noise_floor_per_bin[k]);
                        } else {
                            noise_floor_per_bin[k] += FLOOR_DECAY * (mag - noise_floor_per_bin[k]);
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
                    let bin_burst_onset = max_bin_excess > 3.0 && bin_burst_count >= 3;
                    let onset_detected = flux_onset && bin_burst_onset;

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

                    let energy_rising = frame_energy > energy_ema * 1.5;

                    // Dev-tools-only decision telemetry. The control flow
                    // itself doesn't read these post-event; they exist to
                    // feed the rerun status label below.
                    #[cfg(feature = "dev-tools")]
                    let mut tick_dist_beats = f64::INFINITY;
                    #[cfg(feature = "dev-tools")]
                    let mut suppressed_by_tick_dev = false;

                    if onset_detected {
                        // Stamp first — the calibrated beat is what we
                        // compare against the metronome's tick history.
                        let window_centre_offset =
                            -(available_samples as i64 - window_size as i64 / 2);
                        let velocity =
                            (current_flux.max(max_bin_excess * 5.0) / 50.0).clamp(0.0, 1.0);
                        let event = transport.stamp_onset(window_centre_offset, velocity);

                        let bpm = transport.get_bpm() as f64;
                        let tick_guard_beats = TICK_GUARD_S * bpm / 60.0;
                        let tick_dist = transport.nearest_tick_distance_beats(event.beat_position);
                        let suppressed_by_tick = tick_dist < tick_guard_beats;

                        #[cfg(feature = "dev-tools")]
                        {
                            tick_dist_beats = tick_dist;
                            suppressed_by_tick_dev = suppressed_by_tick;
                        }

                        if !suppressed_by_tick && energy_rising && frames_since_onset >= 3 {
                            if !calibration_done {
                                let target = calibration_target.load(Ordering::Relaxed);
                                if target == 0 {
                                    // Click hasn't fired yet — pre-cal noise, ignore.
                                    log::trace!("pre-calibration onset ignored (target not set)");
                                } else {
                                    let sr_f64 = transport.get_sample_rate() as f64;
                                    let calibration_samples = event.output_samples - target;
                                    let residual_ms = calibration_samples as f64 * 1000.0 / sr_f64;
                                    log::info!(
                                        "beat_pos: {}, transport beat: {}, target_samples: {}, event_samples: {}",
                                        event.beat_position,
                                        transport.get_accumulated_beats(),
                                        target,
                                        event.output_samples
                                    );
                                    log::info!(
                                        "onset calibration: residual={:.1}ms ({} samples) at target frame {}",
                                        residual_ms,
                                        calibration_samples,
                                        target,
                                    );
                                    // Reject implausible residuals (negative or > 500 ms).
                                    // These indicate a spurious onset unrelated to the click.
                                    let max_cal = (sr_f64 * 0.5) as i64;
                                    if calibration_samples < 0 || calibration_samples > max_cal {
                                        log::warn!(
                                            "onset calibration: rejected implausible residual ({:.1}ms) — retrying",
                                            residual_ms,
                                        );
                                    } else {
                                        transport.set_calibration_offset(calibration_samples);
                                        calibration_done = true;
                                        onset_pending.store(false, Ordering::Relaxed);
                                        onset_fired = true;
                                    }
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
                                onset_fired = true;
                            }
                        }
                    }

                    #[cfg(feature = "dev-tools")]
                    {
                        let bpm_f = transport.get_bpm() as f64;
                        let tick_dist_ms = if tick_dist_beats.is_finite() {
                            tick_dist_beats * 60_000.0 / bpm_f
                        } else {
                            f64::INFINITY
                        };
                        let energy_ratio = if energy_ema > 1e-6 {
                            frame_energy / energy_ema
                        } else {
                            f32::INFINITY
                        };
                        let (status_label, status_color) = if onset_fired {
                            let method = match (flux_onset, bin_burst_onset) {
                                (true, true) => "flux+burst",
                                (true, false) => "flux",
                                (false, true) => "burst",
                                _ => "?",
                            };
                            (
                                format!(
                                    "DETECTED ({}) flux={:.1} burst={}",
                                    method, current_flux, bin_burst_count
                                ),
                                (0u8, 220u8, 0u8),
                            )
                        } else if onset_detected && suppressed_by_tick_dev {
                            (
                                format!("blocked: tick (Δ={:.1} ms)", tick_dist_ms),
                                (220, 80, 80),
                            )
                        } else if onset_detected && !energy_rising {
                            (
                                format!("blocked: energy ({:.2}×)", energy_ratio),
                                (220, 80, 80),
                            )
                        } else if onset_detected
                            && !suppressed_by_tick_dev
                            && energy_rising
                            && frames_since_onset < 3
                        {
                            (
                                format!("blocked: frame gate (gap={})", frames_since_onset),
                                (220, 120, 80),
                            )
                        } else if current_flux > 0.0 || bin_burst_count > 0 {
                            // Had candidate evidence but no gate fired
                            // — most commonly the flux tracker's adaptive
                            // threshold rejected it.
                            (
                                format!(
                                    "candidate: flux={:.1} (tracker rejected), burst={}/{}",
                                    current_flux,
                                    bin_burst_count,
                                    if bin_burst_count >= 3 { "≥3" } else { "<3" }
                                ),
                                (220, 180, 60),
                            )
                        } else {
                            ("idle".to_string(), (110, 110, 110))
                        };

                        Self::dbg_log_rerun(
                            &rec,
                            rerun_frame,
                            &current_mags,
                            bin_width,
                            &noise_floor_per_bin,
                            &dev_flux_bins,
                            &dev_burst_bins,
                            &status_label,
                            status_color,
                            onset_fired,
                        );
                    }

                    if onset_fired || onset_detected && frames_since_onset < 3 {
                        frames_since_onset = 0;
                    } else {
                        frames_since_onset = frames_since_onset.saturating_add(1);
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

#[cfg(feature = "dev-tools")]
impl OnsetDetector {
    /// Log onset detection data to the Rerun live viewer.
    ///
    /// Layered annotations explain the pipeline:
    /// - `spectrum/flux_bins`   yellow dots — bins contributing positive
    ///   spectral flux (would have raised `current_flux`).
    /// - `spectrum/burst_bins`  red dots    — bins that exceeded their
    ///   per-bin floor by `BIN_BURST_RATIO` and got jumped.
    /// - `spectrum/status`      colored dot + label summarizing the
    ///   decision: which detector(s) fired and whether/why the event was
    ///   gated (tick proximity, energy non-rising, tracker rejection).
    fn dbg_log_rerun(
        rec: &rerun::RecordingStream,
        frame: usize,
        mags: &[f32],
        bin_width: f32,
        noise_floor: &[f32],
        flux_bins: &[(f32, f32)],
        burst_bins: &[(f32, f32)],
        status_label: &str,
        status_color: (u8, u8, u8),
        event_emitted: bool,
    ) {
        let min_freq = 20.0f32;
        let max_freq = 20_000.0f32;
        rec.set_time_sequence("frame", frame as i64);

        let min_bin = ((min_freq / bin_width).ceil() as usize).max(1);
        let max_bin = ((max_freq / bin_width).floor() as usize).min(mags.len().saturating_sub(1));

        // 1. Spectrum Line (Gray)
        let spectrum_points: Vec<[f32; 2]> = (min_bin..=max_bin)
            .map(|b| [(b as f32 * bin_width) / 1000.0, mags[b]])
            .collect();
        let _ = rec.log(
            "spectrum/line",
            &rerun::LineStrips2D::new([spectrum_points])
                .with_colors([rerun::Color::from_rgb(100, 100, 100)]),
        );

        // 2. Noise Floor (Red/Brown)
        let nf_points: Vec<[f32; 2]> = (min_bin..=max_bin)
            .map(|b| [(b as f32 * bin_width) / 1000.0, noise_floor[b]])
            .collect();
        let _ = rec.log(
            "spectrum/noise_floor",
            &rerun::LineStrips2D::new([nf_points])
                .with_colors([rerun::Color::from_rgb(161, 75, 75)]),
        );

        // 3a. Flux-contributing bins (yellow, smaller — many per frame)
        if !flux_bins.is_empty() {
            let positions: Vec<[f32; 2]> =
                flux_bins.iter().map(|&(f, m)| [f / 1000.0, m]).collect();
            let _ = rec.log(
                "spectrum/flux_bins",
                &rerun::Points2D::new(positions)
                    .with_colors([rerun::Color::from_rgb(220, 200, 60)])
                    .with_radii([rerun::Radius::new_scene_units(0.012)]),
            );
        } else {
            let _ = rec.log("spectrum/flux_bins", &rerun::Points2D::clear_fields());
        }

        // 3b. Burst-causing bins (red, larger — the bins that actually
        // crossed the per-bin BIN_BURST_RATIO and got their floor jumped)
        if !burst_bins.is_empty() {
            let positions: Vec<[f32; 2]> =
                burst_bins.iter().map(|&(f, m)| [f / 1000.0, m]).collect();
            let _ = rec.log(
                "spectrum/burst_bins",
                &rerun::Points2D::new(positions)
                    .with_colors([rerun::Color::from_rgb(255, 80, 80)])
                    .with_radii([rerun::Radius::new_scene_units(0.025)]),
            );
        } else {
            let _ = rec.log("spectrum/burst_bins", &rerun::Points2D::clear_fields());
        }

        // 4. Status indicator + label.
        let status_pos = [min_freq / 1000.0, 0.0];
        let (r, g, b) = status_color;
        let radius = if event_emitted { 0.12 } else { 0.07 };
        let _ = rec.log(
            "spectrum/status",
            &rerun::Points2D::new([status_pos])
                .with_labels([status_label])
                .with_colors([rerun::Color::from_rgb(r, g, b)])
                .with_radii([rerun::Radius::new_scene_units(radius)]),
        );
    }
}
