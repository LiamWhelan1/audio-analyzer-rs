use std::{
    sync::{
        Arc,
        atomic::{AtomicI8, Ordering},
    },
    thread,
    time::Duration,
};

use crossbeam_channel::Sender;
use rtrb::{Consumer, Producer};

use crate::{
    audio_io::SlotPool,
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
    ) {
        self.state.store(1, Ordering::Relaxed);
        let state = self.state.clone();

        let window_size = 512;
        let hop_size = 128;
        let ring_buffer_len = 4096;

        thread::spawn(move || {
            let hann = Self::hann_window(window_size);
            let mut fft_processor = FftProcessor::new(window_size);

            let mut ring_buffer: Vec<f32> = vec![0.0; ring_buffer_len];
            let mut ring_write_pos = 0;
            let mut ring_read_pos = 0;
            let mut available_samples = 0;

            let mut time_domain_window: Vec<f32> = vec![0.0; window_size];
            let mut prev_magnitude: Vec<f32> = vec![0.0; window_size / 2 + 1];

            let mut tracker = FluxTracker::new(1.5, 0.70, 0.80);

            let min_inter_onset_samples = 2048;
            let mut samples_since_onset = min_inter_onset_samples;

            // ── Goal 2: energy EMA for harmonic-variation gate ────────────
            // Tracks the recent background energy level.  A genuine onset
            // must show a rapid energy increase relative to this baseline;
            // sustained notes with shifting harmonics do not.
            let mut energy_ema: f32 = 0.0;

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
                    slots.release(idx);
                    let _ = reclaim.send(idx);
                    new_data = true;
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
                    let half_size = window_size / 2 + 1;

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

                    // Silence gate: suppress flux on pure-noise frames.
                    // After AGC the signal is boosted, so this threshold is
                    // kept low (5.0) to avoid blanking soft-note onsets during
                    // the AGC warm-up period or on quiet playing.
                    if frame_energy < 10.0 {
                        current_flux = 0.0;
                    }

                    // ── Goal 2: update energy EMA ─────────────────────────
                    // Use asymmetric memory: rise fast so genuine attacks are
                    // compared against a baseline that hasn't caught up yet,
                    // decay slower so sustained notes keep the EMA elevated.
                    let ema_memory = if frame_energy > energy_ema { 0.7 } else { 0.9 };
                    energy_ema = energy_ema * ema_memory + frame_energy * (1.0 - ema_memory);

                    if samples_since_onset >= min_inter_onset_samples {
                        if tracker.update(current_flux) {
                            // ── Goal 1: suppress acoustic echo of device output ──
                            // ── Goal 2: require a rapid energy increase ──────────
                            // energy_ema tracks the recent baseline; an onset must
                            // show at least a 2.5× jump above it.  Harmonic shifts
                            // in sustained notes (throat singing, etc.) keep the
                            // ratio near 1.0 and are therefore suppressed.
                            let energy_rising = frame_energy > energy_ema * 1.25;

                            if !suppressed_by_output && energy_rising {
                                // The onset centre is window_size/2 samples behind the
                                // current ring-buffer write position.  Negate so
                                // stamp_onset shifts the beat position backwards in time.
                                let window_centre_offset = -(window_size as i64 / 2);
                                let velocity = (current_flux / 50.0).clamp(0.0, 1.0);
                                let event = transport.stamp_onset(window_centre_offset, velocity);

                                log::trace!(
                                    "onset @ beat {:.4} (raw offset {})",
                                    event.beat_position,
                                    event.raw_sample_offset,
                                );

                                let _ = onset_tx.push(event);
                                samples_since_onset = 0;
                            }
                        }
                    } else {
                        tracker.update(current_flux);
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
