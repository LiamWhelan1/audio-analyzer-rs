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
    audio_io::SlotPool, audio_io::output::MusicalTransport, dsp::fft::FftProcessor, traits::Worker,
};

/// Detects onsets from incoming audio buffers and reports beat timestamps.
pub struct OnsetDetector {
    state: Arc<AtomicI8>,
    handle: u8,
}

impl Worker for OnsetDetector {
    /// Stop the detector and return its handle.
    fn stop(&mut self) -> u8 {
        self.state.store(-1, Ordering::Relaxed);
        self.handle
    }

    /// Pause the detector and return its handle.
    fn pause(&mut self) -> u8 {
        self.state.store(0, Ordering::Relaxed);
        self.handle
    }

    /// Start or resume the detector and return its handle.
    fn start(&mut self) -> u8 {
        self.state.store(1, Ordering::Relaxed);
        self.handle
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
    pub fn new(handle: u8) -> Self {
        OnsetDetector {
            state: Arc::new(AtomicI8::new(0)),
            handle,
        }
    }

    /// Spawn the detection thread which reads audio slots and pushes beat timestamps.
    ///
    /// - `transport` provides timing information.
    /// - `slots` is the shared buffer pool.
    /// - `cons` receives available slot indices.
    /// - `reclaim` is used to return slots to the free pool.
    /// - `onset_tx` receives detected beat timestamps as `f64`.
    pub fn detect_onsets(
        &mut self,
        transport: Arc<MusicalTransport>,
        slots: Arc<SlotPool>,
        mut cons: Consumer<usize>,
        reclaim: Sender<usize>,
        mut onset_tx: Producer<f64>,
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

            let min_inter_onset_samples = 4500;
            let mut samples_since_onset = min_inter_onset_samples;

            while state.load(Ordering::Relaxed) != -1 {
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

                    if frame_energy < 10.0 {
                        current_flux = 0.0;
                    }

                    if samples_since_onset >= min_inter_onset_samples {
                        if tracker.update(current_flux) {
                            let current_beat = (transport.get_accumulated_beats()
                                - 0.17 * (transport.get_bpm() as f64 / 60.0))
                                .max(0.0);
                            log::trace!("{current_beat:.4}");
                            if let Err(_) = onset_tx.push(current_beat) {}

                            samples_since_onset = 0;
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
