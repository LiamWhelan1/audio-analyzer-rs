use std::{
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
        let hann = Self::hann_window(slot_len);
        let mut windowed: Vec<f32> = vec![0.0; slot_len];
        let mut output: Vec<Complex<f32>> = vec![Complex { re: 0.0, im: 0.0 }; slot_len / 2 + 1];
        let mut fft = FftProcessor::new(slot_len);
        let freqs: Vec<f32> = (0..=slot_len / 2 + 1)
            .map(|i| (i as f32) * (sr as f32) / (slot_len as f32))
            .collect();
        let sender = self.sender.clone();
        let mut prev_freq = 0.0_f32;
        thread::spawn(move || {
            while state.load(Ordering::Relaxed) != -1 || !cons.is_empty() {
                if state.load(Ordering::Relaxed) == 0 {
                    match cons.pop() {
                        Ok(idx) => {
                            // log::trace!(
                            //     "[CONSUMER DEBUG] consumer {} popping slot {}",
                            //     handle,
                            //     idx
                            // );
                            if slots.release(idx) {
                                let _ = reclaim.send(idx);
                            }
                        }
                        Err(_) => thread::sleep(Duration::from_millis(5)),
                    }
                } else {
                    match cons.pop() {
                        Ok(idx) => {
                            // log::trace!(
                            //     "[CONSUMER DEBUG] consumer {} popping slot {}",
                            //     handle,
                            //     idx
                            // );
                            debug_assert!(idx < slots.slots.len());
                            unsafe {
                                let slot_slice: &mut [f32] = &mut *slots.slots[idx].get();
                                // let count = CALLBACK_COUNT.load(Ordering::Relaxed);
                                // if count % 200 == 0 {
                                //     super::super::analysis::handle_audio_input(&slot_slice, 1);
                                //     let min =
                                //         slot_slice.iter().fold(f32::INFINITY, |a, &s| a.min(s));
                                //     let max =
                                //         slot_slice.iter().fold(f32::NEG_INFINITY, |a, &s| a.max(s));
                                //     log::debug!("Amplitude range = {min:.3}..{max:.3}, channels=1");
                                // }
                                for i in 0..slot_len {
                                    windowed[i] = slot_slice[i] * hann[i];
                                }
                                output.copy_from_slice(fft.process_forward(&mut windowed));
                                let mut max_freq = 0usize;
                                let mut max_amp = 0.0_f32;
                                let mut min_amp = 1.0_f32;
                                for (i, cplx) in output.iter().enumerate() {
                                    let amp = cplx.norm_sqr();
                                    if amp > max_amp {
                                        max_amp = amp;
                                        max_freq = i;
                                    }
                                    if amp < min_amp {
                                        min_amp = amp;
                                    }
                                }
                                if max_amp - min_amp < 0.5 || max_freq <= 1 {
                                    if slots.release(idx) {
                                        // log::trace!(
                                        //     "[CONSUMER DEBUG] consumer {} released slot {}",
                                        //     handle,
                                        //     idx
                                        // );
                                        let _ = reclaim.send(idx);
                                    }
                                    continue;
                                }
                                let mut freq = 0.0_f32;
                                let mut tot_amp = 0.0_f32;
                                let rng = 10;
                                for (i, cplx) in output[max_freq.max(rng) - rng
                                    ..max_freq.min(output.len() - rng) + rng]
                                    .iter()
                                    .enumerate()
                                {
                                    let amp = cplx.norm_sqr().powf(2.5);
                                    freq += amp * freqs[max_freq.max(rng) - rng + i];
                                    tot_amp += amp;
                                }
                                freq /= tot_amp;
                                // if max_amp > 2.5 && max_freq != prev_freq {
                                //     let _ = sender.send(Note::from_freq(max_freq, base_freq));
                                // }
                                if freq != prev_freq {
                                    prev_freq = freq;
                                    let _ = sender.send(Note::from_freq(freq, base_freq));
                                }
                                // let freq = pitch::yin_pitch(&slot_slice, sr as f32, 20.0, 20000.0);
                                // if let Some(f) = freq {
                                //     if f != prev_freq {
                                //         prev_freq = f;
                                //         let _ = sender.send(Note::from_freq(f, base_freq));
                                //     }
                                // }
                            }
                            if slots.release(idx) {
                                // log::trace!(
                                //     "[CONSUMER DEBUG] consumer {} released slot {}",
                                //     handle,
                                //     idx
                                // );
                                let _ = reclaim.send(idx);
                            }
                        }
                        Err(_) => {
                            thread::sleep(Duration::from_millis(5));
                        }
                    }
                }
            }
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
