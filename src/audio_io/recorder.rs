use std::{
    fs::File,
    io::BufWriter,
    sync::{
        Arc,
        atomic::{AtomicI8, Ordering},
    },
    thread,
    time::Duration,
};

use crossbeam_channel::Sender;
use hound::WavSpec;
use rtrb::Consumer;

use crate::audio_io::SlotPool;

// static CALLBACK_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub struct Recorder {
    state: Arc<AtomicI8>,
    handle: u8,
}

impl crate::traits::Worker for Recorder {
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

impl Recorder {
    pub fn start_record(
        slots: Arc<SlotPool>,
        mut cons: Consumer<usize>,
        handle: u8,
        reclaim: Sender<usize>,
        spec: WavSpec,
        path: String,
    ) -> Self {
        let recorder = Recorder {
            state: Arc::new(AtomicI8::new(1)),
            handle,
        };
        let state = recorder.state.clone();
        println!("recorder initialized");
        thread::spawn(move || {
            let file = File::create(path).unwrap();
            let bufw = BufWriter::new(file);
            let mut writer = hound::WavWriter::new(bufw, spec).unwrap();
            // batch write parameters
            let mut slots_written = 0usize;
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
                                // if count % 50000 == 0 {
                                //     let min =
                                //         slot_slice.iter().fold(f32::INFINITY, |a, &s| a.min(s));
                                //     let max =
                                //         slot_slice.iter().fold(f32::NEG_INFINITY, |a, &s| a.max(s));
                                //     log::debug!("Amplitude range = {min:.3}..{max:.3}, channels=1");
                                // }
                                for &s in slot_slice.iter() {
                                    let s_i16 = (s.max(-1.0).min(1.0) * i16::MAX as f32) as i16;
                                    writer.write_sample(s_i16).ok();
                                }
                            }
                            slots_written += 1;
                            if slots.release(idx) {
                                // log::trace!(
                                //     "[CONSUMER DEBUG] consumer {} released slot {}",
                                //     handle,
                                //     idx
                                // );
                                let _ = reclaim.send(idx);
                            }
                            // periodic flush policy: flush every few slots
                            if slots_written >= 8 {
                                writer.flush().ok();
                                slots_written = 0;
                            }
                        }
                        Err(_) => {
                            thread::sleep(Duration::from_millis(5));
                        }
                    }
                }
            }
            writer.flush().ok();
            writer.finalize().ok();
            println!("recorder finalized");
        });
        recorder
    }
}
