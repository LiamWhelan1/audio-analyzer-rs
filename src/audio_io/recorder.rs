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

use crate::audio_io::SlotPool;
use crossbeam_channel::Sender;
use hound::WavSpec;
use rtrb::Consumer;

/// Records audio from the shared `SlotPool` into a WAV file on a background thread.
pub struct Recorder {
    state: Arc<AtomicI8>,
    handle: u8,
}

impl crate::traits::Worker for Recorder {
    /// Stop the recorder and return its handle.
    fn stop(&mut self) -> u8 {
        self.state.store(-1, Ordering::Relaxed);
        self.handle
    }

    /// Pause the recorder and return its handle.
    fn pause(&mut self) -> u8 {
        self.state.store(0, Ordering::Relaxed);
        self.handle
    }

    /// Start or resume the recorder and return its handle.
    fn start(&mut self) -> u8 {
        self.state.store(1, Ordering::Relaxed);
        self.handle
    }
}

impl Recorder {
    /// Spawn a recorder that writes incoming slots to `path` as WAV with `spec`.
    ///
    /// - `slots` is the shared buffer pool.
    /// - `cons` receives indices of filled slots.
    /// - `handle` is the worker id for this recorder.
    /// - `reclaim` is used to return freed slots.
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
            let mut slots_written = 0usize;
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
                } else {
                    match cons.pop() {
                        Ok(idx) => {
                            debug_assert!(idx < slots.slots.len());
                            unsafe {
                                let slot_slice: &mut [f32] = &mut *slots.slots[idx].get();
                                for &s in slot_slice.iter() {
                                    let s_i16 = (s.max(-1.0).min(1.0) * i16::MAX as f32) as i16;
                                    writer.write_sample(s_i16).ok();
                                }
                            }
                            slots_written += 1;
                            if slots.release(idx) {
                                let _ = reclaim.send(idx);
                            }
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
