pub mod recorder;
pub mod stft;

use anyhow::Result;
use cpal::traits::*;
use crossbeam_channel::{Sender, unbounded};
use hound::{self, WavSpec};
use rtrb::{Consumer, Producer, RingBuffer};
use std::cell::UnsafeCell;
use std::fs::File;
use std::io::BufWriter;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::generators::Note;

// static CALLBACK_COUNT: AtomicU64 = AtomicU64::new(0);
// static LAST_CALLBACK_TS: AtomicU64 = AtomicU64::new(0);

pub struct SlotPool {
    slots: Vec<UnsafeCell<Box<[f32]>>>,
    // Use a wider atomic to avoid wrapping under concurrent updates and to
    // make underflow checks easier. usize is natural for counters.
    use_counts: Vec<AtomicUsize>, // track how many consumers still hold it
}

unsafe impl Send for SlotPool {}
unsafe impl Sync for SlotPool {}

impl SlotPool {
    fn new(pool_size: usize, slot_len: usize) -> Arc<Self> {
        let mut slots = Vec::with_capacity(pool_size);
        let mut counts = Vec::with_capacity(pool_size);
        for _ in 0..pool_size {
            slots.push(UnsafeCell::new(vec![0.0f32; slot_len].into_boxed_slice()));
            counts.push(AtomicUsize::new(0));
        }
        Arc::new(Self {
            slots,
            use_counts: counts,
        })
    }

    /// Acquire extra references for fan-out.
    #[inline]
    fn acquire(&self, idx: usize, consumers: usize) {
        // store the number of consumers that will read this slot. We use
        // SeqCst to keep ordering simple across threads.
        self.use_counts[idx].store(consumers, Ordering::SeqCst);
        // log::trace!("[SLOT DEBUG] acquire(): slot {} ref {}", idx, consumers);
    }
    #[inline]
    fn release(&self, idx: usize) -> bool {
        // returns true if this was the last release
        // Use fetch_update so we never underflow the counter.
        let res = self.use_counts[idx].fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| {
            if v == 0 { None } else { Some(v - 1) }
        });

        match res {
            Ok(prev) => {
                debug_assert!(prev > 0, "[SLOT DEBUG] release(): prev should be > 0");
                if prev == 1 {
                    // fully released
                    // log::trace!("[SLOT DEBUG] release(): slot {} fully released", idx);
                    return true;
                }
                // log::trace!("[SLOT DEBUG] release(): slot {} decremented", idx);
                false
            }
            Err(current) => {
                // Someone attempted to release when the counter was already 0.
                // This indicates a bug in consumer / reclaim logic; log it so
                // it can be diagnosed rather than silently wrapping.
                log::error!(
                    "[SLOT DEBUG] release(): underflow attempt on slot {} (current={})",
                    idx,
                    current
                );
                false
            }
        }
    }
}

pub struct AudioMeta {
    host: cpal::Host,
    in_dev: cpal::Device,
    out_dev: cpal::Device,
    in_conf: cpal::SupportedStreamConfig,
    out_conf: cpal::SupportedStreamConfig,
    in_sr: u32,
    out_sr: u32,
    in_channels: u16,
    out_channels: u16,
    frames: usize,
    pool: usize,
    slot_len: usize,
}

impl AudioMeta {
    pub fn new() -> Result<Self> {
        let host = cpal::default_host();
        let in_dev = host
            .default_input_device()
            .expect("No input device available");
        let out_dev = host
            .default_output_device()
            .expect("No output device available");
        let in_conf = in_dev.default_input_config()?;
        let out_conf = out_dev.default_output_config()?;
        println!(
            "format: {}, sample_rate: {:?}, channels: {}",
            in_conf.sample_format(),
            in_conf.sample_rate(),
            in_conf.channels()
        );
        println!(
            "format: {}, sample_rate: {:?}, channels: {}",
            out_conf.sample_format(),
            out_conf.sample_rate(),
            out_conf.channels()
        );
        let in_sr = in_conf.sample_rate().0;
        let out_sr = out_conf.sample_rate().0;
        let in_channels = in_conf.channels();
        let out_channels = out_conf.channels();
        let frames = 2048;
        let pool = 32;
        let slot_len = frames * 1 as usize;
        Ok(Self {
            host,
            in_dev,
            out_dev,
            in_conf,
            out_conf,
            in_sr,
            out_sr,
            in_channels,
            out_channels,
            frames,
            pool,
            slot_len,
        })
    }
}

pub struct AudioPipeline {
    meta: AudioMeta,
    pub slots: Arc<SlotPool>,
    running: Arc<AtomicBool>,

    // free_pool: free_prod lives with reclaimer; free_cons is used by input
    free_cons: Option<Consumer<usize>>,

    // reducer ring: input -> reducer
    reducer_prod: Option<Producer<usize>>,

    // reclaimer channel (multi-producer)
    reclaim_tx: Sender<usize>,

    // control channels for reducer (to add/remove producers)
    reducer_add_tx: Sender<(u8, Producer<usize>)>,
    reducer_remove_tx: Sender<u8>,
    available_handles: Vec<u8>,

    // thread handles for joining
    reclaimer_handle: Option<thread::JoinHandle<()>>,
    reducer_handle: Option<thread::JoinHandle<()>>,
    input_stream: Option<cpal::Stream>,
}

impl AudioPipeline {
    pub fn new() -> Result<Self> {
        let meta = AudioMeta::new()?;
        // host/devices checked later in start()
        let slots = SlotPool::new(meta.pool, meta.slot_len);
        // Free ring (we will move free_prod to reclaimer)
        let (mut free_prod, free_cons) = RingBuffer::<usize>::new(meta.pool);
        for i in 0..meta.pool {
            free_prod.push(i).ok();
        }

        // reducer ring between input and reducer
        let (reducer_prod, mut reducer_cons) = RingBuffer::<usize>::new(meta.pool);

        // reclaim channel (multi-producer senders from consumers)
        let (reclaim_tx, reclaim_rx) = unbounded();

        // control channels for adding/removing consumers (producer moved into reducer)
        let (reducer_add_tx, reducer_add_rx) = unbounded::<(u8, Producer<usize>)>();
        let (reducer_remove_tx, reducer_remove_rx) = unbounded::<u8>();
        let available_handles = (0..u8::MAX).collect();
        let running = Arc::new(AtomicBool::new(true));
        let reclaim_rx_out = reclaim_rx.clone();
        // Start reclaimer thread (owns free_prod)
        let running_reclaim = running.clone();
        let reclaimer_handle = thread::spawn(move || {
            // free_prod is the single producer for free ring
            while running_reclaim.load(Ordering::SeqCst) || !reclaim_rx_out.is_empty() {
                match reclaim_rx_out.recv_timeout(Duration::from_millis(20)) {
                    Ok(idx) => {
                        // push back into free pool (spin if full)
                        while free_prod.push(idx).is_err() {
                            thread::sleep(Duration::from_micros(200));
                        }
                    }
                    Err(_) => {}
                }
            }
            // drain remaining messages
            while let Ok(idx) = reclaim_rx_out.try_recv() {
                let _ = free_prod.push(idx);
            }
        });

        // Start reducer thread (owns consumer producers)
        let slots_clone = slots.clone();
        let running_reducer = running.clone();
        let reclaim_tx_for_reducer = reclaim_tx.clone();
        let reducer_handle = thread::spawn(move || {
            // local map of consumers
            let mut consumers: Vec<(u8, Producer<usize>)> = Vec::new();
            // loop: handle control messages and reducer_cons pops
            loop {
                // try to drain add/remove controls quickly
                while let Ok((name, prod)) = reducer_add_rx.try_recv() {
                    consumers.push((name, prod));
                }
                while let Ok(name) = reducer_remove_rx.try_recv() {
                    consumers.retain(|(n, _)| n != &name);
                }

                // process one index if available
                match reducer_cons.pop() {
                    Ok(idx) => {
                        // log::trace!(
                        //     "[REDUCER DEBUG] acquired slot {} for {} consumers",
                        //     idx,
                        //     consumers.len()
                        // );
                        debug_assert!(idx < slots_clone.slots.len());
                        // apply noise reduction in-place
                        unsafe {
                            let slot = &mut *slots_clone.slots[idx].get();
                            // simple smoothing filter
                            for i in 1..slot.len() {
                                slot[i] = 0.6 * slot[i - 1] + 0.4 * slot[i];
                            }
                        }
                        // snapshot consumer count & push to each
                        let count = consumers.len();
                        slots_clone.acquire(idx, count.max(1));
                        if count == 0 {
                            // no consumers: release immediately
                            if slots_clone.release(idx) {
                                let _ = reclaim_tx_for_reducer.send(idx);
                            }
                        } else {
                            // push to each consumer; if a push fails, decrement the slot counter immediately
                            for (_, prod) in consumers.iter_mut() {
                                if let Err(_) = prod.push(idx) {
                                    log::warn!(
                                        "[REDUCER DEBUG] push() failed for consumer; slot {} will be immediately released",
                                        idx
                                    );
                                    if slots_clone.release(idx) {
                                        log::warn!(
                                            "[REDUCER DEBUG] slot {} reclaimed early due to push failure",
                                            idx
                                        );
                                        let _ = reclaim_tx_for_reducer.send(idx);
                                    }
                                } else {
                                    // log::trace!("[REDUCER DEBUG] pushed slot {} to consumer", idx);
                                }
                            }
                        }
                    }
                    Err(_) => {
                        // nothing currently — break on shutdown
                        if !running_reducer.load(Ordering::SeqCst)
                            && reducer_add_rx.is_empty()
                            && reducer_remove_rx.is_empty()
                        {
                            break;
                        }
                        thread::sleep(Duration::from_millis(2));
                    }
                }
            }

            // cleanup: drain reducer_cons, push to reclaim if needed
            while let Ok(idx) = reducer_cons.pop() {
                unsafe {
                    let slot = &mut *slots_clone.slots[idx].get();
                    for i in 1..slot.len() {
                        slot[i] = 0.6 * slot[i - 1] + 0.4 * slot[i];
                    }
                }
                // no consumers => immediate reclaim
                if slots_clone.release(idx) {
                    let _ = reclaim_tx_for_reducer.send(idx);
                }
            }
        });

        Ok(Self {
            meta,
            slots,
            running,
            free_cons: Some(free_cons),
            reducer_prod: Some(reducer_prod),
            reclaim_tx,
            reducer_add_tx,
            reducer_remove_tx,
            available_handles,
            reclaimer_handle: Some(reclaimer_handle),
            reducer_handle: Some(reducer_handle),
            input_stream: None,
        })
    }

    pub fn add_consumer(&mut self, handle: u8) -> Consumer<usize> {
        let (prod, cons) = RingBuffer::<usize>::new(self.meta.pool);
        // send producer into reducer via control channel (reducer owns it)
        self.reducer_add_tx.send((handle, prod)).unwrap();
        cons
    }

    /// Remove a consumer by name.
    pub fn remove_consumer(&mut self, handle: u8) {
        let _ = self.reducer_remove_tx.send(handle);
        self.available_handles.push(handle);
    }

    pub fn shutdown(mut self) {
        self.running.store(false, Ordering::SeqCst);
        if let Some(stream) = self.input_stream.take() {
            drop(stream); // stops callback
        }
        // reducer thread: send a small delay for it to exit
        if let Some(handle) = self.reducer_handle.take() {
            handle.join().ok();
        }
        // ensure reclaimer finishes
        if let Some(handle) = self.reclaimer_handle.take() {
            handle.join().ok();
        }
        println!("shutdown complete");
    }
    pub fn start_input(&mut self) -> anyhow::Result<()> {
        if self.input_stream.is_some() {
            return Ok(());
        }

        let in_dev = &self.meta.in_dev;
        let in_conf = self.meta.in_conf.clone();
        let input_config = in_conf.clone().into();

        let slot_len = self.meta.slot_len;
        let in_channels = self.meta.in_channels as usize;

        let free_cons = self.free_cons.take().unwrap();
        let reducer_prod = self.reducer_prod.take().unwrap();
        let slots = self.slots.clone();
        let running = self.running.clone();

        // We'll dynamically choose the correct stream builder for the format
        let stream = match in_conf.sample_format() {
            cpal::SampleFormat::F32 => Self::build_input_stream::<f32>(
                in_dev,
                &input_config,
                in_channels,
                slot_len,
                free_cons,
                reducer_prod,
                slots,
                running,
            )?,
            cpal::SampleFormat::I16 => Self::build_input_stream::<i16>(
                in_dev,
                &input_config,
                in_channels,
                slot_len,
                free_cons,
                reducer_prod,
                slots,
                running,
            )?,
            cpal::SampleFormat::U16 => Self::build_input_stream::<u16>(
                in_dev,
                &input_config,
                in_channels,
                slot_len,
                free_cons,
                reducer_prod,
                slots,
                running,
            )?,
            format => panic!("Sample format {format} not implemented"),
        };

        stream.play()?;
        self.input_stream = Some(stream);
        Ok(())
    }

    fn build_input_stream<T>(
        device: &cpal::Device,
        config: &cpal::StreamConfig,
        in_channels: usize,
        slot_len: usize,
        mut free_cons: rtrb::Consumer<usize>,
        mut reducer_prod: rtrb::Producer<usize>,
        slots: Arc<SlotPool>,
        running: Arc<AtomicBool>,
    ) -> anyhow::Result<cpal::Stream>
    where
        T: cpal::Sample + cpal::SizedSample + std::fmt::Debug,
        f32: cpal::FromSample<T>,
    {
        // let start_time = Instant::now();
        let mut state = (None::<usize>, 0usize);
        // LAST_CALLBACK_TS.store(0, Ordering::Relaxed);
        let stream = device.build_input_stream(
            config,
            move |data: &[T], _: &cpal::InputCallbackInfo| {
                if !running.load(Ordering::Relaxed) {
                    return;
                }

                let (ref mut current_idx, ref mut write_pos) = state;
                let mut frame_offset = 0usize;
                // Downmix inline
                while frame_offset < data.len() {
                    if current_idx.is_none() {
                        match free_cons.pop() {
                            Ok(idx) => {
                                *current_idx = Some(idx);
                                *write_pos = 0;
                            }
                            Err(_) => return, // buffer full, drop samples
                        }
                    }

                    let idx = current_idx.unwrap();
                    unsafe {
                        let slot_slice: &mut [f32] = &mut *slots.slots[idx].get();
                        let can_write = slot_len - *write_pos;
                        let frames_avail = (data.len() - frame_offset) / in_channels;
                        let frames_to_write =
                            can_write.min(frames_avail * 1 /* mono downmix */);

                        debug_assert!(*write_pos + frames_to_write <= slot_len);

                        for f in 0..frames_to_write {
                            let frame = &data[frame_offset + f * in_channels
                                ..frame_offset + (f + 1) * in_channels];
                            let mixed = frame
                                .iter()
                                .fold(0.0f32, |acc, &s| acc + s.to_sample::<f32>())
                                / in_channels as f32;
                            slot_slice[*write_pos + f] = mixed;
                        }

                        frame_offset += frames_to_write * in_channels;
                        *write_pos += frames_to_write;

                        if *write_pos >= slot_len {
                            let _ = reducer_prod.push(idx);
                            *current_idx = None;
                            *write_pos = 0;
                        }
                    }
                }
            },
            move |err| log::error!("input stream error: {}", err),
            None,
        )?;
        Ok(stream)
    }

    pub fn spawn_recorder(&mut self, path: &str) -> recorder::Recorder {
        let handle = self.available_handles.pop().expect("Out of consumer slots");
        let cons = self.add_consumer(handle);
        let slots = self.slots.clone();
        let reclaim = self.reclaim_tx.clone();
        // writer prepared here
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: self.meta.in_sr,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let path = path.to_string();
        recorder::Recorder::start_record(slots, cons, handle, reclaim, spec, path)
    }

    pub fn spawn_transformer(&mut self, sender: Sender<Vec<f32>>) -> stft::STFT {
        let handle = self.available_handles.pop().expect("Out of consumer slots");
        let cons = self.add_consumer(handle);
        let slots = self.slots.clone();
        let reclaim = self.reclaim_tx.clone();
        let mut stft = stft::STFT::new(handle, sender);
        stft.pitches(slots, self.meta.slot_len, cons, reclaim, self.meta.in_sr);
        stft
    }
}
