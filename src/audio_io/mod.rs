pub mod output;
pub mod recorder;
pub mod stft;

use anyhow::{Result, anyhow};
use cpal::traits::*;
use crossbeam_channel::{Sender, unbounded};
use hound;
use rtrb::{Consumer, Producer, RingBuffer};
use spin;
use std::cell::UnsafeCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

use crate::AnalysisCallback;
use crate::generators::metronome::{Metronome, MetronomeCommand};
use crate::generators::player::{AudioPlayer, PlayerController};
use crate::generators::synth::{SynthCommand, Synthesizer};
use output::{Mixer, MusicalTransport, OutputController};

// ============================================================================
//  SLOT POOL (Shared Memory)
// ============================================================================

pub struct SlotPool {
    slots: Vec<UnsafeCell<Box<[f32]>>>,
    use_counts: Vec<AtomicUsize>,
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

    #[inline]
    fn acquire(&self, idx: usize, consumers: usize) {
        self.use_counts[idx].store(consumers, Ordering::SeqCst);
    }

    #[inline]
    fn release(&self, idx: usize) -> bool {
        let res = self.use_counts[idx].fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| {
            if v == 0 { None } else { Some(v - 1) }
        });

        match res {
            Ok(prev) => prev == 1,
            Err(current) => {
                log::error!(
                    "[SLOT DEBUG] release(): underflow on slot {} (curr={})",
                    idx,
                    current
                );
                false
            }
        }
    }
}

// ============================================================================
//  AUDIO META
// ============================================================================
pub struct AudioMeta {
    host: cpal::Host,
    in_dev: cpal::Device,
    out_dev: cpal::Device,
    in_conf: cpal::SupportedStreamConfig,
    out_conf: cpal::SupportedStreamConfig,
    pub in_sr: u32,
    pub out_sr: u32,
    in_channels: u16,
    out_channels: u16,
    frames: usize,
    pool: usize,
    slot_len: usize,
}

impl AudioMeta {
    pub fn probe() -> Result<Self> {
        let host = cpal::default_host();

        let in_dev = host
            .default_input_device()
            .ok_or_else(|| anyhow!("No input device available"))?;
        let out_dev = host
            .default_output_device()
            .ok_or_else(|| anyhow!("No output device available"))?;

        let in_conf = in_dev.default_input_config()?;
        let out_conf = out_dev.default_output_config()?;

        println!(
            "Found Input: {} ({:?})",
            in_dev.name().unwrap_or_default(),
            in_conf
        );
        println!(
            "Found Output: {} ({:?})",
            out_dev.name().unwrap_or_default(),
            out_conf
        );

        let in_sr = in_conf.sample_rate().0;
        let out_sr = out_conf.sample_rate().0;
        let in_channels = in_conf.channels();
        let out_channels = out_conf.channels();

        let frames = 2048;
        let pool = 32;
        let slot_len = frames;

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
    pub fn update_input(&mut self) -> anyhow::Result<()> {
        self.in_dev = self
            .host
            .default_input_device()
            .ok_or_else(|| anyhow!("No input device available"))?;
        self.in_conf = self.in_dev.default_input_config()?;
        println!(
            "Found Input: {} ({:?})",
            self.in_dev.name().unwrap_or_default(),
            self.in_conf
        );
        self.in_sr = self.in_conf.sample_rate().0;
        self.in_channels = self.in_conf.channels();
        Ok(())
    }
    pub fn update_output(&mut self) -> anyhow::Result<()> {
        self.out_dev = self
            .host
            .default_output_device()
            .ok_or_else(|| anyhow!("No output device available"))?;
        self.out_conf = self.out_dev.default_output_config()?;
        println!(
            "Found Output: {} ({:?})",
            self.out_dev.name().unwrap_or_default(),
            self.out_conf
        );
        self.out_sr = self.out_conf.sample_rate().0;
        self.out_channels = self.out_conf.channels();
        Ok(())
    }
}

// ============================================================================
//  AUDIO PIPELINE
// ============================================================================

pub struct AudioPipeline {
    meta: AudioMeta,
    pub slots: Arc<SlotPool>,

    // Global State
    pub transport: Arc<MusicalTransport>,

    // Input Logic State (Separate from Global Running)
    input_logic_running: Arc<AtomicBool>,

    // Input State
    input_error: Arc<AtomicBool>,
    free_cons: Option<Consumer<usize>>,
    reducer_prod: Option<Producer<usize>>,
    reclaim_tx: Sender<usize>,
    reducer_add_tx: Sender<(u8, Producer<usize>)>,
    reducer_remove_tx: Sender<u8>,

    // Tracks active worker IDs
    active_workers: Vec<u8>,
    // Stack of free IDs to recycle
    available_handles: Vec<u8>,

    // Output State
    running: Arc<AtomicBool>, // Still controls global output callbacks
    output_error: Arc<AtomicBool>,
    mixer: Arc<spin::Mutex<Mixer>>,

    // Threads
    reclaimer_handle: Option<thread::JoinHandle<()>>,
    reducer_handle: Option<thread::JoinHandle<()>>,

    // Streams
    input_stream: Option<cpal::Stream>,
    output_stream: Option<cpal::Stream>,
}

impl AudioPipeline {
    pub fn new() -> Result<Self> {
        let meta = AudioMeta::probe()?;
        let slots = SlotPool::new(meta.pool, meta.slot_len);
        let transport = MusicalTransport::new(120.0);
        let running = Arc::new(AtomicBool::new(true));

        let mixer = Arc::new(spin::Mutex::new(Mixer::new(meta.out_channels as usize)));

        // Channels for communication (Placeholders, overwritten in init_input)
        let (reclaim_tx, _reclaim_rx) = unbounded();
        let (reducer_add_tx, _reducer_add_rx) = unbounded::<(u8, Producer<usize>)>();
        let (reducer_remove_tx, _reducer_remove_rx) = unbounded::<u8>();

        let mut pipeline = Self {
            meta,
            slots,
            running,
            transport,
            input_logic_running: Arc::new(AtomicBool::new(false)),
            input_error: Arc::new(AtomicBool::new(false)),
            free_cons: None,
            reducer_prod: None,
            reclaim_tx,
            reducer_add_tx,
            reducer_remove_tx,
            active_workers: Vec::new(),
            available_handles: (0..u8::MAX).collect(), // Initialize stack of 0-255
            output_error: Arc::new(AtomicBool::new(false)),
            mixer,
            reclaimer_handle: None,
            reducer_handle: None,
            input_stream: None,
            output_stream: None,
        };

        // Initialize input internals immediately so they are ready
        pipeline.init_input_infrastructure();

        Ok(pipeline)
    }

    /// Initializes the Ring Buffers and Helper Threads for the input side.
    /// This allows us to "reboot" the input logic without destroying the Output or Transport.
    fn init_input_infrastructure(&mut self) {
        if self.reclaimer_handle.is_some() || self.reducer_handle.is_some() {
            // Already running
            return;
        }

        // 1. Create Ring Buffers
        let (mut free_prod, free_cons) = RingBuffer::<usize>::new(self.meta.pool);
        for i in 0..self.meta.pool {
            let _ = free_prod.push(i);
        }

        let (reducer_prod, mut reducer_cons) = RingBuffer::<usize>::new(self.meta.pool);

        // 2. Create Channels
        let (reclaim_tx, reclaim_rx) = unbounded();
        let (reducer_add_tx, reducer_add_rx) = unbounded::<(u8, Producer<usize>)>();
        let (reducer_remove_tx, reducer_remove_rx) = unbounded::<u8>();

        // Store transmission sides in struct
        self.reclaim_tx = reclaim_tx;
        self.reducer_add_tx = reducer_add_tx;
        self.reducer_remove_tx = reducer_remove_tx;
        self.free_cons = Some(free_cons);
        self.reducer_prod = Some(reducer_prod);

        // 3. Flags
        self.input_logic_running.store(true, Ordering::SeqCst);
        let running_reclaim = self.input_logic_running.clone();

        // 4. Spawn Reclaimer Thread
        let reclaimer_handle = thread::spawn(move || {
            while running_reclaim.load(Ordering::SeqCst) {
                // We use a short timeout so we can check the exit flag
                if let Ok(idx) = reclaim_rx.recv_timeout(Duration::from_millis(50)) {
                    while free_prod.push(idx).is_err() {
                        thread::sleep(Duration::from_micros(200));
                    }
                }
            }
        });

        // 5. Spawn Reducer Thread
        let slots_clone = self.slots.clone();
        let running_reducer = self.input_logic_running.clone();
        let reclaim_tx_reducer = self.reclaim_tx.clone(); // Use the NEW tx

        let reducer_handle = thread::spawn(move || {
            let mut consumers: Vec<(u8, Producer<usize>)> = Vec::new();

            loop {
                // Handle control messages
                while let Ok((name, prod)) = reducer_add_rx.try_recv() {
                    consumers.push((name, prod));
                }
                while let Ok(name) = reducer_remove_rx.try_recv() {
                    consumers.retain(|(n, _)| n != &name);
                }

                match reducer_cons.pop() {
                    Ok(idx) => {
                        unsafe {
                            let slot = &mut *slots_clone.slots[idx].get();
                            // Simple smoothing (Low-pass)
                            for i in 1..slot.len() {
                                slot[i] = 0.6 * slot[i - 1] + 0.4 * slot[i];
                            }
                        }

                        let count = consumers.len();
                        slots_clone.acquire(idx, count.max(1));

                        if count == 0 {
                            if slots_clone.release(idx) {
                                let _ = reclaim_tx_reducer.send(idx);
                            }
                        } else {
                            for (_, prod) in consumers.iter_mut() {
                                if prod.push(idx).is_err() {
                                    if slots_clone.release(idx) {
                                        let _ = reclaim_tx_reducer.send(idx);
                                    }
                                }
                            }
                        }
                    }
                    Err(_) => {
                        if !running_reducer.load(Ordering::SeqCst) {
                            break;
                        }
                        thread::sleep(Duration::from_millis(2));
                    }
                }
            }

            // Drain remaining on shutdown
            while let Ok(idx) = reducer_cons.pop() {
                if slots_clone.release(idx) {
                    let _ = reclaim_tx_reducer.send(idx);
                }
            }
        });

        self.reclaimer_handle = Some(reclaimer_handle);
        self.reducer_handle = Some(reducer_handle);
    }

    pub fn output_controller(&self) -> OutputController {
        OutputController::new(self.mixer.clone())
    }

    /// Safely shutdown global pipeline, including output.
    pub fn shutdown(mut self) {
        self.running.store(false, Ordering::SeqCst);
        self.stop_input(); // Helper to kill input threads
        self.output_stream = None;
        println!("Pipeline shutdown complete.");
    }

    /// Forces the input stream and infrastructure to stop.
    /// This is called on Shutdown OR on critical Error (e.g. Device Unplugged).
    /// It does NOT check `active_workers` because if the device is gone, the workers are dead anyway.
    pub fn stop_input(&mut self) {
        // FIX: Removed the early return check (self.active_workers.len() > 0).
        // If this function is called, it means we MUST stop (either error or shutdown).

        // 1. Drop the CPAL Stream (Stops microphone)
        if let Some(stream) = self.input_stream.take() {
            drop(stream);
            println!("Input stream stopped and dropped.");
        }

        // 2. Stop the Helper Threads (Save CPU)
        self.input_logic_running.store(false, Ordering::SeqCst);

        if let Some(h) = self.reclaimer_handle.take() {
            h.join().ok();
        }
        if let Some(h) = self.reducer_handle.take() {
            h.join().ok();
        }

        // 3. Clean up infrastructure
        self.free_cons = None;
        self.reducer_prod = None;

        // FIX: Reset worker tracking. If the infrastructure died, old workers are zombies.
        // We reset the handle pool so the NEXT workers get fresh IDs.
        self.active_workers.clear();
        self.available_handles = (0..u8::MAX).collect();

        println!("Input logic threads stopped and worker handles reset.");
    }

    /// Checks if there are any active workers. If none, saves CPU by stopping input.
    fn try_auto_stop_input(&mut self) {
        if self.active_workers.is_empty() {
            println!("No active input workers. Stopping input stream to save resources.");
            self.stop_input();
        }
    }

    pub fn stop_output(&mut self) {
        if let Some(stream) = self.output_stream.take() {
            drop(stream);
            println!("Output stream stopped and dropped.");
        }
    }

    pub fn start_input(&mut self) -> Result<()> {
        // 1. Check for async errors or invalid device
        if self.input_error.load(Ordering::Relaxed)
            || self.meta.in_dev.default_input_config().is_err()
        {
            println!("⚠️  Detected async input error. Restarting input...");
            // This calls stop_input, which now CORRECTLY tears everything down
            // regardless of whether workers "think" they are active.
            self.stop_input();
        }

        if self.input_stream.is_some() {
            return Ok(());
        }

        // 2. Check if infrastructure is missing (was stopped)
        if self.free_cons.is_none() {
            println!("Input infrastructure missing. Reinitializing...");
            self.meta.update_input()?;
            self.init_input_infrastructure();
        }

        // 3. Build Stream
        match self.build_input_stream_internal() {
            Ok(s) => {
                s.play()?;
                self.input_stream = Some(s);
                self.input_error.store(false, Ordering::Relaxed);
                Ok(())
            }
            Err(e) => {
                println!("⚠️  Input stream build failed: {}.", e);
                // Attempt one clean retry
                self.stop_input();
                self.init_input_infrastructure();
                let s = self.build_input_stream_internal()?;
                s.play()?;
                self.input_stream = Some(s);
                Ok(())
            }
        }
    }

    pub fn start_output(&mut self) -> Result<()> {
        if self.output_error.load(Ordering::Relaxed)
            || self.meta.out_dev.default_output_config().is_err()
        {
            println!("⚠️  Detected async output error. Restarting output...");
            self.stop_output();
            self.meta.update_output()?;
        }

        if self.output_stream.is_some() {
            return Ok(());
        }

        match self.build_output_stream_internal() {
            Ok(s) => {
                s.play()?;
                self.output_stream = Some(s);
                self.output_error.store(false, Ordering::Relaxed);
                Ok(())
            }
            Err(e) => {
                println!("⚠️  Output stream build failed: {}", e);
                // Retry once
                self.stop_output();
                let s = self.build_output_stream_internal()?;
                s.play()?;
                self.output_stream = Some(s);
                Ok(())
            }
        }
    }

    fn build_input_stream_internal(&mut self) -> Result<cpal::Stream> {
        // We take these out. If the stream dies, they are dropped.
        // But stop_input() clears them anyway, and start_input() calls init() to recreate them.
        let free_cons = self
            .free_cons
            .take()
            .expect("Pipeline logic error: missing free_cons");
        let reducer_prod = self
            .reducer_prod
            .take()
            .expect("Pipeline logic error: missing reducer_prod");

        let config: cpal::StreamConfig = self.meta.in_conf.clone().into();
        let err_flag = self.input_error.clone();
        let transport = self.transport.clone();
        let running = self.input_logic_running.clone(); // Use input-specific flag

        let err_fn = move |err| {
            log::error!("Input stream error: {}", err);
            err_flag.store(true, Ordering::Relaxed);
        };

        let stream = match self.meta.in_conf.sample_format() {
            cpal::SampleFormat::F32 => Self::build_input_stream_impl::<f32>(
                &self.meta.in_dev,
                &config,
                self.meta.in_channels as usize,
                self.meta.slot_len,
                free_cons,
                reducer_prod,
                self.slots.clone(),
                running,
                transport,
                err_fn,
            )?,
            cpal::SampleFormat::I16 => Self::build_input_stream_impl::<i16>(
                &self.meta.in_dev,
                &config,
                self.meta.in_channels as usize,
                self.meta.slot_len,
                free_cons,
                reducer_prod,
                self.slots.clone(),
                running,
                transport,
                err_fn,
            )?,
            cpal::SampleFormat::U16 => Self::build_input_stream_impl::<u16>(
                &self.meta.in_dev,
                &config,
                self.meta.in_channels as usize,
                self.meta.slot_len,
                free_cons,
                reducer_prod,
                self.slots.clone(),
                running,
                transport,
                err_fn,
            )?,
            f => return Err(anyhow!("Unsupported sample format: {}", f)),
        };
        Ok(stream)
    }

    fn build_input_stream_impl<T>(
        device: &cpal::Device,
        config: &cpal::StreamConfig,
        channels: usize,
        slot_len: usize,
        mut free_cons: Consumer<usize>,
        mut reducer_prod: Producer<usize>,
        slots: Arc<SlotPool>,
        running: Arc<AtomicBool>,
        transport: Arc<MusicalTransport>,
        err_fn: impl FnMut(cpal::StreamError) + Send + 'static,
    ) -> Result<cpal::Stream>
    where
        T: cpal::Sample + cpal::SizedSample,
        f32: cpal::FromSample<T>,
    {
        let mut state = (None::<usize>, 0usize);

        let stream = device.build_input_stream(
            config,
            move |data: &[T], _: &cpal::InputCallbackInfo| {
                let frames = data.len() / channels;
                transport.tick_input(frames as i64);

                if !running.load(Ordering::Relaxed) {
                    return;
                }

                let (ref mut current_idx, ref mut write_pos) = state;
                let mut frame_offset = 0;

                while frame_offset < data.len() {
                    if current_idx.is_none() {
                        match free_cons.pop() {
                            Ok(idx) => {
                                *current_idx = Some(idx);
                                *write_pos = 0;
                            }
                            Err(_) => return,
                        }
                    }

                    let idx = current_idx.unwrap();
                    unsafe {
                        let slot_slice = &mut *slots.slots[idx].get();
                        let frames_avail = (data.len() - frame_offset) / channels;
                        let space_left = slot_len - *write_pos;
                        let frames_to_write = space_left.min(frames_avail);

                        for f in 0..frames_to_write {
                            let start = frame_offset + f * channels;
                            let frame = &data[start..start + channels];
                            let mixed = frame
                                .iter()
                                .fold(0.0f32, |acc, s| acc + s.to_sample::<f32>())
                                / channels as f32;
                            slot_slice[*write_pos + f] = mixed;
                        }

                        frame_offset += frames_to_write * channels;
                        *write_pos += frames_to_write;

                        if *write_pos >= slot_len {
                            let _ = reducer_prod.push(idx);
                            *current_idx = None;
                            *write_pos = 0;
                        }
                    }
                }
            },
            err_fn,
            None,
        )?;

        Ok(stream)
    }

    fn build_output_stream_internal(&mut self) -> Result<cpal::Stream> {
        let config: cpal::StreamConfig = self.meta.out_conf.clone().into();
        let err_flag = self.output_error.clone();

        let transport = self.transport.clone();
        let mixer = self.mixer.clone();
        let running = self.running.clone();
        let channels = self.meta.out_channels as usize;

        let err_fn = move |err| {
            log::error!("Output stream error: {}", err);
            err_flag.store(true, Ordering::Relaxed);
        };

        let stream = match self.meta.out_conf.sample_format() {
            cpal::SampleFormat::F32 => Self::build_output_stream_impl::<f32>(
                &self.meta.out_dev,
                &config,
                channels,
                self.meta.out_sr,
                transport,
                mixer,
                running,
                err_fn,
            )?,
            cpal::SampleFormat::I16 => Self::build_output_stream_impl::<i16>(
                &self.meta.out_dev,
                &config,
                channels,
                self.meta.out_sr,
                transport,
                mixer,
                running,
                err_fn,
            )?,
            cpal::SampleFormat::U16 => Self::build_output_stream_impl::<u16>(
                &self.meta.out_dev,
                &config,
                channels,
                self.meta.out_sr,
                transport,
                mixer,
                running,
                err_fn,
            )?,
            f => return Err(anyhow!("Unsupported sample format: {}", f)),
        };
        Ok(stream)
    }

    fn build_output_stream_impl<T>(
        device: &cpal::Device,
        config: &cpal::StreamConfig,
        channels: usize,
        sr: u32,
        transport: Arc<MusicalTransport>,
        mixer: Arc<spin::Mutex<Mixer>>,
        running: Arc<AtomicBool>,
        err_fn: impl FnMut(cpal::StreamError) + Send + 'static,
    ) -> Result<cpal::Stream>
    where
        T: cpal::Sample + cpal::SizedSample + cpal::FromSample<f32>,
    {
        let default_buf_size = 1024 * channels;
        let mut local_f32_buffer = vec![0.0f32; default_buf_size];

        let stream = device.build_output_stream(
            config,
            move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
                let frames = data.len() / channels;
                transport.tick_output(frames as i64, sr as f32);

                if !running.load(Ordering::Relaxed) {
                    data.fill(T::EQUILIBRIUM);
                    return;
                }

                if let Some(mut mixer) = mixer.try_lock() {
                    if local_f32_buffer.len() != data.len() {
                        local_f32_buffer.resize(data.len(), 0.0);
                    }

                    mixer.process(&mut local_f32_buffer, channels);

                    for (f32_sample, out_sample) in local_f32_buffer.iter().zip(data.iter_mut()) {
                        *out_sample = T::from_sample(*f32_sample);
                    }
                } else {
                    data.fill(T::EQUILIBRIUM);
                }
            },
            err_fn,
            None,
        )?;

        Ok(stream)
    }

    pub fn add_consumer(&mut self, handle: u8) -> Consumer<usize> {
        let (prod, cons) = RingBuffer::<usize>::new(self.meta.pool);
        self.reducer_add_tx.send((handle, prod)).unwrap();
        cons
    }

    pub fn remove_consumer(&mut self, handle: u8) {
        // Send removal signal
        let _ = self.reducer_remove_tx.send(handle);

        // Remove from tracking
        self.active_workers.retain(|&h| h != handle);

        // Recycle the ID for future use
        self.available_handles.push(handle);

        // FIX: Check if we can power down inputs now
        self.try_auto_stop_input();
    }

    pub fn spawn_recorder(&mut self, path: &str) -> Result<recorder::Recorder> {
        // FIX: Use stack to get ID. Pop returns None if empty (max consumers reached).
        let handle = self
            .available_handles
            .pop()
            .expect("Max consumers reached (255)");

        self.active_workers.push(handle);
        let cons = self.add_consumer(handle);

        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: self.meta.in_sr,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        Ok(recorder::Recorder::start_record(
            self.slots.clone(),
            cons,
            handle,
            self.reclaim_tx.clone(),
            spec,
            path.to_string(),
        ))
    }

    pub fn spawn_transformer(&mut self, callback: AnalysisCallback) -> Result<stft::STFT> {
        // FIX: Use stack to get ID.
        let handle = self
            .available_handles
            .pop()
            .expect("Max consumers reached (255)");

        self.active_workers.push(handle);
        let cons = self.add_consumer(handle);

        let mut stft = stft::STFT::new(handle);
        stft.pitch(
            self.slots.clone(),
            self.meta.slot_len,
            cons,
            self.reclaim_tx.clone(),
            self.meta.in_sr,
            callback,
        );
        Ok(stft)
    }

    pub fn spawn_metronome(&mut self, bpm: f32) -> Producer<MetronomeCommand> {
        let controller = self.output_controller();
        let (met, met_controller) = Metronome::new(self.meta.out_sr, bpm, self.transport.clone());
        controller.add_source(Box::new(met));
        met_controller
    }
    pub fn spawn_synthesizer(&mut self) -> Producer<SynthCommand> {
        let (synth, producer) = Synthesizer::new(self.meta.out_sr, self.transport.clone());
        self.output_controller().add_source(Box::new(synth));
        producer
    }
    pub fn spawn_player(&mut self) -> PlayerController {
        let (player, controller) = AudioPlayer::new(self.meta.out_sr);
        self.output_controller().add_source(Box::new(player));
        controller
    }
}
