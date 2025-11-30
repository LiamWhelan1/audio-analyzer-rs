pub mod output; // <-- NEW: Output system module
pub mod recorder;
pub mod stft;

use anyhow::{Result, anyhow};
use cpal::traits::*;
use crossbeam_channel::{Sender, unbounded};
use hound;
use rtrb::{Consumer, Producer, RingBuffer};
use spin; // <-- NEW: For real-time safe mutex
use std::cell::UnsafeCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

use crate::AnalysisCallback;
use crate::audio_io::output::{BeatStrength, Metronome};
use output::{AudioSource, Mixer, OutputController}; // <-- NEW: Import output wrappers

// ============================================================================
//  STREAM TIMER
// ============================================================================

// This object will be shared (via Arc) between
// the input and output callbacks.
pub struct StreamTimer {
    // Tracks total frames processed by the output (master)
    output_frames: AtomicI64,
    // Tracks total frames processed by the input (slave)
    input_frames: AtomicI64,
}

impl StreamTimer {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            output_frames: AtomicI64::new(0),
            input_frames: AtomicI64::new(0),
        })
    }
    // Call this from the output callback
    pub fn tick_output(&self, num_frames: i64) {
        self.output_frames.fetch_add(num_frames, Ordering::Relaxed);
    }

    // Call this from the input callback
    pub fn tick_input(&self, num_frames: i64) {
        self.input_frames.fetch_add(num_frames, Ordering::Relaxed);
    }

    // This is the magic: calculate the current drift.
    // A positive value means the input is "ahead" of the output.
    // A negative value means it's "behind".
    pub fn get_drift_samples(&self) -> i64 {
        let in_t = self.input_frames.load(Ordering::Relaxed);
        let out_t = self.output_frames.load(Ordering::Relaxed);
        in_t - out_t
    }
}

unsafe impl Send for StreamTimer {}
unsafe impl Sync for StreamTimer {}

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

#[derive(Clone)]
pub struct AudioMeta {
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
    /// Probes the system for the current default devices and configuration.
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

        // Pipeline Constants
        let frames = 2048;
        let pool = 32;
        // We store Mono in the slots, so slot length is just frame count
        let slot_len = frames;

        Ok(Self {
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

// ============================================================================
//  AUDIO PIPELINE
// ============================================================================

pub struct AudioPipeline {
    meta: AudioMeta,
    pub slots: Arc<SlotPool>,
    running: Arc<AtomicBool>,
    timer: Arc<StreamTimer>,

    // Input State
    input_error: Arc<AtomicBool>, // Tracks async stream errors
    free_cons: Option<Consumer<usize>>,
    reducer_prod: Option<Producer<usize>>,
    reclaim_tx: Sender<usize>,
    reducer_add_tx: Sender<(u8, Producer<usize>)>,
    reducer_remove_tx: Sender<u8>,
    available_handles: Vec<u8>,

    // Output State
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
        // Just delegate to the internal builder
        Self::build_fresh()
    }

    /// Internal builder that creates a completely new pipeline state.
    fn build_fresh() -> Result<Self> {
        let meta = AudioMeta::probe()?;
        let slots = SlotPool::new(meta.pool, meta.slot_len);
        let timer = StreamTimer::new();

        // 1. Free Ring (Reclaimer -> Input)
        let (mut free_prod, free_cons) = RingBuffer::<usize>::new(meta.pool);
        for i in 0..meta.pool {
            let _ = free_prod.push(i);
        }

        // 2. Reducer Ring (Input -> Reducer)
        let (reducer_prod, mut reducer_cons) = RingBuffer::<usize>::new(meta.pool);

        // 3. Channels
        let (reclaim_tx, reclaim_rx) = unbounded();
        let (reducer_add_tx, reducer_add_rx) = unbounded::<(u8, Producer<usize>)>();
        let (reducer_remove_tx, reducer_remove_rx) = unbounded::<u8>();

        let running = Arc::new(AtomicBool::new(true));

        // Input state
        let input_error = Arc::new(AtomicBool::new(false));

        // Output state
        let output_error = Arc::new(AtomicBool::new(false));
        let mixer = Arc::new(spin::Mutex::new(Mixer::new(meta.out_channels as usize)));

        // 4. Reclaimer Thread
        let running_reclaim = running.clone();
        let reclaimer_handle = thread::spawn(move || {
            while running_reclaim.load(Ordering::SeqCst) || !reclaim_rx.is_empty() {
                if let Ok(idx) = reclaim_rx.recv_timeout(Duration::from_millis(20)) {
                    while free_prod.push(idx).is_err() {
                        thread::sleep(Duration::from_micros(200));
                    }
                }
            }
        });

        // 5. Reducer Thread
        let slots_clone = slots.clone();
        let running_reducer = running.clone();
        let reclaim_tx_reducer = reclaim_tx.clone();

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
                        // Optional: Signal Processing / Noise Reduction here
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
                                    // Downstream buffer full? reclaim immediately
                                    if slots_clone.release(idx) {
                                        let _ = reclaim_tx_reducer.send(idx);
                                    }
                                }
                            }
                        }
                    }
                    Err(_) => {
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

            // Drain remaining on shutdown
            while let Ok(idx) = reducer_cons.pop() {
                if slots_clone.release(idx) {
                    let _ = reclaim_tx_reducer.send(idx);
                }
            }
        });

        Ok(Self {
            meta,
            slots,
            running,
            timer,
            input_error,
            free_cons: Some(free_cons),
            reducer_prod: Some(reducer_prod),
            reclaim_tx,
            reducer_add_tx,
            reducer_remove_tx,
            available_handles: (0..u8::MAX).collect(),
            output_error,
            mixer,
            reclaimer_handle: Some(reclaimer_handle),
            reducer_handle: Some(reducer_handle),
            input_stream: None,
            output_stream: None,
        })
    }

    /// Public getter for the output controller
    pub fn output_controller(&self) -> OutputController {
        OutputController::new(self.mixer.clone())
    }

    /// Completely resets the pipeline.
    /// Stops old threads, probes new devices, and builds fresh infrastructure.
    fn reset_pipeline(&mut self) -> Result<()> {
        println!("🔄  Resetting Audio Pipeline...");

        // 1. Stop existing threads
        self.shutdown_internal();

        // 2. Build new state
        let new_pipe = Self::build_fresh()?;

        // 3. Replace Self fields
        self.meta = new_pipe.meta;
        self.slots = new_pipe.slots;
        self.running = new_pipe.running;
        self.timer = new_pipe.timer;

        self.input_error = new_pipe.input_error;
        self.free_cons = new_pipe.free_cons;
        self.reducer_prod = new_pipe.reducer_prod;
        self.reclaim_tx = new_pipe.reclaim_tx;
        self.reducer_add_tx = new_pipe.reducer_add_tx;
        self.reducer_remove_tx = new_pipe.reducer_remove_tx;
        self.available_handles = new_pipe.available_handles;

        self.output_error = new_pipe.output_error;
        self.mixer = new_pipe.mixer;

        self.reclaimer_handle = new_pipe.reclaimer_handle;
        self.reducer_handle = new_pipe.reducer_handle;
        self.input_stream = None;
        self.output_stream = None;

        println!(
            "✅  Pipeline Reset Complete. New Devices: In={}, Out={}",
            self.meta.in_dev.name().unwrap_or_default(),
            self.meta.out_dev.name().unwrap_or_default()
        );
        Ok(())
    }

    fn shutdown_internal(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        // Drop streams explicitly
        self.input_stream = None;
        self.output_stream = None;
        // Join threads
        if let Some(h) = self.reducer_handle.take() {
            h.join().ok();
        }
        if let Some(h) = self.reclaimer_handle.take() {
            h.join().ok();
        }
    }

    pub fn shutdown(mut self) {
        self.shutdown_internal();
        println!("Pipeline shutdown complete.");
    }

    pub fn stop_input(&mut self) {
        if self.available_handles.len() != 255 {
            return;
        }
        if let Some(stream) = self.input_stream.take() {
            drop(stream);
            println!("Input stream stopped and dropped.");
        }
    }

    pub fn stop_output(&mut self) {
        if let Some(stream) = self.output_stream.take() {
            drop(stream);
            println!("Output stream stopped and dropped.");
        }
    }

    pub fn start_input(&mut self) -> Result<()> {
        // 1. Check for async errors (e.g. device unplugged during last run)
        if self.input_error.load(Ordering::Relaxed) {
            println!("⚠️  Detected async input error. Resetting...");
            self.reset_pipeline()?;
        }

        // 3. Probe device validity
        // If the stored device is no longer valid (unplugged), this returns Error.
        if self.meta.in_dev.default_input_config().is_err() {
            println!("⚠️  Stored input device invalid. Resetting...");
            self.reset_pipeline()?;
        }

        if self.input_stream.is_some() {
            return Ok(());
        }
        // 2. Check for broken state (e.g. stop_input was called, consuming rings)
        if self.free_cons.is_none() {
            println!("⚠️  Pipeline in stopped state. Resetting...");
            self.reset_pipeline()?;
        }

        // 4. Attempt to build stream
        match self.build_input_stream_internal() {
            Ok(s) => {
                s.play()?;
                self.input_stream = Some(s);
                // Reset error flag on success
                self.input_error.store(false, Ordering::Relaxed);
                Ok(())
            }
            Err(e) => {
                println!(
                    "⚠️  Input stream build failed: {}. Attempting fresh reset...",
                    e
                );
                // Fallback: Try full reset and build again
                self.reset_pipeline()?;
                let s = self.build_input_stream_internal()?;
                s.play()?;
                self.input_stream = Some(s);
                Ok(())
            }
        }
    }

    pub fn start_output(&mut self) -> Result<()> {
        if self.output_error.load(Ordering::Relaxed) {
            println!("⚠️  Detected async output error. Resetting...");
            self.reset_pipeline()?;
        }

        if self.meta.out_dev.default_output_config().is_err() {
            println!("⚠️  Stored output device invalid. Resetting...");
            self.reset_pipeline()?;
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
                println!(
                    "⚠️  Output stream build failed: {}. Attempting fresh reset...",
                    e
                );
                self.reset_pipeline()?;
                let s = self.build_output_stream_internal()?;
                s.play()?;
                self.output_stream = Some(s);
                Ok(())
            }
        }
    }

    fn build_input_stream_internal(&mut self) -> Result<cpal::Stream> {
        // We must take() these. If this method fails after taking,
        // the pipeline is broken (missing rings) and MUST be reset next time.
        let free_cons = self
            .free_cons
            .take()
            .expect("Pipeline broken: missing free_cons");
        let reducer_prod = self
            .reducer_prod
            .take()
            .expect("Pipeline broken: missing reducer_prod");

        let config: cpal::StreamConfig = self.meta.in_conf.clone().into();
        let err_flag = self.input_error.clone();

        // ** ADDED TIMER **
        let timer = self.timer.clone();

        let err_fn = move |err| {
            log::error!("Input stream error: {}", err);
            // Signal async error so next start_input() knows to reset
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
                self.running.clone(),
                timer, // <-- Pass timer
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
                self.running.clone(),
                timer, // <-- Pass timer
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
                self.running.clone(),
                timer, // <-- Pass timer
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
        timer: Arc<StreamTimer>, // <-- Receive timer
        err_fn: impl FnMut(cpal::StreamError) + Send + 'static,
    ) -> Result<cpal::Stream>
    where
        T: cpal::Sample + cpal::SizedSample,
        f32: cpal::FromSample<T>,
    {
        let mut state = (None::<usize>, 0usize); // (current_slot_idx, write_pos)

        let stream = device.build_input_stream(
            config,
            move |data: &[T], _: &cpal::InputCallbackInfo| {
                // ** TICK TIMER **
                // We tick at the *start* of the callback
                let frames = data.len() / channels;
                timer.tick_input(frames as i64);

                if !running.load(Ordering::Relaxed) {
                    return;
                }

                let (ref mut current_idx, ref mut write_pos) = state;
                let mut frame_offset = 0;

                while frame_offset < data.len() {
                    // 1. Get a buffer slot if we don't have one
                    if current_idx.is_none() {
                        match free_cons.pop() {
                            Ok(idx) => {
                                *current_idx = Some(idx);
                                *write_pos = 0;
                            }
                            Err(_) => return, // Buffer full, drop samples
                        }
                    }

                    let idx = current_idx.unwrap();
                    // 2. Write / Downmix
                    unsafe {
                        let slot_slice = &mut *slots.slots[idx].get();
                        let frames_avail = (data.len() - frame_offset) / channels;
                        let space_left = slot_len - *write_pos;
                        let frames_to_write = space_left.min(frames_avail);

                        for f in 0..frames_to_write {
                            let start = frame_offset + f * channels;
                            let frame = &data[start..start + channels];
                            // Downmix to mono f32
                            let mixed = frame
                                .iter()
                                .fold(0.0f32, |acc, s| acc + s.to_sample::<f32>())
                                / channels as f32;
                            slot_slice[*write_pos + f] = mixed;
                        }

                        frame_offset += frames_to_write * channels;
                        *write_pos += frames_to_write;

                        // 3. If slot full, push to reducer
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

    // --- NEW: Output Stream Functions ---

    fn build_output_stream_internal(&mut self) -> Result<cpal::Stream> {
        let config: cpal::StreamConfig = self.meta.out_conf.clone().into();
        let err_flag = self.output_error.clone();

        let timer = self.timer.clone();
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
                timer,
                mixer,
                running,
                err_fn,
            )?,
            cpal::SampleFormat::I16 => Self::build_output_stream_impl::<i16>(
                &self.meta.out_dev,
                &config,
                channels,
                timer,
                mixer,
                running,
                err_fn,
            )?,
            cpal::SampleFormat::U16 => Self::build_output_stream_impl::<u16>(
                &self.meta.out_dev,
                &config,
                channels,
                timer,
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
        timer: Arc<StreamTimer>,
        mixer: Arc<spin::Mutex<Mixer>>,
        running: Arc<AtomicBool>,
        err_fn: impl FnMut(cpal::StreamError) + Send + 'static,
    ) -> Result<cpal::Stream>
    where
        T: cpal::Sample + cpal::SizedSample + cpal::FromSample<f32>,
    {
        // --- THIS IS THE FIX ---
        // Pre-allocate a local f32 buffer here, outside the callback.
        // We guess a buffer size. It will resize if needed, but ideally only once.
        let default_buf_size = 1024 * channels;
        let mut local_f32_buffer = vec![0.0f32; default_buf_size];

        let stream = device.build_output_stream(
            config,
            move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
                let frames = data.len() / channels;
                // ** TICK TIMER **
                timer.tick_output(frames as i64);

                if !running.load(Ordering::Relaxed) {
                    data.fill(T::EQUILIBRIUM);
                    return;
                }

                // We use try_lock() to avoid blocking the audio thread.
                // If the main thread is busy adding/removing sources,
                // we'll just play silence for one buffer.
                if let Some(mut mixer) = mixer.try_lock() {
                    // --- THIS LOGIC IS REPLACED ---
                    // // Get the mixer's internal f32 buffer, ensuring it's the right size
                    // let f32_buffer = mixer.get_f32_buffer(data.len());
                    //
                    // // Run all audio sources (metronome, synths, etc.)
                    // mixer.process(f32_buffer, channels);
                    //
                    // // Convert our f32 mixed audio into the hardware sample format
                    // for (f32_sample, out_sample) in f32_buffer.iter().zip(data.iter_mut()) {
                    //     *out_sample = T::from_sample(*f32_sample);
                    // }

                    // --- WITH THIS ---

                    // 1. Ensure our local buffer is the right size.
                    // This won't allocate unless the hardware buffer size changes.
                    if local_f32_buffer.len() != data.len() {
                        local_f32_buffer.resize(data.len(), 0.0);
                    }

                    // 2. Run the mixer. It will fill `local_f32_buffer`
                    //    and use its *own* internal `scratch_buf` for mixing.
                    //    The borrows no longer conflict.
                    mixer.process(&mut local_f32_buffer, channels);

                    // 3. Convert from our local f32 buffer into the hardware buffer `data`.
                    for (f32_sample, out_sample) in local_f32_buffer.iter().zip(data.iter_mut()) {
                        *out_sample = T::from_sample(*f32_sample);
                    }
                } else {
                    // Failed to get lock, play silence
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
        let _ = self.reducer_remove_tx.send(handle);
        self.available_handles.push(handle);
    }

    // Worker Spawners (Delegated to modules)
    pub fn spawn_recorder(&mut self, path: &str) -> Result<recorder::Recorder> {
        let handle = self.available_handles.pop().expect("Max consumers reached");
        let cons = self.add_consumer(handle);

        let spec = hound::WavSpec {
            channels: 1, // Our slots are mono
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
        let handle = self.available_handles.pop().expect("Max consumers reached");
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

    pub fn spawn_metronome(&mut self, bpm: f32) -> Producer<output::MetronomeCommand> {
        let controller = self.output_controller();
        let (met, met_controller) = Metronome::new(
            self.meta.out_sr,
            bpm,
            vec![
                BeatStrength::Strong,
                BeatStrength::Weak,
                BeatStrength::Weak,
                BeatStrength::Weak,
            ],
            None,
        );
        controller.add_source(Box::new(met));
        met_controller
    }
    pub fn spawn_synthesizer(&mut self) -> Producer<output::SynthCommand> {
        let (synth, producer) = output::Synthesizer::new(self.meta.out_sr);

        // Add to mixer
        self.output_controller().add_source(Box::new(synth));

        producer
    }
}
