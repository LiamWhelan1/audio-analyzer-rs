use anyhow::Result;
use cpal::Stream;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread;
use std::time::Duration;

use crate::analysis::pitch::yin_pitch;
use crate::audio_io::spawn_input_stream;
use crate::dsp::fft::FftProcessor;
use crate::resample::Resampler;
use crate::types::{AudioChunk, Sample};
use crossbeam_channel::{Receiver, Sender, unbounded};
use rtrb::{Consumer, RingBuffer};

/// Represents a simplified analysis result that can be sent to the UI layer.
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    pub pitch_hz: Option<f32>,
    pub magnitude: f32,
    pub timestamp_ms: f64,
}

/// Main pipeline handle — manages audio + processing threads.
pub struct AudioPipeline {
    input_stream: Option<Stream>,
    output_stream: Option<Stream>,
    tx: Option<Sender<Vec<Sample>>>,
    rx: Option<Receiver<Vec<Sample>>>,
}

impl AudioPipeline {
    pub fn new() -> Self {
        Self {
            input_stream: None,
            output_stream: None,
            tx: None,
            rx: None,
        }
    }

    /// Initialize the audio pipeline and spawn threads.
    pub fn start() -> Result<Self> {
        let shutdown_flag = Arc::new(AtomicBool::new(false));

        // 1. Create a ring buffer for audio data (64 chunks capacity)
        let (prod, cons) = RingBuffer::<AudioChunk>::new(64);

        // 2. Start audio input stream (pushes chunks into ring)
        let _input_stream = spawn_input_stream(prod)?;

        // 3. Start analysis worker thread (pulls from ring)
        let (tx, rx) = unbounded();
        let worker_shutdown = shutdown_flag.clone();
        thread::spawn(move || worker_thread(worker_shutdown, tx, cons));

        // 4. Return pipeline handle
        Ok(Self {
            result_rx: rx,
            shutdown_flag,
        })
    }

    /// Request shutdown gracefully.
    pub fn stop(self) {
        self.shutdown_flag.store(true, Ordering::SeqCst);
        // CPAL stream drops automatically, thread ends on next check.
    }
}

fn worker_thread(
    stop_flag: Arc<AtomicBool>,
    tx: Sender<AnalysisResult>,
    mut cons: Consumer<AudioChunk>,
) {
    // Parameters
    let target_sr = 44100usize;
    let frame_size = 2048;
    let hop_size = 512;
    let min_freq = 50.0;
    let max_freq = 2000.0;

    let mut resampler: Option<Resampler> = None;
    let mut fft = FftProcessor::new(frame_size);

    let mut frame_buffer: Vec<Sample> = Vec::with_capacity(frame_size);
    let mut last_timestamp_ms = 0.0;

    while !stop_flag.load(Ordering::Relaxed) {
        if let Some(chunk) = cons.pop().ok() {
            // Lazy-init resampler if needed
            if resampler.is_none() {
                resampler = Some(Resampler::new(
                    chunk.sample_rate as usize,
                    target_sr,
                    chunk.channels,
                    hop_size,
                ));
            }

            // Convert interleaved -> mono for analysis
            let mono: Vec<Sample> = if chunk.channels == 1 {
                chunk.samples.clone()
            } else {
                chunk
                    .samples
                    .chunks(chunk.channels)
                    .map(|c| c.iter().sum::<f32>() / c.len() as f32)
                    .collect()
            };

            // Resample to target SR if needed
            let mut processed = mono;
            if let Some(rsm) = &mut resampler {
                let frames = vec![processed.clone()]; // single channel
                if let Ok(out) = rsm.process(frames) {
                    processed = out[0].clone();
                }
            }

            // Frame accumulation
            frame_buffer.extend_from_slice(&processed);
            while frame_buffer.len() >= frame_size {
                let mut frame: Vec<f32> = frame_buffer[..frame_size].to_vec();

                // Apply Hann window
                for (i, s) in frame.iter_mut().enumerate() {
                    let w = 0.5
                        - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / frame_size as f32).cos();
                    *s *= w;
                }

                // --- Pitch detection ---
                let pitch = yin_pitch(&frame, target_sr as f32, min_freq, max_freq);

                // --- FFT Magnitude ---
                let spectrum = fft.process(&mut frame);
                let mag = spectrum.iter().map(|c| c.norm()).sum::<f32>() / spectrum.len() as f32;

                // --- Package result ---
                last_timestamp_ms += (hop_size as f64 / target_sr as f64) * 1000.0;
                let result = AnalysisResult {
                    pitch_hz: pitch,
                    magnitude: mag,
                    timestamp_ms: last_timestamp_ms,
                };

                let _ = tx.send(result);

                // Shift buffer
                frame_buffer.drain(..hop_size);
            }
        } else {
            // No data — sleep briefly to avoid busy waiting
            thread::sleep(Duration::from_millis(2));
        }
    }
    println!("Worker thread shutting down.");
}
