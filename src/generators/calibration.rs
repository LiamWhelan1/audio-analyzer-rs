use std::sync::{
    Arc,
    atomic::{AtomicI64, Ordering},
};

use crate::{audio_io::timing::MusicalTransport, traits::AudioSource};

use super::{MIN_ENVELOPE, TWO_PI};

/// One-shot click used to measure end-to-end audio round-trip latency.
///
/// Schedules a brief sinusoidal click at a future output frame, publishes
/// the actual frame number on a shared atomic, and finishes once the
/// envelope decays.  The `OnsetDetector` reads the atomic to learn the
/// expected frame and stores the residual on the transport as
/// `calibration_offset_samples`.
///
/// Unlike the `Metronome`, this source deliberately does **not** call
/// `notify_tick_at_frame` — we want the click to be detected, not
/// suppressed by the echo-gate.
pub struct CalibrationClick {
    transport: Arc<MusicalTransport>,
    sample_rate: f32,
    target_frame: i64,
    actual_frame: Arc<AtomicI64>,
    fired: bool,
    finished: bool,
    phase: f32,
    envelope: f32,
    decay_rate: f32,
    volume: f32,
}

impl CalibrationClick {
    /// Create a click scheduled `delay_samples` ahead of the current output
    /// frame.  `actual_frame` is the shared atomic the detector polls for
    /// the target frame number.
    pub fn new(
        transport: Arc<MusicalTransport>,
        sample_rate: f32,
        delay_samples: i64,
        actual_frame: Arc<AtomicI64>,
        volume: f32,
    ) -> Self {
        let target_frame = transport.get_output_frames() + delay_samples;
        // 50 ms decay — sharp attack, brief tail, easy onset target.
        let decay_samples_count = sample_rate * 0.05;
        let decay_rate = MIN_ENVELOPE.powf(1.0 / decay_samples_count);
        Self {
            transport,
            sample_rate,
            target_frame,
            actual_frame,
            fired: false,
            finished: false,
            phase: 0.0,
            envelope: 1.0,
            decay_rate,
            volume,
        }
    }
}

impl AudioSource for CalibrationClick {
    fn is_finished(&self) -> bool {
        self.finished
    }

    fn process(&mut self, buffer: &mut [f32], channels: usize) {
        if self.finished {
            return;
        }

        let total_frames = (buffer.len() / channels) as i64;
        let buffer_start_frame = self.transport.get_output_frames() - total_frames;

        let start_offset: usize = if !self.fired {
            let off = self.target_frame - buffer_start_frame;
            if off < 0 {
                // Target was in a past buffer — fire immediately at frame 0.
                self.actual_frame
                    .store(buffer_start_frame, Ordering::SeqCst);
                self.fired = true;
                0
            } else if off < total_frames {
                self.actual_frame.store(self.target_frame, Ordering::SeqCst);
                self.fired = true;
                off as usize
            } else {
                // Not yet — emit silence this buffer.
                return;
            }
        } else {
            0
        };

        const FREQ: f32 = 2500.0;
        let phase_inc = (FREQ * TWO_PI) / self.sample_rate;
        for f in start_offset..(total_frames as usize) {
            let s = (self.phase * phase_inc).sin() * self.volume * self.envelope;
            self.phase += 1.0;
            self.envelope *= self.decay_rate;
            let base = f * channels;
            for ch in 0..channels {
                buffer[base + ch] += s;
            }
            if self.envelope <= MIN_ENVELOPE {
                self.finished = true;
                break;
            }
        }
    }
}
