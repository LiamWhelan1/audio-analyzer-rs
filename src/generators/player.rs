use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use rtrb::{Consumer, Producer, RingBuffer};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::traits::AudioSource;

#[derive(Clone, Debug)]
/// Commands for controlling the `AudioPlayer` running on the audio thread.
pub enum PlayerCommand {
    /// Send new audio data (Samples, Sample Rate, Channels)
    LoadTrack(Arc<Vec<f32>>, u32, usize),
    Play,
    Pause,
    Stop,
    /// Seek to a specific time in seconds
    Seek(f64),
}

/// AudioSource implementation that plays decoded audio buffers.
pub struct AudioPlayer {
    playing: bool,
    finished: bool,
    position_frames: f64,

    samples: Arc<Vec<f32>>,
    sample_rate: u32,
    source_channels: usize,

    system_sample_rate: f32,
    playback_rate_ratio: f64,

    command_rx: Consumer<PlayerCommand>,
}

impl AudioPlayer {
    /// Construct a new `AudioPlayer` and its controller.
    pub fn new(system_sample_rate: u32) -> (Self, PlayerController) {
        let (prod, cons) = RingBuffer::new(64);

        let player = Self {
            playing: false,
            finished: false,
            position_frames: 0.0,
            samples: Arc::new(Vec::new()),
            sample_rate: 44100,
            source_channels: 2,
            system_sample_rate: system_sample_rate as f32,
            playback_rate_ratio: 1.0,
            command_rx: cons,
        };

        let controller = PlayerController { command_tx: prod };

        (player, controller)
    }

    /// Poll and handle pending `PlayerCommand`s from the controller.
    fn handle_commands(&mut self) {
        while let Ok(cmd) = self.command_rx.pop() {
            match cmd {
                PlayerCommand::LoadTrack(data, sr, channels) => {
                    self.samples = data;
                    self.sample_rate = sr;
                    self.source_channels = channels;
                    self.position_frames = 0.0;
                    self.playing = false;
                    self.playback_rate_ratio =
                        self.sample_rate as f64 / self.system_sample_rate as f64;
                }
                PlayerCommand::Play => self.playing = true,
                PlayerCommand::Pause => self.playing = false,
                PlayerCommand::Stop => {
                    self.playing = false;
                    self.position_frames = 0.0;
                }
                PlayerCommand::Seek(time_secs) => {
                    let target_frame = time_secs * self.sample_rate as f64;
                    let max_frame = (self.samples.len() / self.source_channels) as f64;
                    self.position_frames = target_frame.clamp(0.0, max_frame);
                }
            }
        }
    }
}

impl AudioSource for AudioPlayer {
    /// Returns true when the player has been explicitly finished/stopped.
    fn is_finished(&self) -> bool {
        self.finished
    }

    /// Fill `buffer` with resampled/interpolated audio from the loaded track.
    fn process(&mut self, buffer: &mut [f32], output_channels: usize) {
        self.handle_commands();

        if !self.playing || self.samples.is_empty() {
            return;
        }

        let num_frames = buffer.len() / output_channels;
        let total_source_frames = self.samples.len() / self.source_channels;

        for frame_idx in 0..num_frames {
            if self.position_frames as usize >= total_source_frames - 1 {
                self.playing = false;
                self.position_frames = 0.0;
                break;
            }

            let index_floor = self.position_frames.floor() as usize;
            let frac = (self.position_frames - index_floor as f64) as f32;
            let next_index = index_floor + 1;

            let current_sample_idx = index_floor * self.source_channels;
            let next_sample_idx = next_index * self.source_channels;

            for channel in 0..output_channels {
                let src_ch = if channel < self.source_channels {
                    channel
                } else {
                    0
                };

                let sample_current = self.samples[current_sample_idx + src_ch];
                let sample_next = self.samples[next_sample_idx + src_ch];

                let sample = sample_current + frac * (sample_next - sample_current);

                buffer[frame_idx * output_channels + channel] += sample;
            }

            self.position_frames += self.playback_rate_ratio;
        }
    }
}

pub struct PlayerController {
    command_tx: Producer<PlayerCommand>,
}

impl PlayerController {
    /// Send a `Play` command to the player.
    pub fn play(&mut self) {
        let _ = self.command_tx.push(PlayerCommand::Play);
    }

    /// Send a `Pause` command to the player.
    pub fn pause(&mut self) {
        let _ = self.command_tx.push(PlayerCommand::Pause);
    }

    /// Send a `Stop` command to the player.
    pub fn stop(&mut self) {
        let _ = self.command_tx.push(PlayerCommand::Stop);
    }

    /// Seek the player to `time_in_seconds`.
    pub fn seek(&mut self, time_in_seconds: f64) {
        let _ = self.command_tx.push(PlayerCommand::Seek(time_in_seconds));
    }

    /// Load and decode an audio file, then push it to the audio thread.
    pub fn load_file<P: AsRef<Path>>(&mut self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let src = File::open(path)?;
        let mss = MediaSourceStream::new(Box::new(src), Default::default());
        let hint = Hint::new();

        let format_opts = FormatOptions::default();
        let metadata_opts = MetadataOptions::default();
        let decoder_opts = DecoderOptions::default();

        let probed =
            symphonia::default::get_probe().format(&hint, mss, &format_opts, &metadata_opts)?;

        let mut format = probed.format;
        let track = format.default_track().ok_or("No track found")?;
        let track_id = track.id;
        let mut decoder =
            symphonia::default::get_codecs().make(&track.codec_params, &decoder_opts)?;

        let mut audio_buffer: Vec<f32> = Vec::new();
        let mut sample_rate = 44100;
        let mut channels = 2;

        loop {
            let packet = match format.next_packet() {
                Ok(packet) => packet,
                Err(Error::IoError(_)) => break,
                Err(Error::ResetRequired) => continue,
                Err(err) => return Err(Box::new(err)),
            };

            if packet.track_id() != track_id {
                continue;
            }

            match decoder.decode(&packet) {
                Ok(decoded) => {
                    let spec = *decoded.spec();
                    sample_rate = spec.rate;
                    channels = spec.channels.count();

                    let duration = decoded.capacity();

                    let mut buf =
                        symphonia::core::audio::SampleBuffer::<f32>::new(duration as u64, spec);
                    buf.copy_interleaved_ref(decoded);

                    audio_buffer.extend_from_slice(buf.samples());
                }
                Err(Error::IoError(_)) => break,
                Err(Error::DecodeError(_)) => continue,
                Err(err) => return Err(Box::new(err)),
            }
        }

        let _ = self.command_tx.push(PlayerCommand::LoadTrack(
            Arc::new(audio_buffer),
            sample_rate,
            channels,
        ));

        Ok(())
    }
}
