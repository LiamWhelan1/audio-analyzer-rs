pub mod drone;
pub mod metronome;
use std::fmt;

use crate::types::{AudioChunk, Sample};

#[derive(Clone, Copy)]
pub enum Waveform {
    Sine,
    Square,
    Sawtooth,
    Triangle,
}

#[derive(Debug)]
pub enum Name {
    C,
    D,
    E,
    F,
    G,
    A,
    B,
}

pub enum Accidental {
    Sharp,
    Flat,
    DoubleSharp,
    DoubleFlat,
    Natural,
}

pub struct Note {
    name: Name,
    accidental: Option<Accidental>,
    octave: u8,
}

impl Note {
    pub fn new(note: &str) -> Self {
        let (name, accidental, octave) = Self::parse_note_name(note);
        Self {
            name,
            accidental,
            octave,
        }
    }

    fn parse_note_name(note: &str) -> (Name, Option<Accidental>, u8) {
        let chars: Vec<char> = note.chars().collect();
        if chars.len() < 2 {
            panic!("Invalid note format");
        }

        let name = match chars[0] {
            'C' => Name::C,
            'D' => Name::D,
            'E' => Name::E,
            'F' => Name::F,
            'G' => Name::G,
            'A' => Name::A,
            'B' => Name::B,
            _ => panic!("Invalid note name"),
        };

        let (accidental, octave_start) = if chars[1] == '#' {
            (Some(Accidental::Sharp), 2)
        } else if chars[1] == 'b' {
            (Some(Accidental::Flat), 2)
        } else if chars.len() > 2 && chars[1] == 'x' {
            (Some(Accidental::DoubleSharp), 2)
        } else if chars.len() > 2 && chars[1] == 'B' {
            (Some(Accidental::DoubleFlat), 2)
        } else if chars[1] == 'n' {
            (Some(Accidental::Natural), 2)
        } else {
            (None, 1)
        };

        let octave: u8 = chars[octave_start..]
            .iter()
            .collect::<String>()
            .parse()
            .expect("Invalid octave");

        (name, accidental, octave)
    }

    pub fn get_frequency(&self, base_freq: Option<f32>) -> f32 {
        let mut num_semis = match self.name {
            Name::C => -9,
            Name::D => -7,
            Name::E => -5,
            Name::F => -4,
            Name::G => -2,
            Name::A => 0,
            Name::B => 2,
        };
        num_semis += if let Some(acc) = &self.accidental {
            match acc {
                Accidental::Natural => 0,
                Accidental::Sharp => 1,
                Accidental::Flat => -1,
                Accidental::DoubleSharp => 2,
                Accidental::DoubleFlat => -2,
            }
        } else {
            0
        };
        num_semis += (self.octave as i32 - 4) * 12;
        let base = base_freq.unwrap_or(440.0);
        base * 2.0_f32.powf(num_semis as f32 / 12.0)
    }
}

impl fmt::Display for Note {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let accidental = if let Some(acc) = &self.accidental {
            match acc {
                Accidental::Sharp => "#",
                Accidental::Flat => "b",
                Accidental::Natural => "",
                Accidental::DoubleFlat => "bb",
                Accidental::DoubleSharp => "x",
            }
        } else {
            ""
        };
        write!(f, "{:?}{}{}", self.name, accidental, self.octave)
    }
}

pub struct ADSR {
    pub attack: f32,  // seconds
    pub decay: f32,   // seconds
    pub sustain: f32, // sustain level (0.0 - 1.0)
    pub release: f32, // seconds
}

impl ADSR {
    pub fn amplitude(&self, t: f32, duration: f32) -> f32 {
        if t < self.attack {
            // Linear ramp attack
            t / self.attack
        } else if t < self.attack + self.decay {
            // Decay toward sustain
            let d = (t - self.attack) / self.decay;
            1.0 + (self.sustain - 1.0) * d
        } else if t < duration - self.release {
            // Sustain
            self.sustain
        } else if t < duration {
            // Release
            let d = (t - (duration - self.release)) / self.release;
            self.sustain * (1.0 - d)
        } else {
            0.0
        }
    }
}

fn osc_sample(waveform: Waveform, phase: f32, harmonics: usize) -> f32 {
    match waveform {
        Waveform::Sine => phase.sin(),
        Waveform::Square => {
            // Fourier series for square wave
            (1..=harmonics)
                .step_by(2) // odd harmonics only
                .map(|k| (phase * k as f32).sin() / k as f32)
                .sum::<f32>()
                * (4.0 / std::f32::consts::PI)
        }
        Waveform::Sawtooth => {
            // Fourier series for sawtooth
            (1..=harmonics)
                .map(|k| -(phase * k as f32).sin() / k as f32)
                .sum::<f32>()
                * (2.0 / std::f32::consts::PI)
        }
        Waveform::Triangle => {
            // Fourier series for triangle wave
            (1..=harmonics)
                .step_by(2) // odd harmonics
                .map(|k| {
                    (phase * k as f32).sin() * (-1.0_f32).powi(((k - 1) / 2) as i32)
                        / (k * k) as f32
                })
                .sum::<f32>()
                * (8.0 / (std::f32::consts::PI * std::f32::consts::PI))
        }
    }
}

pub fn create_wave(
    waveform: Waveform,
    freq: f32,
    amplitude: f32,
    duration_secs: f32,
    sample_rate: u32,
    channels: u16,
    adsr: Option<&ADSR>,
    harmonics: usize, // used for non-sine waves
) -> AudioChunk {
    let total_frames = (duration_secs * sample_rate as f32).round() as usize;
    let mut samples = Vec::with_capacity(total_frames * channels as usize);

    let mut phase = 0.0;
    let phase_inc = 2.0 * std::f32::consts::PI * freq / sample_rate as f32;

    for n in 0..total_frames {
        let t = n as f32 / sample_rate as f32;
        let mut sample = osc_sample(waveform, phase, harmonics);

        // Envelope
        if let Some(env) = &adsr {
            sample *= env.amplitude(t, duration_secs);
        }

        sample *= amplitude;
        phase += phase_inc;
        if phase > 2.0 * std::f32::consts::PI {
            phase -= 2.0 * std::f32::consts::PI;
        }

        for _ in 0..channels {
            samples.push(sample);
        }
    }

    AudioChunk {
        samples,
        sample_rate,
        channels: channels.into(),
    }
}
