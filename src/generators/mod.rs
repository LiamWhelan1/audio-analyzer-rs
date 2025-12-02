pub mod metronome;
pub mod synth;
use std::f32::consts::PI;
use std::fmt::{self, write};

// ============================================================================
//  CONSTANTS & UTILS
// ============================================================================

const TWO_PI: f32 = 2.0 * PI;
const MIN_ENVELOPE: f32 = 0.001;
// Standard MIDI max velocity
const MAX_MIDI_VELOCITY: f32 = 127.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Waveform {
    Sine,
    Square,
    Sawtooth,
    Triangle,
}

#[derive(Clone, Debug)]
pub struct SynthNote {
    pub freq: f32,
    pub duration_beats: f32,
    pub onset_beats: f32,
    pub velocity: f32,
    pub waveform: Waveform,
}

pub struct MidiNote {
    midi: u8,
    cents: f32,
}

impl MidiNote {
    pub fn new(midi: u8, cents: f32) -> Self {
        Self { midi, cents }
    }

    pub fn from_freq(freq: f32, base_freq: Option<f32>) -> Self {
        let base = base_freq.unwrap_or(440.0);
        let base = base * 2.0_f32.powf(-4.75);
        let log = (freq / base).log2() * 1200.0;
        let mut cents = log % 100.0;
        cents = if cents < 50.0 {
            cents
        } else {
            -(100.0 - cents)
        };
        let midi = (log / 100.0).round() as u8 + 12;
        Self { midi, cents }
    }

    pub fn from_note(note: &Note) -> Self {
        let freq = note.to_freq(None);
        Self::from_freq(freq, None)
    }

    pub fn from_note_name(name: &str) -> Self {
        let note = Note::new(name);
        Self::from_note(&note)
    }

    pub fn to_freq(&self, base_freq: Option<f32>) -> f32 {
        let base = base_freq.unwrap_or(440.0);
        base * 2.0_f32.powf((self.midi as f32 - 69.0 + (self.cents / 100.0)) / 12.0)
    }
}

impl std::fmt::Display for MidiNote {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {}{:.4}",
            self.midi,
            if self.cents >= 0.0 { "+" } else { "" },
            self.cents
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Name {
    C,
    D,
    E,
    F,
    G,
    A,
    B,
}

#[derive(Clone, Copy)]
pub enum Accidental {
    Sharp,
    Flat,
    DoubleSharp,
    DoubleFlat,
    Natural,
}

pub enum Quality {
    Major,
    HarmonicMinor,
    MelodicMinor,
    NaturalMinor,
    Ionian,
    Dorian,
    Phrygian,
    Lydian,
    Mixolydian,
    Aeolian,
    Locrain,
}

pub struct Key {
    pub name: Name,
    pub accidental: Option<Accidental>,
    pub quality: Quality,
    pub semis_map: [u8; 7],
}

impl Key {
    pub fn new(key: &str) -> Self {
        let key = key.to_owned();
        let mut split = key.split_whitespace();
        let first = split.next().expect("invalid format");
        let name = match first.chars().next().expect("invalid format") {
            'C' => Name::C,
            'D' => Name::D,
            'E' => Name::E,
            'F' => Name::F,
            'G' => Name::G,
            'A' => Name::A,
            'B' => Name::B,
            _ => panic!("Invalid note name"),
        };
        let accidental = if let Some(a) = first.chars().next() {
            match a {
                '#' => Some(Accidental::Sharp),
                'x' => Some(Accidental::DoubleSharp),
                'b' => Some(Accidental::Flat),
                'n' => Some(Accidental::Natural),
                'B' => Some(Accidental::DoubleFlat),
                _ => panic!("Invalid note name"),
            }
        } else {
            None
        };
        let second = split.next().unwrap_or("Major");
        let quality = match second {
            "Major" => Quality::Major,
            "Minor" => Quality::NaturalMinor,
            "Harmonic" => Quality::HarmonicMinor,
            "Melodic" => Quality::MelodicMinor,
            "Ionian" => Quality::Ionian,
            "Dorian" => Quality::Dorian,
            "Phrygian" => Quality::Phrygian,
            "Lydian" => Quality::Lydian,
            "Mixolydian" => Quality::Mixolydian,
            "Aeolian" => Quality::Aeolian,
            "Locrian" => Quality::Locrain,
            _ => panic!("Invalid key"),
        };
        let semis_map = match quality {
            Quality::Major => [2, 2, 1, 2, 2, 2, 1],
            Quality::NaturalMinor => [2, 1, 2, 2, 1, 2, 2],
            Quality::HarmonicMinor => [2, 1, 2, 2, 1, 3, 1],
            Quality::MelodicMinor => [2, 1, 2, 2, 2, 2, 1],
            Quality::Ionian => [2, 2, 1, 2, 2, 2, 1],
            Quality::Dorian => [2, 1, 2, 2, 2, 1, 2],
            Quality::Phrygian => [1, 2, 2, 2, 1, 2, 2],
            Quality::Lydian => [2, 2, 2, 1, 2, 2, 1],
            Quality::Mixolydian => [2, 2, 1, 2, 2, 1, 2],
            Quality::Aeolian => [2, 1, 2, 2, 1, 2, 2],
            Quality::Locrain => [1, 2, 2, 1, 2, 2, 2],
        };
        Self {
            name,
            accidental,
            quality,
            semis_map,
        }
    }
}

pub struct Note {
    name: Name,
    accidental: Option<Accidental>,
    octave: u8,
    cents: f32,
}

impl Note {
    pub fn new(note: &str) -> Self {
        let (name, accidental, octave) = Self::parse_note_name(note);
        Self {
            name,
            accidental,
            octave,
            cents: 0.0,
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

    pub fn to_freq(&self, base_freq: Option<f32>) -> f32 {
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
        base * 2.0_f32.powf((num_semis as f32 + (self.cents / 100.0)) / 12.0)
    }

    pub fn from_freq(freq: f32, base_freq: Option<f32>) -> Self {
        let base = base_freq.unwrap_or(440.0);
        let base = base * 2.0_f32.powf(-4.75);
        let log = (freq / base).log2() * 1200.0;
        let octave = ((log + 50.0) / 1200.0) as u8;
        let semis = ((log / 100.0).round() % 12.0) as usize;
        let mut cents = log % 100.0;
        cents = if cents < 50.0 {
            cents
        } else {
            -(100.0 - cents)
        };
        let notes = [
            (Name::C, None),
            (Name::C, Some(Accidental::Sharp)),
            (Name::D, None),
            (Name::D, Some(Accidental::Sharp)),
            (Name::E, None),
            (Name::F, None),
            (Name::F, Some(Accidental::Sharp)),
            (Name::G, None),
            (Name::G, Some(Accidental::Sharp)),
            (Name::A, None),
            (Name::A, Some(Accidental::Sharp)),
            (Name::B, None),
        ];
        let (name, accidental) = notes[semis];
        Self {
            name,
            accidental,
            octave,
            cents,
        }
    }
    pub fn get_name(&self) -> String {
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
        format!("{:?}{}{}", self.name, accidental, self.octave)
    }
    pub fn get_cents(&self) -> f32 {
        self.cents
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
        write!(
            f,
            "{:?}{}{} {}{:.3}",
            self.name,
            accidental,
            self.octave,
            if self.cents >= 0.0 { "+" } else { "" },
            self.cents
        )
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
) {
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
}
