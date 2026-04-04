use std::fmt;

use crate::analysis::tuner::TuningSystem;
use serde::Serialize;

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

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Interval {
    name: IntType,
    accuracy: f32,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub enum IntType {
    Min2,
    Maj2,
    Min3,
    Maj3,
    Per4,
    Aug4,
    Per5,
    Min6,
    Maj6,
    Min7,
    Maj7,
    Per8,
}

impl Interval {
    pub fn new(freqs: &Vec<f32>, system: Option<TuningSystem>) -> Self {
        let mut ratio = freqs[1] / freqs[0];
        while ratio > 2.0 {
            ratio /= 2.0;
        }
        let ratios: [f32; 13] = match system {
            Some(TuningSystem::JustIntonation) => [
                1.0, 1.0667, 1.125, 1.2, 1.25, 1.3333, 1.4063, 1.5, 1.6, 1.6667, 1.8, 1.875, 2.0,
            ],
            _ => [
                1.0, 1.0595, 1.1225, 1.1892, 1.2599, 1.3348, 1.4142, 1.4983, 1.5874, 1.6818,
                1.7818, 1.8877, 2.0,
            ],
        };
        let (idx, closest_ratio): (usize, &f32) = ratios
            .iter()
            .enumerate()
            .min_by(|a, b| {
                let diff_a = (ratio - *a.1).abs();
                let diff_b = (ratio - *b.1).abs();
                diff_a.total_cmp(&diff_b)
            })
            .unwrap();
        let name = match idx {
            0 => IntType::Per8,
            1 => IntType::Min2,
            2 => IntType::Maj2,
            3 => IntType::Min3,
            4 => IntType::Maj3,
            5 => IntType::Per4,
            6 => IntType::Aug4,
            7 => IntType::Per5,
            8 => IntType::Min6,
            9 => IntType::Maj6,
            10 => IntType::Min7,
            11 => IntType::Maj7,
            12 => IntType::Per8,
            _ => panic!("Not possible"),
        };
        return Self {
            name,
            accuracy: (closest_ratio - ratio) * 100.0,
        };
    }

    pub fn get_name(&self) -> String {
        format!("{:?}", self.name)
    }

    pub fn get_accuracy(&self) -> f32 {
        self.accuracy
    }
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
