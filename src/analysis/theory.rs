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
    /// Parse a note name such as `"C#4"` or `"Bb3"`.
    ///
    /// Returns `Ok(Self)` on success or an `Err` with a descriptive message on
    /// invalid input — never panics.
    pub fn try_new(note: &str) -> Result<Self, String> {
        let (name, accidental, octave) = Self::parse_note_name(note)?;
        Ok(Self {
            name,
            accidental,
            octave,
            cents: 0.0,
        })
    }

    /// Panicking wrapper around [`try_new`] for callers that treat a bad note
    /// name as a programming error.  Prefer [`try_new`] when the name comes
    /// from external input.
    pub fn new(note: &str) -> Self {
        Self::try_new(note)
            .unwrap_or_else(|e| panic!("Note::new called with invalid note \"{note}\": {e}"))
    }

    fn parse_note_name(note: &str) -> Result<(Name, Option<Accidental>, u8), String> {
        // Note names are always ASCII — work with bytes to avoid Vec<char> allocation.
        let bytes = note.as_bytes();
        if bytes.len() < 2 {
            return Err(format!(
                "Note name \"{note}\" is too short — expected format like \"C#4\" or \"A4\""
            ));
        }

        let name = match bytes[0] {
            b'C' => Name::C,
            b'D' => Name::D,
            b'E' => Name::E,
            b'F' => Name::F,
            b'G' => Name::G,
            b'A' => Name::A,
            b'B' => Name::B,
            other => {
                return Err(format!(
                    "Invalid note letter '{}' in \"{note}\" — expected one of C D E F G A B",
                    other as char
                ))
            }
        };

        let (accidental, octave_start) = if bytes[1] == b'#' {
            (Some(Accidental::Sharp), 2)
        } else if bytes[1] == b'b' {
            (Some(Accidental::Flat), 2)
        } else if bytes.len() > 2 && bytes[1] == b'x' {
            (Some(Accidental::DoubleSharp), 2)
        } else if bytes.len() > 2 && bytes[1] == b'B' {
            (Some(Accidental::DoubleFlat), 2)
        } else if bytes[1] == b'n' {
            (Some(Accidental::Natural), 2)
        } else {
            (None, 1)
        };

        let octave_str = &note[octave_start..];
        let octave: u8 = octave_str.parse().map_err(|_| {
            format!(
                "Invalid octave \"{octave_str}\" in \"{note}\" — expected a number like 4"
            )
        })?;

        Ok((name, accidental, octave))
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
    /// Identify the interval between two frequencies.
    ///
    /// Requires at least two elements; if `freqs` has fewer than two elements,
    /// or if the first frequency is zero, returns a perfect-octave default
    /// rather than panicking.
    pub fn new(freqs: &[f32], system: Option<TuningSystem>) -> Self {
        if freqs.len() < 2 || freqs[0] == 0.0 {
            return Self {
                name: IntType::Per8,
                accuracy: 0.0,
            };
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    // ── Note identification ────────────────────────────────────────────────────

    #[test]
    fn note_from_freq_a4_identifies_correctly() {
        let note = Note::from_freq(440.0, None);
        assert_eq!(note.get_name(), "A4");
    }

    #[test]
    fn note_from_freq_a4_cents_near_zero() {
        let note = Note::from_freq(440.0, None);
        assert!(
            note.get_cents().abs() < 2.0,
            "A4 at exactly 440 Hz should have cents near 0, got {}",
            note.get_cents()
        );
    }

    #[test]
    fn note_from_freq_c4_identifies_correctly() {
        let note = Note::from_freq(261.626, None);
        assert_eq!(note.get_name(), "C4");
    }

    #[test]
    fn note_from_freq_c_sharp_4_identifies_correctly() {
        // C#4 = 261.626 * 2^(1/12) ≈ 277.18 Hz
        let c_sharp_4 = 261.626_f32 * 2.0_f32.powf(1.0 / 12.0);
        let note = Note::from_freq(c_sharp_4, None);
        assert_eq!(note.get_name(), "C#4");
    }

    #[test]
    fn note_cents_always_within_fifty_semitone_range() {
        // Cents should always be in [-50, 50] — one rule of the representation.
        let freqs = [261.63, 293.66, 329.63, 349.23, 392.0, 440.0, 493.88, 523.25];
        for freq in freqs {
            let cents = Note::from_freq(freq, None).get_cents();
            assert!(
                cents >= -50.0 && cents <= 50.0,
                "cents {} out of range for freq {}",
                cents,
                freq
            );
        }
    }

    // ── Note::new / to_freq round-trips ───────────────────────────────────────

    #[test]
    fn note_new_a4_to_freq_round_trip() {
        let freq = Note::new("A4").to_freq(None);
        assert!(
            (freq - 440.0).abs() < 0.1,
            "A4.to_freq() should be ~440 Hz, got {}",
            freq
        );
    }

    #[test]
    fn note_new_c4_to_freq_round_trip() {
        let freq = Note::new("C4").to_freq(None);
        // C4 = 440 * 2^(-9/12) ≈ 261.63 Hz
        assert!(
            (freq - 261.63).abs() < 0.5,
            "C4.to_freq() should be ~261.63 Hz, got {}",
            freq
        );
    }

    #[test]
    fn note_new_with_sharp_correct_frequency() {
        // C#4 is 1 semitone above C4
        let c4 = Note::new("C4").to_freq(None);
        let c_sharp_4 = Note::new("C#4").to_freq(None);
        let expected_ratio = 2.0_f32.powf(1.0 / 12.0);
        assert!(
            (c_sharp_4 / c4 - expected_ratio).abs() < 0.001,
            "C#4/C4 ratio should be 2^(1/12), got {}",
            c_sharp_4 / c4
        );
    }

    #[test]
    fn note_new_with_flat_correct_frequency() {
        // Bb3 is 1 semitone below B3
        let b3 = Note::new("B3").to_freq(None);
        let bb3 = Note::new("Bb3").to_freq(None);
        let expected_ratio = 2.0_f32.powf(-1.0 / 12.0);
        assert!(
            (bb3 / b3 - expected_ratio).abs() < 0.001,
            "Bb3/B3 ratio should be 2^(-1/12), got {}",
            bb3 / b3
        );
    }

    // ── Note::try_new ────────────────────────────────────────────────────────────

    #[test]
    fn note_try_new_valid_note_returns_ok() {
        let result = Note::try_new("A4");
        assert!(result.is_ok(), "try_new(\"A4\") should succeed");
        assert_eq!(result.unwrap().get_name(), "A4");
    }

    #[test]
    fn note_try_new_invalid_name_returns_err_not_panic() {
        let result = Note::try_new("X4");
        assert!(
            result.is_err(),
            "try_new(\"X4\") should return Err, not panic"
        );
        let msg = result.err().unwrap();
        assert!(
            msg.contains("X") || msg.contains("note") || msg.contains("invalid") || msg.to_lowercase().contains("invalid"),
            "error message should describe the problem, got: {msg}"
        );
    }

    #[test]
    fn note_try_new_too_short_returns_err_not_panic() {
        let result = Note::try_new("A");
        assert!(
            result.is_err(),
            "try_new(\"A\") (no octave) should return Err, not panic"
        );
    }

    #[test]
    fn note_try_new_empty_string_returns_err_not_panic() {
        let result = Note::try_new("");
        assert!(result.is_err(), "try_new(\"\") should return Err, not panic");
    }

    // ── Interval detection ────────────────────────────────────────────────────

    #[test]
    fn interval_perfect_fifth_equal_temperament() {
        let c4 = 261.63_f32;
        let g4 = c4 * 2.0_f32.powf(7.0 / 12.0); // exact ET perfect fifth
        let interval = Interval::new(&[c4, g4], None);
        assert_eq!(interval.get_name(), "Per5");
    }

    #[test]
    fn interval_octave_equal_temperament() {
        let c4 = 261.63_f32;
        let c5 = c4 * 2.0;
        let interval = Interval::new(&[c4, c5], None);
        assert_eq!(interval.get_name(), "Per8");
    }

    #[test]
    fn interval_major_third_equal_temperament() {
        let c4 = 261.63_f32;
        let e4 = c4 * 2.0_f32.powf(4.0 / 12.0);
        let interval = Interval::new(&[c4, e4], None);
        assert_eq!(interval.get_name(), "Maj3");
    }

    #[test]
    fn interval_minor_third_equal_temperament() {
        let c4 = 261.63_f32;
        let eb4 = c4 * 2.0_f32.powf(3.0 / 12.0);
        let interval = Interval::new(&[c4, eb4], None);
        assert_eq!(interval.get_name(), "Min3");
    }

    #[test]
    fn interval_perfect_fourth_equal_temperament() {
        let c4 = 261.63_f32;
        let f4 = c4 * 2.0_f32.powf(5.0 / 12.0);
        let interval = Interval::new(&[c4, f4], None);
        assert_eq!(interval.get_name(), "Per4");
    }

    // ── Interval edge cases ───────────────────────────────────────────────────

    #[test]
    fn interval_new_single_freq_does_not_panic() {
        let result = std::panic::catch_unwind(|| Interval::new(&[440.0_f32], None));
        assert!(
            result.is_ok(),
            "Interval::new with fewer than 2 frequencies must not panic"
        );
    }

    #[test]
    fn interval_new_empty_slice_does_not_panic() {
        let result = std::panic::catch_unwind(|| Interval::new(&[], None));
        assert!(
            result.is_ok(),
            "Interval::new with empty slice must not panic"
        );
    }

    // ── MidiNote ─────────────────────────────────────────────────────────────

    #[test]
    fn midi_note_from_freq_a4_round_trips() {
        let midi = MidiNote::from_freq(440.0, None);
        let freq_back = midi.to_freq(None);
        assert!(
            (freq_back - 440.0).abs() < 1.0,
            "MidiNote round-trip for A4 should return ~440 Hz, got {}",
            freq_back
        );
    }

    #[test]
    fn midi_note_from_freq_c4_round_trips() {
        let midi = MidiNote::from_freq(261.63, None);
        let freq_back = midi.to_freq(None);
        assert!(
            (freq_back - 261.63).abs() < 1.0,
            "MidiNote round-trip for C4 should return ~261.63 Hz, got {}",
            freq_back
        );
    }
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
