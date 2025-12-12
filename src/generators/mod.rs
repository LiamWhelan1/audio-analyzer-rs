pub mod metronome;
pub mod player;
pub mod synth;

use std::fmt;
use std::path::Path;
use std::{f32::consts::PI, fs};

use crate::audio_io::stft::TuningSystem;
use metronome::BeatStrength;
use midly::{MidiMessage, Smf, Timing, TrackEventKind};

/// Two pi constant used by generators.
const TWO_PI: f32 = 2.0 * PI;
/// Minimum envelope threshold for tick/voice retention.
const MIN_ENVELOPE: f32 = 0.001;
/// Standard MIDI max velocity.
const MAX_MIDI_VELOCITY: f32 = 127.0;

#[derive(Clone, Copy, Debug)]
/// High-level synthesizer instrument choices.
pub enum Instrument {
    Piano,
    Violin,
}

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
    /// The start time of the note *relative to the start of the measure*
    pub start_beat_in_measure: f32,
    pub duration_beats: f32,
    pub velocity: f32, // Normalized 0.0 - 1.0
    pub instrument: Instrument,
}

/// Instrument-specific synthesis parameters.
struct InstrumentParams {
    /// Attack time in seconds.
    attack_sec: f32,
    /// Decay time in seconds.
    decay_sec: f32,
    /// Sustain level (0.0 - 1.0).
    sustain_level: f32,
    /// Release time in seconds.
    release_sec: f32,
    /// Mix amount used to blend timbral components.
    timbre_mix: f32,
}

/// ADSR envelope states used by voices.
enum EnvState {
    Attack,
    Decay,
    Sustain,
    Release,
    Finished,
}

#[derive(Clone, Debug)]
/// A sequencer measure containing notes and timing info.
pub struct Measure {
    pub notes: Vec<SynthNote>,
    pub time_signature: (u8, u8),
    pub bpm: f32,
    pub global_start_beat: f64,
}

impl Measure {
    pub fn duration_beats(&self) -> f64 {
        (self.time_signature.0 as f64) * 4.0 / (self.time_signature.1 as f64)
    }

    /// Return a simple metronome pattern for the measure (downbeat strong).
    pub fn get_pattern(&self) -> Vec<BeatStrength> {
        let (num, _denom) = self.time_signature;
        let mut p = Vec::new();
        p.push(BeatStrength::Strong);
        for _ in 1..num {
            p.push(BeatStrength::Weak);
        }
        p
    }
}

/// Parse a MIDI file and convert tracks into sequencer `Measure`s.
pub fn load_midi_file(path: &Path, instrument: Instrument) -> Result<Vec<Measure>, String> {
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    let smf = Smf::parse(&bytes).map_err(|e| e.to_string())?;

    let ticks_per_beat = match smf.header.timing {
        Timing::Metrical(t) => t.as_int() as f64,
        _ => return Err("Timecode timing not supported, only Metrical".to_string()),
    };

    struct AbsEvent<'a> {
        abs_tick: u64,
        kind: TrackEventKind<'a>,
    }
    let mut events = Vec::new();

    for track in smf.tracks.iter() {
        let mut abs_tick = 0;
        for event in track {
            abs_tick += event.delta.as_int() as u64;
            events.push(AbsEvent {
                abs_tick,
                kind: event.kind,
            });
        }
    }
    events.sort_by_key(|e| e.abs_tick);

    let mut measures = Vec::new();
    let mut current_time_sig = (4, 4);
    let mut current_bpm = 120.0;

    let mut active_notes: [Option<(u64, u8)>; 128] = [None; 128];
    struct AbsNote {
        note: u8,
        start_beat: f64,
        end_beat: f64,
        velocity: f32,
    }
    let mut final_notes_abs = Vec::new();

    struct MapChange {
        beat: f64,
        new_val: f32,
    }
    struct SigChange {
        beat: f64,
        num: u8,
        den: u8,
    }
    let mut sig_changes = Vec::new();
    let mut bpm_changes = Vec::new();

    for event in events {
        let beat = event.abs_tick as f64 / ticks_per_beat;
        match event.kind {
            TrackEventKind::Meta(midly::MetaMessage::Tempo(micros)) => {
                let bpm = 60_000_000.0 / micros.as_int() as f32;
                bpm_changes.push(MapChange { beat, new_val: bpm });
            }
            TrackEventKind::Meta(midly::MetaMessage::TimeSignature(n, d, _, _)) => {
                let denom = 2u8.pow(d as u32);
                sig_changes.push(SigChange {
                    beat,
                    num: n,
                    den: denom,
                });
            }
            TrackEventKind::Midi { message, .. } => match message {
                MidiMessage::NoteOn { key, vel } => {
                    let k = key.as_int() as usize;
                    let v = vel.as_int();
                    if v > 0 {
                        active_notes[k] = Some((event.abs_tick, v));
                    } else if let Some((start_tick, start_vel)) = active_notes[k] {
                        let start_b = start_tick as f64 / ticks_per_beat;
                        final_notes_abs.push(AbsNote {
                            note: key.as_int(),
                            start_beat: start_b,
                            end_beat: beat,
                            velocity: start_vel as f32 / 127.0,
                        });
                        active_notes[k] = None;
                    }
                }
                MidiMessage::NoteOff { key, .. } => {
                    let k = key.as_int() as usize;
                    if let Some((start_tick, start_vel)) = active_notes[k] {
                        let start_b = start_tick as f64 / ticks_per_beat;
                        final_notes_abs.push(AbsNote {
                            note: key.as_int(),
                            start_beat: start_b,
                            end_beat: beat,
                            velocity: start_vel as f32 / 127.0,
                        });
                        active_notes[k] = None;
                    }
                }
                _ => {}
            },
            _ => {}
        }
    }

    let max_beat = final_notes_abs
        .iter()
        .map(|n| n.end_beat)
        .fold(0.0, f64::max);
    let mut cursor = 0.0;
    let mut sig_idx = 0;
    let mut bpm_idx = 0;

    while cursor < max_beat || cursor == 0.0 {
        if sig_idx < sig_changes.len() && sig_changes[sig_idx].beat <= cursor + 0.001 {
            current_time_sig = (sig_changes[sig_idx].num, sig_changes[sig_idx].den);
            sig_idx += 1;
        }
        if bpm_idx < bpm_changes.len() && bpm_changes[bpm_idx].beat <= cursor + 0.001 {
            current_bpm = bpm_changes[bpm_idx].new_val;
            bpm_idx += 1;
        }

        let beats_in_measure = (current_time_sig.0 as f64) * 4.0 / (current_time_sig.1 as f64);
        let end_of_measure = cursor + beats_in_measure;
        let mut measure_notes = Vec::new();
        for n in &final_notes_abs {
            if n.start_beat >= cursor && n.start_beat < end_of_measure {
                let freq = 440.0 * 2.0_f32.powf((n.note as f32 - 69.0) / 12.0);
                measure_notes.push(SynthNote {
                    freq,
                    start_beat_in_measure: (n.start_beat - cursor) as f32,
                    duration_beats: (n.end_beat - n.start_beat) as f32,
                    velocity: n.velocity,
                    instrument,
                });
            }
        }

        measures.push(Measure {
            notes: measure_notes,
            time_signature: current_time_sig,
            bpm: current_bpm,
            global_start_beat: cursor,
        });

        cursor = end_of_measure;
        if beats_in_measure <= 0.0 {
            break;
        }
    }

    Ok(measures)
}

#[derive(PartialEq)]
pub enum TunerMode {
    MultiPitch,
    SinglePitch,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntervalQuality {
    Major,
    Minor,
    Perfect,
    Augmented,
    Diminished,
}

pub struct Interval {
    pub bass_note: Note,
    pub top_note: Note,
    pub size: u8,
    pub quality: IntervalQuality,
    pub semis: u8,
    pub accuracy: f32,
}

impl Interval {
    pub fn new(
        freqs: Vec<f32>,
        base_freq: Option<f32>,
        system: Option<TuningSystem>,
    ) -> Option<Self> {
        Self::from_freqs(freqs, base_freq, system)
    }

    pub fn from_freqs(
        freqs: Vec<f32>,
        base_freq: Option<f32>,
        _system: Option<TuningSystem>,
    ) -> Option<Self> {
        if freqs.len() != 2 {
            return None;
        }

        let bass = freqs[0];
        let top = freqs[1];

        // Ensure bass is lower than top, swap if needed for interval logic
        let (bass, top) = if bass < top { (bass, top) } else { (top, bass) };

        let bass_note = Note::from_freq(bass, base_freq);
        let top_note = Note::from_freq(top, base_freq);

        // 1. Calculate Semitones
        // Formula: 12 * log2(f2 / f1)
        let semitone_float = 12.0 * (top / bass).log2();
        let semis = semitone_float.round() as u8;

        // 2. Determine Size and Quality based on semitones (Standard 12-TET mapping)
        // Note: This matches standard Western theory. Tritone is mapped to Diminished 5th here.
        let (size, quality) = match semis % 12 {
            0 => (1, IntervalQuality::Perfect),    // Unison
            1 => (2, IntervalQuality::Minor),      // m2
            2 => (2, IntervalQuality::Major),      // M2
            3 => (3, IntervalQuality::Minor),      // m3
            4 => (3, IntervalQuality::Major),      // M3
            5 => (4, IntervalQuality::Perfect),    // P4
            6 => (5, IntervalQuality::Diminished), // d5 (Tritone)
            7 => (5, IntervalQuality::Perfect),    // P5
            8 => (6, IntervalQuality::Minor),      // m6
            9 => (6, IntervalQuality::Major),      // M6
            10 => (7, IntervalQuality::Minor),     // m7
            11 => (7, IntervalQuality::Major),     // M7
            _ => (8, IntervalQuality::Perfect), // Octave (should be caught by mod, but for safety)
        };

        // Adjust size for octaves (e.g. 13 semitones is a m2 + octave, so size is 9)
        let octave_shift = (semis / 12) * 7;
        let adjusted_size = size + octave_shift;

        // 3. Calculate Accuracy
        // We calculate deviation in "cents". 1 semitone = 100 cents.
        // If the note is perfectly on the grid, diff is 0.0.
        // Accuracy score: 1.0 is perfect, 0.0 is 50 cents off (quarter tone).
        let diff_semis = (semitone_float - semis as f32).abs();
        let accuracy = (1.0 - (diff_semis * 2.0)).max(0.0);

        Some(Self {
            bass_note,
            top_note,
            size: adjusted_size,
            quality,
            semis,
            accuracy,
        })
    }

    pub fn get_accuracy(&self) -> f32 {
        self.accuracy
    }

    pub fn get_name(&self) -> String {
        let s = match (self.size - 1) % 7 {
            0 => "Oct", // Or Unison if semis is 0
            1 => "2nd",
            2 => "3rd",
            3 => "4th",
            4 => "5th",
            5 => "6th",
            6 => "7th",
            _ => "Unison",
        };

        // Handling Unison edge case
        let s = if self.size == 1 { "Uni" } else { s };

        let q = match self.quality {
            IntervalQuality::Augmented => "+",
            IntervalQuality::Diminished => "o",
            IntervalQuality::Major => "M",
            IntervalQuality::Minor => "m",
            IntervalQuality::Perfect => "P",
        };
        format!("{q} {s}")
    }
}

// ==========================================
//               CHORD LOGIC
// ==========================================

pub struct Chord {
    pub name: String,
    pub root_note: Note,
    pub intervals: Vec<Interval>,
    pub quality: String, // e.g. "Major Triad", "Minor 7", "Inverted"
}

impl Chord {
    /// Constructs a chord from a sorted vector of frequencies.
    /// It calculates intervals between neighbors (0->1, 1->2, etc.)
    /// and matches the sequence of intervals to known chord shapes.
    pub fn new(mut freqs: Vec<f32>) -> Option<Self> {
        if freqs.len() < 3 {
            return None; // Need at least 3 notes for a chord (triad)
        }

        // Sort frequencies low to high
        freqs.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Create the stack of intervals
        let mut intervals = Vec::new();
        let mut semitone_stack = Vec::new();

        for i in 0..freqs.len() - 1 {
            let pair = vec![freqs[i], freqs[i + 1]];
            if let Some(interval) = Interval::from_freqs(pair, None, None) {
                semitone_stack.push(interval.semis);
                intervals.push(interval);
            }
        }

        // Identify Chord based on semitone stack
        // The tuple result is (Chord Name, Index of Root Note in the freq list)
        let (quality_name, root_index) = match semitone_stack.as_slice() {
            // --- TRIADS ---
            [4, 3] => ("Major", 0),           // Root Pos: M3, m3 (C E G)
            [3, 5] => ("Major / 1st Inv", 2), // 1st Inv: m3, P4 (E G C) -> Root is Top (index 2)
            [5, 4] => ("Major / 2nd Inv", 1), // 2nd Inv: P4, M3 (G C E) -> Root is Mid (index 1)

            [3, 4] => ("Minor", 0),           // Root Pos: m3, M3 (C Eb G)
            [4, 5] => ("Minor / 1st Inv", 2), // 1st Inv: M3, P4 (Eb G C) -> Root is Top (index 2)
            [5, 3] => ("Minor / 2nd Inv", 1), // 2nd Inv: P4, m3 (G C Eb) -> Root is Mid (index 1)

            [3, 3] => ("Diminished", 0), // m3, m3

            // --- 7th CHORDS (4 Notes) ---
            [4, 3, 4] => ("Major 7", 0),    // M3, m3, M3 (C E G B)
            [3, 4, 3] => ("Minor 7", 0),    // m3, M3, m3 (C Eb G Bb)
            [4, 3, 3] => ("Dominant 7", 0), // M3, m3, m3 (C E G Bb)

            // --- SPECIFIC EXAMPLE FROM PROMPT ---
            // "P4 then m6 is a minor chord on whatever the middle frequency is"
            // P4 = 5 semis, m6 = 8 semis.
            // Notes: 0, 5, 13. (e.g., G, C, Ab).
            // Usually this is Fm(add9) no 5th, or some quartal harmony,
            // but we map it per user request:
            [5, 8] => ("User Defined Minor", 1),

            _ => ("Unknown", 0),
        };

        // Determine the actual root note based on the index returned
        // The root_index refers to the index in the sorted `freqs` vector
        let root_freq = freqs[root_index];
        let root_note = Note::from_freq(root_freq, None);

        let name = format!("{} {}", root_note.get_name(), quality_name);

        Some(Self {
            name,
            root_note,
            intervals,
            quality: quality_name.to_string(),
        })
    }
}
