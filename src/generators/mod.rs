pub mod metronome;
pub mod player;
pub mod synth;
use std::fmt;
use std::path::Path;
use std::{f32::consts::PI, fs};

use metronome::BeatStrength;
use midly::{MidiMessage, Smf, Timing, TrackEventKind};

// ============================================================================
//  CONSTANTS & UTILS
// ============================================================================

const TWO_PI: f32 = 2.0 * PI;
const MIN_ENVELOPE: f32 = 0.001;
// Standard MIDI max velocity
const MAX_MIDI_VELOCITY: f32 = 127.0;

#[derive(Clone, Copy, Debug)]
pub enum Instrument {
    Piano,
    Violin,
    // Add more instruments here later (e.g., Flute, Bass, etc.)
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

struct InstrumentParams {
    // Envelope times (in seconds)
    attack_sec: f32,
    decay_sec: f32,
    sustain_level: f32, // The target level after decay (0.0 to 1.0)
    release_sec: f32,

    // Timbre characteristics
    timbre_mix: f32, // A value used to mix fundamental and harmonics/noise
}

enum EnvState {
    Attack,
    Decay,
    Sustain,
    Release,
    Finished,
}

#[derive(Clone, Debug)]
pub struct Measure {
    pub notes: Vec<SynthNote>,
    pub time_signature: (u8, u8), // (numerator, denominator)
    pub bpm: f32,
    pub global_start_beat: f64, // Absolute beat where this measure starts
}

impl Measure {
    pub fn duration_beats(&self) -> f64 {
        (self.time_signature.0 as f64) * 4.0 / (self.time_signature.1 as f64)
    }

    /// Returns the metronome pattern for this measure
    pub fn get_pattern(&self) -> Vec<BeatStrength> {
        let (num, _denom) = self.time_signature;
        let mut p = Vec::new();

        // Standard logic: Downbeat is Strong.
        p.push(BeatStrength::Strong);

        // Simple logic for remaining beats
        for _ in 1..num {
            p.push(BeatStrength::Weak);
        }
        p
    }
}

pub fn load_midi_file(path: &Path) -> Result<Vec<Measure>, String> {
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    let smf = Smf::parse(&bytes).map_err(|e| e.to_string())?;

    let ticks_per_beat = match smf.header.timing {
        Timing::Metrical(t) => t.as_int() as f64,
        _ => return Err("Timecode timing not supported, only Metrical".to_string()),
    };

    // 1. Flatten all tracks into a single event list with absolute ticks
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
    // Sort by time so we process in order
    events.sort_by_key(|e| e.abs_tick);

    // 2. Process Events into Notes and Measures
    let mut measures = Vec::new();

    // Default Context
    let mut current_time_sig = (4, 4);
    let mut current_bpm = 120.0;

    // Temporary storage for active Note-Ons to pair with Note-Offs
    // Key: MIDI Note Number, Value: (Start Tick, Velocity)
    let mut active_notes: [Option<(u64, u8)>; 128] = [None; 128];

    struct AbsNote {
        note: u8,
        start_beat: f64,
        end_beat: f64,
        velocity: f32,
    }
    let mut final_notes_abs = Vec::new();

    // Map Changes
    struct MapChange {
        beat: f64,
        new_val: f32,
    } // Used for BPM
    struct SigChange {
        beat: f64,
        num: u8,
        den: u8,
    }

    let mut sig_changes = Vec::new();
    let mut bpm_changes = Vec::new();

    // Pass 1: Extract Notes and Map Changes
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
            TrackEventKind::Midi { message, .. } => {
                match message {
                    MidiMessage::NoteOn { key, vel } => {
                        let k = key.as_int() as usize;
                        let v = vel.as_int();
                        if v > 0 {
                            active_notes[k] = Some((event.abs_tick, v));
                        } else {
                            // Velocity 0 is Note Off
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
                }
            }
            _ => {}
        }
    }

    // Pass 2: Slice into Measures
    let max_beat = final_notes_abs
        .iter()
        .map(|n| n.end_beat)
        .fold(0.0, f64::max);
    let mut cursor = 0.0;
    let mut sig_idx = 0;
    let mut bpm_idx = 0;

    // While we haven't covered the full duration
    while cursor < max_beat || cursor == 0.0 {
        // Update Time Sig if needed
        if sig_idx < sig_changes.len() && sig_changes[sig_idx].beat <= cursor + 0.001 {
            current_time_sig = (sig_changes[sig_idx].num, sig_changes[sig_idx].den);
            sig_idx += 1;
        }
        // Update BPM if needed
        if bpm_idx < bpm_changes.len() && bpm_changes[bpm_idx].beat <= cursor + 0.001 {
            current_bpm = bpm_changes[bpm_idx].new_val;
            bpm_idx += 1;
        }

        let beats_in_measure = (current_time_sig.0 as f64) * 4.0 / (current_time_sig.1 as f64);
        let end_of_measure = cursor + beats_in_measure;
        let instrument = Instrument::Piano;
        // Collect notes that start within this measure
        let mut measure_notes = Vec::new();
        for n in &final_notes_abs {
            // Note belongs to this measure if it starts here
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

        // Safety break for infinite loops if file is empty or calc error
        if beats_in_measure <= 0.0 {
            break;
        }
    }

    Ok(measures)
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
