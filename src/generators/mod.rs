pub mod metronome;
pub mod player;
pub mod synth;

use std::path::Path;
use std::{f32::consts::PI, fs};

use metronome::BeatStrength;
use midly::{MidiMessage, Smf, Timing, TrackEventKind};

use crate::AudioEngineError;

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

impl Instrument {
    pub fn from(s: &str) -> Result<Instrument, AudioEngineError> {
        let instrument = s.to_lowercase();
        match instrument.as_str() {
            "piano" => Ok(Self::Piano),
            "violin" => Ok(Self::Violin),
            _ => Err(AudioEngineError::Internal {
                msg: format!("Instrument '{instrument}' is unavailable"),
            }),
        }
    }
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
#[derive(Copy, Clone, PartialEq, Debug)]
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
pub fn load_midi_file(
    path: &Path,
    instrument: Instrument,
    bpm: Option<f32>,
) -> Result<Vec<Measure>, String> {
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
    let mut current_bpm = bpm.unwrap_or(120.0);

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
                println!("BPM: {bpm}");
            }
            TrackEventKind::Meta(midly::MetaMessage::TimeSignature(n, d, _, _)) => {
                let denom = 2u8.pow(d as u32);
                sig_changes.push(SigChange {
                    beat,
                    num: n,
                    den: denom,
                });
                println!("Time signature: {n}/{denom}");
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
    let bpm_ratio = current_bpm
        / bpm_changes
            .get(0)
            .unwrap_or(&MapChange {
                beat: 0.0,
                new_val: current_bpm,
            })
            .new_val;

    while cursor < max_beat || cursor == 0.0 {
        if sig_idx < sig_changes.len() && sig_changes[sig_idx].beat <= cursor + 0.001 {
            current_time_sig = (sig_changes[sig_idx].num, sig_changes[sig_idx].den);
            sig_idx += 1;
        }
        if bpm_idx < bpm_changes.len() && bpm_changes[bpm_idx].beat <= cursor + 0.001 {
            current_bpm = bpm_changes[bpm_idx].new_val * bpm_ratio;
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
    println!("{measures:?}");
    Ok(measures)
}
