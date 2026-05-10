//! Pure resolver: turns a TrackedNoteStart into a MatchOutcome given the buffer.

use crate::practice::buffer::{Candidate, CandidateKind, MeasureBuffer, SlotStatus};
use crate::practice::types::{MatchOutcome, TrackedNoteStart};

pub const MIN_MATCH_SCORE: i32 = 80;
pub const DOUBLED_NOTE_FRESHNESS: f64 = 0.5;

pub fn resolve(
    tracked: &TrackedNoteStart,
    buf: &MeasureBuffer,
    frontier: (usize, usize),
) -> MatchOutcome {
    let cands = buf.candidates(tracked.start_beat, frontier);

    // Rule 1: any Pending slot whose duration window contains tracked.start_beat
    // → match (regardless of pitch). Pick the closest.
    let in_window_pending: Vec<&Candidate> = cands.iter()
        .filter(|c| matches!(c.kind, CandidateKind::InWindow) && c.status == SlotStatus::Pending)
        .collect();
    if !in_window_pending.is_empty() {
        let best = in_window_pending.iter()
            .min_by(|a, b| {
                let da = (tracked.start_beat - a.expected.beat_position).abs();
                let db = (tracked.start_beat - b.expected.beat_position).abs();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        return MatchOutcome::Matched {
            key: best.key,
            timing_err: tracked.start_beat - best.expected.beat_position,
            pitch_correct: tracked.midi_note == best.expected.midi_note,
            upgrade: false,
            skipped_keys: Vec::new(),
        };
    }

    // (Tasks 15–17 add upgrade, doubled, look-ahead, extra cases.)
    MatchOutcome::ExtraNote { during: None }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::practice::buffer::MeasureBuffer;
    use crate::practice::types::{StartSource, TrackedNoteStart};
    use crate::generators::{Instrument, Measure, SynthNote};
    use std::sync::Arc;

    fn measure_with_notes(notes: Vec<(f32, f32, f32)>, start: f64) -> Measure {
        // notes: (start_beat_in_measure, duration, freq)
        Measure {
            notes: notes.iter().map(|&(s, d, f)| SynthNote {
                freq: f,
                start_beat_in_measure: s,
                duration_beats: d,
                velocity: 0.5,
                instrument: Instrument::Piano,
            }).collect(),
            time_signature: (4, 4),
            bpm: 120.0,
            global_start_beat: start,
        }
    }

    fn ts(midi: u8, beat: f64) -> TrackedNoteStart {
        TrackedNoteStart {
            seq: 0,
            midi_note: midi,
            start_beat: beat,
            start_source: StartSource::Onset,
            initial_cents: 0.0,
        }
    }

    #[test]
    fn in_window_correct_pitch_matches_pending() {
        // C4 expected at beat 0, duration 1.
        let measures = Arc::new(vec![measure_with_notes(vec![(0.0, 1.0, 261.626)], 0.0)]);
        let buf = MeasureBuffer::new(measures, 0, 0);
        let out = resolve(&ts(60, 0.05), &buf, (0, 0));
        match out {
            MatchOutcome::Matched { key, pitch_correct, .. } => {
                assert_eq!(key, (0, 0));
                assert!(pitch_correct);
            }
            other => panic!("expected Matched, got {other:?}"),
        }
    }

    #[test]
    fn in_window_wrong_pitch_matches_pitch_correct_false() {
        let measures = Arc::new(vec![measure_with_notes(vec![(0.0, 1.0, 261.626)], 0.0)]);
        let buf = MeasureBuffer::new(measures, 0, 0);
        let out = resolve(&ts(62, 0.05), &buf, (0, 0));
        match out {
            MatchOutcome::Matched { key, pitch_correct, .. } => {
                assert_eq!(key, (0, 0));
                assert!(!pitch_correct);
            }
            other => panic!("expected Matched, got {other:?}"),
        }
    }
}
