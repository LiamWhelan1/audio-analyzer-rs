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

    // Rule 2: any Matched(false) slot whose duration window contains the beat
    // AND pitch is exact → upgrade.
    let upgrade_target = cands.iter().find(|c| {
        matches!(c.kind, CandidateKind::InWindow)
            && matches!(c.status, SlotStatus::Matched { pitch_correct: false })
            && tracked.midi_note == c.expected.midi_note
    });
    if let Some(c) = upgrade_target {
        return MatchOutcome::Matched {
            key: c.key,
            timing_err: tracked.start_beat - c.expected.beat_position,
            pitch_correct: true,
            upgrade: true,
            skipped_keys: Vec::new(),
        };
    }

    // Rule 3: Matched(true) slot, exact pitch, within freshness → DoubledNote.
    let doubled_target = cands.iter().find(|c| {
        matches!(c.kind, CandidateKind::InWindow)
            && matches!(c.status, SlotStatus::Matched { pitch_correct: true })
            && tracked.midi_note == c.expected.midi_note
            && buf.slot(c.key).and_then(|s| s.matched_start_beat).map_or(false, |msb| {
                tracked.start_beat - msb <= DOUBLED_NOTE_FRESHNESS
            })
    });
    if let Some(c) = doubled_target {
        return MatchOutcome::DoubledNote { key: c.key };
    }

    // (Task 17 adds look-ahead and extra cases.)
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
    fn doubled_note_within_freshness_emits_doubled() {
        let measures = Arc::new(vec![measure_with_notes(vec![(0.0, 1.0, 261.626)], 0.0)]);
        let mut buf = MeasureBuffer::new(measures, 0, 0);
        buf.record_match((0, 0), &ts(60, 0.0), true);
        let out = resolve(&ts(60, 0.2), &buf, (0, 0));
        assert_eq!(out, MatchOutcome::DoubledNote { key: (0, 0) });
    }

    #[test]
    fn doubled_note_beyond_freshness_emits_extra() {
        let measures = Arc::new(vec![measure_with_notes(vec![(0.0, 4.0, 261.626)], 0.0)]);
        let mut buf = MeasureBuffer::new(measures, 0, 0);
        buf.record_match((0, 0), &ts(60, 0.0), true);
        // Beat 0.6 is > DOUBLED_NOTE_FRESHNESS (0.5) past matched_start_beat.
        let out = resolve(&ts(60, 0.6), &buf, (0, 0));
        matches!(out, MatchOutcome::ExtraNote { .. }); // not Doubled
    }

    #[test]
    fn upgrade_promotes_matched_false_to_true() {
        let measures = Arc::new(vec![measure_with_notes(vec![(0.0, 1.0, 261.626)], 0.0)]);
        let mut buf = MeasureBuffer::new(measures, 0, 0);
        // Slot is currently Matched(false).
        buf.record_match((0, 0), &ts(62, 0.05), false);

        let out = resolve(&ts(60, 0.10), &buf, (0, 0));
        match out {
            MatchOutcome::Matched { key, pitch_correct, upgrade, .. } => {
                assert_eq!(key, (0, 0));
                assert!(pitch_correct);
                assert!(upgrade);
            }
            other => panic!("expected Matched (upgrade), got {other:?}"),
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
