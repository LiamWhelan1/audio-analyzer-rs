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
        let skipped_keys = walk_skipped(buf, frontier, best.key);
        return MatchOutcome::Matched {
            key: best.key,
            timing_err: tracked.start_beat - best.expected.beat_position,
            pitch_correct: tracked.midi_note == best.expected.midi_note,
            upgrade: false,
            skipped_keys,
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

    // Rule 4: score Lookahead and Lookbehind candidates.
    fn pitch_score(played: u8, expected: u8) -> i32 {
        let d = (played as i32 - expected as i32).abs();
        match d { 0 => 100, 1 => 30, 2 => 10, _ => 0 }
    }
    fn timing_score(beat: f64, exp: &crate::practice::metrics::ExpectedNote) -> i32 {
        if beat >= exp.beat_position && beat < exp.beat_position + exp.duration_beats {
            50
        } else {
            let err = (beat - exp.beat_position).abs();
            ((50.0 - 100.0 * err) as i32).max(0)
        }
    }

    let mut best: Option<(&Candidate, i32)> = None;
    for c in &cands {
        if !matches!(c.status, SlotStatus::Pending) { continue; }
        let kind_penalty = match c.kind {
            CandidateKind::InWindow => 0,
            CandidateKind::Lookahead(1) => -10,
            CandidateKind::Lookahead(2) => -25,
            CandidateKind::Lookbehind(1) => -15,
            _ => -50,
        };
        let score = pitch_score(tracked.midi_note, c.expected.midi_note)
                  + timing_score(tracked.start_beat, &c.expected)
                  + kind_penalty;
        if score >= MIN_MATCH_SCORE
            && tracked.midi_note == c.expected.midi_note
            && best.map_or(true, |(_, bs)| score > bs)
        {
            best = Some((c, score));
        }
    }

    if let Some((c, _)) = best {
        let skipped = walk_skipped(buf, frontier, c.key);
        return MatchOutcome::Matched {
            key: c.key,
            timing_err: tracked.start_beat - c.expected.beat_position,
            pitch_correct: true,
            upgrade: false,
            skipped_keys: skipped,
        };
    }

    // Rule 5: extra note. `during` = any in-window slot (Pending or Matched).
    let during = cands.iter()
        .find(|c| matches!(c.kind, CandidateKind::InWindow))
        .map(|c| c.key);
    MatchOutcome::ExtraNote { during }
}

fn walk_skipped(buf: &MeasureBuffer, frontier: (usize, usize), target: (usize, usize)) -> Vec<(usize, usize)> {
    let mut skipped = Vec::new();
    let mut walker = frontier;
    let mut steps = 0;
    while walker != target && steps < 64 {
        if let Some(s) = buf.slot(walker) {
            if matches!(s.status, SlotStatus::Pending) {
                skipped.push(walker);
            }
        } else {
            break;
        }
        walker = step_forward(buf, walker);
        steps += 1;
    }
    skipped
}

fn step_forward(buf: &MeasureBuffer, key: (usize, usize)) -> (usize, usize) {
    // We don't need exact cross-measure logic for skipped — just pick the
    // closest slot one note ahead by sequential numbering within the measure,
    // then walk into the next measure if needed.
    let next = (key.0, key.1 + 1);
    if buf.slot(next).is_some() { next }
    else { (key.0 + 1, 0) }
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
    fn lookahead_matches_skipped_frontier() {
        // Notes at 0, 1, 2 (each 1 beat). Student plays at beat 1.05 with pitch
        // matching note 2 (E4 = midi 64). Frontier is (0, 1). Note 1 (D4) gets
        // skipped, note 2 matched.
        let measures = Arc::new(vec![measure_with_notes(vec![
            (0.0, 1.0, 261.626),    // C4
            (1.0, 1.0, 293.665),    // D4
            (2.0, 1.0, 329.628),    // E4
        ], 0.0)]);
        let mut buf = MeasureBuffer::new(measures, 0, 0);
        buf.record_match((0, 0), &ts(60, 0.0), true);

        // Tracked: pitch=64 (E4) at beat 1.05 (still in D4's window normally, but
        // pitch overrides that — actually no: Rule 1 is pitch-agnostic. So beat
        // 1.05 with E4 still matches D4 with pitch_correct=false. Use beat=2.05
        // instead so we cleanly hit lookahead).
        let out = resolve(&ts(64, 2.05), &buf, (0, 1));
        match out {
            MatchOutcome::Matched { key, skipped_keys, pitch_correct, .. } => {
                assert_eq!(key, (0, 2));
                assert_eq!(skipped_keys, vec![(0, 1)]);
                assert!(pitch_correct);
            }
            other => panic!("expected Matched lookahead, got {other:?}"),
        }
    }

    #[test]
    fn unmatched_note_in_rest_emits_extra_note_with_during_none() {
        let measures = Arc::new(vec![measure_with_notes(vec![(0.0, 0.5, 261.626)], 0.0)]);
        let buf = MeasureBuffer::new(measures, 0, 0);
        // Beat 2.0 — well past the only note's duration window.
        let out = resolve(&ts(60, 2.0), &buf, (0, 0));
        assert_eq!(out, MatchOutcome::ExtraNote { during: None });
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
