//! 3-measure ring buffer with per-slot match state.

use std::collections::HashMap;
use std::sync::Arc;

use crate::generators::Measure;
use crate::practice::metrics::{ExpectedNote, MeasureData};
use crate::practice::types::TrackedNoteStart;

const LOOKAHEAD_NOTES: u8 = 2;
const LOOKBEHIND_NOTES: u8 = 1;

#[derive(Clone, Debug, PartialEq)]
pub enum SlotStatus {
    Pending,
    Matched { pitch_correct: bool },
    Missed,
}

#[derive(Clone, Debug)]
pub struct NoteSlot {
    pub status: SlotStatus,
    pub matched_start_beat: Option<f64>,
    pub matched_seq: Option<u64>,
}

#[derive(Clone, Copy, Debug)]
pub enum CandidateKind {
    InWindow,
    Lookahead(u8),
    Lookbehind(u8),
}

#[derive(Clone, Debug)]
pub struct Candidate {
    pub key: (usize, usize),
    pub expected: ExpectedNote,
    pub status: SlotStatus,
    pub kind: CandidateKind,
}

pub struct MeasureBuffer {
    measures: Arc<Vec<Measure>>,
    practice_start: usize,
    practice_end: usize,
    past_idx: Option<usize>,
    current_idx: usize,
    future_idx: Option<usize>,
    slots: HashMap<(usize, usize), NoteSlot>,
    /// Set once `practice_end` has been aged out. Subsequent `advance()` calls
    /// return empty so the polling loop sees a clean terminal state.
    done: bool,
}

impl MeasureBuffer {
    pub fn new(measures: Arc<Vec<Measure>>, practice_start: usize, practice_end: usize) -> Self {
        let future_idx = if practice_start < practice_end {
            Some(practice_start + 1)
        } else {
            None
        };
        let mut buf = Self {
            measures,
            practice_start,
            practice_end,
            past_idx: None,
            current_idx: practice_start,
            future_idx,
            slots: HashMap::new(),
            done: false,
        };
        buf.populate_slots(practice_start);
        if let Some(f) = future_idx {
            buf.populate_slots(f);
        }
        buf
    }

    pub fn current_idx(&self) -> usize { self.current_idx }
    pub fn past_idx(&self) -> Option<usize> { self.past_idx }
    pub fn future_idx(&self) -> Option<usize> { self.future_idx }
    pub fn slot(&self, key: (usize, usize)) -> Option<&NoteSlot> { self.slots.get(&key) }
    pub fn measures(&self) -> &[Measure] { &self.measures }
    pub fn practice_end(&self) -> usize { self.practice_end }
    pub fn is_done(&self) -> bool { self.done }

    /// Return the measure index whose half-open duration window
    /// `[global_start_beat, global_start_beat + duration_beats)` contains `beat`,
    /// scanning the three active measures (past / current / future). Falls back
    /// to `current_idx` if none of them claim the beat (e.g. `beat` lies in the
    /// count-off, before practice_start, or beyond the last loaded measure).
    pub fn measure_for_beat(&self, beat: f64) -> usize {
        for m_idx in [self.past_idx, Some(self.current_idx), self.future_idx]
            .iter()
            .copied()
            .flatten()
        {
            let m = &self.measures[m_idx];
            let start = m.global_start_beat;
            let end = start + m.duration_beats();
            if beat >= start && beat < end {
                return m_idx;
            }
        }
        self.current_idx
    }

    pub fn record_match(&mut self, key: (usize, usize), tracked: &TrackedNoteStart, pitch_correct: bool) {
        if let Some(slot) = self.slots.get_mut(&key) {
            slot.status = SlotStatus::Matched { pitch_correct };
            slot.matched_start_beat = Some(tracked.start_beat);
            slot.matched_seq = Some(tracked.seq);
        }
    }

    pub fn upgrade_match(&mut self, key: (usize, usize), tracked: &TrackedNoteStart) {
        if let Some(slot) = self.slots.get_mut(&key) {
            slot.status = SlotStatus::Matched { pitch_correct: true };
            slot.matched_start_beat = Some(tracked.start_beat);
            slot.matched_seq = Some(tracked.seq);
        }
    }

    pub fn mark_missed(&mut self, key: (usize, usize)) {
        if let Some(slot) = self.slots.get_mut(&key) {
            slot.status = SlotStatus::Missed;
        }
    }

    /// Lowest-key slot strictly after `frontier` whose status is `Pending`.
    /// Walks current → future measure boundary if needed.
    pub fn next_pending_after(&self, frontier: (usize, usize)) -> Option<(usize, usize)> {
        // Within current measure first, then future measure if any.
        let candidates_iter = std::iter::once(self.current_idx)
            .chain(self.future_idx.into_iter());
        for m_idx in candidates_iter {
            let n_count = self.measures[m_idx].notes.len();
            let start = if m_idx == frontier.0 { frontier.1 + 1 } else { 0 };
            for n_idx in start..n_count {
                let key = (m_idx, n_idx);
                if let Some(slot) = self.slots.get(&key) {
                    if slot.status == SlotStatus::Pending {
                        return Some(key);
                    }
                }
            }
        }
        None
    }

    /// All slots whose duration window (half-open [start, start+dur))
    /// covers `beat`, plus the next LOOKAHEAD_NOTES not-yet-Matched slots
    /// after `frontier`, plus the previous LOOKBEHIND_NOTES already-passed
    /// slots before `frontier`. Slots of any status (Pending/Matched/Missed)
    /// are returned; the matcher's scoring rules differentiate based on status.
    pub fn candidates(&self, beat: f64, frontier: (usize, usize)) -> Vec<Candidate> {
        let measure_indices: Vec<usize> = [self.past_idx, Some(self.current_idx), self.future_idx]
            .iter()
            .copied()
            .flatten()
            .collect();

        // Collect every slot in the active measures with its expected note.
        let mut all: Vec<(usize, usize, ExpectedNote)> = Vec::new();
        for m_idx in measure_indices {
            let expected = build_expected_notes(&self.measures[m_idx]);
            for (n_idx, exp) in expected.into_iter().enumerate() {
                all.push((m_idx, n_idx, exp));
            }
        }

        // Sort by beat position.
        all.sort_by(|a, b| {
            a.2.beat_position
                .partial_cmp(&b.2.beat_position)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Find frontier's index in `all`.
        let frontier_pos = all.iter().position(|&(m, n, _)| (m, n) == frontier);

        let mut out = Vec::new();
        for (i, (m_idx, n_idx, exp)) in all.iter().enumerate() {
            let key = (*m_idx, *n_idx);
            let Some(slot) = self.slots.get(&key) else { continue };
            let in_window = beat >= exp.beat_position
                && beat < exp.beat_position + exp.duration_beats;

            let kind = if in_window {
                CandidateKind::InWindow
            } else if let Some(fp) = frontier_pos {
                let delta = i as isize - fp as isize;
                if delta > 0 && delta <= LOOKAHEAD_NOTES as isize {
                    CandidateKind::Lookahead(delta as u8)
                } else if delta < 0 && (-delta) <= LOOKBEHIND_NOTES as isize {
                    CandidateKind::Lookbehind((-delta) as u8)
                } else {
                    continue;
                }
            } else {
                continue;
            };

            out.push(Candidate {
                key,
                expected: exp.clone(),
                status: slot.status.clone(),
                kind,
            });
        }
        out
    }

    fn populate_slots(&mut self, m_idx: usize) {
        if m_idx >= self.measures.len() { return; }
        for (n_idx, _note) in self.measures[m_idx].notes.iter().enumerate() {
            self.slots.insert(
                (m_idx, n_idx),
                NoteSlot {
                    status: SlotStatus::Pending,
                    matched_start_beat: None,
                    matched_seq: None,
                },
            );
        }
    }

    /// Cycle past/current/future when transport_beat ≥ current_measure_end.
    /// Returns 0 or 1 aged-out measures (as MeasureData skeletons — only
    /// `measure_index` and `expected_notes` are populated; the per-tick
    /// accumulators (notes, onsets, dynamics, durations, doubled_seqs) are
    /// filled in by the ModeController in Phase 4).
    pub fn advance(&mut self, transport_beat: f64) -> Vec<MeasureData> {
        if self.done {
            return Vec::new();
        }
        let current_end = self.measures[self.current_idx].global_start_beat
            + self.measures[self.current_idx].duration_beats();

        if transport_beat < current_end {
            return Vec::new();
        }

        let aged_idx = self.current_idx;
        let expected_notes = build_expected_notes(&self.measures[aged_idx]);

        // Drop any slot for the past measure (one cycle of look-behind already elapsed).
        if let Some(p) = self.past_idx {
            self.slots.retain(|(m, _), _| *m != p);
        }

        // Cycle forward.
        self.past_idx = Some(self.current_idx);
        if let Some(f) = self.future_idx {
            self.current_idx = f;
        }
        self.future_idx = if self.current_idx < self.practice_end {
            Some(self.current_idx + 1)
        } else {
            None
        };
        if let Some(f) = self.future_idx {
            self.populate_slots(f);
        }

        // Terminal state: the measure that just aged out is the last one in
        // the practice range. Block any further advance() calls so the loop
        // doesn't keep yielding the same measure on every tick.
        if aged_idx == self.practice_end {
            self.done = true;
        }

        // Note: any still-Pending slot in the now-past measure stays in `slots`
        // for the look-behind window. Its status will be flipped to Missed by the
        // ModeController during this cycle (the ModeController calls mark_missed
        // for each remaining Pending slot in `aged_idx` before draining MeasureData).

        vec![MeasureData {
            measure_index: aged_idx as u32,
            onsets: Vec::new(),
            notes: Vec::new(),
            dynamics: Vec::new(),
            expected_notes,
            note_durations: Vec::new(),
            doubled_note_seqs: Vec::new(),
        }]
    }
}

fn build_expected_notes(measure: &Measure) -> Vec<ExpectedNote> {
    measure
        .notes
        .iter()
        .map(|n| ExpectedNote {
            beat_position: measure.global_start_beat + n.start_beat_in_measure as f64,
            duration_beats: n.duration_beats as f64,
            midi_note: freq_to_midi(n.freq),
            dynamic: velocity_to_dynamic(n.velocity),
        })
        .collect()
}

fn freq_to_midi(freq: f32) -> u8 {
    (69.0 + 12.0 * (freq / 440.0).log2()).round().clamp(0.0, 127.0) as u8
}

fn velocity_to_dynamic(velocity: f32) -> Option<crate::audio_io::dynamics::DynamicLevel> {
    use crate::audio_io::dynamics::DynamicLevel;
    match velocity {
        v if v <= 0.0 => None,
        v if v < 0.125 => Some(DynamicLevel::Ppp),
        v if v < 0.25 => Some(DynamicLevel::Pp),
        v if v < 0.375 => Some(DynamicLevel::P),
        v if v < 0.5 => Some(DynamicLevel::Mp),
        v if v < 0.625 => Some(DynamicLevel::Mf),
        v if v < 0.75 => Some(DynamicLevel::F),
        v if v < 0.875 => Some(DynamicLevel::Ff),
        _ => Some(DynamicLevel::Fff),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generators::SynthNote;

    fn dummy_measure(global_start_beat: f64, num_notes: usize) -> Measure {
        Measure {
            notes: (0..num_notes)
                .map(|i| SynthNote {
                    freq: 440.0,
                    start_beat_in_measure: i as f32,
                    duration_beats: 1.0,
                    velocity: 0.5,
                    instrument: crate::generators::Instrument::Piano,
                })
                .collect(),
            time_signature: (4, 4),
            bpm: 120.0,
            global_start_beat,
        }
    }

    #[test]
    fn new_populates_current_and_future_slots() {
        let measures = Arc::new(vec![
            dummy_measure(0.0, 2),
            dummy_measure(4.0, 3),
            dummy_measure(8.0, 1),
        ]);
        let buf = MeasureBuffer::new(measures, 0, 2);
        assert_eq!(buf.current_idx(), 0);
        assert_eq!(buf.future_idx(), Some(1));
        assert_eq!(buf.past_idx(), None);
        assert!(buf.slot((0, 0)).is_some());
        assert!(buf.slot((0, 1)).is_some());
        assert!(buf.slot((1, 0)).is_some());
        assert!(buf.slot((1, 2)).is_some());
        assert!(buf.slot((2, 0)).is_none()); // not yet in the buffer
        assert_eq!(buf.slot((0, 0)).unwrap().status, SlotStatus::Pending);
    }

    #[test]
    fn advance_yields_aged_out_when_transport_crosses_measure_end() {
        let measures = Arc::new(vec![
            dummy_measure(0.0, 1),
            dummy_measure(4.0, 1),
            dummy_measure(8.0, 1),
        ]);
        let mut buf = MeasureBuffer::new(measures, 0, 2);

        // Inside first measure — no advance.
        let aged = buf.advance(2.0);
        assert!(aged.is_empty());
        assert_eq!(buf.current_idx(), 0);

        // Cross into measure 1.
        let aged = buf.advance(4.5);
        assert_eq!(aged.len(), 1);
        assert_eq!(aged[0].measure_index, 0);
        assert_eq!(buf.current_idx(), 1);
        assert_eq!(buf.past_idx(), Some(0));
        assert_eq!(buf.future_idx(), Some(2));
        // Slots for the aged-out measure are removed (or scheduled for removal on next advance).
        // Per the look-behind window rule, slots are kept for one cycle as Lookbehind.
        // For this assertion we just verify the slot is no longer in the active forward path —
        // the look-behind cycle will be exercised in Task 7 (candidates).
    }

    #[test]
    fn advance_yields_measure_data_with_expected_notes() {
        let measures = Arc::new(vec![dummy_measure(0.0, 2), dummy_measure(4.0, 1)]);
        let mut buf = MeasureBuffer::new(measures, 0, 1);
        let aged = buf.advance(4.5);
        assert_eq!(aged.len(), 1);
        // Two unplayed slots in measure 0 → expected_notes len = 2 in MeasureData.
        assert_eq!(aged[0].expected_notes.len(), 2);
    }

    fn fake_tracked(midi: u8, beat: f64) -> TrackedNoteStart {
        TrackedNoteStart {
            seq: 0,
            midi_note: midi,
            start_beat: beat,
            start_source: crate::practice::types::StartSource::Onset,
            initial_cents: 0.0,
        }
    }

    #[test]
    fn record_match_sets_slot_matched_correct() {
        let measures = Arc::new(vec![dummy_measure(0.0, 2)]);
        let mut buf = MeasureBuffer::new(measures, 0, 0);
        let t = fake_tracked(60, 0.0);
        buf.record_match((0, 0), &t, true);
        let s = buf.slot((0, 0)).unwrap();
        assert_eq!(s.status, SlotStatus::Matched { pitch_correct: true });
        assert_eq!(s.matched_start_beat, Some(0.0));
        assert_eq!(s.matched_seq, Some(0));
    }

    #[test]
    fn upgrade_match_promotes_pitch_correct_false_to_true() {
        let measures = Arc::new(vec![dummy_measure(0.0, 1)]);
        let mut buf = MeasureBuffer::new(measures, 0, 0);
        let mut t1 = fake_tracked(61, 0.05);
        t1.seq = 1;
        let mut t2 = fake_tracked(60, 0.1);
        t2.seq = 2;
        buf.record_match((0, 0), &t1, false);
        buf.upgrade_match((0, 0), &t2);
        let s = buf.slot((0, 0)).unwrap();
        assert_eq!(s.status, SlotStatus::Matched { pitch_correct: true });
        assert_eq!(s.matched_seq, Some(2));
    }

    #[test]
    fn mark_missed_sets_slot_missed() {
        let measures = Arc::new(vec![dummy_measure(0.0, 2)]);
        let mut buf = MeasureBuffer::new(measures, 0, 0);
        buf.mark_missed((0, 0));
        assert_eq!(buf.slot((0, 0)).unwrap().status, SlotStatus::Missed);
    }

    #[test]
    fn next_pending_after_finds_next_unmatched_slot() {
        let measures = Arc::new(vec![dummy_measure(0.0, 4)]);
        let mut buf = MeasureBuffer::new(measures, 0, 0);
        let t = fake_tracked(60, 0.0);
        buf.record_match((0, 0), &t, true);
        assert_eq!(buf.next_pending_after((0, 0)), Some((0, 1)));
        buf.mark_missed((0, 1));
        assert_eq!(buf.next_pending_after((0, 0)), Some((0, 2)));
    }

    #[test]
    fn next_pending_after_crosses_measure_boundary() {
        let measures = Arc::new(vec![dummy_measure(0.0, 1), dummy_measure(4.0, 2)]);
        let buf = MeasureBuffer::new(measures, 0, 1);
        assert_eq!(buf.next_pending_after((0, 0)), Some((1, 0)));
    }

    #[test]
    fn measure_for_beat_returns_correct_measure() {
        let measures = Arc::new(vec![
            dummy_measure(0.0, 1),
            dummy_measure(4.0, 1),
            dummy_measure(8.0, 1),
        ]);
        // Practice 0..=2 so all three measures land in the past/current/future window
        // after one advance.
        let mut buf = MeasureBuffer::new(measures, 0, 2);
        // Before any advance: only current (0) and future (1) are active.
        assert_eq!(buf.measure_for_beat(2.0), 0);
        assert_eq!(buf.measure_for_beat(5.0), 1);
        // Beat in measure 2's range — not yet in buffer's three-measure window,
        // so falls back to current_idx.
        assert_eq!(buf.measure_for_beat(9.0), 0);
        // Boundary: 4.0 lies in measure 1's [4, 8) window, not measure 0's [0, 4).
        assert_eq!(buf.measure_for_beat(4.0), 1);
        // Advance into measure 1 → past=0, current=1, future=2.
        let _ = buf.advance(4.5);
        assert_eq!(buf.measure_for_beat(2.0), 0);   // look-behind
        assert_eq!(buf.measure_for_beat(5.0), 1);   // current
        assert_eq!(buf.measure_for_beat(9.0), 2);   // look-ahead
    }

    #[test]
    fn advance_marks_done_after_aging_out_practice_end() {
        let measures = Arc::new(vec![dummy_measure(0.0, 1), dummy_measure(4.0, 1)]);
        let mut buf = MeasureBuffer::new(measures, 0, 1);
        assert!(!buf.is_done());
        // Cross into measure 1 (still inside practice range).
        let aged = buf.advance(4.5);
        assert_eq!(aged.len(), 1);
        assert!(!buf.is_done(), "should not be done after aging out non-final measure");
        // Cross past measure 1 (which IS practice_end).
        let aged = buf.advance(8.5);
        assert_eq!(aged.len(), 1);
        assert!(buf.is_done(), "should be done after aging out practice_end");
        // Subsequent advances are no-ops.
        let aged = buf.advance(20.0);
        assert!(aged.is_empty(), "advance() must yield empty once done");
    }

    #[test]
    fn candidates_includes_in_window_lookahead_and_lookbehind() {
        let measures = Arc::new(vec![dummy_measure(0.0, 4)]);
        let mut buf = MeasureBuffer::new(measures, 0, 0);
        // Mark slot 0 as Matched (it'll show up as Lookbehind from frontier=(0,1)).
        buf.record_match((0, 0), &fake_tracked(60, 0.0), true);

        // Frontier is (0, 1). beat=1.5 is inside slot 1's duration window [1,2)
        // and slot 2 is +1 lookahead, slot 3 is +2 lookahead.
        let cands = buf.candidates(1.5, (0, 1));
        let keys: Vec<_> = cands.iter().map(|c| (c.key, format!("{:?}", c.kind))).collect();
        assert!(keys.iter().any(|(k, _)| *k == (0, 1)));    // InWindow
        assert!(keys.iter().any(|(k, _)| *k == (0, 2)));    // Lookahead(1)
        assert!(keys.iter().any(|(k, _)| *k == (0, 3)));    // Lookahead(2)
        assert!(keys.iter().any(|(k, _)| *k == (0, 0)));    // Lookbehind(1)
    }
}
