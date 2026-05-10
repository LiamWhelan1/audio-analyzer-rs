//! 3-measure ring buffer with per-slot match state.

use std::collections::HashMap;
use std::sync::Arc;

use crate::generators::Measure;
use crate::practice::metrics::{ExpectedNote, MeasureData};
use crate::practice::types::TrackedNoteStart;

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
}
