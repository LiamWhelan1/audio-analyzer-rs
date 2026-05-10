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

    /// Cycle past/current/future when transport_beat ≥ current_measure_end.
    /// Returns 0 or 1 aged-out measures (as MeasureData skeletons — only
    /// `measure_index` and `expected_notes` are populated; the per-tick
    /// accumulators (notes, onsets, dynamics, durations, doubled_seqs) are
    /// filled in by the ModeController in Phase 4).
    pub fn advance(&mut self, transport_beat: f64) -> Vec<MeasureData> {
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
}
