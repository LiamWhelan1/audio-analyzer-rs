//! ModeController — orchestrates conditioner → matcher → clock and applies
//! ClockActions through a per-mode filter.

use std::collections::HashMap;
use std::sync::Arc;

use crate::audio_io::dynamics::DynamicLevel;
use crate::audio_io::timing::{MusicalTransport, OnsetEvent};
use crate::practice::buffer::{MeasureBuffer, SlotStatus};
use crate::practice::clock::{ClockConfig, ClockManager};
use crate::practice::conditioner::InputConditioner;
use crate::practice::matcher;
use crate::practice::metrics::{DynamicsEvent, MeasureData, NoteEvent};
use crate::practice::types::{
    ClockAction, ConditionerEvent, MatchOutcome, PracticeMode, TrackedNoteEnd, TrackedNoteStart, TunerFrame,
};

#[derive(Clone, Debug)]
pub struct MatchedSnapshot {
    pub measure_idx: usize,
    pub note_idx_in_measure_data: usize,
    pub expected_duration: f64,
    pub expected_midi: u8,
}

pub struct ModeController {
    pub mode: PracticeMode,
    pub transport: Arc<MusicalTransport>,
    pub conditioner: InputConditioner,
    pub buffer: MeasureBuffer,
    pub clock: ClockManager,
    pub frontier: (usize, usize),

    pub in_progress_played_notes: HashMap<usize, Vec<NoteEvent>>,
    pub in_progress_onsets:       HashMap<usize, Vec<OnsetEvent>>,
    pub in_progress_dynamics:     HashMap<usize, Vec<DynamicsEvent>>,
    pub in_progress_durations:    HashMap<usize, Vec<Option<f64>>>,
    pub in_progress_doubled_seqs: HashMap<usize, Vec<u64>>,

    pub match_log: HashMap<u64, MatchedSnapshot>,

    pub last_dynamic_level: Option<DynamicLevel>,
}

pub struct TickInputs<'a> {
    pub transport_beat: f64,
    pub tuner_frame: Option<&'a TunerFrame>,
    pub new_onsets: &'a [OnsetEvent],
    pub dynamic_level: DynamicLevel,
}

pub struct TickOutputs {
    pub aged_measures: Vec<MeasureData>,
    pub events: Vec<ConditionerEvent>,
    pub outcomes: Vec<(MatchOutcome, TrackedNoteStart)>,
}

impl ModeController {
    pub fn new(
        mode: PracticeMode,
        transport: Arc<MusicalTransport>,
        conditioner: InputConditioner,
        buffer: MeasureBuffer,
        clock: ClockManager,
        practice_start: usize,
    ) -> Self {
        Self {
            mode, transport, conditioner, buffer, clock,
            frontier: (practice_start, 0),
            in_progress_played_notes: HashMap::new(),
            in_progress_onsets:       HashMap::new(),
            in_progress_dynamics:     HashMap::new(),
            in_progress_durations:    HashMap::new(),
            in_progress_doubled_seqs: HashMap::new(),
            match_log: HashMap::new(),
            last_dynamic_level: None,
        }
    }

    /// One polling tick. Returns aged-out measures and intermediate events
    /// (for testing / feedback emission).
    pub fn tick(&mut self, inputs: TickInputs) -> TickOutputs {
        // (Implemented in Task 24.)
        let _ = inputs;
        TickOutputs { aged_measures: vec![], events: vec![], outcomes: vec![] }
    }
}

#[cfg(test)]
mod tests {
    // Integration tests added in Task 25 onward.
}
