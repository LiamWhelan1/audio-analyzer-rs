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
        let mut outputs = TickOutputs { aged_measures: vec![], events: vec![], outcomes: vec![] };

        // 1. Conditioner.
        let events = self.conditioner.ingest(inputs.tuner_frame, inputs.new_onsets);

        // 2. Onsets accumulator (raw).
        self.in_progress_onsets.entry(self.buffer.current_idx())
            .or_default()
            .extend_from_slice(inputs.new_onsets);

        // 3. Dynamics accumulator.
        if inputs.dynamic_level != DynamicLevel::Silence
            && self.last_dynamic_level != Some(inputs.dynamic_level)
        {
            self.in_progress_dynamics.entry(self.buffer.current_idx())
                .or_default()
                .push(DynamicsEvent {
                    beat_position: inputs.transport_beat,
                    level: inputs.dynamic_level,
                });
            self.last_dynamic_level = Some(inputs.dynamic_level);
        }

        // 4. Process each ConditionerEvent.
        for ev in &events {
            match ev {
                ConditionerEvent::Started(t) => {
                    let outcome = matcher::resolve(t, &self.buffer, self.frontier);
                    self.handle_outcome(t, &outcome, inputs.transport_beat);
                    outputs.outcomes.push((outcome, t.clone()));
                }
                ConditionerEvent::Ended(t) => {
                    self.handle_ended(t);
                }
            }
        }
        outputs.events = events;

        // 5. Tick-level clock check (stop trigger, etc.).
        let tick_actions = self.clock.on_tick(&self.buffer, self.frontier, inputs.transport_beat, self.mode);
        for a in tick_actions { self.apply_action(&a); }

        // 6. Buffer advance.
        let aged = self.buffer.advance(inputs.transport_beat);
        for mut m in aged {
            let mi = m.measure_index as usize;
            // Mark any remaining Pending slots Missed.
            let mut to_miss: Vec<(usize, usize)> = (0..m.expected_notes.len())
                .filter(|i| matches!(self.buffer.slot((mi, *i)).map(|s| &s.status),
                                     Some(SlotStatus::Pending)))
                .map(|i| (mi, i))
                .collect();
            for k in to_miss.drain(..) {
                self.buffer.mark_missed(k);
            }
            // Drain in-progress accumulators.
            m.onsets    = self.in_progress_onsets.remove(&mi).unwrap_or_default();
            m.notes     = self.in_progress_played_notes.remove(&mi).unwrap_or_default();
            m.dynamics  = self.in_progress_dynamics.remove(&mi).unwrap_or_default();
            m.note_durations    = self.in_progress_durations.remove(&mi).unwrap_or_default();
            m.doubled_note_seqs = self.in_progress_doubled_seqs.remove(&mi).unwrap_or_default();
            outputs.aged_measures.push(m);
        }

        outputs
    }

    fn handle_outcome(&mut self, t: &TrackedNoteStart, outcome: &MatchOutcome, transport_beat: f64) {
        let mi = self.buffer.current_idx();
        self.in_progress_played_notes.entry(mi).or_default().push(NoteEvent {
            beat_position: t.start_beat,
            midi_note: t.midi_note,
            avg_cents: t.initial_cents,
        });
        self.in_progress_durations.entry(mi).or_default().push(None);
        let note_idx = self.in_progress_played_notes[&mi].len() - 1;

        let actions = match outcome {
            MatchOutcome::Matched { key, pitch_correct, upgrade, skipped_keys, .. } => {
                for k in skipped_keys { self.buffer.mark_missed(*k); }
                if *upgrade {
                    self.buffer.upgrade_match(*key, t);
                } else {
                    self.buffer.record_match(*key, t, *pitch_correct);
                }
                self.frontier = step_forward(&self.buffer, *key);
                // Record for later End hold-check.
                let exp = expected_for(&self.buffer, *key);
                self.match_log.insert(t.seq, MatchedSnapshot {
                    measure_idx: key.0,
                    note_idx_in_measure_data: note_idx,
                    expected_duration: exp.duration_beats,
                    expected_midi: exp.midi_note,
                });
                self.clock.on_match(outcome, &exp, transport_beat, self.mode)
            }
            MatchOutcome::DoubledNote { key } => {
                self.in_progress_doubled_seqs.entry(mi).or_default().push(t.seq);
                let slot = self.buffer.slot(*key).cloned();
                slot.map(|s| self.clock.on_doubled(&s, self.mode)).unwrap_or_default()
            }
            MatchOutcome::ExtraNote { .. } => self.clock.on_extra(),
        };
        for a in actions { self.apply_action(&a); }
    }

    fn handle_ended(&mut self, t: &TrackedNoteEnd) {
        let Some(snap) = self.match_log.remove(&t.seq) else { return };
        let mi = snap.measure_idx;
        if let Some(notes) = self.in_progress_played_notes.get_mut(&mi) {
            if let Some(n) = notes.get_mut(snap.note_idx_in_measure_data) {
                let actual_duration = t.end_beat - n.beat_position;
                n.avg_cents = t.avg_cents;
                if let Some(durs) = self.in_progress_durations.get_mut(&mi) {
                    if let Some(d) = durs.get_mut(snap.note_idx_in_measure_data) {
                        *d = Some(actual_duration);
                    }
                }
                let _ = actual_duration;
            }
        }
    }

    fn apply_action(&self, action: &ClockAction) {
        match (self.mode, action) {
            (PracticeMode::Performance, _) => {}
            (_, ClockAction::SeekToBeat(b)) => self.transport.seek_to_beat(*b),
            (PracticeMode::FollowAlong, ClockAction::Stop) => self.transport.stop(),
            (PracticeMode::Rubato, ClockAction::Stop) => {}
            (_, ClockAction::Play) => self.transport.play(),
            (_, ClockAction::SetBpm(bpm)) => self.transport.set_bpm(*bpm),
        }
    }
}

fn step_forward(buf: &MeasureBuffer, key: (usize, usize)) -> (usize, usize) {
    let next = (key.0, key.1 + 1);
    if buf.slot(next).is_some() { next }
    else { (key.0 + 1, 0) }
}

fn expected_for(buf: &MeasureBuffer, key: (usize, usize)) -> crate::practice::metrics::ExpectedNote {
    let m = &buf.measures()[key.0];
    let n = &m.notes[key.1];
    crate::practice::metrics::ExpectedNote {
        beat_position: m.global_start_beat + n.start_beat_in_measure as f64,
        duration_beats: n.duration_beats as f64,
        midi_note: (69.0 + 12.0 * (n.freq / 440.0).log2()).round().clamp(0.0, 127.0) as u8,
        dynamic: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use crate::audio_io::dynamics::DynamicLevel;
    use crate::generators::{Instrument, Measure, SynthNote};

    fn three_quarter_note_measure() -> Arc<Vec<Measure>> {
        Arc::new(vec![Measure {
            notes: vec![
                SynthNote { freq: 261.626, start_beat_in_measure: 0.0, duration_beats: 1.0, velocity: 0.5, instrument: Instrument::Piano },
                SynthNote { freq: 293.665, start_beat_in_measure: 1.0, duration_beats: 1.0, velocity: 0.5, instrument: Instrument::Piano },
                SynthNote { freq: 329.628, start_beat_in_measure: 2.0, duration_beats: 1.0, velocity: 0.5, instrument: Instrument::Piano },
            ],
            time_signature: (4,4), bpm: 120.0, global_start_beat: 0.0,
        }])
    }

    fn make_mc(mode: PracticeMode) -> ModeController {
        let transport = MusicalTransport::new(120.0, 48000.0);
        transport.play();
        let measures = three_quarter_note_measure();
        let buffer = MeasureBuffer::new(measures, 0, 0);
        let conditioner = InputConditioner::new(transport.clone());
        let clock = ClockManager::new(transport.clone(), ClockConfig::default(), 120.0);
        ModeController::new(mode, transport, conditioner, buffer, clock, 0)
    }

    fn frame_with(midis: Vec<(u8, f64)>, beat: f64) -> TunerFrame {
        TunerFrame { notes: midis, tuner_beat: beat }
    }

    #[test]
    fn followalong_perfect_play_advances_frontier() {
        let mut mc = make_mc(PracticeMode::FollowAlong);
        // 5 frames stable on C4 → emit Started; matcher sees in-window Pending → Matched.
        for i in 0..5 {
            let _ = mc.tick(TickInputs {
                transport_beat: i as f64 * 0.02,
                tuner_frame: Some(&frame_with(vec![(60, 0.0)], i as f64 * 0.02)),
                new_onsets: &[],
                dynamic_level: DynamicLevel::Silence,
            });
        }
        assert_eq!(mc.frontier, (0, 1));
    }

    #[test]
    fn performance_mode_does_not_call_seek_or_stop() {
        let mut mc = make_mc(PracticeMode::Performance);
        let initial_beat = mc.transport.get_accumulated_beats();
        // Bunch of frames with severe timing error.
        for i in 0..5 {
            let _ = mc.tick(TickInputs {
                transport_beat: 5.0 + i as f64 * 0.02,    // way past expected beat 0
                tuner_frame: Some(&frame_with(vec![(60, 0.0)], 5.0 + i as f64 * 0.02)),
                new_onsets: &[],
                dynamic_level: DynamicLevel::Silence,
            });
        }
        // Performance mode never seeks.
        assert!((mc.transport.get_accumulated_beats() - initial_beat).abs() < 1e-6);
    }
}
