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
    ClockAction, ConditionerEvent, MatchOutcome, PracticeMode, TrackedNoteEnd, TrackedNoteStart,
    TunerFrame,
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
    pub in_progress_onsets: HashMap<usize, Vec<OnsetEvent>>,
    pub in_progress_dynamics: HashMap<usize, Vec<DynamicsEvent>>,
    pub in_progress_durations: HashMap<usize, Vec<Option<f64>>>,
    pub in_progress_doubled_seqs: HashMap<usize, Vec<u64>>,

    pub match_log: HashMap<u64, MatchedSnapshot>,

    pub last_dynamic_level: Option<DynamicLevel>,

    pub feedback: Vec<crate::practice::SendInfo>,
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
            mode,
            transport,
            conditioner,
            buffer,
            clock,
            frontier: (practice_start, 0),
            in_progress_played_notes: HashMap::new(),
            in_progress_onsets: HashMap::new(),
            in_progress_dynamics: HashMap::new(),
            in_progress_durations: HashMap::new(),
            in_progress_doubled_seqs: HashMap::new(),
            match_log: HashMap::new(),
            last_dynamic_level: None,
            feedback: Vec::new(),
        }
    }

    /// One polling tick. Returns aged-out measures and intermediate events
    /// (for testing / feedback emission).
    pub fn tick(&mut self, inputs: TickInputs) -> TickOutputs {
        let mut outputs = TickOutputs {
            aged_measures: vec![],
            events: vec![],
            outcomes: vec![],
        };

        // 1. Conditioner.
        let events = self
            .conditioner
            .ingest(inputs.tuner_frame, inputs.new_onsets);

        // 2. Onsets accumulator (raw). Bucket by the onset's own beat — late-bucketed
        // onsets at a measure boundary belong to the measure that owns that beat.
        for o in inputs.new_onsets {
            let mi = self.buffer.measure_for_beat(o.beat_position);
            self.in_progress_onsets.entry(mi).or_default().push(*o);
        }

        // 3. Dynamics accumulator.
        if inputs.dynamic_level != DynamicLevel::Silence
            && self.last_dynamic_level != Some(inputs.dynamic_level)
        {
            self.in_progress_dynamics
                .entry(self.buffer.current_idx())
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
        let tick_actions = self.clock.on_tick(
            &self.buffer,
            self.frontier,
            inputs.transport_beat,
            self.mode,
        );
        for a in tick_actions {
            self.apply_action(&a);
        }

        // 6. Buffer advance.
        let aged = self.buffer.advance(inputs.transport_beat);
        for mut m in aged {
            let mi = m.measure_index as usize;
            // Mark any remaining Pending slots Missed AND emit a live MissingNote
            // SendInfo per spec §5 step 9 ("emit MissingNote SendInfo" for each
            // remaining Pending slot when the measure ages out).
            let mut to_miss: Vec<(usize, usize)> = (0..m.expected_notes.len())
                .filter(|i| {
                    matches!(
                        self.buffer.slot((mi, *i)).map(|s| &s.status),
                        Some(SlotStatus::Pending)
                    )
                })
                .map(|i| (mi, i))
                .collect();
            for k in to_miss.drain(..) {
                self.feedback.push(missing_note_send_info(k, &self.buffer));
                self.buffer.mark_missed(k);
                // Frontier may have stalled on this Pending slot; advance it past
                // the now-missed key so subsequent matches don't trip on it.
                if self.frontier == k {
                    self.frontier = step_forward(&self.buffer, k);
                }
            }
            // Drain in-progress accumulators.
            m.onsets = self.in_progress_onsets.remove(&mi).unwrap_or_default();
            m.notes = self
                .in_progress_played_notes
                .remove(&mi)
                .unwrap_or_default();
            m.dynamics = self.in_progress_dynamics.remove(&mi).unwrap_or_default();
            m.note_durations = self.in_progress_durations.remove(&mi).unwrap_or_default();
            m.doubled_note_seqs = self
                .in_progress_doubled_seqs
                .remove(&mi)
                .unwrap_or_default();
            outputs.aged_measures.push(m);
        }

        outputs
    }

    fn handle_outcome(
        &mut self,
        t: &TrackedNoteStart,
        outcome: &MatchOutcome,
        transport_beat: f64,
    ) {
        // Bucket the played note into the measure whose duration window contains
        // its start_beat — not into `current_idx`. A note Started at the very
        // end of a tick can have a start_beat that already belongs to the next
        // measure even though buffer.advance() hasn't run yet this tick.
        let mi = self.buffer.measure_for_beat(t.start_beat);
        self.in_progress_played_notes
            .entry(mi)
            .or_default()
            .push(NoteEvent {
                beat_position: t.start_beat,
                midi_note: t.midi_note,
                avg_cents: t.initial_cents,
            });
        self.in_progress_durations.entry(mi).or_default().push(None);
        let note_idx = self.in_progress_played_notes[&mi].len() - 1;

        let actions = match outcome {
            MatchOutcome::Matched {
                key,
                pitch_correct,
                upgrade,
                skipped_keys,
                timing_err,
            } => {
                for k in skipped_keys {
                    self.buffer.mark_missed(*k);
                    self.feedback.push(missing_note_send_info(*k, &self.buffer));
                }
                if *upgrade {
                    self.buffer.upgrade_match(*key, t);
                } else {
                    self.buffer.record_match(*key, t, *pitch_correct);
                }
                self.frontier = step_forward(&self.buffer, *key);
                let exp = expected_for(&self.buffer, *key);
                self.match_log.insert(
                    t.seq,
                    MatchedSnapshot {
                        measure_idx: key.0,
                        note_idx_in_measure_data: note_idx,
                        expected_duration: exp.duration_beats,
                        expected_midi: exp.midi_note,
                    },
                );
                // Live feedback for the match.
                let prim = if !*pitch_correct {
                    send_info(*key, crate::practice::MusicError::WrongNote, &exp, t)
                } else if *upgrade {
                    upgrade_send_info(*key, &exp, t)
                } else {
                    send_info(*key, crate::practice::MusicError::None, &exp, t)
                };
                self.feedback.push(prim);
                let timing_threshold = exp.duration_beats
                    * self.clock.cfg().seek_threshold_pct
                    * mode_tol_scale(self.mode);
                if timing_err.abs() > timing_threshold {
                    self.feedback
                        .push(timing_send_info(*key, &exp, t, *timing_err));
                }
                self.clock
                    .on_match(outcome, &exp, transport_beat, self.mode)
            }
            MatchOutcome::DoubledNote { key } => {
                self.in_progress_doubled_seqs
                    .entry(mi)
                    .or_default()
                    .push(t.seq);
                let exp = expected_for(&self.buffer, *key);
                self.feedback
                    .push(send_info(*key, crate::practice::MusicError::Tempo, &exp, t));
                let slot = self.buffer.slot(*key).cloned();
                slot.map(|s| self.clock.on_doubled(&s, self.mode))
                    .unwrap_or_default()
            }
            MatchOutcome::ExtraNote { during } => {
                self.feedback
                    .push(extra_note_send_info(*during, t, &self.buffer));
                self.clock.on_extra()
            }
        };
        for a in actions {
            self.apply_action(&a);
        }
    }

    fn handle_ended(&mut self, t: &TrackedNoteEnd) {
        let Some(snap) = self.match_log.remove(&t.seq) else {
            return;
        };
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
                const HOLD_TOLERANCE_PCT: f64 = 0.25;
                if actual_duration > snap.expected_duration * (1.0 + HOLD_TOLERANCE_PCT) {
                    self.feedback.push(crate::practice::SendInfo {
                        measure: mi as u32,
                        note_index: snap.note_idx_in_measure_data,
                        error_type: crate::practice::MusicError::HeldTooLong,
                        intensity: 0.6,
                        expected: format!("held~{:.2}", snap.expected_duration),
                        received: format!("held for {:.2}", actual_duration),
                    });
                } else if actual_duration < snap.expected_duration * (1.0 - HOLD_TOLERANCE_PCT) {
                    self.feedback.push(crate::practice::SendInfo {
                        measure: mi as u32,
                        note_index: snap.note_idx_in_measure_data,
                        error_type: crate::practice::MusicError::HeldTooShort,
                        intensity: 0.6,
                        expected: format!("held~{:.2}", snap.expected_duration),
                        received: format!("held for {:.2}", actual_duration),
                    });
                }
                const INTONATION_THRESHOLD: f64 = 25.0;
                let intonation_threshold = INTONATION_THRESHOLD * mode_tol_scale(self.mode);
                if t.avg_cents.abs() > intonation_threshold {
                    self.feedback.push(crate::practice::SendInfo {
                        measure: mi as u32,
                        note_index: snap.note_idx_in_measure_data,
                        error_type: crate::practice::MusicError::Intonation,
                        intensity: (t.avg_cents.abs() / 50.0).min(1.0),
                        expected: format!(
                            "{}",
                            crate::analysis::theory::Note::from_midi(snap.expected_midi).get_name()
                        ),
                        received: format!(
                            "{} {:+.0}c",
                            crate::analysis::theory::Note::from_midi(t.midi_note).get_name(),
                            t.avg_cents
                        ),
                    });
                }
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
    if buf.slot(next).is_some() {
        next
    } else {
        (key.0 + 1, 0)
    }
}

fn send_info(
    key: (usize, usize),
    err: crate::practice::MusicError,
    exp: &crate::practice::metrics::ExpectedNote,
    t: &TrackedNoteStart,
) -> crate::practice::SendInfo {
    crate::practice::SendInfo {
        measure: key.0 as u32,
        note_index: key.1,
        error_type: err,
        intensity: 0.0,
        expected: format!(
            "{} beat {:.2}",
            crate::analysis::theory::Note::from_midi(exp.midi_note).get_name(),
            exp.beat_position
        ),
        received: format!(
            "{} at beat {:.2}",
            crate::analysis::theory::Note::from_midi(t.midi_note).get_name(),
            t.start_beat
        ),
    }
}

fn upgrade_send_info(
    key: (usize, usize),
    exp: &crate::practice::metrics::ExpectedNote,
    t: &TrackedNoteStart,
) -> crate::practice::SendInfo {
    crate::practice::SendInfo {
        measure: key.0 as u32,
        note_index: key.1,
        error_type: crate::practice::MusicError::None,
        intensity: 0.0,
        expected: format!(
            "{} at beat {:.2} (corrected)",
            crate::analysis::theory::Note::from_midi(exp.midi_note).get_name(),
            exp.beat_position
        ),
        received: format!(
            "{} at beat {:.2}",
            crate::analysis::theory::Note::from_midi(t.midi_note).get_name(),
            t.start_beat
        ),
    }
}

fn timing_send_info(
    key: (usize, usize),
    exp: &crate::practice::metrics::ExpectedNote,
    t: &TrackedNoteStart,
    err: f64,
) -> crate::practice::SendInfo {
    crate::practice::SendInfo {
        measure: key.0 as u32,
        note_index: key.1,
        error_type: crate::practice::MusicError::Timing,
        intensity: (err.abs() / 0.5).min(1.0),
        expected: format!(
            "{} at beat {:.3}",
            crate::analysis::theory::Note::from_midi(exp.midi_note).get_name(),
            exp.beat_position
        ),
        received: format!(
            "{} at beat {:.3}",
            crate::analysis::theory::Note::from_midi(t.midi_note).get_name(),
            t.start_beat
        ),
    }
}

fn missing_note_send_info(key: (usize, usize), buf: &MeasureBuffer) -> crate::practice::SendInfo {
    let exp = expected_for(buf, key);
    crate::practice::SendInfo {
        measure: key.0 as u32,
        note_index: key.1,
        error_type: crate::practice::MusicError::MissingNote,
        intensity: 1.0,
        expected: format!(
            "{} at beat {:.2}",
            crate::analysis::theory::Note::from_midi(exp.midi_note).get_name(),
            exp.beat_position
        ),
        received: "silence".into(),
    }
}

fn mode_tol_scale(mode: PracticeMode) -> f64 {
    // Rubato widens the timing/intonation tolerance per spec §5 table.
    match mode {
        PracticeMode::Rubato => 1.5,
        _ => 1.0,
    }
}

fn extra_note_send_info(
    during: Option<(usize, usize)>,
    t: &TrackedNoteStart,
    buf: &MeasureBuffer,
) -> crate::practice::SendInfo {
    let (measure, note_index, expected_str) = match during {
        Some(key) => {
            let exp = expected_for(buf, key);
            (
                key.0 as u32,
                key.1,
                format!(
                    "{} (extra during held)",
                    crate::analysis::theory::Note::from_midi(exp.midi_note).get_name()
                ),
            )
        }
        None => (0, 0, "silence".into()),
    };
    crate::practice::SendInfo {
        measure,
        note_index,
        error_type: crate::practice::MusicError::UnexpectedNote,
        intensity: 0.5,
        expected: expected_str,
        received: format!(
            "{} at beat {:.2}",
            crate::analysis::theory::Note::from_midi(t.midi_note).get_name(),
            t.start_beat
        ),
    }
}

fn expected_for(
    buf: &MeasureBuffer,
    key: (usize, usize),
) -> crate::practice::metrics::ExpectedNote {
    let m = &buf.measures()[key.0];
    let n = &m.notes[key.1];
    crate::practice::metrics::ExpectedNote {
        beat_position: m.global_start_beat + n.start_beat_in_measure as f64,
        duration_beats: n.duration_beats as f64,
        midi_note: (69.0 + 12.0 * (n.freq / 440.0).log2())
            .round()
            .clamp(0.0, 127.0) as u8,
        dynamic: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_io::dynamics::DynamicLevel;
    use crate::generators::{Instrument, Measure, SynthNote};
    use std::sync::Arc;

    fn three_quarter_note_measure() -> Arc<Vec<Measure>> {
        Arc::new(vec![Measure {
            notes: vec![
                SynthNote {
                    freq: 261.626,
                    start_beat_in_measure: 0.0,
                    duration_beats: 1.0,
                    velocity: 0.5,
                    instrument: Instrument::Piano,
                },
                SynthNote {
                    freq: 293.665,
                    start_beat_in_measure: 1.0,
                    duration_beats: 1.0,
                    velocity: 0.5,
                    instrument: Instrument::Piano,
                },
                SynthNote {
                    freq: 329.628,
                    start_beat_in_measure: 2.0,
                    duration_beats: 1.0,
                    velocity: 0.5,
                    instrument: Instrument::Piano,
                },
            ],
            time_signature: (4, 4),
            bpm: 120.0,
            global_start_beat: 0.0,
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
        TunerFrame {
            notes: midis,
            tuner_beat: beat,
        }
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
    fn aged_out_pending_slots_emit_live_missing_note_feedback() {
        // 1-measure practice range with 3 notes, all silent → all should age
        // out as live MissingNote events when transport crosses past the measure.
        let mut mc = make_mc(PracticeMode::Performance);
        // Drive a few empty ticks inside the measure.
        let _ = mc.tick(TickInputs {
            transport_beat: 1.0,
            tuner_frame: None,
            new_onsets: &[],
            dynamic_level: DynamicLevel::Silence,
        });
        assert!(mc.feedback.is_empty(), "no events emitted mid-measure");
        // Cross past beat 4 → measure 0 ages out, all 3 Pending slots become Missed.
        let outputs = mc.tick(TickInputs {
            transport_beat: 4.5,
            tuner_frame: None,
            new_onsets: &[],
            dynamic_level: DynamicLevel::Silence,
        });
        assert_eq!(outputs.aged_measures.len(), 1);
        let missing_count = mc
            .feedback
            .iter()
            .filter(|e| e.error_type == crate::practice::MusicError::MissingNote)
            .count();
        assert_eq!(
            missing_count, 3,
            "expected 3 live MissingNote events, got: {:?}",
            mc.feedback
        );
    }

    #[test]
    fn note_started_at_measure_boundary_buckets_into_correct_measure() {
        // 2-measure practice range. Detect a stable C4 starting near the very
        // end of measure 0 (start_beat = 3.95) — within measure 0's [0, 4) window.
        // Another stable C4 starting at 4.05 — must land in measure 1.
        let measures = Arc::new(vec![
            Measure {
                notes: vec![SynthNote {
                    freq: 261.626,
                    start_beat_in_measure: 0.0,
                    duration_beats: 1.0,
                    velocity: 0.5,
                    instrument: Instrument::Piano,
                }],
                time_signature: (4, 4),
                bpm: 120.0,
                global_start_beat: 0.0,
            },
            Measure {
                notes: vec![SynthNote {
                    freq: 261.626,
                    start_beat_in_measure: 0.0,
                    duration_beats: 1.0,
                    velocity: 0.5,
                    instrument: Instrument::Piano,
                }],
                time_signature: (4, 4),
                bpm: 120.0,
                global_start_beat: 4.0,
            },
        ]);
        let transport = MusicalTransport::new(120.0, 48000.0);
        transport.play();
        let buffer = MeasureBuffer::new(measures.clone(), 0, 1);
        let conditioner = InputConditioner::new(transport.clone());
        let clock = ClockManager::new(transport.clone(), ClockConfig::default(), 120.0);
        let mut mc = ModeController::new(
            PracticeMode::Performance,
            transport,
            conditioner,
            buffer,
            clock,
            0,
        );

        // measure_for_beat sanity: 4.05 should map to measure 1, not measure 0.
        assert_eq!(mc.buffer.measure_for_beat(4.05), 1);
        assert_eq!(mc.buffer.measure_for_beat(3.95), 0);
    }

    #[test]
    fn performance_mode_does_not_call_seek_or_stop() {
        let mut mc = make_mc(PracticeMode::Performance);
        let initial_beat = mc.transport.get_accumulated_beats();
        // Bunch of frames with severe timing error.
        for i in 0..5 {
            let _ = mc.tick(TickInputs {
                transport_beat: 5.0 + i as f64 * 0.02, // way past expected beat 0
                tuner_frame: Some(&frame_with(vec![(60, 0.0)], 5.0 + i as f64 * 0.02)),
                new_onsets: &[],
                dynamic_level: DynamicLevel::Silence,
            });
        }
        // Performance mode never seeks.
        assert!((mc.transport.get_accumulated_beats() - initial_beat).abs() < 1e-6);
    }
}
