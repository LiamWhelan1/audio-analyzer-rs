//! ClockManager — owns student-tempo state and decides transport actions.
//! Returns ClockActions; does NOT call transport methods directly.

use std::sync::Arc;

use crate::audio_io::timing::MusicalTransport;
use crate::practice::buffer::{MeasureBuffer, NoteSlot, SlotStatus};
use crate::practice::metrics::ExpectedNote;
use crate::practice::types::{ClockAction, MatchOutcome, PracticeMode};

#[derive(Clone, Copy, Debug)]
pub struct ClockConfig {
    pub seek_threshold_pct: f64,
    pub bpm_change_threshold_pct: f64,
    pub bpm_change_streak: usize,
    pub stop_lead_epsilon: f64,
    pub seek_landing_epsilon: f64,
    pub ewma_alpha: f32,
}

impl Default for ClockConfig {
    fn default() -> Self {
        Self {
            seek_threshold_pct: 0.15,
            bpm_change_threshold_pct: 0.08,
            bpm_change_streak: 3,
            stop_lead_epsilon: 0.001,
            seek_landing_epsilon: 0.001,
            ewma_alpha: 0.4,
        }
    }
}

pub struct ClockManager {
    transport: Arc<MusicalTransport>,
    cfg: ClockConfig,
    bpm_ewma: f32,
    streak_late: usize,
    streak_early: usize,
    last_match_real_beat: Option<f64>,
    last_match_expected_beat: Option<f64>,
    stopped_for_unplayed: bool,
    /// Set in `on_tick` when frontier is Pending and transport has crossed it.
    /// Cleared in `on_match`. When `Some`, `t_stu_bpm()` returns this instead
    /// of `bpm_ewma`. NOT folded into the persistent EWMA (per spec §4 (b)).
    hesitation_tempo: Option<f32>,
}

impl ClockManager {
    pub fn new(transport: Arc<MusicalTransport>, cfg: ClockConfig, initial_bpm: f32) -> Self {
        Self {
            transport,
            cfg,
            bpm_ewma: initial_bpm,
            streak_late: 0,
            streak_early: 0,
            last_match_real_beat: None,
            last_match_expected_beat: None,
            stopped_for_unplayed: false,
            hesitation_tempo: None,
        }
    }

    pub fn t_stu_bpm(&self) -> f32 {
        self.hesitation_tempo.unwrap_or(self.bpm_ewma)
    }
    pub fn cfg(&self) -> &ClockConfig { &self.cfg }

    pub fn on_doubled(&mut self, slot: &NoteSlot, mode: PracticeMode) -> Vec<ClockAction> {
        if mode == PracticeMode::Performance { return Vec::new(); }
        let Some(matched_beat) = slot.matched_start_beat else { return Vec::new() };
        vec![
            ClockAction::SeekToBeat(matched_beat + self.cfg.seek_landing_epsilon),
            ClockAction::Play,
        ]
    }

    pub fn on_extra(&mut self) -> Vec<ClockAction> { Vec::new() }

    pub fn on_tick(
        &mut self,
        buf: &MeasureBuffer,
        frontier: (usize, usize),
        transport_beat: f64,
        mode: PracticeMode,
    ) -> Vec<ClockAction> {
        // Hesitation tempo: always compute when frontier is Pending and overdue.
        // This runs in every mode (it's a read-only display value, not a clock action).
        let frontier_pending = buf.slot(frontier)
            .map_or(false, |s| s.status == SlotStatus::Pending);
        if frontier_pending {
            let frontier_beat = {
                let m = &buf.measures()[frontier.0];
                m.global_start_beat + m.notes[frontier.1].start_beat_in_measure as f64
            };
            if transport_beat > frontier_beat {
                // Use the last anchor; if none, use frontier - 1 measure as a rough origin.
                if let (Some(prev_real), Some(prev_exp)) =
                    (self.last_match_real_beat, self.last_match_expected_beat)
                {
                    let real_diff = transport_beat - prev_real;
                    let exp_diff = frontier_beat - prev_exp;
                    if real_diff > 1e-6 && exp_diff > 0.0 {
                        let current_bpm = self.transport.get_bpm();
                        self.hesitation_tempo = Some((exp_diff / real_diff) as f32 * current_bpm);
                    }
                }
            } else {
                self.hesitation_tempo = None;
            }
        } else {
            self.hesitation_tempo = None;
        }

        // Stop trigger — FollowAlong only.
        if mode != PracticeMode::FollowAlong { return Vec::new(); }
        if self.stopped_for_unplayed { return Vec::new(); }
        if !frontier_pending { return Vec::new(); }

        let Some(next) = buf.next_pending_after(frontier) else { return Vec::new() };
        let next_beat = {
            let m = &buf.measures()[next.0];
            m.global_start_beat + m.notes[next.1].start_beat_in_measure as f64
        };

        if transport_beat >= next_beat - self.cfg.stop_lead_epsilon {
            self.stopped_for_unplayed = true;
            return vec![ClockAction::Stop];
        }
        Vec::new()
    }

    pub fn on_match(
        &mut self,
        outcome: &MatchOutcome,
        expected: &ExpectedNote,
        transport_beat: f64,
        mode: PracticeMode,
    ) -> Vec<ClockAction> {
        let MatchOutcome::Matched { timing_err, .. } = *outcome else {
            return Vec::new();
        };
        let mut actions = Vec::new();
        let current_bpm = self.transport.get_bpm();

        // T_stu update.
        if let (Some(prev_real), Some(prev_exp)) = (self.last_match_real_beat, self.last_match_expected_beat) {
            let real_diff = transport_beat - prev_real;
            let exp_diff = expected.beat_position - prev_exp;
            if real_diff > 1e-6 {
                let local_tempo = ((exp_diff / real_diff) as f32) * current_bpm;
                self.bpm_ewma = self.cfg.ewma_alpha * local_tempo
                              + (1.0 - self.cfg.ewma_alpha) * self.bpm_ewma;

                let pct = self.cfg.bpm_change_threshold_pct as f32;
                if local_tempo < current_bpm * (1.0 - pct) {
                    self.streak_late += 1;
                    self.streak_early = 0;
                } else if local_tempo > current_bpm * (1.0 + pct) {
                    self.streak_early += 1;
                    self.streak_late = 0;
                } else {
                    self.streak_late = 0;
                    self.streak_early = 0;
                }
            }
        }
        self.last_match_real_beat = Some(transport_beat);
        self.last_match_expected_beat = Some(expected.beat_position);
        // Match received → clear any in-flight hesitation override.
        self.hesitation_tempo = None;

        // Seek decision.
        match mode {
            PracticeMode::Performance => {}
            PracticeMode::FollowAlong => {
                let threshold = expected.duration_beats * self.cfg.seek_threshold_pct;
                let must_seek = timing_err.abs() > threshold || self.stopped_for_unplayed;
                if must_seek {
                    let target = if transport_beat < expected.beat_position {
                        expected.beat_position - self.cfg.seek_landing_epsilon
                    } else {
                        expected.beat_position + self.cfg.seek_landing_epsilon
                    };
                    actions.push(ClockAction::SeekToBeat(target));
                }
                actions.push(ClockAction::Play);
                self.stopped_for_unplayed = false;
            }
            PracticeMode::Rubato => {
                let target = if transport_beat < expected.beat_position {
                    expected.beat_position - self.cfg.seek_landing_epsilon
                } else {
                    expected.beat_position + self.cfg.seek_landing_epsilon
                };
                actions.push(ClockAction::SeekToBeat(target));
                actions.push(ClockAction::Play);
            }
        }

        // set_bpm gate.
        if mode != PracticeMode::Performance
            && (self.streak_late >= self.cfg.bpm_change_streak
                || self.streak_early >= self.cfg.bpm_change_streak)
        {
            let pct = self.cfg.bpm_change_threshold_pct as f32;
            let dev = (self.bpm_ewma - current_bpm).abs() / current_bpm.max(1.0);
            if dev > pct {
                actions.push(ClockAction::SetBpm(self.bpm_ewma));
                self.streak_late = 0;
                self.streak_early = 0;
            }
        }

        actions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk() -> ClockManager {
        let t = MusicalTransport::new(120.0, 48000.0);
        ClockManager::new(t, ClockConfig::default(), 120.0)
    }

    #[test]
    fn initial_t_stu_equals_initial_bpm() {
        let cm = mk();
        assert!((cm.t_stu_bpm() - 120.0).abs() < 1e-6);
    }

    fn matched(beat: f64, key: (usize, usize)) -> MatchOutcome {
        let _ = beat;
        MatchOutcome::Matched {
            key,
            timing_err: 0.0,
            pitch_correct: true,
            upgrade: false,
            skipped_keys: vec![],
        }
    }

    fn exp(beat: f64, dur: f64) -> ExpectedNote {
        ExpectedNote { beat_position: beat, duration_beats: dur, midi_note: 60, dynamic: None }
    }

    #[test]
    fn local_tempo_drags_when_real_time_diff_exceeds_expected() {
        let mut cm = mk();
        // First match — establishes anchor, no EWMA update.
        let _ = cm.on_match(&matched(0.0, (0,0)), &exp(0.0, 1.0), 0.0, PracticeMode::FollowAlong);
        // Second match: expected diff = 1.0 beats; real diff = 1.5 beats.
        // Local tempo = 1.0/1.5 * 120 = 80 BPM.
        // EWMA = 0.4 * 80 + 0.6 * 120 = 104.
        let _ = cm.on_match(&matched(1.5, (0,1)), &exp(1.0, 1.0), 1.5, PracticeMode::FollowAlong);
        assert!((cm.t_stu_bpm() - 104.0).abs() < 0.5);
    }

    #[test]
    fn doubled_in_followalong_emits_seek_back_plus_play() {
        let mut cm = mk();
        let slot = NoteSlot {
            status: SlotStatus::Matched { pitch_correct: true },
            matched_start_beat: Some(2.0),
            matched_seq: Some(0),
        };
        let actions = cm.on_doubled(&slot, PracticeMode::FollowAlong);
        let seek = actions.iter().find_map(|a| match a {
            ClockAction::SeekToBeat(b) => Some(*b), _ => None,
        });
        assert_eq!(seek, Some(2.001));
        assert!(actions.iter().any(|a| matches!(a, ClockAction::Play)));
    }

    #[test]
    fn doubled_in_performance_emits_nothing() {
        let mut cm = mk();
        let slot = NoteSlot {
            status: SlotStatus::Matched { pitch_correct: true },
            matched_start_beat: Some(2.0),
            matched_seq: Some(0),
        };
        let actions = cm.on_doubled(&slot, PracticeMode::Performance);
        assert!(actions.is_empty());
    }

    #[test]
    fn extra_emits_no_actions_in_any_mode() {
        let mut cm = mk();
        assert!(cm.on_extra().is_empty());
    }

    #[test]
    fn stop_fires_when_transport_approaches_next_after_pending_frontier() {
        use crate::practice::buffer::MeasureBuffer;
        use crate::generators::{Instrument, Measure, SynthNote};
        let measures = std::sync::Arc::new(vec![Measure {
            notes: vec![
                SynthNote { freq: 261.626, start_beat_in_measure: 0.0, duration_beats: 1.0, velocity: 0.5, instrument: Instrument::Piano },
                SynthNote { freq: 293.665, start_beat_in_measure: 1.0, duration_beats: 1.0, velocity: 0.5, instrument: Instrument::Piano },
                SynthNote { freq: 329.628, start_beat_in_measure: 2.0, duration_beats: 1.0, velocity: 0.5, instrument: Instrument::Piano },
            ],
            time_signature: (4, 4), bpm: 120.0, global_start_beat: 0.0,
        }]);
        let buf = MeasureBuffer::new(measures, 0, 0);
        let mut cm = mk();
        let frontier = (0, 1);    // beat 1 still Pending; beat 2 is the next-after.
        let actions = cm.on_tick(&buf, frontier, 1.999, PracticeMode::FollowAlong);
        assert!(actions.iter().any(|a| matches!(a, ClockAction::Stop)));
    }

    #[test]
    fn within_15pct_threshold_no_seek() {
        let mut cm = mk();
        let _ = cm.on_match(&matched(0.0, (0,0)), &exp(0.0, 1.0), 0.0, PracticeMode::FollowAlong);
        let actions = cm.on_match(
            &MatchOutcome::Matched { key: (0,1), timing_err: 0.10, pitch_correct: true, upgrade: false, skipped_keys: vec![] },
            &exp(1.0, 1.0),
            1.10,
            PracticeMode::FollowAlong,
        );
        assert!(actions.iter().all(|a| !matches!(a, ClockAction::SeekToBeat(_))));
    }

    #[test]
    fn early_outside_threshold_seeks_to_beat_minus_epsilon() {
        let mut cm = mk();
        let _ = cm.on_match(&matched(0.0, (0,0)), &exp(0.0, 1.0), 0.0, PracticeMode::FollowAlong);
        // Student played at transport.beat 0.7, "matched" at expected.beat 1.0 → 0.3 early.
        let actions = cm.on_match(
            &MatchOutcome::Matched { key: (0,1), timing_err: -0.3, pitch_correct: true, upgrade: false, skipped_keys: vec![] },
            &exp(1.0, 1.0),
            0.7,
            PracticeMode::FollowAlong,
        );
        let seek = actions.iter().find_map(|a| match a {
            ClockAction::SeekToBeat(b) => Some(*b),
            _ => None,
        }).expect("expected SeekToBeat");
        assert!((seek - (1.0 - 0.001)).abs() < 1e-9);
    }

    #[test]
    fn late_outside_threshold_seeks_to_beat_plus_epsilon() {
        let mut cm = mk();
        let _ = cm.on_match(&matched(0.0, (0,0)), &exp(0.0, 1.0), 0.0, PracticeMode::FollowAlong);
        let actions = cm.on_match(
            &MatchOutcome::Matched { key: (0,1), timing_err: 0.3, pitch_correct: true, upgrade: false, skipped_keys: vec![] },
            &exp(1.0, 1.0),
            1.3,
            PracticeMode::FollowAlong,
        );
        let seek = actions.iter().find_map(|a| match a {
            ClockAction::SeekToBeat(b) => Some(*b),
            _ => None,
        }).expect("expected SeekToBeat");
        assert!((seek - (1.0 + 0.001)).abs() < 1e-9);
    }

    #[test]
    fn three_late_notes_in_a_row_emit_set_bpm() {
        let mut cm = mk();
        // Anchor.
        let _ = cm.on_match(&matched(0.0, (0,0)), &exp(0.0, 1.0), 0.0, PracticeMode::FollowAlong);
        // Three late notes: 1.5, 3.0, 4.5 — local tempo = 80 each → eventually
        // |80 vs 120| = 33% > 8% AND streak = 3 → SetBpm.
        let mut last_actions: Vec<ClockAction> = vec![];
        for (i, (real, expected_b)) in [(1.5, 1.0), (3.0, 2.0), (4.5, 3.0)].iter().enumerate() {
            last_actions = cm.on_match(
                &matched(*real, (0, i+1)),
                &exp(*expected_b, 1.0),
                *real,
                PracticeMode::FollowAlong,
            );
        }
        assert!(
            last_actions.iter().any(|a| matches!(a, ClockAction::SetBpm(_))),
            "expected SetBpm in {last_actions:?}"
        );
    }
}
