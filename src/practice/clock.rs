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

        // (Seek decisions — Task 21.)
        let _ = timing_err;

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
