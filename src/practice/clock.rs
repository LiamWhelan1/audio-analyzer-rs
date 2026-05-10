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
}
