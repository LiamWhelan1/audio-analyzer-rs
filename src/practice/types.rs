//! Shared types crossing component boundaries: practice mode, conditioner
//! events, match outcomes, clock actions. No logic.

/// Practice mode — controls which clock actions are applied.
#[derive(serde::Serialize, Clone, Copy, Debug, PartialEq)]
pub enum PracticeMode {
    FollowAlong,
    Performance,
    Rubato,
}

impl PracticeMode {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "followalong" | "follow_along" | "follow-along" => Some(Self::FollowAlong),
            "performance" => Some(Self::Performance),
            "rubato" => Some(Self::Rubato),
            _ => None,
        }
    }
}

/// One tuner analysis hop — decoded from `TunerOutput` for the conditioner.
#[derive(Clone, Debug)]
pub struct TunerFrame {
    pub notes: Vec<(u8, f64)>,    // (midi_note, cents)
    pub tuner_beat: f64,          // already calibrated
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StartSource {
    Onset,
    StableFiveFrame,
    TransientCluster,
}

#[derive(Clone, Debug)]
pub struct TrackedNoteStart {
    pub seq: u64,
    pub midi_note: u8,
    pub start_beat: f64,
    pub start_source: StartSource,
    pub initial_cents: f64,
}

#[derive(Clone, Debug)]
pub struct TrackedNoteEnd {
    pub seq: u64,
    pub midi_note: u8,
    pub end_beat: f64,
    pub avg_cents: f64,
    pub frame_count: u32,
}

#[derive(Clone, Debug)]
pub enum ConditionerEvent {
    Started(TrackedNoteStart),
    Ended(TrackedNoteEnd),
}

#[derive(Clone, Debug, PartialEq)]
pub enum MatchOutcome {
    Matched {
        key: (usize, usize),
        timing_err: f64,
        pitch_correct: bool,
        upgrade: bool,
        skipped_keys: Vec<(usize, usize)>,
    },
    DoubledNote { key: (usize, usize) },
    ExtraNote { during: Option<(usize, usize)> },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ClockAction {
    SeekToBeat(f64),
    Stop,
    Play,
    SetBpm(f32),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn practice_mode_parses_case_insensitive() {
        assert_eq!(PracticeMode::from_str("FollowAlong"), Some(PracticeMode::FollowAlong));
        assert_eq!(PracticeMode::from_str("performance"), Some(PracticeMode::Performance));
        assert_eq!(PracticeMode::from_str("RUBATO"), Some(PracticeMode::Rubato));
        assert_eq!(PracticeMode::from_str("invalid"), None);
    }
}
