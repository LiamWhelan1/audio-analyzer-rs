//! 3-tier note-start conditioner with per-pitch state machines, transient
//! cluster tracking, and vibrato glide cents folding.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use crate::audio_io::timing::{MusicalTransport, OnsetEvent};
use crate::practice::types::{
    ConditionerEvent, StartSource, TrackedNoteEnd, TrackedNoteStart, TunerFrame,
};

pub const STABLE_FRAMES: u32 = 5;
pub const END_FRAMES: u32 = 5;
pub const ONSET_CLAIM_WINDOW: f64 = 0.05;
pub const CLUSTER_MIN_TRANSIENTS: usize = 4;
pub const CLUSTER_FRAME_WINDOW: u64 = 10;
pub const RECENT_ONSET_RETENTION_BEATS: f64 = 0.5;

#[derive(Clone, Debug)]
struct ActiveBody {
    seq: u64,
    start_beat: f64,
    start_source: StartSource,
    cents_sum: f64,
    frame_count: u32,
}

#[derive(Clone, Debug)]
enum PitchState {
    StartPending {
        frames: u32,
        first_frame_beat: f64,
        first_frame_seq: u64,
        cents_buffer: Vec<f64>,
    },
    Active(ActiveBody),
    EndPending {
        absent_frames: u32,
        first_absence_beat: f64,
        carry: ActiveBody,
    },
}

pub struct InputConditioner {
    _transport: Arc<MusicalTransport>,
    pitches: HashMap<u8, PitchState>,
    recent_onsets: VecDeque<OnsetEvent>,
    transient_log: VecDeque<(u64, f64, u8)>,    // (seq, beat, midi)
    frame_seq: u64,
    next_event_seq: u64,
    last_tuner_beat: Option<f64>,
}

impl InputConditioner {
    pub fn new(transport: Arc<MusicalTransport>) -> Self {
        Self {
            _transport: transport,
            pitches: HashMap::new(),
            recent_onsets: VecDeque::new(),
            transient_log: VecDeque::new(),
            frame_seq: 0,
            next_event_seq: 0,
            last_tuner_beat: None,
        }
    }

    pub fn ingest(
        &mut self,
        tuner_frame: Option<&TunerFrame>,
        new_onsets: &[OnsetEvent],
    ) -> Vec<ConditionerEvent> {
        // Always ingest onsets (10 ms tick rate).
        for o in new_onsets {
            self.recent_onsets.push_back(*o);
        }

        // Run the per-pitch machine only on new tuner frames.
        let Some(frame) = tuner_frame else { return Vec::new() };
        if Some(frame.tuner_beat) == self.last_tuner_beat {
            return Vec::new();
        }
        self.last_tuner_beat = Some(frame.tuner_beat);
        self.frame_seq += 1;

        // Drop stale onsets relative to the current frame.
        let cutoff = frame.tuner_beat - RECENT_ONSET_RETENTION_BEATS;
        while self.recent_onsets.front().map_or(false, |o| o.beat_position < cutoff) {
            self.recent_onsets.pop_front();
        }
        // Drop stale transient_log entries.
        let seq_cutoff = self.frame_seq.saturating_sub(CLUSTER_FRAME_WINDOW);
        while self.transient_log.front().map_or(false, |&(s, _, _)| s < seq_cutoff) {
            self.transient_log.pop_front();
        }

        // (State transitions implemented in Task 10.)
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_transport() -> Arc<MusicalTransport> {
        MusicalTransport::new(120.0, 48000.0)
    }

    #[test]
    fn ingest_with_no_inputs_returns_empty() {
        let mut c = InputConditioner::new(mk_transport());
        assert!(c.ingest(None, &[]).is_empty());
    }

    #[test]
    fn ingest_dedups_repeat_tuner_frames() {
        let mut c = InputConditioner::new(mk_transport());
        let f = TunerFrame { notes: vec![], tuner_beat: 1.0 };
        let _ = c.ingest(Some(&f), &[]);
        // Same beat → no further work.
        let events = c.ingest(Some(&f), &[]);
        assert!(events.is_empty());
    }
}
