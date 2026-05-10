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

        let mut events = Vec::new();
        let present: std::collections::HashSet<u8> = frame.notes.iter().map(|(m, _)| *m).collect();
        let cents_by_midi: HashMap<u8, f64> = frame.notes.iter().copied().collect();

        // 1. Pitches present in the frame.
        for &m in &present {
            let cents = *cents_by_midi.get(&m).unwrap_or(&0.0);
            let entry = self.pitches.remove(&m);
            let new_state = match entry {
                None => PitchState::StartPending {
                    frames: 1,
                    first_frame_beat: frame.tuner_beat,
                    first_frame_seq: self.frame_seq,
                    cents_buffer: vec![cents],
                },
                Some(PitchState::StartPending {
                    frames,
                    first_frame_beat,
                    first_frame_seq,
                    mut cents_buffer,
                }) => {
                    cents_buffer.push(cents);
                    let new_frames = frames + 1;
                    if new_frames >= STABLE_FRAMES {
                        // Tier cascade is implemented in Task 11. For now, always StableFiveFrame.
                        let seq = self.next_event_seq;
                        self.next_event_seq += 1;
                        let avg = cents_buffer.iter().sum::<f64>() / cents_buffer.len() as f64;
                        let cents_sum = cents_buffer.iter().sum::<f64>();
                        events.push(ConditionerEvent::Started(TrackedNoteStart {
                            seq,
                            midi_note: m,
                            start_beat: first_frame_beat,
                            start_source: StartSource::StableFiveFrame,
                            initial_cents: avg,
                        }));
                        let _ = first_frame_seq;
                        PitchState::Active(ActiveBody {
                            seq,
                            start_beat: first_frame_beat,
                            start_source: StartSource::StableFiveFrame,
                            cents_sum,
                            frame_count: STABLE_FRAMES,
                        })
                    } else {
                        PitchState::StartPending {
                            frames: new_frames,
                            first_frame_beat,
                            first_frame_seq,
                            cents_buffer,
                        }
                    }
                }
                Some(PitchState::Active(mut body)) => {
                    body.cents_sum += cents;
                    body.frame_count += 1;
                    PitchState::Active(body)
                }
                Some(PitchState::EndPending { carry, .. }) => {
                    // Resume — absence was a brief gap.
                    PitchState::Active(carry)
                }
            };
            self.pitches.insert(m, new_state);
        }

        // 2. Pitches in the map but missing from the frame.
        let missing: Vec<u8> = self.pitches.keys()
            .copied()
            .filter(|m| !present.contains(m))
            .collect();
        for m in missing {
            let entry = self.pitches.remove(&m).unwrap();
            let new_state_opt = match entry {
                PitchState::StartPending { first_frame_beat, first_frame_seq, .. } => {
                    // Transient: log it for cluster detection.
                    self.transient_log.push_back((first_frame_seq, first_frame_beat, m));
                    None
                }
                PitchState::Active(carry) => Some(PitchState::EndPending {
                    absent_frames: 1,
                    first_absence_beat: frame.tuner_beat,
                    carry,
                }),
                PitchState::EndPending { absent_frames, first_absence_beat, carry } => {
                    let new_count = absent_frames + 1;
                    if new_count >= END_FRAMES {
                        let avg_cents = if carry.frame_count > 0 {
                            carry.cents_sum / carry.frame_count as f64
                        } else { 0.0 };
                        events.push(ConditionerEvent::Ended(TrackedNoteEnd {
                            seq: carry.seq,
                            midi_note: m,
                            end_beat: first_absence_beat,
                            avg_cents,
                            frame_count: carry.frame_count,
                        }));
                        None
                    } else {
                        Some(PitchState::EndPending {
                            absent_frames: new_count,
                            first_absence_beat,
                            carry,
                        })
                    }
                }
            };
            if let Some(s) = new_state_opt {
                self.pitches.insert(m, s);
            }
        }

        events
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

    fn frame(midis_cents: Vec<(u8, f64)>, beat: f64) -> TunerFrame {
        TunerFrame { notes: midis_cents, tuner_beat: beat }
    }

    #[test]
    fn stable_5_frames_emits_started_with_first_frame_beat() {
        let mut c = InputConditioner::new(mk_transport());
        // 4 frames not enough.
        for i in 0..4 {
            let f = frame(vec![(60, 0.0)], i as f64 * 0.02);
            assert!(c.ingest(Some(&f), &[]).is_empty());
        }
        // 5th frame confirms.
        let f = frame(vec![(60, 0.0)], 4.0 * 0.02);
        let evs = c.ingest(Some(&f), &[]);
        assert_eq!(evs.len(), 1);
        match &evs[0] {
            ConditionerEvent::Started(s) => {
                assert_eq!(s.midi_note, 60);
                assert!((s.start_beat - 0.0).abs() < 1e-9);     // first_frame_beat
                assert_eq!(s.start_source, StartSource::StableFiveFrame);
            }
            _ => panic!("expected Started"),
        }
    }

    #[test]
    fn pitch_disappearing_for_5_frames_emits_ended() {
        let mut c = InputConditioner::new(mk_transport());
        for i in 0..5 {
            let _ = c.ingest(Some(&frame(vec![(60, 0.0)], i as f64 * 0.02)), &[]);
        }
        // Now silent for 5 frames.
        let mut end_event: Option<TrackedNoteEnd> = None;
        for i in 5..10 {
            let evs = c.ingest(Some(&frame(vec![], i as f64 * 0.02)), &[]);
            for e in evs {
                if let ConditionerEvent::Ended(e) = e {
                    end_event = Some(e);
                }
            }
        }
        let e = end_event.expect("expected Ended event");
        assert_eq!(e.midi_note, 60);
        // first_absence_beat = 5 * 0.02 = 0.10.
        assert!((e.end_beat - 0.10).abs() < 1e-9);
    }
}
