# Practice Engine Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor [src/practice/mod.rs](../../../src/practice/mod.rs) to implement the 3-tier note conditioner, 3-measure ring buffer, polyphonic matcher, ClockManager with hybrid `T_stu`, and three execution modes (FollowAlong, Performance, Rubato) per the design spec.

**Architecture:** Single-threaded polling loop (10 ms cadence). Each tick calls layered components in order: `InputConditioner → Matcher → ClockManager → feedback emission`. ModeController owns orchestration and applies `ClockAction`s through a per-mode filter. `PracticeSession` public API preserved; only `new` gains a `mode` parameter (replacing `wait_for_onset`).

**Tech Stack:** Rust 2021. Standard library (`HashMap`, `VecDeque`). `parking_lot::RwLock` for the existing tuner/dynamics output handles. Existing `MusicalTransport` from [src/audio_io/timing.rs](../../../src/audio_io/timing.rs). Existing `Tuner` (already supports `MultiPitch` mode). Existing `OnsetDetection`. Tests use `cargo test --lib practice::` and `cargo test --lib audio_io::`.

**Spec:** [docs/superpowers/specs/2026-05-09-practice-engine-refactor-design.md](../specs/2026-05-09-practice-engine-refactor-design.md)

---

## File structure

**Created files (under `src/practice/`):**

| File | Responsibility |
|---|---|
| `types.rs` | Shared event/match/action enums and `TunerFrame`. No logic. |
| `buffer.rs` | `MeasureBuffer` — 3-measure ring + per-slot match state. |
| `conditioner.rs` | `InputConditioner` — per-pitch state machines, tier cascade, vibrato glide. |
| `matcher.rs` | Pure `resolve()` function over `(TrackedNoteStart, &MeasureBuffer, frontier)`. |
| `clock.rs` | `ClockManager` — `T_stu`, seek/stop/play/set_bpm decisions. Returns `ClockAction`s. |
| `mode.rs` | `ModeController` — orchestrator + per-tick algorithm + mode filter. |
| `session.rs` | `SessionState` (FFI-shared, mutex-guarded) split out from `mod.rs`. |

**Modified files:**

| File | Changes |
|---|---|
| `src/practice/mod.rs` | Strip down to `PracticeSession`; declare new sub-modules; remove inline `SessionState` and `run_session`. |
| `src/practice/metrics.rs` | Add `note_durations`, `doubled_note_seqs` to `MeasureData`. Extend `MusicError` and `Metrics`. |
| `src/audio_io/timing.rs` | Add `MusicalTransport::calibrated_beat`. |
| `src/lib.rs` | Replace `wait_for_onset: bool` parameter with `mode: String`. |

---

## Phase 1 — Foundations: types, ring buffer, transport helper

End-of-phase milestone: existing functionality intact; `MeasureBuffer` drives measure advance; new types/helpers compile but aren't yet wired into detection logic.

### Task 1: Add `MusicalTransport::calibrated_beat`

**Files:**
- Modify: `src/audio_io/timing.rs` (add method after `seek_to_beat`)

- [ ] **Step 1: Write the failing test**

In `src/audio_io/timing.rs` `mod tests`, add:

```rust
#[test]
fn test_calibrated_beat_subtracts_total_latency() {
    let t = MusicalTransport::new(120.0, 48000.0);
    t.set_input_latency(480);                // 10 ms
    t.set_output_latency(240);               // 5 ms
    t.set_calibration_offset(96);            // 2 ms residual

    // 120 BPM = 2 beats/sec. Total latency = 816 samples / 48000 = 0.017 s.
    // Beats of latency = 0.017 * 2 = 0.034.
    let calibrated = t.calibrated_beat(4.0);
    let expected = 4.0 - (816.0 / 48000.0) * (120.0 / 60.0);
    assert!(
        (calibrated - expected).abs() < 1e-9,
        "expected {expected}, got {calibrated}"
    );
}

#[test]
fn test_calibrated_beat_zero_latency_passthrough() {
    let t = MusicalTransport::new(120.0, 48000.0);
    assert!((t.calibrated_beat(2.5) - 2.5).abs() < 1e-9);
}
```

- [ ] **Step 2: Verify the tests fail**

Run: `cargo test --lib audio_io::timing::tests::test_calibrated_beat -- --nocapture`
Expected: `error[E0599]: no method named `calibrated_beat` found`.

- [ ] **Step 3: Implement `calibrated_beat`**

In `src/audio_io/timing.rs`, add after `seek_to_beat` (around line 428):

```rust
/// Apply input + output latency + calibration_offset compensation to a raw
/// beat position obtained from non-onset sources (e.g. the tuner pipeline).
/// Onset events from `stamp_onset` are already calibrated and must NOT be
/// passed through this helper.
pub fn calibrated_beat(&self, raw_beat: f64) -> f64 {
    let sr = self.get_sample_rate() as f64;
    let bpm = self.get_bpm() as f64;
    let beats_per_sample = bpm / (60.0 * sr);
    let input_lat = self.input_latency_samples.load(Ordering::Relaxed);
    let output_lat = self.output_latency_samples.load(Ordering::Relaxed);
    let calibration = self.calibration_offset_samples.load(Ordering::Relaxed);
    let latency_beats = (input_lat + output_lat) as f64 * beats_per_sample;
    let calibration_beats = calibration as f64 * beats_per_sample;
    raw_beat - latency_beats - calibration_beats
}
```

- [ ] **Step 4: Verify tests pass**

Run: `cargo test --lib audio_io::timing::tests::test_calibrated_beat`
Expected: 2 passing.

- [ ] **Step 5: Commit**

```bash
git add src/audio_io/timing.rs
git commit -m "Add MusicalTransport::calibrated_beat helper"
```

---

### Task 2: Create `practice/types.rs` with shared types

**Files:**
- Create: `src/practice/types.rs`
- Modify: `src/practice/mod.rs` (line 13 area: declare module)

- [ ] **Step 1: Write `types.rs`**

Create `src/practice/types.rs`:

```rust
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
    DoubledNote {
        key: (usize, usize),
    },
    ExtraNote {
        during: Option<(usize, usize)>,
    },
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
```

- [ ] **Step 2: Declare module**

In `src/practice/mod.rs`, find `pub mod metrics;` (line 13) and replace with:

```rust
pub mod metrics;
pub mod types;
```

- [ ] **Step 3: Verify build + tests**

```
cargo test --lib practice::types::
```
Expected: 1 passing.

- [ ] **Step 4: Commit**

```bash
git add src/practice/types.rs src/practice/mod.rs
git commit -m "Add practice::types module with shared event types"
```

---

### Task 3: Extend `MeasureData` with `note_durations` and `doubled_note_seqs`

**Files:**
- Modify: `src/practice/metrics.rs` (extend `MeasureData` struct + update existing call sites)

- [ ] **Step 1: Add fields to `MeasureData`**

In `src/practice/metrics.rs`, find the `MeasureData` struct (around line 51) and add two fields:

```rust
pub struct MeasureData {
    pub measure_index: u32,
    pub onsets: Vec<OnsetEvent>,
    pub notes: Vec<NoteEvent>,
    pub dynamics: Vec<DynamicsEvent>,
    pub expected_notes: Vec<ExpectedNote>,
    /// Parallel to `notes`: actual measured duration (end_beat - start_beat),
    /// or None if the corresponding TrackedNoteEnd never arrived (still held,
    /// session ended mid-note, etc.).
    pub note_durations: Vec<Option<f64>>,
    /// Sequence numbers of `notes` entries that were flagged as DoubledNote
    /// (start-restart) by the matcher.
    pub doubled_note_seqs: Vec<u64>,
}
```

- [ ] **Step 2: Update existing constructions**

Find every `MeasureData {` literal and add the two new fields with empty `Vec::new()` defaults. Locations:

- `src/practice/mod.rs` around line 877 (the `data = MeasureData { ... }` in `run_session`).
- `src/practice/metrics.rs` test module around line 657, 670, 683, 695, 712, 723, 738, 752, 768, 781, 802, 818, 836, 853, 917, 943 — every test uses `MeasureData { ... }`.

For each, add `note_durations: vec![]` and `doubled_note_seqs: vec![]` to the struct literal.

- [ ] **Step 3: Verify all existing tests still pass**

```
cargo test --lib practice::metrics
cargo test --lib practice::tests
```
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add src/practice/mod.rs src/practice/metrics.rs
git commit -m "Add note_durations and doubled_note_seqs fields to MeasureData"
```

---

### Task 4: `MeasureBuffer` skeleton with `NoteSlot` and constructor

**Files:**
- Create: `src/practice/buffer.rs`
- Modify: `src/practice/mod.rs` (declare `pub mod buffer;`)

- [ ] **Step 1: Write skeleton + constructor test**

Create `src/practice/buffer.rs`:

```rust
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
}
```

- [ ] **Step 2: Declare module**

In `src/practice/mod.rs`:

```rust
pub mod metrics;
pub mod types;
pub mod buffer;
```

- [ ] **Step 3: Verify**

```
cargo test --lib practice::buffer
```
Expected: 1 passing.

- [ ] **Step 4: Commit**

```bash
git add src/practice/buffer.rs src/practice/mod.rs
git commit -m "Add MeasureBuffer skeleton with slot population"
```

---

### Task 5: `MeasureBuffer::advance` cycles measures forward

**Files:**
- Modify: `src/practice/buffer.rs`

- [ ] **Step 1: Write the failing test**

Add to `buffer.rs` `mod tests`:

```rust
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
    // Slots for the aged-out measure are removed.
    assert!(buf.slot((0, 0)).is_none());
}

#[test]
fn advance_marks_pending_slots_missed_on_age_out() {
    let measures = Arc::new(vec![dummy_measure(0.0, 2), dummy_measure(4.0, 1)]);
    let mut buf = MeasureBuffer::new(measures, 0, 1);
    let aged = buf.advance(4.5);
    assert_eq!(aged.len(), 1);
    // Two unplayed slots in measure 0 → expected_notes len = 2 in MeasureData.
    assert_eq!(aged[0].expected_notes.len(), 2);
}
```

- [ ] **Step 2: Verify failure**

```
cargo test --lib practice::buffer::tests::advance_
```
Expected: FAIL — `advance` undefined.

- [ ] **Step 3: Implement `advance`**

Add to `impl MeasureBuffer`:

```rust
/// Cycle past/current/future when transport_beat ≥ current_measure_end.
/// Returns 0 or 1 aged-out measures (as MeasureData).
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

    // Build the MeasureData skeleton with just the expected_notes; the
    // ModeController populates the per-tick accumulators (notes, onsets,
    // dynamics, durations, doubled_seqs).
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
```

Add the helper at the bottom of `buffer.rs`:

```rust
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
```

(These are duplicated from `mod.rs`; Task 7 will move them to a shared location.)

- [ ] **Step 4: Verify tests pass**

```
cargo test --lib practice::buffer
```
Expected: 3 passing.

- [ ] **Step 5: Commit**

```bash
git add src/practice/buffer.rs
git commit -m "Implement MeasureBuffer::advance with measure cycling"
```

---

### Task 6: `record_match`, `upgrade_match`, `mark_missed`, `next_pending_after`

**Files:**
- Modify: `src/practice/buffer.rs`

- [ ] **Step 1: Write tests**

Add to `buffer.rs` `mod tests`:

```rust
fn fake_tracked(midi: u8, beat: f64) -> TrackedNoteStart {
    TrackedNoteStart {
        seq: 0,
        midi_note: midi,
        start_beat: beat,
        start_source: crate::practice::types::StartSource::Onset,
        initial_cents: 0.0,
    }
}

#[test]
fn record_match_sets_slot_matched_correct() {
    let measures = Arc::new(vec![dummy_measure(0.0, 2)]);
    let mut buf = MeasureBuffer::new(measures, 0, 0);
    let t = fake_tracked(60, 0.0);
    buf.record_match((0, 0), &t, true);
    let s = buf.slot((0, 0)).unwrap();
    assert_eq!(s.status, SlotStatus::Matched { pitch_correct: true });
    assert_eq!(s.matched_start_beat, Some(0.0));
    assert_eq!(s.matched_seq, Some(0));
}

#[test]
fn upgrade_match_promotes_pitch_correct_false_to_true() {
    let measures = Arc::new(vec![dummy_measure(0.0, 1)]);
    let mut buf = MeasureBuffer::new(measures, 0, 0);
    let t1 = fake_tracked(61, 0.05);
    let t2 = fake_tracked(60, 0.1);
    buf.record_match((0, 0), &t1, false);
    buf.upgrade_match((0, 0), &t2);
    let s = buf.slot((0, 0)).unwrap();
    assert_eq!(s.status, SlotStatus::Matched { pitch_correct: true });
    assert_eq!(s.matched_seq, Some(0));   // t2.seq
}

#[test]
fn mark_missed_sets_slot_missed() {
    let measures = Arc::new(vec![dummy_measure(0.0, 2)]);
    let mut buf = MeasureBuffer::new(measures, 0, 0);
    buf.mark_missed((0, 0));
    assert_eq!(buf.slot((0, 0)).unwrap().status, SlotStatus::Missed);
}

#[test]
fn next_pending_after_finds_next_unmatched_slot() {
    let measures = Arc::new(vec![dummy_measure(0.0, 4)]);
    let mut buf = MeasureBuffer::new(measures, 0, 0);
    let t = fake_tracked(60, 0.0);
    buf.record_match((0, 0), &t, true);
    assert_eq!(buf.next_pending_after((0, 0)), Some((0, 1)));
    buf.mark_missed((0, 1));
    assert_eq!(buf.next_pending_after((0, 0)), Some((0, 2)));
}

#[test]
fn next_pending_after_crosses_measure_boundary() {
    let measures = Arc::new(vec![dummy_measure(0.0, 1), dummy_measure(4.0, 2)]);
    let buf = MeasureBuffer::new(measures, 0, 1);
    assert_eq!(buf.next_pending_after((0, 0)), Some((1, 0)));
}
```

- [ ] **Step 2: Implement methods**

Add to `impl MeasureBuffer`:

```rust
pub fn record_match(&mut self, key: (usize, usize), tracked: &TrackedNoteStart, pitch_correct: bool) {
    if let Some(slot) = self.slots.get_mut(&key) {
        slot.status = SlotStatus::Matched { pitch_correct };
        slot.matched_start_beat = Some(tracked.start_beat);
        slot.matched_seq = Some(tracked.seq);
    }
}

pub fn upgrade_match(&mut self, key: (usize, usize), tracked: &TrackedNoteStart) {
    if let Some(slot) = self.slots.get_mut(&key) {
        slot.status = SlotStatus::Matched { pitch_correct: true };
        slot.matched_start_beat = Some(tracked.start_beat);
        slot.matched_seq = Some(tracked.seq);
    }
}

pub fn mark_missed(&mut self, key: (usize, usize)) {
    if let Some(slot) = self.slots.get_mut(&key) {
        slot.status = SlotStatus::Missed;
    }
}

/// Lowest-key slot strictly after `frontier` whose status is `Pending`.
/// Walks current → future measure boundary if needed.
pub fn next_pending_after(&self, frontier: (usize, usize)) -> Option<(usize, usize)> {
    // Within current measure first.
    let candidates = [self.current_idx]
        .into_iter()
        .chain(self.future_idx.into_iter());
    for m_idx in candidates {
        let n_count = self.measures[m_idx].notes.len();
        let start = if m_idx == frontier.0 { frontier.1 + 1 } else { 0 };
        for n_idx in start..n_count {
            let key = (m_idx, n_idx);
            if let Some(slot) = self.slots.get(&key) {
                if slot.status == SlotStatus::Pending {
                    return Some(key);
                }
            }
        }
    }
    None
}
```

- [ ] **Step 3: Verify tests pass**

```
cargo test --lib practice::buffer
```
Expected: 8 passing.

- [ ] **Step 4: Commit**

```bash
git add src/practice/buffer.rs
git commit -m "Implement MeasureBuffer slot-mutation methods + next_pending_after"
```

---

### Task 7: `MeasureBuffer::candidates` — overlap, look-ahead, look-behind

**Files:**
- Modify: `src/practice/buffer.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn candidates_includes_in_window_lookahead_and_lookbehind() {
    let measures = Arc::new(vec![dummy_measure(0.0, 4)]);
    let mut buf = MeasureBuffer::new(measures, 0, 0);
    // Mark slot 0 as Matched (it'll show up as Lookbehind from frontier=(0,1)).
    buf.record_match((0, 0), &fake_tracked(60, 0.0), true);

    // Frontier is (0, 1). beat=1.5 is inside slot 1's duration window [1,2)
    // and slot 2 is +1 lookahead, slot 3 is +2 lookahead.
    let cands = buf.candidates(1.5, (0, 1));
    let keys: Vec<_> = cands.iter().map(|c| (c.key, c.kind.clone())).collect();
    assert!(keys.iter().any(|(k, _)| *k == (0, 1)));    // InWindow
    assert!(keys.iter().any(|(k, _)| *k == (0, 2)));    // Lookahead(1)
    assert!(keys.iter().any(|(k, _)| *k == (0, 3)));    // Lookahead(2)
    assert!(keys.iter().any(|(k, _)| *k == (0, 0)));    // Lookbehind(1)
}
```

(Add `#[derive(Clone)]` on `CandidateKind` in `buffer.rs` near its definition so the test can `.clone()` it.)

- [ ] **Step 2: Implement `candidates`**

Add to `impl MeasureBuffer`:

```rust
const LOOKAHEAD_NOTES: u8 = 2;
const LOOKBEHIND_NOTES: u8 = 1;

pub fn candidates(&self, beat: f64, frontier: (usize, usize)) -> Vec<Candidate> {
    let mut out = Vec::new();
    let measure_indices: Vec<usize> = [self.past_idx, Some(self.current_idx), self.future_idx]
        .iter().copied().flatten().collect();

    // First: collect every slot in the active measures with its expected note.
    let mut all: Vec<(usize, usize, ExpectedNote)> = Vec::new();
    for m_idx in measure_indices {
        let expected = build_expected_notes(&self.measures[m_idx]);
        for (n_idx, exp) in expected.into_iter().enumerate() {
            all.push((m_idx, n_idx, exp));
        }
    }

    // Sort by beat position so we can compute distance from frontier in note count.
    all.sort_by(|a, b| a.2.beat_position.partial_cmp(&b.2.beat_position).unwrap_or(std::cmp::Ordering::Equal));

    // Find frontier's index in `all`.
    let frontier_pos = all.iter().position(|&(m, n, _)| (m, n) == frontier);

    for (i, (m_idx, n_idx, exp)) in all.iter().enumerate() {
        let key = (*m_idx, *n_idx);
        let Some(slot) = self.slots.get(&key) else { continue };
        let in_window = beat >= exp.beat_position
            && beat < exp.beat_position + exp.duration_beats;

        let kind = if in_window {
            CandidateKind::InWindow
        } else if let Some(fp) = frontier_pos {
            let delta = i as isize - fp as isize;
            if delta > 0 && delta <= LOOKAHEAD_NOTES as isize {
                CandidateKind::Lookahead(delta as u8)
            } else if delta < 0 && (-delta) <= LOOKBEHIND_NOTES as isize {
                CandidateKind::Lookbehind((-delta) as u8)
            } else {
                continue;
            }
        } else {
            continue;
        };

        out.push(Candidate {
            key,
            expected: exp.clone(),
            status: slot.status.clone(),
            kind,
        });
    }
    out
}
```

You'll also need `#[derive(Clone)]` on `ExpectedNote` in `metrics.rs` for `exp.clone()` to compile.

- [ ] **Step 3: Verify**

```
cargo test --lib practice::buffer
```
Expected: 9 passing.

- [ ] **Step 4: Commit**

```bash
git add src/practice/buffer.rs src/practice/metrics.rs
git commit -m "Implement MeasureBuffer::candidates with lookahead/lookbehind"
```

---

### Task 8: Wire `MeasureBuffer` into existing `run_session` (no behavior change)

**Goal:** replace inline `current_measure_idx` advance with `MeasureBuffer.advance` call. Existing pitch-detection / matching code untouched. Existing tests must continue passing.

**Files:**
- Modify: `src/practice/mod.rs`

- [ ] **Step 1: Add `MeasureBuffer` to `SessionState`**

Find `SessionState` in `src/practice/mod.rs` (line 97). Replace its `current_measure_idx: usize` field reference site (line 100) with both fields side-by-side temporarily — keep `current_measure_idx` for the FFI surface, add a new `buffer: Option<MeasureBuffer>` for orchestration:

```rust
struct SessionState {
    practice_start: usize,
    practice_end: usize,
    current_measure_idx: usize,     // mirrored from buffer.current_idx for poll_transport
    buffer: Option<crate::practice::buffer::MeasureBuffer>,
    // ... rest unchanged
}
```

In `SessionState::new`, initialize `buffer: None`. In `PracticeSession::start` (around line 339), after creating the new SessionState, populate the buffer:

```rust
*self.state.lock().map_err(lock_err)? = SessionState::new(start, end, /* ... */);
{
    let mut s = self.state.lock().map_err(lock_err)?;
    s.buffer = Some(crate::practice::buffer::MeasureBuffer::new(
        self.measures.clone(),
        start,
        end,
    ));
}
```

- [ ] **Step 2: Replace the `if beat >= measure_end { ... }` block in `run_session`**

In `run_session` at the bottom (around line 851), replace:

```rust
if beat >= measure_end {
    s.finalize_pending_note();
    let finished_idx = s.current_measure_idx;
    let expected = build_expected_notes(&measures[finished_idx]);
    let done = finished_idx >= s.practice_end;
    if !done {
        s.current_measure_idx += 1;
        s.onset_received_this_measure = false;
        // ... existing onset-wait-mode code ...
    }
    let data = MeasureData { /* ... */ };
    // ...
}
```

with:

```rust
if beat >= measure_end {
    s.finalize_pending_note();
    let finished_idx = s.current_measure_idx;
    let expected = build_expected_notes(&measures[finished_idx]);
    let done = finished_idx >= s.practice_end;

    // Drive MeasureBuffer to age out the finished measure.
    if let Some(buf) = s.buffer.as_mut() {
        let _aged = buf.advance(beat);
        // (We continue using the existing per-tick accumulators for now;
        // Task 26 will move this into ModeController.)
        if !done {
            s.current_measure_idx = buf.current_idx();
        }
    } else if !done {
        s.current_measure_idx += 1;
    }
    s.onset_received_this_measure = false;
    // (rest of the block unchanged)
```

- [ ] **Step 3: Verify all existing tests still pass**

```
cargo test --lib practice
```
Expected: all passing (no behavioral change).

- [ ] **Step 4: Commit**

```bash
git add src/practice/mod.rs
git commit -m "Wire MeasureBuffer into existing run_session for measure advance"
```

---

### Phase 1 milestone

```bash
cargo test --lib
cargo build --release
```
Expected: full test suite green, binary builds. Commit any cleanup:

```bash
git commit --allow-empty -m "Phase 1 complete: foundations laid"
```

---

## Phase 2 — InputConditioner

End-of-phase milestone: the inline `pitch-stability hysteresis` block in `run_session` is replaced by a call to `InputConditioner::ingest`. Polyphonic detection is exercised. Vibrato glide is correct. Existing single-pitch behavior is preserved as a degenerate case.

### Task 9: `InputConditioner` skeleton + state types

**Files:**
- Create: `src/practice/conditioner.rs`
- Modify: `src/practice/mod.rs` (declare `pub mod conditioner;`)

- [ ] **Step 1: Write skeleton + smoke test**

Create `src/practice/conditioner.rs`:

```rust
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

        // (State transitions implemented in next tasks.)
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
```

- [ ] **Step 2: Declare module**

In `src/practice/mod.rs`:

```rust
pub mod metrics;
pub mod types;
pub mod buffer;
pub mod conditioner;
```

- [ ] **Step 3: Verify**

```
cargo test --lib practice::conditioner
```
Expected: 2 passing.

- [ ] **Step 4: Commit**

```bash
git add src/practice/conditioner.rs src/practice/mod.rs
git commit -m "Add InputConditioner skeleton with onset/frame ingestion"
```

---

### Task 10: Per-pitch state transitions (Idle → StartPending → Active → EndPending → Idle)

**Files:**
- Modify: `src/practice/conditioner.rs`

- [ ] **Step 1: Write tests for stable-pitch start (no onset, no cluster)**

Add to `mod tests`:

```rust
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
```

- [ ] **Step 2: Implement the state machine in `ingest`**

Replace the `// (State transitions implemented in next tasks.)` placeholder in `ingest` with:

```rust
let mut events = Vec::new();
let mut present: std::collections::HashSet<u8> = frame.notes.iter().map(|(m, _)| *m).collect();
let cents_by_midi: HashMap<u8, f64> = frame.notes.iter().copied().collect();

// 1. Pitches present in the frame.
for &m in &present.clone() {
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
                // Tier cascade — implemented in Task 11. For now, always StableFiveFrame.
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
```

(`present` is already declared above the match block; remove `let mut` from it since we're not mutating.)

- [ ] **Step 3: Verify tests pass**

```
cargo test --lib practice::conditioner
```
Expected: 4 passing.

- [ ] **Step 4: Commit**

```bash
git add src/practice/conditioner.rs
git commit -m "Implement per-pitch state transitions in InputConditioner"
```

---

### Task 11: Tier cascade (Onset / TransientCluster / StableFiveFrame)

**Files:**
- Modify: `src/practice/conditioner.rs`

- [ ] **Step 1: Write tests**

```rust
#[test]
fn onset_within_window_tags_start_source_onset_and_uses_onset_beat() {
    let mut c = InputConditioner::new(mk_transport());
    // Onset arrives just before the pitch stabilizes.
    let onset = OnsetEvent { beat_position: 0.01, raw_sample_offset: 0, velocity: 0.7 };
    let _ = c.ingest(None, &[onset]);

    for i in 0..5 {
        let f = frame(vec![(60, 0.0)], 0.02 + i as f64 * 0.02);
        let evs = c.ingest(Some(&f), &[]);
        if i == 4 {
            match &evs[0] {
                ConditionerEvent::Started(s) => {
                    assert_eq!(s.start_source, StartSource::Onset);
                    assert!((s.start_beat - 0.01).abs() < 1e-9);
                }
                _ => panic!(),
            }
        }
    }
}

#[test]
fn four_transients_then_stable_pitch_uses_transient_cluster() {
    let mut c = InputConditioner::new(mk_transport());
    // Push 4 transients (each lasting 1 frame, so they hit StartPending → Idle).
    for i in 0..4 {
        let f1 = frame(vec![(50 + i, 0.0)], i as f64 * 0.02);
        let _ = c.ingest(Some(&f1), &[]);
        let f2 = frame(vec![], (i as f64 + 0.5) * 0.02);
        let _ = c.ingest(Some(&f2), &[]);
    }
    // Now the actual note: stable for 5 frames.
    let mut started: Option<TrackedNoteStart> = None;
    for i in 0..5 {
        let f = frame(vec![(60, 0.0)], 0.5 + i as f64 * 0.02);
        for e in c.ingest(Some(&f), &[]) {
            if let ConditionerEvent::Started(s) = e {
                started = Some(s);
            }
        }
    }
    let s = started.expect("expected Started");
    assert_eq!(s.start_source, StartSource::TransientCluster);
    // start_beat should equal the FIRST transient's beat (0.0).
    assert!((s.start_beat - 0.0).abs() < 1e-9);
}
```

- [ ] **Step 2: Replace the `// Tier cascade — implemented in Task 11.` block**

In the `StartPending` confirmation branch from Task 10, replace the inline `StartSource::StableFiveFrame` decision with a call to a new helper. Add at the bottom of `impl InputConditioner`:

```rust
fn run_tier_cascade(
    &mut self,
    midi: u8,
    first_frame_beat: f64,
    first_frame_seq: u64,
) -> (f64, StartSource) {
    // 1. Onset claim.
    if let Some(idx) = self.recent_onsets.iter().position(|o| {
        (o.beat_position - first_frame_beat).abs() < ONSET_CLAIM_WINDOW
    }) {
        let claimed = self.recent_onsets.remove(idx).unwrap();
        return (claimed.beat_position, StartSource::Onset);
    }
    let _ = midi; // suppress unused if we ever add midi-aware filtering

    // 2. Transient cluster.
    let cutoff_seq = first_frame_seq.saturating_sub(CLUSTER_FRAME_WINDOW);
    let cluster: Vec<_> = self.transient_log.iter()
        .filter(|(s, _, _)| *s >= cutoff_seq)
        .copied()
        .collect();
    if cluster.len() >= CLUSTER_MIN_TRANSIENTS {
        let first_beat = cluster[0].1;
        // Drain consumed entries.
        self.transient_log.retain(|(s, _, _)| *s < cutoff_seq);
        return (first_beat, StartSource::TransientCluster);
    }

    // 3. Stable five frame.
    (first_frame_beat, StartSource::StableFiveFrame)
}
```

Then in the `StartPending` confirmation branch, replace:

```rust
// Tier cascade — implemented in Task 11. For now, always StableFiveFrame.
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
PitchState::Active(ActiveBody {
    seq,
    start_beat: first_frame_beat,
    start_source: StartSource::StableFiveFrame,
    cents_sum,
    frame_count: STABLE_FRAMES,
})
```

with:

```rust
let (start_beat, start_source) = self.run_tier_cascade(m, first_frame_beat, first_frame_seq);
let seq = self.next_event_seq;
self.next_event_seq += 1;
let avg = cents_buffer.iter().sum::<f64>() / cents_buffer.len() as f64;
let cents_sum = cents_buffer.iter().sum::<f64>();
events.push(ConditionerEvent::Started(TrackedNoteStart {
    seq,
    midi_note: m,
    start_beat,
    start_source,
    initial_cents: avg,
}));
PitchState::Active(ActiveBody {
    seq,
    start_beat,
    start_source,
    cents_sum,
    frame_count: STABLE_FRAMES,
})
```

- [ ] **Step 3: Verify**

```
cargo test --lib practice::conditioner
```
Expected: 6 passing.

- [ ] **Step 4: Commit**

```bash
git add src/practice/conditioner.rs
git commit -m "Implement 3-tier cascade (Onset, TransientCluster, StableFiveFrame)"
```

---

### Task 12: Vibrato glide cents folding

**Files:**
- Modify: `src/practice/conditioner.rs`

- [ ] **Step 1: Write the test**

```rust
#[test]
fn vibrato_glide_folds_cents_into_outgoing_note() {
    let mut c = InputConditioner::new(mk_transport());
    // C4 stable for 5 frames at avg +30 cents.
    for i in 0..5 {
        let f = frame(vec![(60, 30.0)], i as f64 * 0.02);
        let _ = c.ingest(Some(&f), &[]);
    }
    // Now glide upward: 4 frames where C4 disappears and C#4 appears with progressively higher cents.
    // C#4 cents -50 → -40 → -30 → -20 (rising → folded into C4 as +50, +60, +70, +80).
    for (i, c_sharp_cents) in [(0, -50.0), (1, -40.0), (2, -30.0), (3, -20.0)].iter() {
        let f = frame(vec![(61, *c_sharp_cents)], 5.0 * 0.02 + *i as f64 * 0.02);
        let _ = c.ingest(Some(&f), &[]);
    }
    // 5th frame of C#4 → confirms; C4 must Ended at this pivot.
    let evs = c.ingest(
        Some(&frame(vec![(61, -10.0)], 9.0 * 0.02)),
        &[],
    );
    let mut got_end_c4 = false;
    let mut got_start_csharp = false;
    for e in evs {
        match e {
            ConditionerEvent::Ended(t) if t.midi_note == 60 => {
                // avg_cents should be > 30 because we folded the climbing C#4 cents into it.
                assert!(t.avg_cents > 30.0, "expected folded glide to raise avg, got {}", t.avg_cents);
                got_end_c4 = true;
            }
            ConditionerEvent::Started(t) if t.midi_note == 61 => {
                // C#4's own cents start fresh — they should NOT include the high folded values.
                got_start_csharp = true;
            }
            _ => {}
        }
    }
    assert!(got_end_c4, "expected Ended(C4)");
    assert!(got_start_csharp, "expected Started(C#4)");
}
```

- [ ] **Step 2: Implement glide folding**

In the per-frame state machine (in `ingest`), after the loop over present pitches, add a glide-folding pass. Restructure the pitch-handling logic so that **for each frame**, before the per-pitch transitions:

```rust
// Glide folding: for each EndPending pitch P_old, if any other pitch P_new is
// in StartPending this frame, fold P_new's cents into P_old.
let end_pending_midis: Vec<u8> = self.pitches.iter()
    .filter_map(|(m, s)| matches!(s, PitchState::EndPending { .. }).then_some(*m))
    .collect();
for old_m in &end_pending_midis {
    for &(new_m, new_cents) in &frame.notes {
        if new_m == *old_m { continue; }
        // Only fold if new_m is currently StartPending (or about to be).
        let new_is_start_pending = matches!(
            self.pitches.get(&new_m),
            Some(PitchState::StartPending { .. })
        ) || !self.pitches.contains_key(&new_m);
        if !new_is_start_pending { continue; }
        let folded = (new_m as f64 - *old_m as f64) * 100.0 + new_cents;
        if let Some(PitchState::EndPending { carry, .. }) = self.pitches.get_mut(old_m) {
            carry.cents_sum += folded;
            carry.frame_count += 1;
        }
    }
}
```

Place this block **immediately after** `self.frame_seq += 1;` and the cleanup loops, BEFORE the per-pitch state transitions begin.

Also, the cascade-confirmation branch needs a special case: when `P_new` confirms while some `P_old` is in `EndPending`, emit `Ended(P_old)` with `end_beat = P_new.first_frame_beat` immediately, **before** running the cascade for `P_new`. Modify the StartPending confirmation branch to first scan for any EndPending pitch and finalize it:

```rust
// Inside the StartPending confirmation branch, before run_tier_cascade:
let pivot_beat = first_frame_beat;
let to_end: Vec<u8> = self.pitches.iter()
    .filter_map(|(om, s)| matches!(s, PitchState::EndPending { .. }).then_some(*om))
    .collect();
for old_m in to_end {
    if let Some(PitchState::EndPending { carry, .. }) = self.pitches.remove(&old_m) {
        let avg = if carry.frame_count > 0 { carry.cents_sum / carry.frame_count as f64 } else { 0.0 };
        events.push(ConditionerEvent::Ended(TrackedNoteEnd {
            seq: carry.seq,
            midi_note: old_m,
            end_beat: pivot_beat,
            avg_cents: avg,
            frame_count: carry.frame_count,
        }));
    }
}
```

- [ ] **Step 3: Verify**

```
cargo test --lib practice::conditioner
```
Expected: 7 passing.

- [ ] **Step 4: Commit**

```bash
git add src/practice/conditioner.rs
git commit -m "Implement vibrato glide cents folding in InputConditioner"
```

---

### Task 13: Wire `InputConditioner` into `run_session`, replace inline pitch-stability hysteresis

**Goal:** the inline `match (midi_now, locked_midi) { ... }` block in `run_session` (lines ~699–763) is removed in favor of `conditioner.ingest`. `current_notes` is now populated from `Started` events; the existing `finalize_pending_note` and per-tick cents accumulation are removed.

**Files:**
- Modify: `src/practice/mod.rs`

- [ ] **Step 1: Add conditioner to `SessionState`**

In `SessionState`:

```rust
struct SessionState {
    // ... existing fields ...
    conditioner: Option<crate::practice::conditioner::InputConditioner>,
}
```

In `SessionState::new`, init `conditioner: None`. In `PracticeSession::start`, after creating the buffer:

```rust
s.conditioner = Some(crate::practice::conditioner::InputConditioner::new(self.transport.clone()));
```

- [ ] **Step 2: Replace the inline hysteresis in `run_session`**

In `run_session`, the section from line 670 (`let prev_note_count = s.current_notes.len();`) through line 764 (end of the `match (midi_now, locked_midi)` block) should be replaced by:

```rust
// Build a TunerFrame if the tuner produced a new analysis hop.
let calibrated_tuner_beat = transport.calibrated_beat(tuner_beat);
let tuner_frame_opt: Option<crate::practice::types::TunerFrame> = {
    let last = s.last_tuner_beat;
    if last == Some(calibrated_tuner_beat) {
        None
    } else {
        s.last_tuner_beat = Some(calibrated_tuner_beat);
        let pairs: Vec<(u8, f64)> = note_names.iter().zip(note_cents.iter())
            .filter_map(|(n, c)| note_name_to_midi(n).map(|m| (m, *c)))
            .collect();
        Some(crate::practice::types::TunerFrame {
            notes: pairs,
            tuner_beat: calibrated_tuner_beat,
        })
    }
};

let prev_note_count = s.current_notes.len();
let conditioner_events = if let Some(cond) = s.conditioner.as_mut() {
    cond.ingest(tuner_frame_opt.as_ref(), &new_onsets)
} else {
    Vec::new()
};

for ev in conditioner_events {
    use crate::practice::types::ConditionerEvent;
    match ev {
        ConditionerEvent::Started(t) => {
            s.current_notes.push(NoteEvent {
                beat_position: t.start_beat,
                midi_note: t.midi_note,
                avg_cents: t.initial_cents,
            });
        }
        ConditionerEvent::Ended(t) => {
            // Update the matching NoteEvent's avg_cents.
            if let Some(n) = s.current_notes.iter_mut()
                .rev()
                .find(|n| n.midi_note == t.midi_note)
            {
                n.avg_cents = t.avg_cents;
            }
        }
    }
}
```

Add `last_tuner_beat: Option<f64>` to `SessionState` (init to `None`).

- [ ] **Step 3: Remove dead code**

Delete `finalize_pending_note` from `impl SessionState`. Delete the `last_note_*`, `pending_*`, `cents_sum`, `cents_count` fields from `SessionState`. The existing references in `run_session` to `s.finalize_pending_note()` (in measure-end branch and in match-arm cleanup) become no-ops; remove them.

- [ ] **Step 4: Run all existing tests**

```
cargo test --lib practice
```
Expected: most tests pass; some live-feedback tests may need adjustment because the live `[live]` feedback emission still references `s.current_notes.len() > prev_note_count` — that path still works because we push to `current_notes` from Started events.

If any test fails, fix it inline by adjusting the test data (e.g., feeding 5+ frames of a stable pitch instead of 1).

- [ ] **Step 5: Commit**

```bash
git add src/practice/mod.rs
git commit -m "Wire InputConditioner into run_session, remove inline hysteresis"
```

---

### Phase 2 milestone

```bash
cargo test --lib
cargo build --release
git commit --allow-empty -m "Phase 2 complete: InputConditioner replaces inline pitch hysteresis"
```

---

## Phase 3 — Matcher + ClockManager

### Task 14: Matcher skeleton with overlap-rule and exact-pitch matching

**Files:**
- Create: `src/practice/matcher.rs`
- Modify: `src/practice/mod.rs` (declare `pub mod matcher;`)

- [ ] **Step 1: Write `matcher.rs` with one initial test**

Create `src/practice/matcher.rs`:

```rust
//! Pure resolver: turns a TrackedNoteStart into a MatchOutcome given the buffer.

use crate::practice::buffer::{Candidate, CandidateKind, MeasureBuffer, SlotStatus};
use crate::practice::types::{MatchOutcome, TrackedNoteStart};

pub const MIN_MATCH_SCORE: i32 = 80;
pub const DOUBLED_NOTE_FRESHNESS: f64 = 0.5;

pub fn resolve(
    tracked: &TrackedNoteStart,
    buf: &MeasureBuffer,
    frontier: (usize, usize),
) -> MatchOutcome {
    let cands = buf.candidates(tracked.start_beat, frontier);

    // Rule 1: any Pending slot whose duration window contains tracked.start_beat
    // → match (regardless of pitch). Pick the closest.
    let in_window_pending: Vec<&Candidate> = cands.iter()
        .filter(|c| matches!(c.kind, CandidateKind::InWindow) && c.status == SlotStatus::Pending)
        .collect();
    if !in_window_pending.is_empty() {
        let best = in_window_pending.iter()
            .min_by(|a, b| {
                let da = (tracked.start_beat - a.expected.beat_position).abs();
                let db = (tracked.start_beat - b.expected.beat_position).abs();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        return MatchOutcome::Matched {
            key: best.key,
            timing_err: tracked.start_beat - best.expected.beat_position,
            pitch_correct: tracked.midi_note == best.expected.midi_note,
            upgrade: false,
            skipped_keys: Vec::new(),
        };
    }

    // (Tasks 15–17 add upgrade, doubled, look-ahead, extra cases.)
    MatchOutcome::ExtraNote { during: None }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::practice::buffer::MeasureBuffer;
    use crate::practice::types::{StartSource, TrackedNoteStart};
    use crate::generators::{Instrument, Measure, SynthNote};
    use std::sync::Arc;

    fn measure_with_notes(notes: Vec<(f32, f32, f32)>, start: f64) -> Measure {
        // notes: (start_beat_in_measure, duration, freq)
        Measure {
            notes: notes.iter().map(|&(s, d, f)| SynthNote {
                freq: f,
                start_beat_in_measure: s,
                duration_beats: d,
                velocity: 0.5,
                instrument: Instrument::Piano,
            }).collect(),
            time_signature: (4, 4),
            bpm: 120.0,
            global_start_beat: start,
        }
    }

    fn ts(midi: u8, beat: f64) -> TrackedNoteStart {
        TrackedNoteStart {
            seq: 0,
            midi_note: midi,
            start_beat: beat,
            start_source: StartSource::Onset,
            initial_cents: 0.0,
        }
    }

    #[test]
    fn in_window_correct_pitch_matches_pending() {
        // C4 expected at beat 0, duration 1.
        let measures = Arc::new(vec![measure_with_notes(vec![(0.0, 1.0, 261.626)], 0.0)]);
        let buf = MeasureBuffer::new(measures, 0, 0);
        let out = resolve(&ts(60, 0.05), &buf, (0, 0));
        match out {
            MatchOutcome::Matched { key, pitch_correct, .. } => {
                assert_eq!(key, (0, 0));
                assert!(pitch_correct);
            }
            other => panic!("expected Matched, got {other:?}"),
        }
    }

    #[test]
    fn in_window_wrong_pitch_matches_pitch_correct_false() {
        let measures = Arc::new(vec![measure_with_notes(vec![(0.0, 1.0, 261.626)], 0.0)]);
        let buf = MeasureBuffer::new(measures, 0, 0);
        let out = resolve(&ts(62, 0.05), &buf, (0, 0));
        match out {
            MatchOutcome::Matched { key, pitch_correct, .. } => {
                assert_eq!(key, (0, 0));
                assert!(!pitch_correct);
            }
            other => panic!("expected Matched, got {other:?}"),
        }
    }
}
```

- [ ] **Step 2: Declare module**

In `mod.rs`:

```rust
pub mod metrics;
pub mod types;
pub mod buffer;
pub mod conditioner;
pub mod matcher;
```

- [ ] **Step 3: Verify**

```
cargo test --lib practice::matcher
```
Expected: 2 passing.

- [ ] **Step 4: Commit**

```bash
git add src/practice/matcher.rs src/practice/mod.rs
git commit -m "Add matcher::resolve with in-window Pending rule"
```

---

### Task 15: Upgrade case — `Matched(false)` → `Matched(true)`

**Files:**
- Modify: `src/practice/matcher.rs`, `src/practice/buffer.rs`

- [ ] **Step 1: Write the test**

Add to `matcher.rs` `mod tests`:

```rust
#[test]
fn upgrade_promotes_matched_false_to_true() {
    let measures = Arc::new(vec![measure_with_notes(vec![(0.0, 1.0, 261.626)], 0.0)]);
    let mut buf = MeasureBuffer::new(measures, 0, 0);
    // Slot is currently Matched(false).
    buf.record_match((0, 0), &ts(62, 0.05), false);

    let out = resolve(&ts(60, 0.10), &buf, (0, 0));
    match out {
        MatchOutcome::Matched { key, pitch_correct, upgrade, .. } => {
            assert_eq!(key, (0, 0));
            assert!(pitch_correct);
            assert!(upgrade);
        }
        other => panic!("expected Matched (upgrade), got {other:?}"),
    }
}
```

- [ ] **Step 2: Implement the upgrade rule in `resolve`**

After the "in_window_pending" block but before the fallthrough, add:

```rust
// Rule 2: any Matched(false) slot whose duration window contains the beat
// AND pitch is exact → upgrade.
let upgrade_target = cands.iter().find(|c| {
    matches!(c.kind, CandidateKind::InWindow)
        && matches!(c.status, SlotStatus::Matched { pitch_correct: false })
        && tracked.midi_note == c.expected.midi_note
});
if let Some(c) = upgrade_target {
    return MatchOutcome::Matched {
        key: c.key,
        timing_err: tracked.start_beat - c.expected.beat_position,
        pitch_correct: true,
        upgrade: true,
        skipped_keys: Vec::new(),
    };
}
```

Note: with this change, the `in_window_pending` rule must run first (Pending takes precedence over Matched(false) for upgrade detection). The current order in the file is correct.

- [ ] **Step 3: Verify**

```
cargo test --lib practice::matcher
```
Expected: 3 passing.

- [ ] **Step 4: Commit**

```bash
git add src/practice/matcher.rs
git commit -m "Add upgrade rule (Matched(false) → Matched(true)) to matcher"
```

---

### Task 16: DoubledNote rule with freshness cap

**Files:**
- Modify: `src/practice/matcher.rs`

- [ ] **Step 1: Write tests**

```rust
#[test]
fn doubled_note_within_freshness_emits_doubled() {
    let measures = Arc::new(vec![measure_with_notes(vec![(0.0, 1.0, 261.626)], 0.0)]);
    let mut buf = MeasureBuffer::new(measures, 0, 0);
    buf.record_match((0, 0), &ts(60, 0.0), true);
    let out = resolve(&ts(60, 0.2), &buf, (0, 0));
    assert_eq!(out, MatchOutcome::DoubledNote { key: (0, 0) });
}

#[test]
fn doubled_note_beyond_freshness_emits_extra() {
    let measures = Arc::new(vec![measure_with_notes(vec![(0.0, 4.0, 261.626)], 0.0)]);
    let mut buf = MeasureBuffer::new(measures, 0, 0);
    buf.record_match((0, 0), &ts(60, 0.0), true);
    // Beat 0.6 is > DOUBLED_NOTE_FRESHNESS (0.5) past matched_start_beat.
    let out = resolve(&ts(60, 0.6), &buf, (0, 0));
    matches!(out, MatchOutcome::ExtraNote { .. }); // not Doubled
}
```

- [ ] **Step 2: Implement**

After Rule 2 in `resolve`, add Rule 3:

```rust
// Rule 3: Matched(true) slot, exact pitch, within freshness → DoubledNote.
let doubled_target = cands.iter().find(|c| {
    matches!(c.kind, CandidateKind::InWindow)
        && matches!(c.status, SlotStatus::Matched { pitch_correct: true })
        && tracked.midi_note == c.expected.midi_note
        && buf.slot(c.key).and_then(|s| s.matched_start_beat).map_or(false, |msb| {
            tracked.start_beat - msb <= DOUBLED_NOTE_FRESHNESS
        })
});
if let Some(c) = doubled_target {
    return MatchOutcome::DoubledNote { key: c.key };
}
```

- [ ] **Step 3: Verify**

```
cargo test --lib practice::matcher
```
Expected: 5 passing.

- [ ] **Step 4: Commit**

```bash
git add src/practice/matcher.rs
git commit -m "Add DoubledNote rule with freshness cap to matcher"
```

---

### Task 17: Look-ahead matching (skipped notes) and final ExtraNote fallthrough

**Files:**
- Modify: `src/practice/matcher.rs`

- [ ] **Step 1: Write tests**

```rust
#[test]
fn lookahead_matches_skipped_frontier() {
    // Notes at 0, 1, 2 (each 1 beat). Student plays at beat 1.05 with pitch
    // matching note 2 (E4 = midi 64). Frontier is (0, 1). Note 1 (D4) gets
    // skipped, note 2 matched.
    let measures = Arc::new(vec![measure_with_notes(vec![
        (0.0, 1.0, 261.626),    // C4
        (1.0, 1.0, 293.665),    // D4
        (2.0, 1.0, 329.628),    // E4
    ], 0.0)]);
    let mut buf = MeasureBuffer::new(measures, 0, 0);
    buf.record_match((0, 0), &ts(60, 0.0), true);

    // Tracked: pitch=64 (E4) at beat 1.05 (still in D4's window normally, but
    // pitch overrides that — actually no: Rule 1 is pitch-agnostic. So beat
    // 1.05 with E4 still matches D4 with pitch_correct=false. Use beat=2.05
    // instead so we cleanly hit lookahead).
    let out = resolve(&ts(64, 2.05), &buf, (0, 1));
    match out {
        MatchOutcome::Matched { key, skipped_keys, pitch_correct, .. } => {
            assert_eq!(key, (0, 2));
            assert_eq!(skipped_keys, vec![(0, 1)]);
            assert!(pitch_correct);
        }
        other => panic!("expected Matched lookahead, got {other:?}"),
    }
}

#[test]
fn unmatched_note_in_rest_emits_extra_note_with_during_none() {
    let measures = Arc::new(vec![measure_with_notes(vec![(0.0, 0.5, 261.626)], 0.0)]);
    let buf = MeasureBuffer::new(measures, 0, 0);
    // Beat 2.0 — well past the only note's duration window.
    let out = resolve(&ts(60, 2.0), &buf, (0, 0));
    assert_eq!(out, MatchOutcome::ExtraNote { during: None });
}
```

- [ ] **Step 2: Implement scoring + extra note fallthrough**

Replace the final `MatchOutcome::ExtraNote { during: None }` line with:

```rust
// Rule 4: score Lookahead and Lookbehind candidates.
fn pitch_score(played: u8, expected: u8) -> i32 {
    let d = (played as i32 - expected as i32).abs();
    match d { 0 => 100, 1 => 30, 2 => 10, _ => 0 }
}
fn timing_score(beat: f64, exp: &crate::practice::metrics::ExpectedNote) -> i32 {
    if beat >= exp.beat_position && beat < exp.beat_position + exp.duration_beats {
        50
    } else {
        let err = (beat - exp.beat_position).abs();
        ((50.0 - 100.0 * err) as i32).max(0)
    }
}

let mut best: Option<(&Candidate, i32)> = None;
for c in &cands {
    if !matches!(c.status, SlotStatus::Pending) { continue; }
    let kind_penalty = match c.kind {
        CandidateKind::InWindow => 0,
        CandidateKind::Lookahead(1) => -10,
        CandidateKind::Lookahead(2) => -25,
        CandidateKind::Lookbehind(1) => -15,
        _ => -50,
    };
    let score = pitch_score(tracked.midi_note, c.expected.midi_note)
              + timing_score(tracked.start_beat, &c.expected)
              + kind_penalty;
    if score >= MIN_MATCH_SCORE
        && tracked.midi_note == c.expected.midi_note
        && best.map_or(true, |(_, bs)| score > bs)
    {
        best = Some((c, score));
    }
}

if let Some((c, _)) = best {
    let mut skipped = Vec::new();
    let mut walker = frontier;
    while walker != c.key {
        if let Some(s) = buf.slot(walker) {
            if matches!(s.status, SlotStatus::Pending) {
                skipped.push(walker);
            }
        }
        walker = step_forward(buf, walker);
    }
    return MatchOutcome::Matched {
        key: c.key,
        timing_err: tracked.start_beat - c.expected.beat_position,
        pitch_correct: true,
        upgrade: false,
        skipped_keys: skipped,
    };
}

// Rule 5: extra note. `during` = any in-window slot (Pending or Matched).
let during = cands.iter()
    .find(|c| matches!(c.kind, CandidateKind::InWindow))
    .map(|c| c.key);
MatchOutcome::ExtraNote { during }
```

Add a `step_forward` helper at module scope:

```rust
fn step_forward(buf: &MeasureBuffer, key: (usize, usize)) -> (usize, usize) {
    // We don't need exact cross-measure logic for skipped — just pick the
    // closest slot one note ahead by sequential numbering within the measure,
    // then walk into the next measure if needed.
    let next = (key.0, key.1 + 1);
    if buf.slot(next).is_some() { next }
    else { (key.0 + 1, 0) }
}
```

- [ ] **Step 3: Verify**

```
cargo test --lib practice::matcher
```
Expected: 7 passing.

- [ ] **Step 4: Commit**

```bash
git add src/practice/matcher.rs
git commit -m "Add lookahead matching and ExtraNote fallthrough to matcher"
```

---

### Task 18: `ClockManager` skeleton + `t_stu_bpm`

**Files:**
- Create: `src/practice/clock.rs`
- Modify: `src/practice/mod.rs`

- [ ] **Step 1: Write skeleton + initial test**

Create `src/practice/clock.rs`:

```rust
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
```

- [ ] **Step 2: Declare module**

```rust
pub mod clock;
```

- [ ] **Step 3: Verify**

```
cargo test --lib practice::clock
```

- [ ] **Step 4: Commit**

```bash
git add src/practice/clock.rs src/practice/mod.rs
git commit -m "Add ClockManager skeleton"
```

---

### Task 19: T_stu EWMA update on match + streak tracking

**Files:**
- Modify: `src/practice/clock.rs`

- [ ] **Step 1: Write tests**

```rust
fn matched(beat: f64, key: (usize, usize)) -> MatchOutcome {
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
```

- [ ] **Step 2: Implement `on_match`**

Add to `impl ClockManager`:

```rust
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
```

- [ ] **Step 3: Verify**

```
cargo test --lib practice::clock
```
Expected: 3 passing.

- [ ] **Step 4: Commit**

```bash
git add src/practice/clock.rs
git commit -m "Implement T_stu EWMA + streak-gated SetBpm in ClockManager"
```

---

### Task 20: Unified seek-direction rule + 15% threshold

**Files:**
- Modify: `src/practice/clock.rs`

- [ ] **Step 1: Write tests**

```rust
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
```

- [ ] **Step 2: Implement seek decision in `on_match`**

In `on_match`, replace `let _ = timing_err;` with:

```rust
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
```

- [ ] **Step 3: Verify**

```
cargo test --lib practice::clock
```
Expected: 6 passing.

- [ ] **Step 4: Commit**

```bash
git add src/practice/clock.rs
git commit -m "Implement unified seek rule with 15% threshold in ClockManager"
```

---

### Task 21: Stop trigger via `on_tick`

**Files:**
- Modify: `src/practice/clock.rs`

- [ ] **Step 1: Write the test**

```rust
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
```

- [ ] **Step 2: Implement `on_tick`**

First add a public getter to `MeasureBuffer`:

```rust
// in buffer.rs (in `impl MeasureBuffer`)
pub fn measures(&self) -> &[Measure] { &self.measures }
```

Then in `clock.rs`:

```rust
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
```

- [ ] **Step 3: Verify**

```
cargo test --lib practice::clock
cargo test --lib practice::buffer
```
Expected: all passing.

- [ ] **Step 4: Commit**

```bash
git add src/practice/clock.rs src/practice/buffer.rs
git commit -m "Implement stop trigger in ClockManager::on_tick"
```

---

### Task 22: `on_doubled` and `on_extra`

**Files:**
- Modify: `src/practice/clock.rs`

- [ ] **Step 1: Write tests**

```rust
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
```

- [ ] **Step 2: Implement**

```rust
pub fn on_doubled(&mut self, slot: &NoteSlot, mode: PracticeMode) -> Vec<ClockAction> {
    if mode == PracticeMode::Performance { return Vec::new(); }
    let Some(matched_beat) = slot.matched_start_beat else { return Vec::new() };
    vec![
        ClockAction::SeekToBeat(matched_beat + self.cfg.seek_landing_epsilon),
        ClockAction::Play,
    ]
}

pub fn on_extra(&mut self) -> Vec<ClockAction> { Vec::new() }
```

- [ ] **Step 3: Verify**

```
cargo test --lib practice::clock
```

- [ ] **Step 4: Commit**

```bash
git add src/practice/clock.rs
git commit -m "Implement on_doubled and on_extra in ClockManager"
```

---

### Phase 3 milestone

```bash
cargo test --lib
git commit --allow-empty -m "Phase 3 complete: matcher and clock manager unit-tested"
```

---

## Phase 4 — ModeController + integration

### Task 23: `ModeController` skeleton + `tick_for_test` harness

**Files:**
- Create: `src/practice/mode.rs`
- Modify: `src/practice/mod.rs`

- [ ] **Step 1: Write skeleton**

Create `src/practice/mode.rs`:

```rust
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
```

- [ ] **Step 2: Declare module**

```rust
pub mod mode;
```

- [ ] **Step 3: Build**

```
cargo build --lib
```

- [ ] **Step 4: Commit**

```bash
git add src/practice/mode.rs src/practice/mod.rs
git commit -m "Add ModeController skeleton with per-tick contract"
```

---

### Task 24: Implement `ModeController::tick` algorithm

**Files:**
- Modify: `src/practice/mode.rs`

- [ ] **Step 1: Implement `tick`**

Replace the placeholder `tick` body with:

```rust
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
```

Add helpers in `mode.rs`:

```rust
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
```

- [ ] **Step 2: Build**

```
cargo build --lib
```
Expected: success (compile errors fixed inline if any).

- [ ] **Step 3: Commit**

```bash
git add src/practice/mode.rs
git commit -m "Implement ModeController::tick orchestration"
```

---

### Task 25: ModeController integration tests

**Files:**
- Modify: `src/practice/mode.rs`

- [ ] **Step 1: Write integration tests**

Add to `mod tests`:

```rust
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
```

- [ ] **Step 2: Verify**

```
cargo test --lib practice::mode
```
Expected: 2 passing.

- [ ] **Step 3: Commit**

```bash
git add src/practice/mode.rs
git commit -m "Add ModeController integration tests"
```

---

### Task 26: Add `Tempo`, `HeldTooLong`, `HeldTooShort` to `MusicError`

**Files:**
- Modify: `src/practice/mod.rs` (the `MusicError` enum)

- [ ] **Step 1: Add variants**

In `src/practice/mod.rs`, find the `MusicError` enum (around line 77) and add:

```rust
#[derive(serde::Serialize, Clone, PartialEq, Debug)]
pub enum MusicError {
    Timing,
    WrongNote,
    UnexpectedNote,
    MissingNote,
    Intonation,
    Dynamics,
    Tempo,           // NEW
    HeldTooLong,     // NEW
    HeldTooShort,    // NEW
    None,
}
```

- [ ] **Step 2: Verify build**

```
cargo build --lib
```
Expected: build OK.

- [ ] **Step 3: Commit**

```bash
git add src/practice/mod.rs
git commit -m "Add Tempo, HeldTooLong, HeldTooShort variants to MusicError"
```

---

### Task 27: Live `SendInfo` emission per `ConditionerEvent`

**Files:**
- Modify: `src/practice/mode.rs`, `src/practice/mod.rs`

- [ ] **Step 1: Add a feedback callback hook**

Modify `ModeController` to accept a feedback sink. Add to `ModeController`:

```rust
pub feedback: Vec<crate::practice::SendInfo>,
```

Init to `Vec::new()` in `new`.

- [ ] **Step 2: Emit SendInfo from `handle_outcome`**

In `handle_outcome`, after the `outcome` match, before applying clock actions, emit `SendInfo`. Replace the `actions = match outcome { ... }` section (preserving its return-of-actions behavior) with:

```rust
let actions = match outcome {
    MatchOutcome::Matched { key, pitch_correct, upgrade, skipped_keys, timing_err } => {
        for k in skipped_keys {
            self.buffer.mark_missed(*k);
            self.feedback.push(missing_note_send_info(*k, &self.buffer));
        }
        if *upgrade { self.buffer.upgrade_match(*key, t); } else { self.buffer.record_match(*key, t, *pitch_correct); }
        self.frontier = step_forward(&self.buffer, *key);
        let exp = expected_for(&self.buffer, *key);
        self.match_log.insert(t.seq, MatchedSnapshot {
            measure_idx: key.0,
            note_idx_in_measure_data: note_idx,
            expected_duration: exp.duration_beats,
            expected_midi: exp.midi_note,
        });
        // Live feedback for the match.
        let prim = if !*pitch_correct {
            send_info(*key, crate::practice::MusicError::WrongNote, &exp, t)
        } else {
            send_info(*key, crate::practice::MusicError::None, &exp, t)
        };
        self.feedback.push(prim);
        let timing_threshold = exp.duration_beats * self.clock.cfg().seek_threshold_pct
            * mode_tol_scale(self.mode);
        if timing_err.abs() > timing_threshold {
            self.feedback.push(timing_send_info(*key, &exp, t, *timing_err));
        }
        self.clock.on_match(outcome, &exp, transport_beat, self.mode)
    }
    MatchOutcome::DoubledNote { key } => {
        self.in_progress_doubled_seqs.entry(mi).or_default().push(t.seq);
        let exp = expected_for(&self.buffer, *key);
        self.feedback.push(send_info(*key, crate::practice::MusicError::Tempo, &exp, t));
        let slot = self.buffer.slot(*key).cloned();
        slot.map(|s| self.clock.on_doubled(&s, self.mode)).unwrap_or_default()
    }
    MatchOutcome::ExtraNote { during } => {
        self.feedback.push(extra_note_send_info(*during, t, &self.buffer));
        self.clock.on_extra()
    }
};
```

Add helpers at the bottom of `mode.rs`:

```rust
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
        expected: format!("midi={} beat={:.2}", exp.midi_note, exp.beat_position),
        received: format!("midi={} beat={:.3}", t.midi_note, t.start_beat),
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
        expected: format!("beat {:.2}", exp.beat_position),
        received: format!("beat {:.2}", t.start_beat),
    }
}

fn missing_note_send_info(key: (usize, usize), buf: &MeasureBuffer) -> crate::practice::SendInfo {
    let exp = expected_for(buf, key);
    crate::practice::SendInfo {
        measure: key.0 as u32,
        note_index: key.1,
        error_type: crate::practice::MusicError::MissingNote,
        intensity: 1.0,
        expected: format!("midi={} beat={:.2}", exp.midi_note, exp.beat_position),
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
            (key.0 as u32, key.1, format!("midi={} (extra during held)", exp.midi_note))
        }
        None => (0, 0, "silence".into()),
    };
    crate::practice::SendInfo {
        measure,
        note_index,
        error_type: crate::practice::MusicError::UnexpectedNote,
        intensity: 0.5,
        expected: expected_str,
        received: format!("midi={} beat={:.3}", t.midi_note, t.start_beat),
    }
}
```

- [ ] **Step 3: Emit hold + final-intonation feedback in `handle_ended`**

Replace `handle_ended` body with:

```rust
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
            const HOLD_TOLERANCE_PCT: f64 = 0.25;
            if actual_duration > snap.expected_duration * (1.0 + HOLD_TOLERANCE_PCT) {
                self.feedback.push(crate::practice::SendInfo {
                    measure: mi as u32,
                    note_index: snap.note_idx_in_measure_data,
                    error_type: crate::practice::MusicError::HeldTooLong,
                    intensity: 0.6,
                    expected: format!("dur≈{:.2}", snap.expected_duration),
                    received: format!("dur={:.2}", actual_duration),
                });
            } else if actual_duration < snap.expected_duration * (1.0 - HOLD_TOLERANCE_PCT) {
                self.feedback.push(crate::practice::SendInfo {
                    measure: mi as u32,
                    note_index: snap.note_idx_in_measure_data,
                    error_type: crate::practice::MusicError::HeldTooShort,
                    intensity: 0.6,
                    expected: format!("dur≈{:.2}", snap.expected_duration),
                    received: format!("dur={:.2}", actual_duration),
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
                    expected: format!("midi={}", snap.expected_midi),
                    received: format!("midi={} {:+.0}c", t.midi_note, t.avg_cents),
                });
            }
        }
    }
}
```

- [ ] **Step 4: Verify**

```
cargo test --lib practice
```
Expected: existing tests still green; new behavior compiles.

- [ ] **Step 5: Commit**

```bash
git add src/practice/mode.rs
git commit -m "Emit live SendInfo per ConditionerEvent and Ended"
```

---

### Task 28: Replace `run_session` with `ModeController`-driven loop

**Files:**
- Modify: `src/practice/mod.rs`

- [ ] **Step 1: Replace `run_session` body**

In `src/practice/mod.rs`, replace the entire body of `run_session` (function starting around line 530) with:

```rust
fn run_session(
    transport: Arc<MusicalTransport>,
    tuner_output: Arc<parking_lot::RwLock<crate::analysis::tuner::TunerOutput>>,
    onset: Arc<OnsetDetection>,
    dynamics_output: Arc<parking_lot::RwLock<crate::audio_io::dynamics::DynamicsOutput>>,
    measures: Arc<Vec<Measure>>,
    state: Arc<Mutex<SessionState>>,
    feedback: Arc<Mutex<Vec<SendInfo>>>,
    running: Arc<AtomicBool>,
    ability_level: AbilityLevel,
    mode: crate::practice::types::PracticeMode,
) {
    use crate::practice::buffer::MeasureBuffer;
    use crate::practice::clock::{ClockConfig, ClockManager};
    use crate::practice::conditioner::InputConditioner;
    use crate::practice::mode::{ModeController, TickInputs};
    use crate::practice::types::TunerFrame;

    let (practice_start, practice_end, in_countoff_initial, first_measure_beat) = {
        let s = state.lock().unwrap();
        (s.practice_start, s.practice_end, s.in_countoff, s.first_measure_beat)
    };

    let buffer = MeasureBuffer::new(measures.clone(), practice_start, practice_end);
    let conditioner = InputConditioner::new(transport.clone());
    let clock = ClockManager::new(transport.clone(), ClockConfig::default(), transport.get_bpm());
    let mut mc = ModeController::new(
        mode, transport.clone(), conditioner, buffer, clock, practice_start,
    );

    let mut last_tuner_beat: Option<f64> = None;
    let mut in_countoff = in_countoff_initial;

    while running.load(Ordering::Relaxed) {
        let beat = transport.get_accumulated_beats();

        if in_countoff {
            if beat >= first_measure_beat {
                in_countoff = false;
                if let Ok(mut s) = state.lock() { s.in_countoff = false; }
            } else {
                thread::sleep(Duration::from_millis(10));
                continue;
            }
        }

        let new_onsets: Vec<crate::audio_io::timing::OnsetEvent> = onset.drain_onset_events();
        let (note_names, note_cents, raw_tuner_beat) = {
            let out = tuner_output.read();
            (out.notes.clone(), out.accuracies.iter().map(|&c| c as f64).collect::<Vec<_>>(), out.beat_position)
        };
        let calibrated_tuner_beat = transport.calibrated_beat(raw_tuner_beat);
        let tuner_frame: Option<TunerFrame> = if last_tuner_beat == Some(calibrated_tuner_beat) {
            None
        } else {
            last_tuner_beat = Some(calibrated_tuner_beat);
            let pairs: Vec<(u8, f64)> = note_names.iter().zip(note_cents.iter())
                .filter_map(|(n, c)| note_name_to_midi(n).map(|m| (m, *c)))
                .collect();
            Some(TunerFrame { notes: pairs, tuner_beat: calibrated_tuner_beat })
        };
        let dynamic_level = dynamics_output.read().level;

        let outputs = mc.tick(TickInputs {
            transport_beat: beat,
            tuner_frame: tuner_frame.as_ref(),
            new_onsets: &new_onsets,
            dynamic_level,
        });

        // Drain feedback to the shared queue.
        if !mc.feedback.is_empty() {
            if let Ok(mut g) = feedback.lock() {
                g.extend(std::mem::take(&mut mc.feedback));
            }
        }

        // Push aged-out measures into completed_measures and run end-of-measure feedback.
        if !outputs.aged_measures.is_empty() {
            if let Ok(mut s) = state.lock() {
                for m in &outputs.aged_measures {
                    let eom = generate_measure_feedback(m, ability_level);
                    if let Ok(mut g) = feedback.lock() {
                        g.extend(eom);
                    }
                }
                s.current_measure_idx = mc.buffer.current_idx();
                s.completed_measures.extend(outputs.aged_measures);
            }
        }

        // Done condition.
        if mc.frontier.0 > practice_end {
            running.store(false, Ordering::Relaxed);
            break;
        }

        thread::sleep(Duration::from_millis(10));
    }
}
```

- [ ] **Step 2: Add `mode` parameter to `PracticeSession::new`**

Find `PracticeSession::new` (around line 257) and add `mode: PracticeMode`:

```rust
pub(crate) fn new(
    transport: Arc<MusicalTransport>,
    tuner: Arc<Tuner>,
    onset: Arc<OnsetDetection>,
    dynamics_output: Arc<parking_lot::RwLock<crate::audio_io::dynamics::DynamicsOutput>>,
    midi_path: &str,
    instrument: &str,
    countoff_beats: u32,
    mode: crate::practice::types::PracticeMode,
    ability_level: AbilityLevel,
    bpm: f32,
) -> Result<Self, AudioEngineError> {
```

Remove the `wait_for_onset: bool` parameter. Add `mode` as a field on `PracticeSession`. Pass `mode` into the spawned thread closure.

In `start`, when calling `run_session`, pass `self.mode`.

- [ ] **Step 3: Remove dead state**

Delete from `SessionState`:
- `wait_for_onset: bool`
- `next_expected_beat`, `onset_received_this_measure`, `note_refractory_until`
- `pending_midi`, `pending_count`, `pending_first_beat`
- `current_onsets`, `current_notes`, `current_dynamics` (now in ModeController)
- `last_note_*`, `cents_*`, `last_dynamic`
- `countoff_offsets`, `onset_beat_offset`
- `last_tuner_beat`

Update `SessionState::new` and any test fixtures accordingly.

Delete the entire `generate_measure_feedback` function only if no longer used; but it IS used at end-of-measure for the batch path, so KEEP it. Just delete the dead `finalize_pending_note` helper.

- [ ] **Step 4: Build + run all tests**

```
cargo build --lib
cargo test --lib practice
```

Expected: build OK; existing tests for `freq_to_midi`, `note_name_to_midi`, `midi_to_note_name`, `velocity_to_dynamic`, `closest_onset`, `dynamic_at`, `dynamic_to_i32`, AbilityLevel scaling, generate_measure_feedback variants — all still pass.

The few existing run_session-flavored tests may need to be adapted to feed through `ModeController::tick` directly. Update them inline.

- [ ] **Step 5: Commit**

```bash
git add src/practice/mod.rs
git commit -m "Replace run_session with ModeController-driven polling loop"
```

---

### Task 29: Update FFI in `lib.rs` to take `mode: String` parameter

**Files:**
- Modify: `src/lib.rs`

- [ ] **Step 1: Replace `wait_for_onset` with `mode` in `create_practice_session`**

Find `create_practice_session` (line 692). Replace `wait_for_onset: bool,` with `mode: String,`. Replace the call to `practice_mod::PracticeSession::new` to pass `mode_enum` instead of `wait_for_onset`:

```rust
let mode_enum = match practice_mod::types::PracticeMode::from_str(&mode) {
    Some(m) => m,
    None => return Err(AudioEngineError::Internal {
        msg: format!("Unknown practice mode '{mode}'. Expected: FollowAlong, Performance, Rubato"),
    }),
};

let inner = match practice_mod::PracticeSession::new(
    transport, tuner, onset, dynamics,
    &midi_path, &instrument,
    countoff_beats,
    mode_enum,
    level, bpm,
) {
    // ... same as before
};
```

- [ ] **Step 2: Update doc comments and any UniFFI bindings**

If there's a UniFFI .udl file, update the function signature there as well. Run `cargo build --release` to verify.

- [ ] **Step 3: Verify**

```
cargo build --release
cargo test --lib
```

- [ ] **Step 4: Commit**

```bash
git add src/lib.rs
git commit -m "Replace wait_for_onset with mode parameter in PracticeSession FFI"
```

---

### Task 30: Extend `Metrics` with `tempo_err_*` and `hold_err_*`

**Files:**
- Modify: `src/practice/metrics.rs`

- [ ] **Step 1: Add fields**

In the `Metrics` struct (line 62), add:

```rust
pub tempo_err_count: u64,
pub hold_err_count: (u64, u64),
pub tempo_err_measures: Vec<u32>,
pub hold_err_measures:  Vec<u32>,
```

- [ ] **Step 2: Compute them in `Metrics::compute`**

Add helper functions at the bottom of `impl Metrics`:

```rust
fn calc_tempo_err_count(measures: &[MeasureData]) -> u64 {
    measures.iter().map(|m| m.doubled_note_seqs.len() as u64).sum()
}

fn calc_hold_err_count(measures: &[MeasureData]) -> (u64, u64) {
    const HOLD_TOLERANCE_PCT: f64 = 0.25;
    let mut long = 0u64;
    let mut short = 0u64;
    for m in measures {
        for (i, dur_opt) in m.note_durations.iter().enumerate() {
            let Some(actual) = dur_opt else { continue };
            let Some(note) = m.notes.get(i) else { continue };
            // Find the closest expected note to compare durations.
            let exp_dur_opt = m.expected_notes.iter()
                .find(|e| (e.beat_position - note.beat_position).abs() < NOTE_MATCH_WINDOW
                          && e.midi_note == note.midi_note)
                .map(|e| e.duration_beats);
            let Some(exp_dur) = exp_dur_opt else { continue };
            if *actual > exp_dur * (1.0 + HOLD_TOLERANCE_PCT) { long += 1; }
            else if *actual < exp_dur * (1.0 - HOLD_TOLERANCE_PCT) { short += 1; }
        }
    }
    (long, short)
}

fn calc_tempo_err_measures(measures: &[MeasureData]) -> Vec<u32> {
    measures.iter()
        .filter(|m| !m.doubled_note_seqs.is_empty())
        .map(|m| m.measure_index)
        .collect()
}

fn calc_hold_err_measures(measures: &[MeasureData]) -> Vec<u32> {
    const HOLD_TOLERANCE_PCT: f64 = 0.25;
    measures.iter().filter(|m| {
        m.note_durations.iter().enumerate().any(|(i, dur)| {
            let Some(d) = dur else { return false };
            let Some(note) = m.notes.get(i) else { return false };
            let exp_dur = m.expected_notes.iter()
                .find(|e| (e.beat_position - note.beat_position).abs() < NOTE_MATCH_WINDOW
                          && e.midi_note == note.midi_note)
                .map(|e| e.duration_beats);
            let Some(ed) = exp_dur else { return false };
            *d > ed * (1.0 + HOLD_TOLERANCE_PCT) || *d < ed * (1.0 - HOLD_TOLERANCE_PCT)
        })
    }).map(|m| m.measure_index).collect()
}
```

In `Metrics::compute`, before constructing `Self { ... }`, compute:

```rust
let tempo_err_count = Self::calc_tempo_err_count(measures);
let hold_err_count = Self::calc_hold_err_count(measures);
let tempo_err_measures = Self::calc_tempo_err_measures(measures);
let hold_err_measures = Self::calc_hold_err_measures(measures);
```

And add the new fields to the struct literal.

- [ ] **Step 3: Add tests**

```rust
#[test]
fn tempo_err_count_sums_doubled_seqs() {
    let mut m = MeasureData {
        measure_index: 0,
        onsets: vec![],
        notes: vec![],
        dynamics: vec![],
        expected_notes: vec![],
        note_durations: vec![],
        doubled_note_seqs: vec![1, 2, 3],
    };
    let count = Metrics::calc_tempo_err_count(&[m.clone()]);
    assert_eq!(count, 3);
}

#[test]
fn hold_err_count_flags_too_long_and_too_short() {
    let m = MeasureData {
        measure_index: 0,
        onsets: vec![],
        notes: vec![
            note_event(0.0, 60, 0.0),
            note_event(2.0, 64, 0.0),
        ],
        dynamics: vec![],
        expected_notes: vec![
            expected(0.0, 60, 1.0),
            expected(2.0, 64, 1.0),
        ],
        note_durations: vec![Some(1.5), Some(0.5)],   // long, short
        doubled_note_seqs: vec![],
    };
    let (long, short) = Metrics::calc_hold_err_count(&[m]);
    assert_eq!((long, short), (1, 1));
}
```

(Add `#[derive(Clone)]` to `MeasureData` if not already present.)

- [ ] **Step 4: Verify**

```
cargo test --lib practice::metrics
```

- [ ] **Step 5: Commit**

```bash
git add src/practice/metrics.rs
git commit -m "Add tempo_err and hold_err fields to Metrics"
```

---

### Phase 4 milestone

```bash
cargo test --lib
cargo build --release
git commit --allow-empty -m "Phase 4 complete: ModeController + integration done"
```

---

## Final verification

- [ ] **Build the full release binary**

```
cargo build --release
```
Expected: success.

- [ ] **Run the full test suite**

```
cargo test
```
Expected: all green.

- [ ] **Spot-check live integration with a MIDI file**

Run the existing demo or a small smoke check:

```
cargo run --release --example practice_smoke    # if such an example exists; otherwise skip
```

If a smoke example doesn't exist, verify the test_output.wav generation from existing `cargo test` calls is unchanged.

- [ ] **Optional: hand-test each mode with the front-end**

This requires the FE rebuild — out of scope for this plan, but the FFI `create_practice_session` now takes `mode: String`. Verify with the front-end maintainer that the React Native side passes `"FollowAlong"`, `"Performance"`, or `"Rubato"`.

---

## Self-review checklist

- All 6 sections of the spec mapped to tasks: §1 (Tasks 2, 28, 29), §2 (Tasks 9–13), §3 (Tasks 4–7, 14–17), §4 (Tasks 18–22), §5 (Tasks 23–25, 28), §6 (Tasks 26, 27, 30). Phase milestones bracket the spec's four-phase plan.
- New `MusicError` variants and metrics fields included.
- Public API of `PracticeSession::new` updated; FFI signature in `lib.rs` updated.
- `MusicalTransport::calibrated_beat` added (Task 1).
- Vibrato glide rule covered (Task 12).
- Stop-trigger and unified seek-direction rule covered (Tasks 20, 21).
- Mode filter applied via `apply_action` (Task 24).
- Hold-too-long/short detection in Ended events (Task 27).
- End-of-measure batch feedback retained (Task 28 calls `generate_measure_feedback`).
