# Practice Engine Refactor — Design

**Status:** Approved
**Date:** 2026-05-09
**Scope:** `src/practice/mod.rs` and supporting modules; one new method on `MusicalTransport`.

---

## Summary

Refactor the music-practice logic engine in [src/practice/mod.rs](../../../src/practice/mod.rs) to add:

- A **3-tier note-detection conditioner** (Onset / 5-frame stable / 4-transient cluster) that gates pitch identity through 5 STFT hops of stable tuner data while choosing the most accurate timestamp source.
- A **3-measure ring buffer** (past, current, future) with explicit per-slot match status, polyphonic matching, and ±2-note look-ahead/-behind.
- A **`ClockManager`** that controls `MusicalTransport` via `set_bpm`, `seek_to_beat`, `play`, `stop`, derives a hybrid student-timer BPM (`T_stu`), and applies a uniform seek-direction rule based on cursor-vs-beat position.
- Three execution modes: **FollowAlong**, **Performance**, **Rubato**, each with mode-specific clock behavior.
- Live `SendInfo` per matcher event with new `Tempo`, `HeldTooLong`, `HeldTooShort` error variants.
- Polyphonic pitch tracking (the existing `Tuner` already supports `MultiPitch` mode; this refactor unblocks the multi-pitch path the practice session ignores today).

The existing `PracticeSession` public API (`new`, `start`, `stop`, `set_tuner_mode`, `set_bpm`, `poll_transport`, `poll_errors`, `get_metrics`, `is_running`) is preserved verbatim; only `new` gains a `mode: PracticeMode` parameter (replacing the removed `wait_for_onset` flag). The lib-layer FFI surface does not change. The metronome is muted by the front-end / lib layer when entering Rubato — practice code does not own it.

The execution model stays single-threaded (one polling loop, 10 ms cadence) with layered components called sequentially each tick. No new concurrency.

## Glossary

- **STFT hop / frame.** A single tuner analysis output (~23 ms at 2048/44100). The conditioner's "5 frames" rule is 5 of these.
- **Polling tick.** One iteration of the practice loop (10 ms). Faster than the tuner hop rate, so multiple ticks see the same `TunerOutput.beat_position`. Conditioner deduplicates by `beat_position` change.
- **`T_stu`.** Student-timer BPM. A hybrid estimate: per-note-match EWMA when notes are flowing, plus a real-time hesitation-tempo display value when the student is overdue.
- **Frontier.** The lowest-key not-yet-`Matched` slot in the buffer. Advances on each `Matched` outcome.
- **Slot.** Per-`ExpectedNote` state across the 3 active measures. Status is `Pending | Matched { pitch_correct } | Missed`.
- **`ε`.** A small beat offset (default `0.001 beats`) used to land seeks just before/after a target beat.
- **Duration window.** Half-open interval `[exp.beat_position, exp.beat_position + exp.duration_beats)`. A note "in window" if `tracked.start_beat ∈ this interval`.
- **`transport_beat`.** Sampled exactly once at the top of each polling tick (`get_accumulated_beats()`); used as a stable value for all decisions made within that tick. The audio thread is allowed to advance the underlying value mid-tick — we deliberately ignore that drift to get atomic per-tick semantics.

---

## §1 — Module layout, public surface, and shared types

### File layout under `src/practice/`

```
mod.rs          — PracticeSession (public API + thread orchestration), unchanged surface
metrics.rs      — minor additions for new error metrics (§6)
session.rs      — SessionState (FFI-shared) + per-tick orchestration glue
conditioner.rs  — InputConditioner (3-tier per-pitch detection)
buffer.rs       — MeasureBuffer (3-measure ring + per-slot match state)
matcher.rs      — Matcher (resolves note starts against expected notes)
clock.rs        — ClockManager (T_stu, transport seek/stop/play decisions)
mode.rs         — ModeController + PracticeMode + per-tick algorithm
types.rs        — TrackedNoteStart, TrackedNoteEnd, MatchOutcome, ClockAction, etc.
```

### Public API of `PracticeSession`

Preserved verbatim:
- `new(...)` — gains `mode: PracticeMode` parameter; loses `wait_for_onset`.
- `start(start_measure, end_measure)`, `stop()`, `set_tuner_mode(mode)`, `set_bpm(bpm)`.
- `poll_transport()`, `poll_errors()`, `get_metrics()`, `is_running()`.

The UniFFI bindings (`AudioEngine::create_practice_session` and friends) need only a one-line update to pass through `mode`.

### Removed / kept

- `wait_for_onset` flag → **removed**, subsumed by `PracticeMode::FollowAlong`.
- `countoff_beats` → **kept** as count-in only.
- Session-level count-off-offset calibration (`countoff_offsets`, `onset_beat_offset`) → **removed**. The transport's built-in calibration (`stamp_onset` for onsets, `calibrated_beat()` helper for tuner-derived beats) is sufficient.

### One new method on `MusicalTransport`

In [src/audio_io/timing.rs](../../../src/audio_io/timing.rs):

```rust
impl MusicalTransport {
    /// Apply input + output latency + calibration_offset compensation to an
    /// arbitrary beat position obtained from the tuner pipeline (whose raw
    /// `beat_position` is captured at audio-processing time but is not
    /// pre-compensated like onsets are by `stamp_onset`).
    ///
    /// Tier-2 (StableFiveFrame) and tier-3 (TransientCluster) start beats are
    /// passed through this; OnsetEvent.beat_position already comes calibrated.
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
}
```

### Shared types

```rust
pub enum PracticeMode { FollowAlong, Performance, Rubato }

/// Emitted by InputConditioner when a confirmed pitch start lands.
/// Polyphonic chords produce one per pitch, possibly across ticks.
pub struct TrackedNoteStart {
    pub seq: u64,                       // monotonic; Started/Ended are paired by seq
    pub midi_note: u8,
    pub start_beat: f64,                // calibrated; backdated per tier cascade
    pub start_source: StartSource,
    pub initial_cents: f64,             // average of the 5-frame StartPending buffer
}

pub enum StartSource { Onset, StableFiveFrame, TransientCluster }

pub struct TrackedNoteEnd {
    pub seq: u64,
    pub midi_note: u8,
    pub end_beat: f64,                  // first frame of confirmed absence
    pub avg_cents: f64,                 // running average over the note's lifetime
    pub frame_count: u32,
}

pub enum ConditionerEvent {
    Started(TrackedNoteStart),
    Ended(TrackedNoteEnd),
}

pub enum MatchOutcome {
    Matched {
        key: (usize, usize),             // (measure_idx, note_idx)
        timing_err: f64,                 // signed: tracked.start_beat - expected.beat
        pitch_correct: bool,
        upgrade: bool,                   // true iff slot was previously Matched(false)
        skipped_keys: Vec<(usize, usize)>,  // intervening Pending slots → mark Missed
    },
    DoubledNote { key: (usize, usize) },
    ExtraNote   { during: Option<(usize, usize)> },
}

pub enum ClockAction {
    SeekToBeat(f64),
    Stop,
    Play,
    SetBpm(f32),
}
```

`NoteEvent`, `OnsetEvent`, `DynamicsEvent`, `ExpectedNote`, `MeasureData`, `SendInfo`, `MusicError` (extended in §6), `AbilityLevel` are kept from existing code. The `last_note_*`, `pending_*`, `cents_sum` fields on today's `SessionState` move into per-pitch conditioner state.

---

## §2 — InputConditioner

### Inputs per polling tick

- `tuner_frame: Option<&TunerFrame>` — populated only when `TunerOutput.beat_position` advanced since the last tick (deduplication of the ~23 ms tuner hop against the 10 ms polling cadence). Each frame carries `Vec<(midi: u8, cents: f64)>` and `tuner_beat: f64 = transport.calibrated_beat(TunerOutput.beat_position)`.
- `new_onsets: &[OnsetEvent]` — already calibrated by `stamp_onset`.

### Per-pitch state

`HashMap<u8 /* midi */, PitchState>`, sparse; absence of a key = `Idle`.

```rust
enum PitchState {
    StartPending {
        frames: u32,                  // counts up to STABLE_FRAMES (5)
        first_frame_beat: f64,        // calibrated; backdate target
        first_frame_seq: u64,
        cents_buffer: Vec<f64>,       // length == frames
    },
    Active {
        seq: u64,                     // assigned at Start emission
        start_beat: f64,
        start_source: StartSource,
        cents_sum: f64,
        frame_count: u32,
    },
    EndPending {
        absent_frames: u32,           // counts up to END_FRAMES (5)
        first_absence_beat: f64,
        carry: ActiveBody,            // start_beat, start_source, cents_sum, frame_count, seq
    },
}
```

### Auxiliary state

- `recent_onsets: VecDeque<OnsetEvent>` — drops events older than `RECENT_ONSET_RETENTION = 0.5 beats`.
- `transient_log: VecDeque<(seq: u64, beat: f64, midi: u8)>` — entries pushed when a `StartPending` collapses to absent without confirming. Entries older than `CLUSTER_FRAME_WINDOW = 10` frames are dropped.
- `frame_seq: u64` — incremented each time a *new* tuner frame is processed (not per polling tick).
- `next_event_seq: u64` — assigned to each emitted `Started`/`Ended` pair.

### Per-frame state machine

Run only when a new tuner frame arrives. For each midi `m` present in `tuner_frame.notes`:
- Map miss → insert `StartPending { frames: 1, first_frame_beat: tuner_beat, first_frame_seq: frame_seq, cents_buffer: [cents] }`.
- `StartPending(n, fb, .., buf) → StartPending(n+1, fb, .., buf++[cents])`. If `n+1 == STABLE_FRAMES`, run the **tier cascade** (below) and transition to `Active`.
- `Active` → bump `cents_sum`, `frame_count`.
- `EndPending` → restore to `Active`. Cents folded into `carry.cents_sum` during the gap remain there (do not retroactively remove).

For each midi `m` *not* in the new frame but present in the map:
- `StartPending(n, fb, .., m)` → remove from map; push `(seq, fb, m)` to `transient_log`.
- `Active(carry)` → `EndPending { absent_frames: 1, first_absence_beat: tuner_beat, carry }`.
- `EndPending(n, fap, carry) → EndPending(n+1, fap, carry)`. If `n+1 == END_FRAMES`, emit `Ended { seq: carry.seq, midi: m, end_beat: fap, avg_cents: carry.cents_sum / carry.frame_count, frame_count: carry.frame_count }` and remove from map.

### Tier cascade

Run once at the moment `StartPending` confirms (frame `STABLE_FRAMES`) for pitch P. The cascade decides `start_source` and `start_beat`; `initial_cents` is always `avg(cents_buffer)`.

1. **Onset.** Search `recent_onsets` for an event whose `beat_position` falls within `±ONSET_CLAIM_WINDOW` (`0.05 beats`) of `P.first_frame_beat`. If found → `start_beat = onset.beat_position`, `start_source = Onset`. Consume the onset.
2. **TransientCluster.** Else, if `transient_log` contains ≥ `CLUSTER_MIN_TRANSIENTS` (4) entries with `seq ≥ P.first_frame_seq − CLUSTER_FRAME_WINDOW` → `start_beat = transient_log[0].beat`, `start_source = TransientCluster`. Drain consumed entries.
3. **StableFiveFrame.** Else → `start_beat = P.first_frame_beat`, `start_source = StableFiveFrame`.

A `seq = next_event_seq++` is assigned. The `Started` event is emitted; the pitch transitions to `Active { seq, start_beat, start_source, cents_sum: cents_buffer.sum, frame_count: STABLE_FRAMES }`.

### Vibrato glide rule (cross-pitch cents folding)

While pitch `P_old` is in `EndPending` *and* some pitch `P_new ≠ P_old` is in `StartPending`, on each tuner frame the cents reported for `P_new` are translated into `P_old`'s frame:

```
folded_cents = (P_new.midi - P_old.midi) * 100 + P_new_cents
P_old.carry.cents_sum  += folded_cents
P_old.carry.frame_count += 1
```

Two outcomes:
- **`P_new` confirms first** (reaches `STABLE_FRAMES` before `P_old.absent_frames` does): before running the cascade for `P_new`, `P_old` emits `Ended` with `end_beat = P_new.first_frame_beat`, `avg_cents = carry.cents_sum / carry.frame_count` (includes folded glide). `P_new` then runs the cascade and starts fresh — its cents come from its own `cents_buffer` only; the cents that were folded into `P_old` are **not also charged to `P_new`**.
- **`P_new` disappears first** (transient): standard flow. `P_new → Idle` (push to `transient_log`); `P_old` keeps absence-counting.

### Public surface

```rust
pub struct InputConditioner { /* state above + Arc<MusicalTransport> */ }

impl InputConditioner {
    pub fn new(transport: Arc<MusicalTransport>) -> Self;

    /// One polling tick. `tuner_frame` is Some only when beat_position advanced.
    pub fn ingest(
        &mut self,
        tuner_frame: Option<&TunerFrame>,
        new_onsets: &[OnsetEvent],
    ) -> Vec<ConditionerEvent>;
}
```

The conditioner has no awareness of measures, expected notes, modes, or matching. It only converts raw analyzer streams into clean `Started`/`Ended` events with calibrated beats and a stable seq number.

### Tunables (proposed defaults)

- `STABLE_FRAMES = 5`, `END_FRAMES = 5`
- `ONSET_CLAIM_WINDOW = 0.05` beats
- `CLUSTER_MIN_TRANSIENTS = 4`, `CLUSTER_FRAME_WINDOW = 10` frames
- `RECENT_ONSET_RETENTION = 0.5` beats

---

## §3 — MeasureBuffer + Matcher

### MeasureBuffer

The ring holds references into `Arc<Vec<Measure>>` plus per-note slot state across all three active measures.

```rust
pub struct MeasureBuffer {
    measures: Arc<Vec<Measure>>,
    practice_start: usize,
    practice_end: usize,

    past_idx:    Option<usize>,
    current_idx: usize,
    future_idx:  Option<usize>,

    /// (measure_idx, note_idx) → slot
    slots: HashMap<(usize, usize), NoteSlot>,
}

pub struct NoteSlot {
    pub status: SlotStatus,
    pub matched_start_beat: Option<f64>,    // tracked.start_beat that matched
    pub matched_seq: Option<u64>,
}

pub enum SlotStatus {
    Pending,
    Matched { pitch_correct: bool },
    Missed,
}
```

When a measure enters the buffer (initially or via `advance`), all its expected notes get a `Pending` slot. When it ages out, slots for that measure are removed and the measure's `MeasureData` is yielded.

### MeasureBuffer surface

```rust
impl MeasureBuffer {
    pub fn new(measures: Arc<Vec<Measure>>, practice_start: usize, practice_end: usize) -> Self;

    /// Cycle past/current/future when transport_beat ≥ current_measure_end.
    /// Returns 0 or 1 aged-out measures (as MeasureData).
    pub fn advance(&mut self, transport_beat: f64) -> Vec<MeasureData>;

    pub fn record_match(&mut self, key: (usize, usize), tracked: &TrackedNoteStart, pitch_correct: bool);
    pub fn upgrade_match(&mut self, key: (usize, usize), tracked: &TrackedNoteStart);
    pub fn mark_missed(&mut self, key: (usize, usize));

    /// All slots whose duration window (half-open, see glossary) covers `beat`,
    /// plus the next LOOKAHEAD_NOTES not-yet-Matched slots after `frontier`,
    /// plus the previous LOOKBEHIND_NOTES already-passed slots before `frontier`.
    /// Slots of any status (Pending/Matched/Missed) are returned; the matcher's
    /// scoring rules differentiate based on status.
    pub fn candidates(&self, beat: f64, frontier: (usize, usize)) -> Vec<Candidate>;

    /// Lowest-key slot strictly after `frontier` whose status is Pending.
    /// Used by ClockManager for stop-trigger lookup.
    pub fn next_pending_after(&self, frontier: (usize, usize)) -> Option<(usize, usize)>;
}

pub struct Candidate {
    pub key: (usize, usize),
    pub expected: ExpectedNote,
    pub status: SlotStatus,
    pub kind: CandidateKind,    // InWindow | Lookahead(u8) | Lookbehind(u8)
}
```

### Matcher

Pure function over `(TrackedNoteStart, &MeasureBuffer, frontier)`:

```rust
pub fn resolve(tracked: &TrackedNoteStart, buf: &MeasureBuffer, frontier: (usize, usize)) -> MatchOutcome;
```

### Algorithm

1. **Special case: any `Pending` slot's duration window contains `tracked.start_beat`.** This slot is matched regardless of pitch (per agreed §3 (a) rule (ii) — cleaner mirroring of existing "any note played near expected beat" → WrongNote behavior). If multiple Pending slots overlap, pick the one with smallest `|tracked.start_beat - expected.beat|`. Pitch correctness sets `pitch_correct: true | false`.

2. **Upgrade case: any `Matched(false)` slot's duration window contains `tracked.start_beat` and pitch is exact.** Promote → `Matched { pitch_correct: true, upgrade: true, … }`. The slot's status becomes `Matched { pitch_correct: true }`.

3. **DoubledNote case: any `Matched(true)` slot's duration window contains `tracked.start_beat`, pitch is exact, AND `tracked.start_beat - slot.matched_start_beat ≤ DOUBLED_NOTE_FRESHNESS` (0.5 beats).** → `DoubledNote { key }`.

4. **Score the look-ahead/look-behind candidates** for non-overlap matches. Build candidates with `buf.candidates(beat, frontier)` and score each:

   - **Pitch:** exact MIDI = 100; ±1 semitone = 30; ±2 = 10; else 0.
   - **Timing:** if `tracked.start_beat ∈ [exp.beat, exp.beat + exp.duration_beats]` → 50; else `max(0, 50 − 100·|err|)`.
   - **Kind penalty:** `InWindow` 0; `Lookahead(1)` −10, `Lookahead(2)` −25; `Lookbehind(1)` −15.

   If best candidate scores ≥ `MIN_MATCH_SCORE` (80) AND it's `Pending` AND has exact pitch → `Matched { pitch_correct: true, skipped_keys: <Pending between frontier and best> }`.

5. **Otherwise** → `ExtraNote { during: <any Matched(_) or Pending slot whose duration window contains tracked.start_beat, prefer Pending> }`. (Most often `during = None` if the beat lies in a true rest.)

### Frontier

Owned by `ModeController` (not the buffer). Initialized to `(practice_start, 0)`. Advanced to `winner.key + 1` (rolling across measure boundaries) on each `Matched`. `MeasureBuffer::advance` clamps the frontier forward when a measure ages out.

### Look-behind

When a measure ages out, its `Pending` slots are converted to `Missed` and a `MissingNote` `SendInfo` is emitted for each. They remain queryable as `Lookbehind` candidates for one additional `advance` cycle so an extremely late doubled-or-late detection can still resolve to the right slot.

### Polyphony

Falls out for free: a 3-note chord produces 3 independent `TrackedNoteStart`s, each resolved separately. Stacked-but-staggered notes (G4 arriving 0.05 beats after C4/E4) score `InWindow` for the still-`Pending` G4 slot because its duration window covers the lag.

### Tunables

- `MIN_MATCH_SCORE = 80`
- `LOOKAHEAD_NOTES = 2`
- `LOOKBEHIND_NOTES = 1`
- `DOUBLED_NOTE_FRESHNESS = 0.5` beats

---

## §4 — ClockManager

The ClockManager owns student-tempo state and decides what transport actions to fire. It does **not** call `transport.*` directly — it returns `ClockAction`s. The orchestrator (§5) applies them through the mode filter.

### State

```rust
pub struct ClockManager {
    transport: Arc<MusicalTransport>,    // read-only access (get_accumulated_beats, get_bpm)
    cfg: ClockConfig,

    bpm_ewma: f32,                       // initialized to true_bpm
    streak_late: usize,
    streak_early: usize,

    last_match_real_beat:     Option<f64>,  // transport.beat at the moment of last match
    last_match_expected_beat: Option<f64>,

    stopped_for_unplayed: bool,          // true after Stop emitted, until next match releases it
}

pub struct ClockConfig {
    pub seek_threshold_pct: f64,         // 0.15
    pub bpm_change_threshold_pct: f64,   // 0.08
    pub bpm_change_streak: usize,        // 3
    pub stop_lead_epsilon: f64,          // 0.001 beats
    pub seek_landing_epsilon: f64,       // 0.001 beats
    pub ewma_alpha: f32,                 // 0.4
}
```

### Public surface

```rust
impl ClockManager {
    pub fn new(transport: Arc<MusicalTransport>, cfg: ClockConfig, initial_bpm: f32) -> Self;

    pub fn t_stu_bpm(&self) -> f32 { self.bpm_ewma }

    pub fn on_match(
        &mut self,
        outcome: &MatchOutcome,            // Matched variant
        expected: &ExpectedNote,
        transport_beat: f64,
        mode: PracticeMode,
    ) -> Vec<ClockAction>;

    pub fn on_doubled(&mut self, slot: &NoteSlot, mode: PracticeMode) -> Vec<ClockAction>;

    pub fn on_extra(&mut self) -> Vec<ClockAction> { Vec::new() }

    pub fn on_tick(
        &mut self,
        buf: &MeasureBuffer,
        frontier: (usize, usize),
        transport_beat: f64,
        mode: PracticeMode,
    ) -> Vec<ClockAction>;
}
```

### T_stu (hybrid model)

**On confirmed match (in `on_match`):**

```
real_time_diff   = transport_beat - last_match_real_beat
expected_diff    = expected.beat_position - last_match_expected_beat
local_tempo      = (expected_diff / real_time_diff) * current_bpm        // if real_time_diff > 0
bpm_ewma        ← α · local_tempo + (1−α) · bpm_ewma                     // α = 0.4
last_match_real_beat     ← transport_beat
last_match_expected_beat ← expected.beat_position
```

Streaks:
- `streak_late` increments when `local_tempo < current_bpm · (1 − bpm_change_threshold_pct)`. Resets on a non-late note.
- `streak_early` mirror image.

**On tick when overdue (in `on_tick`):**

If `frontier.status == Pending` AND `transport_beat > frontier.expected.beat`, expose a *transient* hesitation tempo via `t_stu_bpm()`:

```
hesitation_tempo = current_bpm * (frontier.expected.beat - last_match_expected_beat)
                                / (transport_beat - last_match_real_beat)
```

This value is **not** folded into the persistent EWMA and does **not** advance streaks. It is purely a live display value for UI/logging. (Confirmed in Q4 (b): "a hesitation should not affect the student bpm.")

### `set_bpm` gate

At the end of `on_match`:
- if `streak_late ≥ bpm_change_streak (3)` OR `streak_early ≥ bpm_change_streak (3)`,
- AND `|bpm_ewma − current_bpm| / current_bpm > bpm_change_threshold_pct (0.08)`,
- AND `mode != Performance`,
- → emit `SetBpm(bpm_ewma)`, reset both streaks.

### Seek decisions

**Unified seek-direction rule (FollowAlong + Rubato):**

```
target_beat = matched.beat - seek_landing_epsilon   if transport_beat < matched.beat
              matched.beat + seek_landing_epsilon   otherwise
```

Rationale: the ε side is chosen so that the natural `did_cross_beat` logic does the right thing — fire the click when transport hasn't yet reached `matched.beat` (seek to `−ε`); avoid re-firing it when transport has already crossed (seek to `+ε`). This single rule handles all three FollowAlong cases (early, late within stop-window, skip-then-match-next-after).

**FollowAlong `on_match`:**

```
must_seek = |timing_err| > expected.duration_beats * 0.15  ||  stopped_for_unplayed
if must_seek:
    emit SeekToBeat(target_beat)
emit Play
stopped_for_unplayed = false
```

**Rubato `on_match`:** always seek; threshold ignored.

```
emit SeekToBeat(target_beat)
emit Play
```

**Performance `on_match`:** no seek/play actions; only T_stu state is updated.

### Stop trigger (FollowAlong only)

Per tick, after frontier is advanced:
- `next_after = buf.next_pending_after(frontier)` — lowest-key strictly-after-frontier slot still `Pending`.
- If `frontier.status == Pending` AND `next_after.is_some()` AND `transport_beat ≥ next_after.beat - stop_lead_epsilon`:
  - emit `Stop`, set `stopped_for_unplayed = true`.

### Doubled note

**FollowAlong + Rubato:** revert. `on_doubled` returns `[SeekToBeat(slot.matched_start_beat + seek_landing_epsilon), Play]`. T_stu is **not** updated for the doubled event.

**Performance:** empty (no clock adjustment).

### Extra note

`on_extra` returns empty in all modes. Per spec: "Ensure that there is no clock adjustment for any mode when an extra note is played."

### First note has no anchor

`last_match_real_beat`/`last_match_expected_beat` are `None` until the first `Matched`. T_stu defaults to `bpm_ewma = current_bpm`; no local tempo computed; no streak update.

### Tunables

- `seek_threshold_pct = 0.15`
- `bpm_change_threshold_pct = 0.08`
- `bpm_change_streak = 3`
- `stop_lead_epsilon = 0.001` beats
- `seek_landing_epsilon = 0.001` beats
- `ewma_alpha = 0.4`

---

## §5 — ModeController and the per-tick algorithm

The `ModeController` is the orchestrator. It is owned by the polling thread (not behind a `Mutex`); thread-shared state (for FFI poll_*) stays in a separate small `Mutex<SessionState>`.

### State

```rust
pub struct ModeController {
    mode: PracticeMode,
    transport: Arc<MusicalTransport>,
    conditioner: InputConditioner,
    buffer: MeasureBuffer,
    clock: ClockManager,

    frontier: (usize, usize),

    // Per-measure event accumulators (built up across ticks before the measure ages out)
    in_progress_played_notes: HashMap<usize, Vec<NoteEvent>>,
    in_progress_onsets:       HashMap<usize, Vec<OnsetEvent>>,
    in_progress_dynamics:     HashMap<usize, Vec<DynamicsEvent>>,
    in_progress_durations:    HashMap<usize, Vec<Option<f64>>>,    // parallel to played_notes
    in_progress_doubled_seqs: HashMap<usize, Vec<u64>>,            // seqs of NoteEvents flagged Tempo

    // Correlate Started/Ended via seq for hold check + finalize avg_cents.
    // Drained on session stop or when a `seq` is much older than current frontier (TTL = 8 beats).
    match_log: HashMap<u64 /* seq */, MatchedSnapshot>,

    // Dedup state
    last_tuner_beat: Option<f64>,
    last_dynamic_level: Option<DynamicLevel>,
}

struct MatchedSnapshot {
    measure_idx: usize,
    note_idx_in_measure_data: usize,        // index into in_progress_played_notes[m]
    expected_duration: f64,
    expected_midi: u8,
}
```

### Per-tick algorithm

Run every 10 ms by the polling thread:

```
1. transport_beat = transport.get_accumulated_beats()

2. Drain inputs:
   - new_onsets    = onset.drain_onset_events()
   - tuner_snap    = tuner_output.read()
   - dyn_level     = dynamics_output.read().level

3. Build TunerFrame (Some only if tuner_snap.beat_position changed since last tick):
   - midis  = parse note names → MIDI numbers
   - cents  = accuracies (f32 → f64)
   - tuner_beat = transport.calibrated_beat(tuner_snap.beat_position)

4. events = conditioner.ingest(tuner_frame, &new_onsets)

5. For each ConditionerEvent:
   - Started(t):
     - Append a NoteEvent { beat_position: t.start_beat, midi_note: t.midi_note,
                            avg_cents: t.initial_cents } to in_progress_played_notes[mi]
       where mi = buffer.measure_for_beat(t.start_beat).
       Append None to in_progress_durations[mi] in the same position.
     - outcome = matcher::resolve(&t, &buffer, frontier)
     - Update buffer.slots:
        Matched { upgrade: false } → buffer.record_match(key, &t, pitch_correct)
        Matched { upgrade: true  } → buffer.upgrade_match(key, &t)
        Matched.skipped_keys       → buffer.mark_missed(k) for each k
        DoubledNote { key }        → in_progress_doubled_seqs[key.0].push(t.seq)  (no slot change)
        ExtraNote                  → no slot change
     - Advance frontier on Matched.
     - Record match_log entry on Matched (for the later End hold-check + final cents).
     - Emit SendInfo per the §6 table.
     - actions = clock.on_match / on_doubled / on_extra (per outcome) with transport_beat, mode.
     - Apply each action through the mode filter (see below).

   - Ended(t):
     - If match_log.contains_key(t.seq):
       - snap = match_log.remove(t.seq)
       - note_ref = &mut in_progress_played_notes[snap.measure_idx][snap.note_idx_in_measure_data]
       - actual_duration = t.end_beat - note_ref.beat_position
       - in_progress_durations[snap.measure_idx][snap.note_idx_in_measure_data] = Some(actual_duration)
       - note_ref.avg_cents = t.avg_cents      // overwrite the provisional initial_cents with final value
       - Emit hold-too-long/short SendInfo if outside HOLD_TOLERANCE_PCT.
       - Emit final Intonation SendInfo if |t.avg_cents| > INTONATION_ERR_THRESHOLD * tol.
     - else (no match log entry — the Start was an ExtraNote, or the entry was TTL-purged): drop silently.

6. Append OnsetEvents to in_progress_onsets[buffer.current_idx]
   (raw — they live with the measure they fired in for metrics).

7. If dyn_level != Silence and dyn_level != last_dynamic_level:
     - Append DynamicsEvent{beat: transport_beat, level: dyn_level} to in_progress_dynamics[current_idx]
     - last_dynamic_level = Some(dyn_level)

8. tick_actions = clock.on_tick(&buffer, frontier, transport_beat, mode)
   Apply each through the mode filter.

9. aged = buffer.advance(transport_beat)   // 0 or 1 measures
   For each aged measure mi:
     - For any remaining Pending slot in mi: buffer.mark_missed; emit MissingNote SendInfo.
     - measure_data = MeasureData {
         measure_index: mi,
         onsets:    in_progress_onsets.remove(mi).unwrap_or_default(),
         notes:     in_progress_played_notes.remove(mi).unwrap_or_default(),
         dynamics:  in_progress_dynamics.remove(mi).unwrap_or_default(),
         note_durations: in_progress_durations.remove(mi).unwrap_or_default(),
         doubled_note_seqs: in_progress_doubled_seqs.remove(mi).unwrap_or_default(),
         expected_notes: build_expected_notes(&measures[mi]),
       }
     - end_of_measure_feedback = generate_measure_feedback(&measure_data, ability_level, mode);
     - completed_measures.push(measure_data);
     - Push end_of_measure_feedback to the SendInfo queue.

10. If frontier > (practice_end, last_note): signal session done.

11. sleep(10 ms)
```

### Mode filter for ClockActions

```rust
fn apply(action: &ClockAction, mode: PracticeMode, transport: &MusicalTransport) {
    match (mode, action) {
        (PracticeMode::Performance, _) => { /* drop all */ }

        (PracticeMode::FollowAlong, ClockAction::SeekToBeat(b)) => transport.seek_to_beat(*b),
        (PracticeMode::FollowAlong, ClockAction::Stop)          => transport.stop(),
        (PracticeMode::FollowAlong, ClockAction::Play)          => transport.play(),
        (PracticeMode::FollowAlong, ClockAction::SetBpm(bpm))   => transport.set_bpm(*bpm),

        (PracticeMode::Rubato,  ClockAction::SeekToBeat(b))     => transport.seek_to_beat(*b),
        (PracticeMode::Rubato,  ClockAction::Play)              => transport.play(),
        (PracticeMode::Rubato,  ClockAction::SetBpm(bpm))       => transport.set_bpm(*bpm),
        (PracticeMode::Rubato,  ClockAction::Stop)              => { /* never stop in Rubato */ }
    }
}
```

`ClockManager` already short-circuits most actions per mode in its decision functions. The filter is defense-in-depth.

### Mode-specific behavior summary

| | FollowAlong | Performance | Rubato |
|---|---|---|---|
| `play()` on detected note | yes | no | yes |
| `stop()` before next-after-frontier | yes | no | no |
| Per-match seek if `\|err\|>15%` | yes (uniform −/+ ε rule) | no | always seek (no threshold) |
| `set_bpm` on streak | yes | no | yes |
| Look-ahead for missed notes | yes (up to 2) | yes (matching only) | yes |
| Doubled-note clock revert | seek + play | none | seek + play |
| Tolerance scaling for feedback | `tol` from `AbilityLevel` | same | `tol × 1.5` |
| Metronome audible | yes | yes | **muted** by FE/lib layer |

### Metronome muting

The metronome is owned by the lib/UniFFI layer. When the user switches to Rubato, the front-end mutes the metronome via existing FE/lib APIs. `PracticeSession` does not own a metronome handle and does not toggle muting itself.

### `PracticeSession` shrinks

`PracticeSession::new` gains `mode: PracticeMode`. `start` configures transport, spawns the polling thread that owns the `ModeController`. The shared `Mutex<SessionState>` for FFI poll_* methods now contains only:

```rust
struct SessionState {
    practice_start: usize,
    practice_end:   usize,
    current_measure_idx: usize,            // mirrored from buffer.current_idx for poll_transport
    completed_measures: Vec<MeasureData>,  // for get_metrics
    in_countoff: bool,
    first_measure_beat: f64,
}
```

The polling thread locks this only briefly when reading inputs (transport handles aren't behind it) and at end-of-tick to publish completed measures. The `ModeController` itself is owned exclusively by the polling thread.

---

## §6 — Feedback emission, hold detection, metrics integration

### MusicError additions

```rust
pub enum MusicError {
    Timing,
    WrongNote,
    UnexpectedNote,
    MissingNote,
    Intonation,
    Dynamics,
    Tempo,             // NEW — doubled-note (start-restart)
    HeldTooLong,       // NEW
    HeldTooShort,      // NEW
    None,
}
```

### Sequence-number plumbing

`TrackedNoteStart` and `TrackedNoteEnd` carry a paired `seq: u64`. `ModeController` keeps `match_log: HashMap<u64, MatchedSnapshot>` to correlate Ends to slots (for hold-too-long/short and for finalizing avg_cents).

### Live `SendInfo` per `ConditionerEvent`

Each `Started` produces 1–N `SendInfo`s:

| MatchOutcome | Primary | Secondary (also if applicable) |
|---|---|---|
| `Matched { pitch_correct: true, upgrade: false }` | `MusicError::None` (success acknowledgment) | `Timing` if `\|err\| > duration·0.15·tol` |
| `Matched { pitch_correct: true, upgrade: true }` | `MusicError::None` with `expected = "<midi> (corrected from earlier <wrong_midi>)"` | `Timing` if applicable |
| `Matched { pitch_correct: false }` | `MusicError::WrongNote` | `Timing` if applicable |
| `Matched { skipped_keys }` (any subset) | one `MusicError::MissingNote` per skipped key | — |
| `DoubledNote` | `MusicError::Tempo` | — |
| `ExtraNote { during: Some(slot) }` | `MusicError::UnexpectedNote` ("expected <slot.midi>, received <played.midi> (extra note during held)") | — |
| `ExtraNote { during: None }` | `MusicError::UnexpectedNote` (`expected = "silence"`) | — |

The `MusicError::None` event for a successful match is emitted live so the front-end has a positive cursor cue per played note. Confirmed: a later error event (Intonation or HeldTooLong) for the same `(measure, note_idx)` overrides the None on the front-end.

Each `Ended` produces 0–2 `SendInfo`s (only when its `seq` correlates to a `Matched` snapshot — `ExtraNote` ends are silent):

| Condition | Emitted |
|---|---|
| `\|t.avg_cents\| > INTONATION_ERR_THRESHOLD · tol` | `MusicError::Intonation` (final, supersedes any provisional intonation feedback) |
| `actual_duration > expected.duration_beats · (1 + HOLD_TOLERANCE_PCT)` | `MusicError::HeldTooLong` |
| `actual_duration < expected.duration_beats · (1 − HOLD_TOLERANCE_PCT)` | `MusicError::HeldTooShort` |

`actual_duration = t.end_beat - matched.start_beat`. `HOLD_TOLERANCE_PCT = 0.25`.

### Wrong-note intonation per spec

`NoteEvent` recorded in `MeasureData.notes` carries `avg_cents` measured against the **played** MIDI. For `Matched { pitch_correct: false }`:
- The expected slot is consumed (so `accuracy_percent` counts it as matched).
- `NoteEvent.midi_note` = played MIDI; `NoteEvent.avg_cents` = deviation from played MIDI.
- `metrics.calc_avg_cent_dev` sums `|avg_cents|` across all detected notes — naturally including wrong-note deviation against its own pitch.

The `WrongNote` SendInfo's `received` shows the played pitch with cents (e.g. "D4 +12c"); `expected` shows the slot's expected pitch.

### End-of-measure feedback batch

When `MeasureBuffer::advance` ages out a measure, `generate_measure_feedback(&MeasureData, AbilityLevel, PracticeMode)` runs as the authoritative reconciliation pass. Its three remaining purposes after live emissions:

1. Authoritative scoring snapshot (frontend can use as canonical record).
2. Dynamics rollup against expected notes (live ticks accumulate; comparison is end-of-measure).
3. Defensive backstop in case a live emission was dropped.

Function signature gains `mode: PracticeMode` so Rubato can apply `tol *= 1.5` (relaxed timing/intonation thresholds in Rubato per the table in §5).

### Metrics additions

Two new fields on `Metrics`:

```rust
pub tempo_err_count: u64,
pub hold_err_count: (u64, u64),        // (held_too_long, held_too_short)
pub tempo_err_measures: Vec<u32>,
pub hold_err_measures:  Vec<u32>,
```

Computed in `Metrics::compute` by:
- `tempo_err_count` / `tempo_err_measures`: derived from doubled-note detection during measure rollup. Since the matcher's DoubledNote outcome doesn't appear in `MeasureData` directly, ModeController stamps a `tempo_err_seqs: Vec<u64>` field onto `MeasureData` recording which played-NoteEvents were doubles. `Metrics::compute` reads this.

```rust
pub struct MeasureData {
    /* existing fields */
    pub note_durations: Vec<Option<f64>>,
    pub doubled_note_seqs: Vec<u64>,
}
```

- `hold_err_count` / `hold_err_measures`: derived from `note_durations` vs `expected_notes[i].duration_beats`.

### Existing metrics functions

The other calculations (`calc_accuracy_percent`, `calc_avg_cent_dev`, `calc_timing_consistency`, `calc_microtiming_skew`, `calc_measure_tempo_map`, `calc_dynamics_*`) continue working unmodified:
- Polyphony works because `note_is_matched` already requires both MIDI + beat-window match.
- Doubled notes don't break accuracy_percent (the `.any()` pattern still finds a match).
- Wrong-note intonation rolls into `calc_avg_cent_dev` correctly.
- Missed notes counted via the existing unmatched-expected logic.

### Tunables added in §6

- `HOLD_TOLERANCE_PCT = 0.25`

---

## Implementation phases

The spec's original phased validation plan maps cleanly to four sequenced refactors. Each phase produces a working binary against which the next phase builds.

### Phase 1 — Ring buffer + data type plumbing

- Add `types.rs` (TrackedNoteStart/End, MatchOutcome, ClockAction).
- Add `MeasureBuffer` in `buffer.rs` with slot tracking.
- Plumb the polling loop to drive `MeasureBuffer::advance` instead of the in-place `current_measure_idx` counter, **without yet** changing detection logic.
- New `MusicalTransport::calibrated_beat`.

**Validation:** existing tests pass; new test injects a `NoteEvent` at beat 4.2 of a 4-beat measure and verifies it resolves against the future_measure's first ExpectedNote.

### Phase 2 — InputConditioner

- `conditioner.rs`: per-pitch state machines, transient-cluster tracker, recent-onsets ring.
- Replace today's `pitch-stability hysteresis` block in `run_session` with `conditioner.ingest`.
- Existing match logic continues running, now consuming `ConditionerEvent`s instead of inline state.

**Validation tests:**
- T-A: an `OnsetEvent` arriving with steady pitch → `Started` with `start_source = Onset`, `start_beat = onset.beat_position` after 5-frame confirmation.
- T-B: 5 frames of the same MIDI with no onset → `Started` with `start_source = StableFiveFrame`, `start_beat = first_frame_beat`.
- T-C: 5 alternating transient pitches followed by a stable pitch → `Started` with `start_source = TransientCluster`, `start_beat` = first transient's beat.
- T-D: vibrato C4 → C#4 glide → one `Ended(C4)` with avg_cents ≈ +55, one `Started(C#4)` with own clean cents.

### Phase 3 — ClockManager + Matcher

- `matcher.rs` and `clock.rs`.
- ModeController stub still single-mode (FollowAlong) for testing.

**Validation tests:**
- Matcher: skip-frontier with pitch-match on `next_after` produces `Matched { skipped_keys: [frontier] }`.
- Matcher: in-window wrong-pitch produces `Matched { pitch_correct: false }`.
- Matcher: mid-held same-pitch hit ≤ 0.5 beats → `DoubledNote`; > 0.5 beats → `ExtraNote { during }`.
- ClockManager: 10% timing error → no `SeekToBeat`. 20% timing error → `SeekToBeat`. Direction follows the unified rule.
- ClockManager: 3 consecutive late notes >8% slow → `SetBpm`.
- ClockManager: stop trigger at `next_after.beat - ε` when frontier still Pending.

### Phase 4 — ModeController + full integration

- `mode.rs` with all three modes wired.
- Mode-specific tolerances and metronome muting (FE-side; doc only here).
- End-of-measure batch updated for new error variants.
- `Metrics` extended with `tempo_err_*` and `hold_err_*`.

**Validation tests:**
- Performance: forced 25% timing error → no transport calls; `T_stu` updated; `MissedNote` errors logged.
- Rubato: 5% timing error → `SeekToBeat(expected.beat ± ε)` always.
- Full mock measure in FollowAlong with a missed note + look-ahead → matcher resolves the next-after, ClockManager emits the correct SeekToBeat sequence, Stop fires before the missed-then-resumed transition.

---

## Open questions / future work

- Tuning the conditioner thresholds (`STABLE_FRAMES`, `CLUSTER_FRAME_WINDOW`, `ONSET_CLAIM_WINDOW`) against a corpus of recorded student takes. Defaults are educated guesses.
- Whether `RecentOnsetRetention = 0.5 beats` should scale with BPM (a fixed window in beats is BPM-stable, so probably no — included here for visibility).
- Doubled-note detection across measure boundaries (rare but possible with very short notes at measure ends). Handled implicitly by the buffer's past+current window; flag if found in testing.
