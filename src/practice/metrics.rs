use serde::Serialize;

use crate::audio_io::dynamics::DynamicLevel;
use crate::audio_io::timing::OnsetEvent;

// ── Thresholds for error-measure classification ───────────────────────────────

/// Maximum allowed mean onset timing error (beats) before a measure is flagged.
pub(crate) const ONSET_TIMING_ERR_THRESHOLD: f64 = 0.15;
/// Minimum note accuracy (0.0–1.0) before a measure is flagged for note errors.
pub(crate) const ACCURACY_ERR_THRESHOLD: f64 = 0.80;
/// Maximum allowed average cent deviation before a measure is flagged.
pub(crate) const INTONATION_ERR_THRESHOLD: f64 = 25.0;
/// Minimum dynamic accuracy (0.0–1.0) before a measure is flagged.
pub(crate) const DYNAMICS_ERR_THRESHOLD: f64 = 0.50;
/// Beat window within which a detected onset/note is considered matched to an expected note.
pub(crate) const NOTE_MATCH_WINDOW: f64 = 0.25;

// ── Input types ───────────────────────────────────────────────────────────────

/// A pitch event emitted by the tuner when a new note is detected.
///
/// `avg_cents` is the running average cent deviation accumulated for this note;
/// it is reset when the next note begins.
pub struct NoteEvent {
    pub beat_position: f64,
    pub midi_note: u8,
    pub avg_cents: f64,
}

/// A dynamic-level change emitted by the dynamics tracker.
pub struct DynamicsEvent {
    pub beat_position: f64,
    pub level: DynamicLevel,
}

/// A single note from the MIDI reference file.
pub struct ExpectedNote {
    /// Beat position of the note onset.
    pub beat_position: f64,
    /// Duration in beats.
    pub duration_beats: f64,
    /// MIDI note number (0–127).
    pub midi_note: u8,
    /// Dynamic marking at this note, if encoded in the MIDI file.
    pub dynamic: Option<DynamicLevel>,
}

/// All accumulated data for one measure from the three live tracker streams
/// and the MIDI reference.
pub struct MeasureData {
    pub measure_index: u32,
    pub onsets: Vec<OnsetEvent>,
    pub notes: Vec<NoteEvent>,
    pub dynamics: Vec<DynamicsEvent>,
    pub expected_notes: Vec<ExpectedNote>,
}

// ── Metrics ───────────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct Metrics {
    pub start_measure: u32,
    pub end_measure: u32,
    pub num_measures: u32,
    pub tempo_bpm: f64,
    /// Percentage (0–100) of MIDI reference notes matched by a detected note.
    pub accuracy_percent: f64,
    /// Mean absolute cent deviation across all detected notes.
    pub avg_cent_dev: f64,
    /// Number of MIDI reference notes with no matching detected note.
    pub num_notes_missed: u64,
    /// Population std dev of onset timing errors in beats. Lower = more consistent.
    pub timing_consistency: f64,
    /// Population std dev of the difference between actual and expected dynamic
    /// level (ppp=0…fff=7) across all notes with a reference dynamic. Lower = more consistent.
    pub dynamics_consistency: f64,
    /// Percentage (0–100) of notes with a reference dynamic where the actual
    /// level is within ±1 step of expected.
    pub dynamics_accuracy: f64,
    pub error_measures: Vec<u32>,
    pub rhythm_err_measures: Vec<u32>,
    pub note_err_measures: Vec<u32>,
    pub intonation_err_measures: Vec<u32>,
    pub dynamics_err_measures: Vec<u32>,
    pub avg_errors_per_measure: f64,
    /// Mean absolute onset timing error in beats.
    pub note_onset_accuracy: f64,
    /// Mean signed timing error in beats. Positive = rushing; negative = dragging.
    pub microtiming_skew: f64,
    /// Tempo stability score (0.0–1.0). Derived from the coefficient of variation
    /// of per-measure effective BPM. 1.0 = perfectly stable.
    pub tempo_stability: f64,
    /// Per-measure effective BPM, estimated by comparing the span of detected onsets
    /// in beat-space to the span of reference notes. Values above `tempo_bpm` indicate
    /// rushing; below indicate dragging.
    pub measure_tempo_map: Vec<f64>,
    /// (min_dynamic, max_dynamic) labels of the dynamic range used across the session.
    pub dynamics_range_used: (String, String),
    /// NOTE: Cannot be determined analytically without repeat-section markers in the
    /// MIDI file. The MIDI reference would need to encode which measures are repetitions
    /// of earlier sections (e.g. via MIDI marker meta-events or a separate section map).
    /// Currently returns 0.0 as a placeholder.
    pub consistency_across_repetitions: f64,
}

impl Metrics {
    /// Compute all metrics from accumulated measure data.
    ///
    /// * `start_measure` / `end_measure` — the measure range practiced.
    /// * `tempo_bpm` — reference tempo from the transport / MIDI file.
    /// * `measures` — per-measure data from the three live trackers and MIDI reference.
    pub fn compute(
        start_measure: u32,
        end_measure: u32,
        tempo_bpm: f64,
        measures: &[MeasureData],
    ) -> Self {
        let num_measures = end_measure.saturating_sub(start_measure) + 1;

        let accuracy_percent = Self::calc_accuracy_percent(measures);
        let avg_cent_dev = Self::calc_avg_cent_dev(measures);
        let num_notes_missed = Self::calc_num_notes_missed(measures);
        let timing_consistency = Self::calc_timing_consistency(measures);
        let dynamics_consistency = Self::calc_dynamics_consistency(measures);
        let dynamics_accuracy = Self::calc_dynamics_accuracy(measures);
        let note_onset_accuracy = Self::calc_note_onset_accuracy(measures);
        let microtiming_skew = Self::calc_microtiming_skew(measures);
        let measure_tempo_map = Self::calc_measure_tempo_map(measures, tempo_bpm);
        let tempo_stability = Self::calc_tempo_stability_from_map(&measure_tempo_map, tempo_bpm);
        let dynamics_range_used = Self::calc_dynamics_range_used(measures);

        let rhythm_err_measures = Self::calc_rhythm_err_measures(measures);
        let note_err_measures = Self::calc_note_err_measures(measures);
        let intonation_err_measures = Self::calc_intonation_err_measures(measures);
        let dynamics_err_measures = Self::calc_dynamics_err_measures(measures);

        let error_measures = {
            let mut all: Vec<u32> = rhythm_err_measures
                .iter()
                .chain(&note_err_measures)
                .chain(&intonation_err_measures)
                .chain(&dynamics_err_measures)
                .copied()
                .collect();
            all.sort_unstable();
            all.dedup();
            all
        };

        let avg_errors_per_measure = if num_measures > 0 {
            error_measures.len() as f64 / num_measures as f64
        } else {
            0.0
        };

        Self {
            start_measure,
            end_measure,
            num_measures,
            tempo_bpm,
            accuracy_percent,
            avg_cent_dev,
            num_notes_missed,
            timing_consistency,
            dynamics_consistency,
            dynamics_accuracy,
            error_measures,
            rhythm_err_measures,
            note_err_measures,
            intonation_err_measures,
            dynamics_err_measures,
            avg_errors_per_measure,
            note_onset_accuracy,
            microtiming_skew,
            tempo_stability,
            measure_tempo_map,
            dynamics_range_used,
            consistency_across_repetitions: 0.0,
        }
    }

    // ── Note accuracy ─────────────────────────────────────────────────────────

    /// Percentage of MIDI reference notes matched by a detected `NoteEvent` within
    /// `NOTE_MATCH_WINDOW` beats with the correct MIDI note number.
    fn calc_accuracy_percent(measures: &[MeasureData]) -> f64 {
        let mut total = 0usize;
        let mut matched = 0usize;
        for m in measures {
            for (ei, _) in m.expected_notes.iter().enumerate() {
                total += 1;
                if Self::note_is_matched(&m.notes, &m.expected_notes, ei, NOTE_MATCH_WINDOW) {
                    matched += 1;
                }
            }
        }
        if total == 0 {
            return 100.0;
        }
        matched as f64 / total as f64 * 100.0
    }

    /// Mean absolute cent deviation across all detected note events.
    fn calc_avg_cent_dev(measures: &[MeasureData]) -> f64 {
        let mut sum = 0.0f64;
        let mut count = 0usize;
        for m in measures {
            for n in &m.notes {
                sum += n.avg_cents.abs();
                count += 1;
            }
        }
        if count == 0 {
            return 0.0;
        }
        sum / count as f64
    }

    /// Number of MIDI reference notes with no matching detected note.
    fn calc_num_notes_missed(measures: &[MeasureData]) -> u64 {
        let mut missed = 0u64;
        for m in measures {
            for (ei, _) in m.expected_notes.iter().enumerate() {
                if !Self::note_is_matched(&m.notes, &m.expected_notes, ei, NOTE_MATCH_WINDOW) {
                    missed += 1;
                }
            }
        }
        missed
    }

    // ── Timing ────────────────────────────────────────────────────────────────

    /// Population std dev of onset timing errors (beats) across all matched onsets.
    /// Lower = more consistent timing.
    fn calc_timing_consistency(measures: &[MeasureData]) -> f64 {
        let errors: Vec<f64> = measures
            .iter()
            .flat_map(|m| {
                m.expected_notes.iter().filter_map(|expected| {
                    Self::closest_onset_within_window(&m.onsets, expected.beat_position)
                        .map(|o| o.beat_position - expected.beat_position)
                })
            })
            .collect();
        Self::std_dev_f64(&errors) as f64
    }

    /// Mean absolute onset timing error in beats across all matched notes.
    fn calc_note_onset_accuracy(measures: &[MeasureData]) -> f64 {
        let mut total_error = 0.0f64;
        let mut count = 0usize;
        for m in measures {
            for expected in &m.expected_notes {
                if let Some(o) =
                    Self::closest_onset_within_window(&m.onsets, expected.beat_position)
                {
                    total_error += (o.beat_position - expected.beat_position).abs();
                    count += 1;
                }
            }
        }
        if count == 0 {
            return 0.0;
        }
        (total_error / count as f64) as f64
    }

    /// Mean signed timing error in beats. Positive = rushing; negative = dragging.
    fn calc_microtiming_skew(measures: &[MeasureData]) -> f64 {
        let mut total_error = 0.0f64;
        let mut count = 0usize;
        for m in measures {
            for expected in &m.expected_notes {
                if let Some(o) =
                    Self::closest_onset_within_window(&m.onsets, expected.beat_position)
                {
                    total_error += o.beat_position - expected.beat_position;
                    count += 1;
                }
            }
        }
        if count == 0 {
            return 0.0;
        }
        (total_error / count as f64) as f64
    }

    /// Per-measure effective BPM, estimated by comparing the beat-space span of detected
    /// onsets to the span of reference notes in the same measure.
    ///
    /// Since `OnsetEvent::beat_position` is already in reference-tempo beat coordinates,
    /// a smaller actual span relative to the expected span indicates rushing, and a larger
    /// span indicates dragging.
    fn calc_measure_tempo_map(measures: &[MeasureData], tempo_bpm: f64) -> Vec<f64> {
        measures
            .iter()
            .map(|m| {
                // Build matched pairs: (expected_beat, actual_onset_beat).
                // Using the closest onset within NOTE_MATCH_WINDOW for each expected note
                // ensures both expected_span and actual_span cover the *same* subset of
                // notes.  Comparing all-expected-span against only-matched-actual-span
                // produces systematic errors whenever the first or last expected note has
                // no matching onset (e.g. beat 12 in measure 3 with a half note).
                let mut matched: Vec<(f64, f64)> = m
                    .expected_notes
                    .iter()
                    .filter_map(|en| {
                        m.onsets
                            .iter()
                            .filter(|o| {
                                (o.beat_position - en.beat_position).abs() < NOTE_MATCH_WINDOW
                            })
                            .min_by(|a, b| {
                                (a.beat_position - en.beat_position)
                                    .abs()
                                    .partial_cmp(&(b.beat_position - en.beat_position).abs())
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .map(|o| (en.beat_position, o.beat_position))
                    })
                    .collect();

                matched.sort_by(|a, b| {
                    a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                });

                if matched.len() < 2 {
                    return tempo_bpm;
                }

                let expected_span = matched.last().unwrap().0 - matched.first().unwrap().0;
                let actual_span = matched.last().unwrap().1 - matched.first().unwrap().1;

                if actual_span < 1e-6 || expected_span < 1e-6 {
                    return tempo_bpm;
                }

                tempo_bpm * expected_span / actual_span
            })
            .collect()
    }

    /// Tempo stability score (0.0–1.0) from coefficient of variation of per-measure BPM.
    /// 1.0 = perfectly stable; 0.0 = ≥100% coefficient of variation.
    fn calc_tempo_stability_from_map(measure_tempo_map: &[f64], tempo_bpm: f64) -> f64 {
        if measure_tempo_map.len() < 2 {
            return 1.0;
        }
        let values: Vec<f64> = measure_tempo_map.iter().map(|&v| v as f64).collect();
        let std = Self::std_dev_f64(&values) as f64;
        let cv = std / tempo_bpm.max(1.0);
        (1.0 - cv.min(1.0)).max(0.0)
    }

    // ── Dynamics ──────────────────────────────────────────────────────────────

    /// Population std dev of (actual_dynamic_level − expected_dynamic_level) across all
    /// notes with a reference dynamic (ppp=0…fff=7). Lower = more consistent.
    fn calc_dynamics_consistency(measures: &[MeasureData]) -> f64 {
        let errors: Vec<f64> = measures
            .iter()
            .flat_map(|m| {
                m.expected_notes.iter().filter_map(|expected| {
                    let exp_dyn = expected.dynamic?;
                    let act_dyn = Self::actual_dynamic_at(&m.dynamics, expected.beat_position)?;
                    Some((Self::dynamic_to_i32(act_dyn) - Self::dynamic_to_i32(exp_dyn)) as f64)
                })
            })
            .collect();
        Self::std_dev_f64(&errors) as f64
    }

    /// Percentage of notes with a reference dynamic where the actual level is
    /// within ±1 step of the expected level.
    fn calc_dynamics_accuracy(measures: &[MeasureData]) -> f64 {
        let mut total = 0usize;
        let mut correct = 0usize;
        for m in measures {
            for expected in &m.expected_notes {
                let Some(exp_dyn) = expected.dynamic else {
                    continue;
                };
                let Some(act_dyn) = Self::actual_dynamic_at(&m.dynamics, expected.beat_position)
                else {
                    continue;
                };
                total += 1;
                if (Self::dynamic_to_i32(act_dyn) - Self::dynamic_to_i32(exp_dyn)).abs() <= 1 {
                    correct += 1;
                }
            }
        }
        if total == 0 {
            return 100.0;
        }
        correct as f64 / total as f64 * 100.0
    }

    /// Min and max dynamic level labels observed across the session, excluding silence.
    fn calc_dynamics_range_used(measures: &[MeasureData]) -> (String, String) {
        let mut min_val: Option<(i32, DynamicLevel)> = None;
        let mut max_val: Option<(i32, DynamicLevel)> = None;

        for m in measures {
            for event in &m.dynamics {
                if event.level == DynamicLevel::Silence {
                    continue;
                }
                let v = Self::dynamic_to_i32(event.level);
                if min_val.map_or(true, |(min, _)| v < min) {
                    min_val = Some((v, event.level));
                }
                if max_val.map_or(true, |(max, _)| v > max) {
                    max_val = Some((v, event.level));
                }
            }
        }

        (
            min_val
                .map(|(_, l)| l.to_string())
                .unwrap_or_else(|| "n/a".into()),
            max_val
                .map(|(_, l)| l.to_string())
                .unwrap_or_else(|| "n/a".into()),
        )
    }

    // ── Error measures ────────────────────────────────────────────────────────

    /// Measures where mean onset timing error exceeds `ONSET_TIMING_ERR_THRESHOLD`.
    fn calc_rhythm_err_measures(measures: &[MeasureData]) -> Vec<u32> {
        measures
            .iter()
            .filter(|m| {
                let errors: Vec<f64> = m
                    .expected_notes
                    .iter()
                    .filter_map(|expected| {
                        Self::closest_onset_within_window(&m.onsets, expected.beat_position)
                            .map(|o| (o.beat_position - expected.beat_position).abs())
                    })
                    .collect();
                if errors.is_empty() {
                    return false;
                }
                errors.iter().sum::<f64>() / errors.len() as f64 > ONSET_TIMING_ERR_THRESHOLD
            })
            .map(|m| m.measure_index)
            .collect()
    }

    /// Measures where note accuracy falls below `ACCURACY_ERR_THRESHOLD`.
    fn calc_note_err_measures(measures: &[MeasureData]) -> Vec<u32> {
        measures
            .iter()
            .filter(|m| {
                let total = m.expected_notes.len();
                if total == 0 {
                    return false;
                }
                let matched = m
                    .expected_notes
                    .iter()
                    .enumerate()
                    .filter(|(ei, _)| {
                        Self::note_is_matched(&m.notes, &m.expected_notes, *ei, NOTE_MATCH_WINDOW)
                    })
                    .count();
                (matched as f64 / total as f64) < ACCURACY_ERR_THRESHOLD
            })
            .map(|m| m.measure_index)
            .collect()
    }

    /// Measures where mean cent deviation exceeds `INTONATION_ERR_THRESHOLD`.
    fn calc_intonation_err_measures(measures: &[MeasureData]) -> Vec<u32> {
        measures
            .iter()
            .filter(|m| {
                if m.notes.is_empty() {
                    return false;
                }
                let avg =
                    m.notes.iter().map(|n| n.avg_cents.abs()).sum::<f64>() / m.notes.len() as f64;
                avg > INTONATION_ERR_THRESHOLD
            })
            .map(|m| m.measure_index)
            .collect()
    }

    /// Measures where dynamics accuracy falls below `DYNAMICS_ERR_THRESHOLD`.
    fn calc_dynamics_err_measures(measures: &[MeasureData]) -> Vec<u32> {
        measures
            .iter()
            .filter(|m| {
                let with_dynamic: Vec<_> = m
                    .expected_notes
                    .iter()
                    .filter(|n| n.dynamic.is_some())
                    .collect();
                if with_dynamic.is_empty() {
                    return false;
                }
                let correct = with_dynamic
                    .iter()
                    .filter(|expected| {
                        let exp_dyn = expected.dynamic.unwrap();
                        Self::actual_dynamic_at(&m.dynamics, expected.beat_position)
                            .map(|act| {
                                (Self::dynamic_to_i32(act) - Self::dynamic_to_i32(exp_dyn)).abs()
                                    <= 1
                            })
                            .unwrap_or(false)
                    })
                    .count();
                (correct as f64 / with_dynamic.len() as f64) < DYNAMICS_ERR_THRESHOLD
            })
            .map(|m| m.measure_index)
            .collect()
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Returns `true` when a detected note constitutes a valid match for `expected_notes[ei]`.
    ///
    /// Two kinds of match are accepted:
    /// * **Exact** — detected MIDI equals `expected[ei].midi_note` and beat is within `window`.
    /// * **Timing-shifted neighbor** — detected MIDI equals `expected[ei-1].midi_note` or
    ///   `expected[ei+1].midi_note` and beat is within `window`.  This means the player
    ///   played the correct note from the sequence but one slot too early or too late —
    ///   a timing error rather than a wrong-note error.
    fn note_is_matched(
        notes: &[NoteEvent],
        expected_notes: &[ExpectedNote],
        ei: usize,
        window: f64,
    ) -> bool {
        let exp_beat = expected_notes[ei].beat_position;
        let exact_midi = expected_notes[ei].midi_note;
        let prev_midi = if ei > 0 { Some(expected_notes[ei - 1].midi_note) } else { None };
        let next_midi = expected_notes.get(ei + 1).map(|e| e.midi_note);

        notes.iter().any(|n| {
            (n.beat_position - exp_beat).abs() < window
                && (n.midi_note == exact_midi
                    || Some(n.midi_note) == prev_midi
                    || Some(n.midi_note) == next_midi)
        })
    }

    /// Returns the closest onset to `target_beat` that falls within `NOTE_MATCH_WINDOW`.
    fn closest_onset_within_window(onsets: &[OnsetEvent], target_beat: f64) -> Option<&OnsetEvent> {
        onsets
            .iter()
            .min_by(|a, b| {
                (a.beat_position - target_beat)
                    .abs()
                    .partial_cmp(&(b.beat_position - target_beat).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .filter(|o| (o.beat_position - target_beat).abs() < NOTE_MATCH_WINDOW)
    }

    /// Most recent dynamic level at or before `beat`, or `None` if no events precede it.
    fn actual_dynamic_at(dynamics: &[DynamicsEvent], beat: f64) -> Option<DynamicLevel> {
        dynamics
            .iter()
            .filter(|d| d.beat_position <= beat)
            .max_by(|a, b| {
                a.beat_position
                    .partial_cmp(&b.beat_position)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|d| d.level)
    }

    /// Map `DynamicLevel` to a signed integer for arithmetic comparison.
    /// `Silence` → -1; `Ppp` → 0; … `Fff` → 7.
    fn dynamic_to_i32(level: DynamicLevel) -> i32 {
        match level {
            DynamicLevel::Silence => -1,
            DynamicLevel::Ppp => 0,
            DynamicLevel::Pp => 1,
            DynamicLevel::P => 2,
            DynamicLevel::Mp => 3,
            DynamicLevel::Mf => 4,
            DynamicLevel::F => 5,
            DynamicLevel::Ff => 6,
            DynamicLevel::Fff => 7,
        }
    }

    /// Population standard deviation of a slice of `f64` values.
    fn std_dev_f64(values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        variance.sqrt()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_io::timing::OnsetEvent;

    // ── Builders ──────────────────────────────────────────────────────────────

    fn onset(beat: f64) -> OnsetEvent {
        OnsetEvent {
            beat_position: beat,
            raw_sample_offset: 0,
            velocity: 0.8,
        }
    }

    fn note_event(beat: f64, midi: u8, cents: f64) -> NoteEvent {
        NoteEvent {
            beat_position: beat,
            midi_note: midi,
            avg_cents: cents,
        }
    }

    fn expected(beat: f64, midi: u8, dur: f64) -> ExpectedNote {
        ExpectedNote {
            beat_position: beat,
            duration_beats: dur,
            midi_note: midi,
            dynamic: None,
        }
    }

    fn expected_with_dyn(beat: f64, midi: u8, dur: f64, level: DynamicLevel) -> ExpectedNote {
        ExpectedNote {
            beat_position: beat,
            duration_beats: dur,
            midi_note: midi,
            dynamic: Some(level),
        }
    }

    fn dyn_event(beat: f64, level: DynamicLevel) -> DynamicsEvent {
        DynamicsEvent {
            beat_position: beat,
            level,
        }
    }

    // ── accuracy_percent ──────────────────────────────────────────────────────

    #[test]
    fn accuracy_percent_all_notes_matched_gives_100() {
        let measures = [MeasureData {
            measure_index: 0,
            onsets: vec![onset(0.0), onset(1.0)],
            notes: vec![note_event(0.0, 60, 0.0), note_event(1.0, 64, 0.0)],
            dynamics: vec![],
            expected_notes: vec![expected(0.0, 60, 1.0), expected(1.0, 64, 1.0)],
        }];
        let pct = Metrics::calc_accuracy_percent(&measures);
        assert!((pct - 100.0).abs() < 1e-9, "expected 100%, got {pct}");
    }

    #[test]
    fn accuracy_percent_no_notes_detected_gives_0() {
        let measures = [MeasureData {
            measure_index: 0,
            onsets: vec![],
            notes: vec![],
            dynamics: vec![],
            expected_notes: vec![expected(0.0, 60, 1.0), expected(1.0, 64, 1.0)],
        }];
        let pct = Metrics::calc_accuracy_percent(&measures);
        assert!((pct - 0.0).abs() < 1e-9, "expected 0%, got {pct}");
    }

    #[test]
    fn accuracy_percent_no_expected_notes_gives_100() {
        // Edge case: no reference notes → nothing to miss → 100%.
        let measures = [MeasureData {
            measure_index: 0,
            onsets: vec![],
            notes: vec![],
            dynamics: vec![],
            expected_notes: vec![],
        }];
        let pct = Metrics::calc_accuracy_percent(&measures);
        assert!((pct - 100.0).abs() < 1e-9, "expected 100%, got {pct}");
    }

    #[test]
    fn accuracy_percent_half_matched_gives_50() {
        let measures = [MeasureData {
            measure_index: 0,
            onsets: vec![onset(0.0)],
            notes: vec![note_event(0.0, 60, 0.0)], // only first note matched
            dynamics: vec![],
            expected_notes: vec![expected(0.0, 60, 1.0), expected(1.0, 64, 1.0)],
        }];
        let pct = Metrics::calc_accuracy_percent(&measures);
        assert!((pct - 50.0).abs() < 1e-9, "expected 50%, got {pct}");
    }

    // ── num_notes_missed ──────────────────────────────────────────────────────

    #[test]
    fn num_notes_missed_zero_when_all_matched() {
        let measures = [MeasureData {
            measure_index: 0,
            onsets: vec![],
            notes: vec![note_event(0.0, 60, 0.0)],
            dynamics: vec![],
            expected_notes: vec![expected(0.0, 60, 1.0)],
        }];
        assert_eq!(Metrics::calc_num_notes_missed(&measures), 0);
    }

    #[test]
    fn num_notes_missed_counts_unmatched_expected_notes() {
        let measures = [MeasureData {
            measure_index: 0,
            onsets: vec![],
            notes: vec![],
            dynamics: vec![],
            expected_notes: vec![expected(0.0, 60, 1.0), expected(1.0, 64, 1.0)],
        }];
        assert_eq!(Metrics::calc_num_notes_missed(&measures), 2);
    }

    // ── avg_cent_dev ──────────────────────────────────────────────────────────

    #[test]
    fn avg_cent_dev_zero_when_perfectly_tuned() {
        let measures = [MeasureData {
            measure_index: 0,
            onsets: vec![],
            notes: vec![note_event(0.0, 60, 0.0), note_event(1.0, 64, 0.0)],
            dynamics: vec![],
            expected_notes: vec![],
        }];
        let dev = Metrics::calc_avg_cent_dev(&measures);
        assert!((dev - 0.0).abs() < 1e-9, "expected 0 cents dev, got {dev}");
    }

    #[test]
    fn avg_cent_dev_computes_mean_absolute_deviation() {
        // Notes deviate by 10 and 30 cents → mean abs = 20.
        let measures = [MeasureData {
            measure_index: 0,
            onsets: vec![],
            notes: vec![note_event(0.0, 60, 10.0), note_event(1.0, 64, -30.0)],
            dynamics: vec![],
            expected_notes: vec![],
        }];
        let dev = Metrics::calc_avg_cent_dev(&measures);
        assert!((dev - 20.0).abs() < 1e-9, "expected 20 cents avg dev, got {dev}");
    }

    #[test]
    fn avg_cent_dev_zero_when_no_notes() {
        let measures = [MeasureData {
            measure_index: 0,
            onsets: vec![],
            notes: vec![],
            dynamics: vec![],
            expected_notes: vec![],
        }];
        assert!((Metrics::calc_avg_cent_dev(&measures) - 0.0).abs() < 1e-9);
    }

    // ── timing_consistency ────────────────────────────────────────────────────

    #[test]
    fn timing_consistency_zero_when_perfectly_on_beat() {
        let measures = [MeasureData {
            measure_index: 0,
            onsets: vec![onset(0.0), onset(1.0)],
            notes: vec![],
            dynamics: vec![],
            expected_notes: vec![expected(0.0, 60, 1.0), expected(1.0, 64, 1.0)],
        }];
        let tc = Metrics::calc_timing_consistency(&measures);
        assert!(tc < 1e-9, "expected timing consistency ~0, got {tc}");
    }

    // ── microtiming_skew ──────────────────────────────────────────────────────

    #[test]
    fn microtiming_skew_positive_when_rushing() {
        // Onsets are 0.1 beats early (beat_position is lower than expected).
        // Wait — onset before expected → onset.beat_position < expected.beat_position
        // skew = onset - expected → negative when dragging, positive when rushing.
        // Rushing = playing early → onset before beat → negative skew value.
        // Let's test rushing: onsets at 0.9 and 1.9 (0.1 beats before expected 1.0 and 2.0).
        let measures = [MeasureData {
            measure_index: 0,
            onsets: vec![onset(0.9), onset(1.9)],
            notes: vec![],
            dynamics: vec![],
            expected_notes: vec![expected(1.0, 60, 1.0), expected(2.0, 64, 1.0)],
        }];
        let skew = Metrics::calc_microtiming_skew(&measures);
        // Skew = mean of (onset - expected) = mean of (-0.1, -0.1) = -0.1
        assert!(
            (skew - (-0.1)).abs() < 1e-9,
            "rushing should give skew ≈ -0.1, got {skew}"
        );
    }

    #[test]
    fn microtiming_skew_zero_when_no_onsets_in_window() {
        let measures = [MeasureData {
            measure_index: 0,
            onsets: vec![],
            notes: vec![],
            dynamics: vec![],
            expected_notes: vec![expected(1.0, 60, 1.0)],
        }];
        let skew = Metrics::calc_microtiming_skew(&measures);
        assert!((skew - 0.0).abs() < 1e-9, "no data should give skew = 0, got {skew}");
    }

    // ── dynamics_accuracy ─────────────────────────────────────────────────────

    #[test]
    fn dynamics_accuracy_100_when_levels_match_within_one_step() {
        let measures = [MeasureData {
            measure_index: 0,
            onsets: vec![],
            notes: vec![],
            dynamics: vec![dyn_event(0.0, DynamicLevel::Mf)],
            expected_notes: vec![expected_with_dyn(0.5, 60, 1.0, DynamicLevel::Mf)],
        }];
        let acc = Metrics::calc_dynamics_accuracy(&measures);
        assert!((acc - 100.0).abs() < 1e-9, "exact match should give 100%, got {acc}");
    }

    #[test]
    fn dynamics_accuracy_0_when_levels_differ_by_more_than_one_step() {
        let measures = [MeasureData {
            measure_index: 0,
            onsets: vec![],
            notes: vec![],
            dynamics: vec![dyn_event(0.0, DynamicLevel::Ppp)],
            // Expected Fff (7 steps away from Ppp=0)
            expected_notes: vec![expected_with_dyn(0.5, 60, 1.0, DynamicLevel::Fff)],
        }];
        let acc = Metrics::calc_dynamics_accuracy(&measures);
        assert!((acc - 0.0).abs() < 1e-9, "large mismatch should give 0%, got {acc}");
    }

    // ── tempo_stability ───────────────────────────────────────────────────────

    #[test]
    fn tempo_stability_is_one_when_all_measures_at_same_tempo() {
        let map = vec![120.0_f64, 120.0, 120.0, 120.0];
        let stability = Metrics::calc_tempo_stability_from_map(&map, 120.0);
        assert!(
            (stability - 1.0).abs() < 1e-9,
            "constant tempo should give stability = 1.0, got {stability}"
        );
    }

    #[test]
    fn tempo_stability_less_than_one_with_variance() {
        let map = vec![100.0_f64, 140.0, 100.0, 140.0]; // wildly inconsistent
        let stability = Metrics::calc_tempo_stability_from_map(&map, 120.0);
        assert!(
            stability < 1.0,
            "variable tempo should give stability < 1.0, got {stability}"
        );
        assert!(stability >= 0.0, "stability must be non-negative, got {stability}");
    }

    // ── std_dev_f64 ───────────────────────────────────────────────────────────

    #[test]
    fn std_dev_f64_zero_for_single_value() {
        assert!((Metrics::std_dev_f64(&[5.0]) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn std_dev_f64_zero_for_identical_values() {
        assert!((Metrics::std_dev_f64(&[3.0, 3.0, 3.0]) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn std_dev_f64_correct_for_known_data() {
        // Population std dev of [2, 4, 4, 4, 5, 5, 7, 9] = 2.0
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let sd = Metrics::std_dev_f64(&data);
        assert!((sd - 2.0).abs() < 1e-6, "expected std dev = 2.0, got {sd}");
    }

    // ── Metrics::compute integration ──────────────────────────────────────────

    #[test]
    fn metrics_compute_perfect_performance() {
        let measures = [MeasureData {
            measure_index: 0,
            onsets: vec![onset(0.0), onset(1.0), onset(2.0), onset(3.0)],
            notes: vec![
                note_event(0.0, 60, 0.0),
                note_event(1.0, 62, 0.0),
                note_event(2.0, 64, 0.0),
                note_event(3.0, 65, 0.0),
            ],
            dynamics: vec![],
            expected_notes: vec![
                expected(0.0, 60, 1.0),
                expected(1.0, 62, 1.0),
                expected(2.0, 64, 1.0),
                expected(3.0, 65, 1.0),
            ],
        }];
        let metrics = Metrics::compute(0, 0, 120.0, &measures);
        assert!((metrics.accuracy_percent - 100.0).abs() < 1e-9);
        assert_eq!(metrics.num_notes_missed, 0);
        assert!((metrics.avg_cent_dev - 0.0).abs() < 1e-9);
    }

    #[test]
    fn metrics_compute_all_notes_missed() {
        let measures = [MeasureData {
            measure_index: 0,
            onsets: vec![],
            notes: vec![],
            dynamics: vec![],
            expected_notes: vec![expected(0.0, 60, 1.0), expected(1.0, 64, 1.0)],
        }];
        let metrics = Metrics::compute(0, 0, 120.0, &measures);
        assert!((metrics.accuracy_percent - 0.0).abs() < 1e-9);
        assert_eq!(metrics.num_notes_missed, 2);
    }
}
