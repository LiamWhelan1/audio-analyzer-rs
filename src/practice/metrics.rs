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
/// Relative tolerance for articulation duration matching (fraction of expected duration).
pub(crate) const ARTICULATION_TOLERANCE: f64 = 0.20;
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
    /// Percentage (0–100) of matched notes whose actual duration is within
    /// 20% of the MIDI reference duration.
    pub articulation_accuracy: f64,
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
    /// * `beats_per_measure` — time signature numerator (e.g. 4 for 4/4).
    /// * `measures` — per-measure data from the three live trackers and MIDI reference.
    pub fn compute(
        start_measure: u32,
        end_measure: u32,
        tempo_bpm: f64,
        beats_per_measure: u32,
        measures: &[MeasureData],
    ) -> Self {
        let num_measures = end_measure.saturating_sub(start_measure) + 1;

        let accuracy_percent = Self::calc_accuracy_percent(measures);
        let avg_cent_dev = Self::calc_avg_cent_dev(measures);
        let num_notes_missed = Self::calc_num_notes_missed(measures);
        let articulation_accuracy = Self::calc_articulation_accuracy(measures, beats_per_measure);
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
            articulation_accuracy,
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
            for expected in &m.expected_notes {
                total += 1;
                if m.notes.iter().any(|n| {
                    n.midi_note == expected.midi_note
                        && (n.beat_position - expected.beat_position).abs() < NOTE_MATCH_WINDOW
                }) {
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
            for expected in &m.expected_notes {
                if !m.notes.iter().any(|n| {
                    n.midi_note == expected.midi_note
                        && (n.beat_position - expected.beat_position).abs() < NOTE_MATCH_WINDOW
                }) {
                    missed += 1;
                }
            }
        }
        missed
    }

    // ── Articulation ──────────────────────────────────────────────────────────

    /// Percentage of matched notes whose actual duration (time to the next `NoteEvent`
    /// within the measure) is within `ARTICULATION_TOLERANCE` of the MIDI reference duration.
    fn calc_articulation_accuracy(measures: &[MeasureData], beats_per_measure: u32) -> f64 {
        let mut total_matched = 0usize;
        let mut articulation_ok = 0usize;

        for m in measures {
            let mut sorted_notes: Vec<&NoteEvent> = m.notes.iter().collect();
            sorted_notes.sort_by(|a, b| {
                a.beat_position
                    .partial_cmp(&b.beat_position)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let measure_end_beat = (m.measure_index + 1) as f64 * beats_per_measure as f64;

            for expected in &m.expected_notes {
                let Some(idx) = sorted_notes.iter().position(|n| {
                    n.midi_note == expected.midi_note
                        && (n.beat_position - expected.beat_position).abs() < NOTE_MATCH_WINDOW
                }) else {
                    continue;
                };

                total_matched += 1;

                let actual_duration = if idx + 1 < sorted_notes.len() {
                    sorted_notes[idx + 1].beat_position - sorted_notes[idx].beat_position
                } else {
                    measure_end_beat - sorted_notes[idx].beat_position
                };

                let ratio = (actual_duration - expected.duration_beats).abs()
                    / expected.duration_beats.max(1e-6);
                if ratio < ARTICULATION_TOLERANCE {
                    articulation_ok += 1;
                }
            }
        }

        if total_matched == 0 {
            return 0.0;
        }
        articulation_ok as f64 / total_matched as f64 * 100.0
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
                let mut actual_beats: Vec<f64> = m.onsets.iter().map(|o| o.beat_position).collect();
                actual_beats.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                let mut expected_beats: Vec<f64> =
                    m.expected_notes.iter().map(|n| n.beat_position).collect();
                expected_beats
                    .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                if actual_beats.len() < 2 || expected_beats.len() < 2 {
                    return tempo_bpm;
                }

                let actual_span = actual_beats.last().unwrap() - actual_beats.first().unwrap();
                let expected_span =
                    expected_beats.last().unwrap() - expected_beats.first().unwrap();

                if actual_span < 1e-6 || expected_span < 1e-6 {
                    return tempo_bpm;
                }

                (tempo_bpm as f64 * expected_span / actual_span) as f64
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
                    .filter(|expected| {
                        m.notes.iter().any(|n| {
                            n.midi_note == expected.midi_note
                                && (n.beat_position - expected.beat_position).abs()
                                    < NOTE_MATCH_WINDOW
                        })
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
