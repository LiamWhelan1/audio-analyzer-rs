use std::time::Duration;

use crate::AudioEngine;
use std::io::{self, Write};

/// Implementation of the UniFFI callback interface for the Tuner.
/// In a real React Native app, this interface would be implemented in Swift or Kotlin.

#[cfg(test)]
mod ffi_tests {
    use super::*;

    #[test]
    fn test_engine_lifecycle() {
        // 1. Create the engine (constructor returns Result<Arc<AudioEngine>>)
        let engine = AudioEngine::new().expect("engine init failed");

        // 2. Start streams
        assert!(engine.start_output().is_ok());

        // 3. Create a worker
        let metronome = engine.create_metronome(120.0).expect("metronome failed");

        // 4. Test methods directly on the object
        assert!(metronome.set_bpm(140.0));
        assert!(metronome.set_volume(0.8));

        // 5. Cleanup is automatic when the Arc goes out of scope
        engine.stop_metronome();
    }

    // ── AudioEngine create/stop guards ────────────────────────────────────────

    #[test]
    fn create_metronome_twice_returns_clear_error() {
        let engine = AudioEngine::new().expect("engine init failed");
        engine.start_output().expect("output stream failed");
        let _ = engine
            .create_metronome(120.0)
            .expect("first metronome failed");

        let result = engine.create_metronome(120.0);
        assert!(result.is_err(), "second create_metronome should fail");
        let err = result.err().unwrap();

        match err {
            crate::AudioEngineError::SpawnFailed { component, msg } => {
                assert_eq!(component, "metronome");
                assert!(
                    msg.to_lowercase().contains("already"),
                    "error message should say 'already active', got: {msg}"
                );
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
        engine.stop_metronome();
    }

    #[test]
    fn stop_metronome_allows_re_creation() {
        let engine = AudioEngine::new().expect("engine init failed");
        engine.start_output().expect("output stream failed");
        engine.create_metronome(120.0).expect("first create failed");
        engine.stop_metronome();
        // Should succeed after stopping
        engine
            .create_metronome(90.0)
            .expect("re-creation after stop failed");
        engine.stop_metronome();
    }

    // ── Metronome pattern int mapping ─────────────────────────────────────────

    #[test]
    fn metronome_set_pattern_accepts_valid_strength_ints() {
        let engine = AudioEngine::new().expect("engine init failed");
        engine.start_output().expect("output stream failed");
        let metronome = engine.create_metronome(120.0).expect("metronome failed");

        // 3=Strong, 2=Medium, 1=Weak, 0=None; unknown values fall back to None
        assert!(metronome.set_pattern(vec![3, 2, 1, 0]));
        assert!(metronome.set_pattern(vec![3, 1, 1, 1])); // typical 4/4
        assert!(metronome.set_pattern(vec![3, 1, 1])); // 3/4
        // Unknown values should not panic — they map to None
        assert!(metronome.set_pattern(vec![99, -1, 42]));

        engine.stop_metronome();
    }

    // ── Tuner mode / system string mapping ───────────────────────────────────

    #[test]
    fn tuner_set_mode_known_and_unknown_strings() {
        let engine = AudioEngine::new().expect("engine init failed");
        engine.start_input().expect("input stream failed");
        let tuner = engine.start_tuner().expect("tuner failed");

        // Known values — must not panic or error
        tuner.set_mode("SinglePitch".to_string());
        tuner.set_mode("MultiPitch".to_string());
        // Unknown value — must silently fall back (MultiPitch), must not panic
        tuner.set_mode("FooBarMode".to_string());

        engine.stop_tuner();
    }

    #[test]
    fn tuner_set_system_known_and_unknown_strings() {
        let engine = AudioEngine::new().expect("engine init failed");
        engine.start_input().expect("input stream failed");
        let tuner = engine.start_tuner().expect("tuner failed");

        tuner.set_system("JustIntonation".to_string());
        tuner.set_system("EqualTemperament".to_string());
        // Unknown value must silently fall back, must not panic
        tuner.set_system("WeirdSystem".to_string());

        engine.stop_tuner();
    }

    #[test]
    fn tuner_poll_output_returns_valid_json_object() {
        let engine = AudioEngine::new().expect("engine init failed");
        engine.start_input().expect("input stream failed");
        let tuner = engine.start_tuner().expect("tuner failed");

        let json = tuner.poll_output();
        let parsed: serde_json::Value =
            serde_json::from_str(&json).expect("poll_output must return valid JSON");

        assert!(
            parsed.is_object(),
            "poll_output must return a JSON object, got: {json}"
        );

        // Required keys present in TunerOutput
        for key in &[
            "label",
            "cents",
            "notes",
            "accuracies",
            "mode",
            "system",
            "base_freq",
            "key",
        ] {
            assert!(
                parsed.get(key).is_some(),
                "poll_output JSON missing key '{key}', got: {json}"
            );
        }

        engine.stop_tuner();
    }

    // ── Synth instrument mapping ───────────────────────────────────────────────

    #[test]
    fn synth_load_file_unknown_instrument_falls_back_to_violin() {
        let engine = AudioEngine::new().expect("engine init failed");
        engine.start_output().expect("output stream failed");
        let synth = engine.create_synth().expect("synth failed");

        // Unknown instrument string must not panic — falls back to Violin
        // (file won't exist, returns false — but must not panic)
        let _ = synth.load_file(
            "/nonexistent.mid".to_string(),
            "UnknownInstrument".to_string(),
        );

        engine.stop_synth();
    }

    #[test]
    fn synth_play_note_zero_velocity_does_not_panic() {
        let engine = AudioEngine::new().expect("engine init failed");
        engine.start_output().expect("output stream failed");
        let synth = engine.create_synth().expect("synth failed");

        // velocity = 0 → NoteOff path; must not panic
        assert!(synth.play_note(440.0, 0.0, "Piano".to_string()));

        engine.stop_synth();
    }

    #[test]
    fn synth_play_note_positive_velocity_does_not_panic() {
        let engine = AudioEngine::new().expect("engine init failed");
        engine.start_output().expect("output stream failed");
        let synth = engine.create_synth().expect("synth failed");

        assert!(synth.play_note(440.0, 64.0, "Piano".to_string()));
        assert!(synth.play_note(440.0, 64.0, "Violin".to_string()));
        // Unknown instrument falls back, must not panic
        assert!(synth.play_note(440.0, 64.0, "Xylophone".to_string()));

        engine.stop_synth();
    }

    // ── poll_dynamics JSON format ──────────────────────────────────────────────

    #[test]
    fn poll_dynamics_returns_valid_json_object_with_required_fields() {
        let engine = AudioEngine::new().expect("engine init failed");
        engine.start_input().expect("input stream failed");

        let json = engine.poll_dynamics();
        let parsed: serde_json::Value =
            serde_json::from_str(&json).expect("poll_dynamics must return valid JSON");

        assert!(
            parsed.is_object(),
            "poll_dynamics must return a JSON object"
        );
        for key in &[
            "level",
            "rms_db",
            "gain_db",
            "session_median_db",
            "noise_floor_db",
        ] {
            assert!(
                parsed.get(key).is_some(),
                "poll_dynamics JSON missing key '{key}'"
            );
        }
    }

    // ── poll_transport JSON format ─────────────────────────────────────────────

    #[test]
    fn poll_transport_returns_valid_json_object_with_required_fields() {
        let engine = AudioEngine::new().expect("engine init failed");
        engine.start_output().expect("output stream failed");

        let json = engine.poll_transport();
        let parsed: serde_json::Value =
            serde_json::from_str(&json).expect("poll_transport must return valid JSON");

        assert!(
            parsed.is_object(),
            "poll_transport must return a JSON object"
        );
        for key in &[
            "beat_position",
            "bpm",
            "is_playing",
            "display_beat_position",
            "current_beat",
            "beat_phase",
            "output_frames",
            "input_frames",
            "drift_samples",
        ] {
            assert!(
                parsed.get(key).is_some(),
                "poll_transport JSON missing key '{key}'"
            );
        }
    }

    // ── PracticeSession::start error messages ─────────────────────────────────

    #[test]
    fn practice_session_start_inverted_range_returns_internal_error_with_clear_message() {
        // We need a PracticeSession to test start() errors.
        // Use a real MIDI file path that exists in the repo for testing.
        // If no test MIDI file exists, we test via the engine route.
        // The error should come from start() before any audio is needed.
        // We can only test this if create_practice_session succeeds.
        // Skip the test gracefully if no MIDI file is available.
        let engine = AudioEngine::new().expect("engine init failed");
        engine.start_input().expect("input stream failed");
        engine.start_output().expect("output stream failed");

        // Try to load a test MIDI file; skip if unavailable.
        let test_midi = std::env::var("TEST_MIDI_PATH").unwrap_or_default();
        if test_midi.is_empty() {
            return; // no MIDI available in this environment
        }

        let session = engine
            .create_practice_session(
                test_midi,
                "Violin".to_string(),
                0,
                false,
                "Advanced".to_string(),
                120.0,
            )
            .expect("practice session creation failed");

        // start_measure > end_measure → should return Internal error with clear message
        let start_result = session.start(5, 2);
        assert!(
            start_result.is_err(),
            "inverted range should return an error"
        );
        let err = start_result.err().unwrap();

        match err {
            crate::AudioEngineError::Internal { msg } => {
                assert!(
                    msg.contains("start_measure")
                        || msg.contains("end_measure")
                        || msg.contains(">"),
                    "error message should describe the range inversion, got: {msg}"
                );
            }
            other => panic!("expected Internal error, got: {other:?}"),
        }
    }
}

/// Interactive CLI simulation for testing the new Object-Oriented API.
pub fn run_cli_simulation() {
    println!("\nCadenza CLI Simulation Layer (UniFFI Mode)");
    println!("Initializing Audio Engine...");

    // Initialize the engine
    let engine = AudioEngine::new().expect("Failed to initialize audio engine");

    println!(concat!(
        "Commands:\n",
        "  met start|stop|bpm   tuner start|stop   synth start|note|stop\n",
        "  player start|load|play|stop   rec start|stop   onset start|stop\n",
        "  practice start        -- guided setup, 2-measure count-off, normal timing\n",
        "  practice start wait   -- same but waits for your onset before advancing\n",
        "  practice stop         -- abort an active session\n",
        "  practice metrics      -- print metrics for the current session\n",
        "  poll dynamics   exit\n",
    ));

    loop {
        print!("\n> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            continue;
        }
        let command = input.trim().to_lowercase();

        match command.as_str() {
            // --- Metronome ---
            "met start" => {
                let _ = engine.create_metronome(120.0);
                println!("Metronome created at 120 BPM.");
            }
            "met stop" => {
                engine.stop_metronome();
                println!("Metronome dropped.");
            }
            "met bpm" => {
                if let Some(m) = engine.active_metronome.lock().unwrap().as_ref() {
                    print!("New BPM: ");
                    io::stdout().flush().unwrap();
                    let mut b = String::new();
                    io::stdin().read_line(&mut b).ok();
                    if let Ok(val) = b.trim().parse::<f32>() {
                        m.set_bpm(val);
                    }
                }
            }
            "poll dynamics" => {
                let engine_dyn = engine.clone();
                std::thread::spawn(move || {
                    loop {
                        let response = engine_dyn.poll_dynamics();
                        println!("{}", response);
                        std::thread::sleep(Duration::from_millis(1000));
                    }
                });
            }
            // --- Synth ---
            "synth start" => {
                let _ = engine.create_synth();
                println!("Synth created.");
            }
            "synth note" => {
                if let Some(s) = engine.active_synth.lock().unwrap().as_ref() {
                    // Logic for a middle C (261.63 Hz) at max velocity
                    s.play_note(261.63, 127.0, "Violin".into());
                    println!("Playing Middle C...");
                }
            }
            "synth stop" => {
                engine.stop_synth();
                println!("Synth dropped.");
            }

            // --- Player ---
            "player start" => {
                let _ = engine.create_player();
                println!("Player created.");
            }
            "player load" => {
                if let Some(p) = engine.active_player.lock().unwrap().as_ref() {
                    print!("File path: ");
                    io::stdout().flush().unwrap();
                    let mut path = String::new();
                    io::stdin().read_line(&mut path).ok();
                    if let Err(e) = p.load_track(path.trim().to_string()) {
                        println!("Load error: {e}");
                    } else {
                        p.play();
                    }
                }
            }
            "player play" => {
                if let Some(p) = engine.active_player.lock().unwrap().as_ref() {
                    p.play();
                }
            }
            "player stop" => {
                engine.stop_player();
                println!("Player dropped.");
            }

            // --- Recording & Analysis ---
            "rec start" => match engine.start_recording("test_output.wav".to_string()) {
                Ok(_) => println!("Recording to test_output.wav"),
                Err(e) => println!("Recording error: {e}"),
            },
            "rec stop" => {
                engine.stop_recording();
                println!("Rec dropped.");
            }
            "onset start" => match engine.start_onset_detection() {
                Ok(_) => println!("Onset detection active."),
                Err(e) => println!("Onset error: {e}"),
            },
            "onset stop" => {
                engine.stop_onset_detection();
                println!("Onset dropped.");
            }
            "tuner start" => {
                if let Ok(_tuner) = engine.start_tuner() {
                    // let move_tuner = Arc::downgrade(&_tuner);
                    // std::thread::spawn(move || {
                    //     loop {
                    //         if let Some(tuner) = move_tuner.upgrade() {
                    //             let response = tuner.poll_output();
                    //             print!("\x1B[2K\rTuner output: {}", response);
                    //             io::stdout().flush().unwrap();
                    //         } else {
                    //             break;
                    //         }
                    //         std::thread::sleep(std::time::Duration::from_millis(100));
                    //     }
                    // });
                }
            }
            "tuner mode: single" => {
                if let Some(tuner) = engine.active_tuner.lock().unwrap().as_ref() {
                    tuner.set_mode("SinglePitch".to_string());
                }
            }
            "tuner mode: multi" => {
                if let Some(tuner) = engine.active_tuner.lock().unwrap().as_ref() {
                    tuner.set_mode("MultiPitch".to_string());
                }
            }
            "tuner stop" => {
                engine.stop_tuner();
                println!("Tuner dropped.");
            }

            // ── Practice session ──────────────────────────────────────────────
            cmd @ ("practice start" | "practice start wait") => {
                if engine.active_practice_session.lock().unwrap().is_some() {
                    println!("A practice session is already active. Use 'practice stop' first.");
                    continue;
                }

                let wait_for_onset = cmd == "practice start wait";

                // --- Gather inputs interactively ---
                macro_rules! prompt {
                    ($msg:expr) => {{
                        print!($msg);
                        io::stdout().flush().unwrap();
                        let mut buf = String::new();
                        io::stdin().read_line(&mut buf).ok();
                        buf.trim().to_string()
                    }};
                }

                let midi_path = prompt!("MIDI file path: ");
                if midi_path.is_empty() {
                    println!("Cancelled.");
                    continue;
                }

                let instrument = {
                    let s = prompt!("Instrument [Piano/Violin, default=Piano]: ");
                    if s.is_empty() { "Piano".to_string() } else { s }
                };

                let ability_level = {
                    let s = prompt!(
                        "Ability level [Beginner/Intermediate/Advanced/Expert, default=Advanced]: "
                    );
                    if s.is_empty() {
                        "Advanced".to_string()
                    } else {
                        s
                    }
                };

                let start_measure: u32 = {
                    let s = prompt!("Start measure (0-indexed, default=0): ");
                    s.parse().unwrap_or(0)
                };

                let end_measure: u32 = {
                    let s = prompt!("End measure (0-indexed): ");
                    match s.parse() {
                        Ok(v) => v,
                        Err(_) => {
                            println!("Invalid end measure — must be a number.");
                            continue;
                        }
                    }
                };

                let countoff_beats: u32 = {
                    let s = prompt!("Count-off beats [default=8, i.e. 2 measures of 4/4]: ");
                    s.parse().unwrap_or(8)
                };

                let bpm: f32 = {
                    let s = prompt!("BPM [default=120.0]: ");
                    s.parse().unwrap_or(120.0)
                };

                // --- Ensure streams are running ---
                let _ = engine.start_input();
                let _ = engine.start_output();

                // Create a metronome so the count-off click is audible.
                // If one already exists this is a no-op (returns Err which we ignore).
                let _ = engine.create_metronome(bpm);

                // --- Create the session ---
                let session = match engine.create_practice_session(
                    midi_path,
                    instrument,
                    countoff_beats,
                    wait_for_onset,
                    ability_level.clone(),
                    bpm,
                ) {
                    Ok(s) => s,
                    Err(e) => {
                        println!("Failed to create practice session: {e}");
                        continue;
                    }
                };

                // --- Start the session ---
                if let Err(e) = session.start(start_measure, end_measure) {
                    println!("Failed to start session: {e}");
                    engine.stop_practice_session();
                    continue;
                }

                let mode_label = if wait_for_onset {
                    "onset-wait"
                } else {
                    "normal"
                };
                println!();
                println!("======================================================");
                println!("  Practice Session Started  ({ability_level} / {mode_label})");
                println!(
                    "  Measures {start_measure}–{end_measure}  |  {countoff_beats}-beat count-off"
                );
                println!("  'practice stop' to abort  |  'practice metrics' for stats");
                println!("======================================================");
                println!();

                // --- Background polling thread (mimics the React Native ~60 Hz loop) ---
                let session_ref = session.clone();
                let engine_ref = engine.clone();
                std::thread::spawn(move || {
                    let mut last_measure_idx: Option<u64> = None;
                    // Use i64 so negative beat positions during count-off are represented.
                    let mut last_beat_floor: Option<i64> = None;
                    let mut was_in_countoff = true;
                    let mut total_feedback_events: u32 = 0;

                    // Helper: print a batch of feedback events.
                    let print_feedback = |arr: Vec<serde_json::Value>, count: &mut u32| {
                        for ev in &arr {
                            *count += 1;
                            let measure = ev["measure"].as_u64().unwrap_or(0);
                            let note_idx = ev["note_index"].as_u64().unwrap_or(0);
                            let err_type = ev["error_type"].as_str().unwrap_or("?");
                            let intensity = ev["intensity"].as_f64().unwrap_or(0.0);
                            let expected = ev["expected"].as_str().unwrap_or("?");
                            let received = ev["received"].as_str().unwrap_or("?");

                            if err_type == "None" {
                                continue;
                            }

                            let marker = if intensity > 0.7 {
                                "!!"
                            } else if intensity > 0.3 {
                                "! "
                            } else {
                                "  "
                            };
                            println!(
                                "  | {marker} [{err_type}] m{measure}:n{note_idx}  \
                                 expected={expected}  received={received}  \
                                 intensity={intensity:.2}"
                            );
                        }
                    };

                    while session_ref.is_running() {
                        // ── Transport (~60 Hz) ───────────────────────────────────
                        let transport_str = session_ref.poll_transport();
                        if let Ok(t) = serde_json::from_str::<serde_json::Value>(&transport_str) {
                            let beat_pos = t["beat_position"].as_f64().unwrap_or(0.0);
                            let beat_floor = beat_pos.floor() as i64; // signed: works for negatives
                            let measure = t["current_measure_idx"].as_u64();
                            let bpm = t["bpm"].as_f64().unwrap_or(0.0);
                            let in_countoff = t["in_countoff"].as_bool().unwrap_or(false);

                            if in_countoff {
                                // Log each new count-off beat using signed floor arithmetic.
                                // beat_pos runs from -(countoff_beats+ε) to 0, so:
                                //   display = floor(beat_pos) + countoff_beats + 1
                                // The very first iteration has beat_pos = -countoff_beats-0.001,
                                // giving display=0 (pre-beat-1 sliver) — suppress it by gating
                                // on display >= 1 rather than clamping, which would double-log
                                // beat 1.
                                if Some(beat_floor) != last_beat_floor {
                                    last_beat_floor = Some(beat_floor);
                                    let display = beat_floor + countoff_beats as i64 + 1;
                                    if display >= 1 && display <= countoff_beats as i64 {
                                        println!("  [count-off] beat {display} (BPM: {bpm:.0})");
                                    }
                                }
                                was_in_countoff = true;
                            } else {
                                // Transition from count-off to performance.
                                if was_in_countoff {
                                    was_in_countoff = false;
                                    println!();
                                    println!("  --- Performance started ---");
                                    println!();
                                }

                                // Log measure changes with a header.
                                if measure != last_measure_idx {
                                    last_measure_idx = measure;
                                    last_beat_floor = None; // reset so we log beat 1 of new measure
                                    println!("  +--- Measure {} ---", measure.unwrap_or(0));
                                }

                                // Log each new beat within the measure.
                                if Some(beat_floor) != last_beat_floor {
                                    last_beat_floor = Some(beat_floor);
                                    println!("  |  beat {beat_pos:.3}");
                                }
                            }
                        }

                        // ── Feedback events (~10 Hz equivalent) ─────────────────
                        let errors_str = session_ref.poll_errors();
                        if errors_str != "[]" {
                            if let Ok(arr) =
                                serde_json::from_str::<Vec<serde_json::Value>>(&errors_str)
                            {
                                print_feedback(arr, &mut total_feedback_events);
                            }
                        }

                        std::thread::sleep(Duration::from_millis(16)); // ~60 Hz
                    }

                    // ── Drain any feedback generated for the last measure ────────
                    // The session sets running=false and breaks immediately after pushing
                    // the final measure's feedback.  The polling loop may exit before
                    // seeing that last poll_errors(), so we do one final drain here.
                    let errors_str = session_ref.poll_errors();
                    if errors_str != "[]" {
                        if let Ok(arr) = serde_json::from_str::<Vec<serde_json::Value>>(&errors_str)
                        {
                            print_feedback(arr, &mut total_feedback_events);
                        }
                    }

                    // ── Session complete ─────────────────────────────────────────
                    println!();
                    println!(
                        "  +--- Session complete ({total_feedback_events} feedback events) ---"
                    );
                    println!();

                    let metrics_str = session_ref.get_metrics();
                    println!("======================================================");
                    println!("  PRACTICE METRICS");
                    println!("======================================================");
                    match serde_json::from_str::<serde_json::Value>(&metrics_str) {
                        Ok(m) if m != serde_json::Value::Object(Default::default()) => {
                            // Print each top-level key on its own line for readability.
                            if let Some(obj) = m.as_object() {
                                // Scalar fields first.
                                let scalars = [
                                    "start_measure",
                                    "end_measure",
                                    "num_measures",
                                    "tempo_bpm",
                                    "accuracy_percent",
                                    "avg_cent_dev",
                                    "num_notes_missed",
                                    "articulation_accuracy",
                                    "timing_consistency",
                                    "note_onset_accuracy",
                                    "microtiming_skew",
                                    "tempo_stability",
                                    "dynamics_accuracy",
                                    "dynamics_consistency",
                                    "avg_errors_per_measure",
                                    "consistency_across_repetitions",
                                ];
                                for key in &scalars {
                                    if let Some(v) = obj.get(*key) {
                                        println!("  {key:<38} {v}");
                                    }
                                }
                                // Array/tuple fields last.
                                let arrays = [
                                    "error_measures",
                                    "rhythm_err_measures",
                                    "note_err_measures",
                                    "intonation_err_measures",
                                    "dynamics_err_measures",
                                    "measure_tempo_map",
                                    "dynamics_range_used",
                                ];
                                for key in &arrays {
                                    if let Some(v) = obj.get(*key) {
                                        println!("  {key:<38} {v}");
                                    }
                                }
                            }
                        }
                        _ => println!("  (no metrics available yet)"),
                    }
                    println!("======================================================");
                    println!();

                    // Clean up the engine slot so a new session can be created.
                    engine_ref.stop_practice_session();
                });
            }

            "practice stop" => {
                engine.stop_practice_session();
                println!("Practice session stopped.");
            }

            "practice metrics" => {
                let guard = engine.active_practice_session.lock().unwrap();
                match guard.as_ref() {
                    None => println!("No active practice session."),
                    Some(session) => {
                        let metrics_str = session.get_metrics();
                        match serde_json::from_str::<serde_json::Value>(&metrics_str) {
                            Ok(m) => {
                                let pretty = serde_json::to_string_pretty(&m)
                                    .unwrap_or_else(|_| metrics_str.clone());
                                println!("{pretty}");
                            }
                            Err(_) => println!("No metrics available yet."),
                        }
                    }
                }
            }

            "exit" => {
                println!("Shutting down engine.");
                break;
            }
            _ => println!("Unknown command."),
        }
    }
}
