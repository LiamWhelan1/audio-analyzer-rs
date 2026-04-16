use std::time::Duration;

use crate::AudioEngine;
use std::io::{self, Write};
use std::sync::Arc;

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
        let _ = engine.create_metronome(120.0).expect("first metronome failed");

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
        engine.create_metronome(90.0).expect("re-creation after stop failed");
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
        assert!(metronome.set_pattern(vec![3, 1, 1]));     // 3/4
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

        assert!(parsed.is_object(), "poll_output must return a JSON object, got: {json}");

        // Required keys present in TunerOutput
        for key in &["label", "cents", "notes", "accuracies", "mode", "system", "base_freq", "key"] {
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
        let _ = synth.load_file("/nonexistent.mid".to_string(), "UnknownInstrument".to_string());

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

        assert!(parsed.is_object(), "poll_dynamics must return a JSON object");
        for key in &["level", "rms_db", "gain_db", "session_median_db", "noise_floor_db"] {
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

        assert!(parsed.is_object(), "poll_transport must return a JSON object");
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
            .create_practice_session(test_midi, "Violin".to_string())
            .expect("practice session creation failed");

        // start_measure > end_measure → should return Internal error with clear message
        let start_result = session.start(5, 2);
        assert!(start_result.is_err(), "inverted range should return an error");
        let err = start_result.err().unwrap();

        match err {
            crate::AudioEngineError::Internal { msg } => {
                assert!(
                    msg.contains("start_measure") || msg.contains("end_measure") || msg.contains(">"),
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

    println!(
        "Commands: audio play, audio pause, rec start, rec stop, tuner start, tuner stop, \
         met start, met stop, synth start, synth note, player start, player load, all exit\n"
    );

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

            "exit" => {
                println!("Shutting down engine.");
                break;
            }
            _ => println!("Unknown command."),
        }
    }
}
