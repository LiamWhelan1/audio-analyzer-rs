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
                    let move_tuner = Arc::downgrade(&_tuner);
                    std::thread::spawn(move || {
                        loop {
                            if let Some(tuner) = move_tuner.upgrade() {
                                let response = tuner.poll_output();
                                print!("\x1B[2K\rTuner output: {}", response);
                                io::stdout().flush().unwrap();
                            } else {
                                break;
                            }
                            std::thread::sleep(std::time::Duration::from_millis(100));
                        }
                    });
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
