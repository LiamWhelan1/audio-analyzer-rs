use std::time::Duration;

use crate::{AudioEngine, Metronome, Player, Synth, Tuner};
use std::io::{self, Write};
use std::sync::Arc;

/// Implementation of the UniFFI callback interface for the Tuner.
/// In a real React Native app, this interface would be implemented in Swift or Kotlin.

#[cfg(test)]
mod ffi_tests {
    use super::*;

    #[test]
    fn test_engine_lifecycle() {
        // 1. Create the engine (constructor returns Arc<AudioEngine>)
        let engine = AudioEngine::new();

        // 2. Start streams
        assert!(engine.start_output());

        // 3. Create a worker
        let metronome = engine.create_metronome(120.0);

        // 4. Test methods directly on the object
        assert!(metronome.set_bpm(140.0));
        assert!(metronome.set_volume(0.8));

        // 5. Cleanup is automatic when the Arc goes out of scope
        metronome.stop();
    }
}

/// Interactive CLI simulation for testing the new Object-Oriented API.
pub fn run_cli_simulation() {
    println!("\nCadenza CLI Simulation Layer (UniFFI Mode)");
    println!("Initializing Audio Engine...");

    // Initialize the engine
    let engine = AudioEngine::new();

    // Store objects in Arcs
    let mut metronome: Option<Arc<Metronome>> = None;
    let mut synth: Option<Arc<Synth>> = None;
    let mut player: Option<Arc<Player>> = None;
    let mut tuner: Option<Arc<Tuner>> = None;

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
                metronome = Some(engine.create_metronome(120.0));
                println!("Metronome created at 120 BPM.");
            }
            "met stop" => {
                if let Some(m) = &metronome {
                    m.stop();
                }
                metronome = None;
                println!("Metronome dropped.");
            }
            "met bpm" => {
                if let Some(m) = &metronome {
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
                synth = Some(engine.create_synth());
                println!("Synth created.");
            }
            "synth note" => {
                if let Some(s) = &synth {
                    // Logic for a middle C (261.63 Hz) at full velocity
                    s.play_note(261.63, 1.0);
                    println!("Playing Middle C...");
                }
            }
            "synth stop" => {
                synth = None;
                println!("Synth dropped.");
            }

            // --- Player ---
            "player start" => {
                player = Some(engine.create_player());
                println!("Player created.");
            }
            "player load" => {
                if let Some(p) = &player {
                    print!("File path: ");
                    io::stdout().flush().unwrap();
                    let mut path = String::new();
                    io::stdin().read_line(&mut path).ok();
                    p.load_track(path.trim().to_string());
                    p.play();
                }
            }

            // --- Recording & Analysis ---
            "rec start" => {
                if engine.start_recording("test_output.wav".to_string()) {
                    println!("Recording to test_output.wav");
                }
            }
            "onset start" => {
                if engine.start_onset_detection() {
                    println!("Onset detection active.");
                }
            }
            "tuner start" => {
                tuner = Some(engine.start_tuner());
                let move_tuner = Arc::downgrade(tuner.as_ref().unwrap());
                std::thread::spawn(move || {
                    loop {
                        if let Some(tuner) = move_tuner.upgrade() {
                            let response = tuner.poll_output();
                            println!("Tuner output: {}", response);
                        } else {
                            break;
                        }
                        std::thread::sleep(std::time::Duration::from_millis(100));
                    }
                });
            }
            "tuner mode: single" => {
                if let Some(tuner) = &tuner {
                    tuner.set_mode("SinglePitch".to_string());
                }
            }
            "tuner mode: multi" => {
                if let Some(tuner) = &tuner {
                    tuner.set_mode("MultiPitch".to_string());
                }
            }
            "tuner stop" => {
                if let Some(tuner) = &tuner {
                    tuner.end();
                }
                tuner = None;
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
