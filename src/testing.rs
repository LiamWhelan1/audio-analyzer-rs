use crate::{
    init_pipeline, pause_command, play_start, record_start, shutdown_pipeline, tune_start,
    worker_pause, worker_start, worker_stop,
};
use std::collections::HashMap;
use std::ffi::CString;
use std::io::{self, Write};

// --- Mock Callback for the Tuner ---
// This simulates the C function that would exist on the iOS/Android side.
extern "C" fn cli_tuner_callback(pitch: f32) {
    // We print with \r (carriage return) to overwrite the line, acting like a real-time display
    print!("\r[Tuner] 🎤 Pitch: {:>7.2} Hz", pitch);
    io::stdout().flush().unwrap();
}

pub fn run_cli_simulation() {
    println!("\n🎼  Cadenza CLI Simulation Layer (FFI Mode)");
    println!("Initializing Audio Pipeline...");

    // 1. Initialize the pipeline via FFI
    let pipeline_handle = init_pipeline();
    if pipeline_handle == 0 {
        eprintln!("❌ Failed to initialize pipeline.");
        return;
    }

    // Store worker handles (i64) mapped by a name
    let mut workers: HashMap<String, i64> = HashMap::new();
    let mut rec_counter = 0;

    println!(
        "Type a command: play | pause | record | tune | stop record | stop tune | stop | exit\n"
    );

    loop {
        print!("\n> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            println!("Error reading input. Try again.");
            continue;
        }

        let command = input.trim().to_lowercase();

        match command.as_str() {
            "play" => {
                if play_start(pipeline_handle) == 0 {
                    println!("▶️  Playback started");
                } else {
                    eprintln!("❌ Error starting playback");
                }
            }
            "pause" => {
                pause_command(pipeline_handle);
                println!("⏸️  Paused");
            }
            "record" => {
                // Simulate creating a temporary path
                let filename = format!("test_rec_{}.wav", rec_counter);
                let c_path = CString::new(filename.clone()).unwrap();

                println!("🎙️  Starting recorder to file: {}", filename);

                // Call FFI: record_start
                let handle = record_start(pipeline_handle, c_path.as_ptr());

                if handle != 0 {
                    workers.insert(format!("rec{}", rec_counter), handle);
                    // Ensure the worker is started (it usually starts on spawn, but good to be explicit)
                    worker_start(handle);
                    rec_counter += 1;
                } else {
                    eprintln!("❌ Error spawning recorder");
                }
            }
            "tune" => {
                println!("🎵  Starting Tuner...");
                // Call FFI: tune_start with our static callback
                let handle = tune_start(pipeline_handle, cli_tuner_callback);

                if handle != 0 {
                    workers.insert("tune".to_string(), handle);
                    worker_start(handle);
                } else {
                    eprintln!("❌ Error spawning tuner");
                }
            }
            "stop record" => {
                let mut to_remove = Vec::new();
                for (name, handle) in &workers {
                    if name.starts_with("rec") {
                        // Stop and release logic
                        worker_stop(*handle);
                        println!("🛑 Stopped {}", name);
                        to_remove.push(name.clone());
                    }
                }
                for name in to_remove {
                    workers.remove(&name);
                }
            }
            "stop tune" => {
                if let Some(handle) = workers.remove("tune") {
                    worker_stop(handle);
                    println!("\n🛑 Tuner stopped");
                } else {
                    println!("Tuner is not running.");
                }
            }
            "pause record" => {
                for (name, handle) in &workers {
                    if name.starts_with("rec") {
                        worker_pause(*handle);
                        println!("Paused {}", name);
                    }
                }
            }
            "start record" => {
                for (name, handle) in &workers {
                    if name.starts_with("rec") {
                        worker_start(*handle);
                        println!("Resumed {}", name);
                    }
                }
            }
            "pause tune" => {
                if let Some(handle) = workers.get("tune") {
                    worker_pause(*handle);
                    println!("\nPaused Tuner");
                }
            }
            "start tune" => {
                if let Some(handle) = workers.get("tune") {
                    worker_start(*handle);
                    println!("Resumed Tuner");
                }
            }
            "metronome" => {
                println!("⚠️  Metronome FFI not yet implemented in lib.rs");
            }
            "drone" => {
                println!("⚠️  Drone FFI not yet implemented in lib.rs");
            }
            "stop" => {
                println!("🛑 Stopping all workers...");
                for (name, handle) in workers.drain() {
                    worker_stop(handle);
                    println!("   Released {}", name);
                }
            }
            "exit" => {
                println!("👋  Exiting simulation.");
                // cleanup remaining workers
                for (_, handle) in workers.drain() {
                    worker_stop(handle);
                }
                // cleanup pipeline
                shutdown_pipeline(pipeline_handle);
                break;
            }
            _ => {
                println!(
                    "Unknown command. Try: play, pause, record, tune, stop record, stop tune, stop, exit"
                )
            }
        }
    }
}
