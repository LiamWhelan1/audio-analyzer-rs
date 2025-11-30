use crate::generators::MidiNote;
use crate::{
    init_pipeline, metronome_set_bpm, metronome_set_pattern, metronome_set_subdivisions,
    metronome_set_volume, metronome_start, metronome_stop, pause_command, play_start, record_start,
    shutdown_pipeline, start_input, synth_load_sequence, synth_pause, synth_play_live,
    synth_resume, synth_start, synth_stop, tune_start, worker_pause, worker_start, worker_stop,
};
use std::collections::HashMap;
use std::ffi::{CString, c_char};
use std::io::{self, Write};

// --- Mock Callback for the Tuner ---
// This simulates the C function that would exist on the iOS/Android side.
extern "C" fn cli_tuner_callback(name: *const c_char, cents: f32) {
    let name = unsafe { crate::cstr_to_rust_str(name) };
    // We print with \r (carriage return) to overwrite the line, acting like a real-time display
    if let Some(n) = name {
        print!(
            "\r[Tuner] 🎤 Name: {} {}{:.4}",
            n,
            if cents >= 0.0 { "+" } else { "" },
            cents
        );
        io::stdout().flush().unwrap();
    }
}

// -----------------------------
// Unit tests for top-level FFI
// -----------------------------
#[cfg(test)]
mod ffi_tests {
    use std::os::raw::c_char;

    // Import the FFI functions from the crate root
    use crate::{
        drone_start, metronome_set_bpm, metronome_set_pattern, metronome_set_subdivisions,
        metronome_set_volume, metronome_start, metronome_stop, pause_command, play_start,
        record_start, shutdown_pipeline, start_input, synth_load_sequence, synth_pause,
        synth_play_live, synth_resume, synth_start, synth_stop, tune_start, worker_pause,
        worker_start, worker_stop,
    };

    extern "C" fn dummy_callback(_name: *const c_char, _cents: f32) {}

    #[test]
    fn invalid_pipeline_handle_behavior() {
        // All of these should handle a "0"/null handle gracefully and not panic.
        assert_eq!(start_input(0), -1);
        assert_eq!(play_start(0), -1);

        // Worker control with invalid handle should return -1
        assert_eq!(worker_start(0), -1);
        assert_eq!(worker_pause(0), -1);
        assert_eq!(worker_stop(0), -1);

        // Record / tune spawn with invalid pipeline should return 0 (means: failed to spawn)
        assert_eq!(record_start(0, std::ptr::null()), 0);
        assert_eq!(tune_start(0, dummy_callback), 0);

        // Metronome with invalid handle should return 0 on start (failure) and -1 for ops
        assert_eq!(metronome_start(0, 120.0), 0);
        assert_eq!(metronome_stop(0), -1);
        assert_eq!(metronome_set_bpm(0, 90.0), -1);
        assert_eq!(metronome_set_volume(0, 0.5), -1);
        assert_eq!(metronome_set_subdivisions(0, std::ptr::null(), 0), -1);
        assert_eq!(metronome_set_pattern(0, std::ptr::null(), 0), -1);

        // Synthesizer handles are similarly defensive for handle==0
        assert_eq!(synth_start(0), 0);
        assert_eq!(synth_stop(0), -1);
        assert_eq!(synth_play_live(0, 440.0, 1.0), -1);
        assert_eq!(synth_load_sequence(0, std::ptr::null(), 0), -1);
        assert_eq!(synth_pause(0), -1);
        assert_eq!(synth_resume(0), -1);

        // Drone stub returns 0 as a TODO placeholder
        assert_eq!(drone_start(440.0), 0);
    }

    #[test]
    fn pause_and_shutdown_no_panic() {
        // pause_command is a global noop (returns 0)
        assert_eq!(pause_command(0), 0);

        // Shutdown with handle 0 should be safe (no panic)
        shutdown_pipeline(0);
    }
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
    let mut met_handles: Vec<i64> = Vec::new();
    let mut synth_handles: Vec<i64> = Vec::new();
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
                println!("Test metronome");
                met_handles.push(metronome_start(pipeline_handle, 60.0));
            }
            "stop met" => {
                if let Some(h) = met_handles.pop() {
                    metronome_stop(h);
                }
            }
            "set bpm" => {
                if let Some(h) = met_handles.first() {
                    println!("bpm? ");
                    let mut bpm = String::new();
                    if io::stdin().read_line(&mut bpm).is_err() {
                        println!("Error reading input.");
                        continue;
                    }
                    if let Ok(bpm) = bpm.trim().parse::<f32>() {
                        metronome_set_bpm(*h, bpm);
                    }
                }
            }
            "set polys" => {
                if let Some(h) = met_handles.first() {
                    println!("polys? ");
                    let mut polys = String::new();
                    if io::stdin().read_line(&mut polys).is_err() {
                        println!("Error reading input.");
                        continue;
                    }
                    let polys: Vec<usize> = polys
                        .trim()
                        .split(" ")
                        .map(|n| n.parse::<usize>().unwrap_or(1))
                        .collect();
                    println!("{polys:?}");
                    metronome_set_subdivisions(*h, polys.as_ptr(), polys.len());
                }
            }
            "drone" => {
                println!("note? ");
                let mut note = String::new();
                if io::stdin().read_line(&mut note).is_err() {
                    println!("Error reading input.");
                    continue;
                }
                if let Ok(note) = note.trim().parse::<u8>() {
                    let freq = MidiNote::new(note, 0.0).to_freq(None);
                    synth_handles.push(synth_start(pipeline_handle));
                    synth_play_live(*synth_handles.last().unwrap(), freq, 67.0);
                }
            }
            "synth start" => {
                let h = synth_start(pipeline_handle);
                if h != 0 {
                    println!("🎹 Synth started: {}", h);
                    synth_handles.push(h);
                } else {
                    eprintln!("❌ Failed to spawn synth");
                }
            }
            "synth stop" => {
                if let Some(h) = synth_handles.pop() {
                    synth_stop(h);
                    println!("🎹 Synth stopped: {}", h);
                } else {
                    println!("No synths running.");
                }
            }
            "synth note" => {
                println!("note? (0-127)");
                let mut note = String::new();
                if io::stdin().read_line(&mut note).is_err() {
                    println!("Error reading input.");
                    continue;
                }
                if let Ok(note) = note.trim().parse::<u8>() {
                    if let Some(&h) = synth_handles.last() {
                        let freq = MidiNote::new(note, 0.0).to_freq(None);
                        // note on
                        synth_play_live(h, freq, 80.0);
                        println!("Played note {} -> {} Hz", note, freq);
                    } else {
                        println!("No synth running. Start one with 'synth start'.");
                    }
                }
            }
            "synth off" => {
                println!("note? (0-127)");
                let mut note = String::new();
                if io::stdin().read_line(&mut note).is_err() {
                    println!("Error reading input.");
                    continue;
                }
                if let Ok(note) = note.trim().parse::<u8>() {
                    if let Some(&h) = synth_handles.last() {
                        let freq = MidiNote::new(note, 0.0).to_freq(None);
                        // note off
                        synth_play_live(h, freq, 0.0);
                        println!("Stopped note {} -> {} Hz", note, freq);
                    } else {
                        println!("No synth running. Start one with 'synth start'.");
                    }
                }
            }
            "synth load" => {
                if let Some(&h) = synth_handles.last() {
                    // Build a tiny sequence of two notes
                    let seq = vec![
                        crate::FFISynthNote {
                            freq: 440.0,
                            duration_ms: 500.0,
                            onset_ms: 0.0,
                            velocity: 80.0,
                        },
                        crate::FFISynthNote {
                            freq: 660.0,
                            duration_ms: 500.0,
                            onset_ms: 500.0,
                            velocity: 80.0,
                        },
                    ];
                    let res = synth_load_sequence(h, seq.as_ptr(), seq.len());
                    if res == 0 {
                        println!("🎹 Loaded sequence into synth {}", h);
                    } else {
                        eprintln!("❌ Failed to load sequence");
                    }
                } else {
                    println!("No synth running. Start one with 'synth start'.");
                }
            }
            "synth pause" => {
                if let Some(&h) = synth_handles.last() {
                    synth_pause(h);
                    println!("🎹 Synth {} paused", h);
                }
            }
            "synth resume" => {
                if let Some(&h) = synth_handles.last() {
                    synth_resume(h);
                    println!("🎹 Synth {} resumed", h);
                }
            }
            "start input" => {
                if start_input(pipeline_handle) == 0 {
                    println!("🎧 Input started");
                } else {
                    eprintln!("❌ Failed to start input");
                }
            }
            "set met pattern" => {
                if let Some(h) = met_handles.first() {
                    println!("pattern? (e.g. 3 1 1 2)");
                    let mut pat = String::new();
                    if io::stdin().read_line(&mut pat).is_err() {
                        println!("Error reading input.");
                        continue;
                    }
                    let vals: Vec<i32> = pat
                        .trim()
                        .split(' ')
                        .filter_map(|s| s.parse::<i32>().ok())
                        .collect();
                    metronome_set_pattern(*h, vals.as_ptr(), vals.len());
                }
            }
            "set met vol" => {
                if let Some(h) = met_handles.first() {
                    println!("volume? (0.0-1.0)");
                    let mut v = String::new();
                    if io::stdin().read_line(&mut v).is_err() {
                        println!("Error reading input.");
                        continue;
                    }
                    if let Ok(vol) = v.trim().parse::<f32>() {
                        metronome_set_volume(*h, vol);
                    }
                }
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
