use crate::generators::MidiNote;
use crate::{
    cstr_to_rust_str, init_pipeline, metronome_set_bpm, metronome_set_pattern,
    metronome_set_subdivisions, metronome_set_volume, metronome_start, metronome_stop, onset_start,
    pause_command, play_start, player_create, player_destroy, player_load_track, player_pause,
    player_play, player_seek, player_stop, record_start, shutdown_pipeline, start_input,
    synth_clear, synth_load_file, synth_pause, synth_play_from, synth_play_live, synth_resume,
    synth_set_vol, synth_start, synth_stop, tune_start, worker_pause, worker_start, worker_stop,
};
use core::slice;
use std::collections::HashMap;
use std::ffi::{CString, c_char};
use std::io::{self, Write};

/// FFI callback that receives tuner results.
extern "C" fn cli_tuner_callback(names: *const *const c_char, len: usize) {
    let notes: Vec<String> = unsafe { slice::from_raw_parts(names, len) }
        .iter()
        .map(|s| unsafe { cstr_to_rust_str(*s).unwrap().to_string() })
        .collect();
    print!("\x1B[2K\r[Tuner] Notes: {:?}", notes);
    io::stdout().flush().unwrap();
}

/// FFI integration tests for pipeline, workers, and FFI handles.
#[cfg(test)]
mod ffi_tests {

    use std::ffi::c_char;

    // Import the FFI functions from the crate root
    use crate::{
        drone_start, metronome_set_bpm, metronome_set_pattern, metronome_set_subdivisions,
        metronome_set_volume, metronome_start, metronome_stop, onset_start, pause_command,
        play_start, record_start, shutdown_pipeline, start_input, synth_load_file, synth_pause,
        synth_play_live, synth_resume, synth_start, synth_stop, tune_start, worker_pause,
        worker_start, worker_stop,
    };

    extern "C" fn dummy_callback(_freqs: *const *const c_char, _len: usize) {}

    /// Test that FFI functions handle null handles gracefully.
    #[test]
    fn invalid_pipeline_handle_behavior() {
        assert_eq!(start_input(0), -1);
        assert_eq!(play_start(0), -1);

        assert_eq!(worker_start(0), -1);
        assert_eq!(worker_pause(0), -1);
        assert_eq!(worker_stop(0), -1);

        assert_eq!(record_start(0, std::ptr::null()), 0);
        assert_eq!(tune_start(0, dummy_callback), 0);
        assert_eq!(onset_start(0), 0);

        assert_eq!(
            metronome_start(
                0,
                0.0,
                vec![].as_ptr(),
                0,
                vec![Vec::new().as_ptr()].as_ptr(),
                vec![].as_ptr(),
                false,
            ),
            0
        );
        assert_eq!(metronome_stop(0), -1);
        assert_eq!(metronome_set_bpm(0, 90.0), -1);
        assert_eq!(metronome_set_volume(0, 0.5), -1);
        assert_eq!(metronome_set_subdivisions(0, std::ptr::null(), 0, 0), -1);
        assert_eq!(metronome_set_pattern(0, std::ptr::null(), 0), -1);

        assert_eq!(synth_start(0), 0);
        assert_eq!(synth_stop(0), -1);
        assert_eq!(synth_play_live(0, 440.0, 1.0), -1);
        assert_eq!(synth_load_file(0, std::ptr::null()), -1);
        assert_eq!(synth_pause(0), -1);
        assert_eq!(synth_resume(0), -1);

        assert_eq!(drone_start(440.0), 0);
    }

    /// Test that pause and shutdown functions don't panic with null handles.
    #[test]
    fn pause_and_shutdown_no_panic() {
        assert_eq!(pause_command(0), 0);
        shutdown_pipeline(0);
    }
}

/// Interactive CLI simulation for testing FFI functionality.
pub fn run_cli_simulation() {
    println!("\nCadenza CLI Simulation Layer (FFI Mode)");
    println!("Initializing Audio Pipeline...");

    // 1. Initialize the pipeline via FFI
    let pipeline_handle = init_pipeline();
    if pipeline_handle == 0 {
        eprintln!("Failed to initialize pipeline.");
        return;
    }

    // Store worker handles (i64) mapped by a name
    let mut workers: HashMap<String, i64> = HashMap::new();
    let mut rec_counter = 0;
    let mut met_handles: Vec<i64> = Vec::new();
    let mut synth_handles: Vec<i64> = Vec::new();
    let mut player_handles: Vec<i64> = Vec::new();
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
            "audio play" => {
                if play_start(pipeline_handle) == 0 {
                    println!("Playback started.");
                } else {
                    eprintln!("Error starting playback");
                }
            }
            "audio pause" => {
                pause_command(pipeline_handle);
                println!("Paused.");
            }
            "recorder start" => {
                let filename = format!("test_rec_{}.wav", rec_counter);
                let c_path = CString::new(filename.clone()).unwrap();

                println!("Starting recorder to file: {}", filename);

                // Call FFI: record_start
                let handle = record_start(pipeline_handle, c_path.as_ptr());

                if handle != 0 {
                    workers.insert(format!("rec{}", rec_counter), handle);
                    // Ensure the worker is started (it usually starts on spawn, but good to be explicit)
                    worker_start(handle);
                    rec_counter += 1;
                } else {
                    eprintln!("Error spawning recorder");
                }
            }
            "tuner start" => {
                println!("Starting tuner...");
                // Call FFI: tune_start with our static callback
                let handle = tune_start(pipeline_handle, cli_tuner_callback);

                if handle != 0 {
                    workers.insert("tune".to_string(), handle);
                    worker_start(handle);
                } else {
                    eprintln!("Error spawning tuner");
                }
            }
            "recorder stop" => {
                let mut to_remove = Vec::new();
                for (name, handle) in &workers {
                    if name.starts_with("rec") {
                        worker_stop(*handle);
                        println!("Stopped {}", name);
                        to_remove.push(name.clone());
                    }
                }
                for name in to_remove {
                    workers.remove(&name);
                }
            }
            "tuner stop" => {
                if let Some(handle) = workers.remove("tune") {
                    worker_stop(handle);
                    println!("\nTuner stopped.");
                } else {
                    println!("Tuner is not running.");
                }
            }
            "recorder pause" => {
                for (name, handle) in &workers {
                    if name.starts_with("rec") {
                        worker_pause(*handle);
                        println!("Paused {}", name);
                    }
                }
            }
            "recorder resume" => {
                for (name, handle) in &workers {
                    if name.starts_with("rec") {
                        worker_start(*handle);
                        println!("Resumed {}", name);
                    }
                }
            }
            "tuner pause" => {
                if let Some(handle) = workers.get("tune") {
                    worker_pause(*handle);
                    println!("\nTuner paused.");
                }
            }
            "tuner resume" => {
                if let Some(handle) = workers.get("tune") {
                    worker_start(*handle);
                    println!("Tuner resumed.");
                }
            }
            "metronome start" => {
                println!("Starting metronome.");
                met_handles.push(metronome_start(
                    pipeline_handle,
                    0.0,
                    vec![].as_ptr(),
                    0,
                    vec![Vec::new().as_ptr()].as_ptr(),
                    vec![].as_ptr(),
                    false,
                ));
            }
            "metronome stop" => {
                if let Some(h) = met_handles.pop() {
                    metronome_stop(h);
                }
            }
            "metronome bpm" => {
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
            "metronome polys" => {
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
                    println!("idx? ");
                    let mut idx = String::new();
                    if io::stdin().read_line(&mut idx).is_err() {
                        println!("Error reading input.");
                        continue;
                    }
                    let idx = idx.trim().parse::<usize>().unwrap();
                    metronome_set_subdivisions(*h, polys.as_ptr(), polys.len(), idx);
                }
            }
            "synth drone" => {
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
                    println!("Synth started: {}", h);
                    synth_handles.push(h);
                } else {
                    eprintln!("Failed to spawn synth");
                }
            }
            "synth stop" => {
                if let Some(h) = synth_handles.pop() {
                    synth_stop(h);
                    println!("Synth stopped: {}", h);
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
            "synth note-off" => {
                println!("note? (0-127)");
                let mut note = String::new();
                if io::stdin().read_line(&mut note).is_err() {
                    println!("Error reading input.");
                    continue;
                }
                if let Ok(note) = note.trim().parse::<u8>() {
                    if let Some(&h) = synth_handles.last() {
                        let freq = MidiNote::new(note, 0.0).to_freq(None);
                        synth_play_live(h, freq, 0.0);
                        println!("Stopped note {} -> {} Hz", note, freq);
                    } else {
                        println!("No synth running. Start one with 'synth start'.");
                    }
                }
            }
            "synth load" => {
                if let Some(&h) = synth_handles.last() {
                    println!("midi path?");
                    let mut path = String::new();
                    if io::stdin().read_line(&mut path).is_err() {
                        println!("Error reading input.");
                        continue;
                    }
                    let path = path.trim();
                    let c_path = CString::new(path).unwrap();
                    if synth_load_file(h, c_path.as_ptr()) == 0 {
                        println!("Loaded {}", path);
                    } else {
                        eprintln!("Failed to load {}", path);
                    }
                } else {
                    println!("No synth running. Start one with 'synth start'.");
                }
            }
            "synth play" => {
                if let Some(&h) = synth_handles.last() {
                    println!("start measure?");
                    let mut measure = String::new();
                    if io::stdin().read_line(&mut measure).is_err() {
                        println!("Error reading input.");
                        continue;
                    }
                    let measure = measure.trim().parse::<usize>().unwrap();
                    if synth_play_from(h, measure) == 0 {
                        println!("Playing from {}", measure);
                    } else {
                        eprintln!("Failed to play from {}", measure);
                    }
                } else {
                    println!("No synth running. Start one with 'synth start'.");
                }
            }
            "synth clear" => {
                if let Some(&h) = synth_handles.last() {
                    synth_clear(h);
                    println!("Synth cleared.");
                }
            }
            "synth pause" => {
                if let Some(&h) = synth_handles.last() {
                    synth_pause(h);
                    println!("Synth paused.");
                }
            }
            "synth resume" => {
                if let Some(&h) = synth_handles.last() {
                    synth_resume(h);
                    println!("Synth resumed.");
                }
            }
            "synth volume" => {
                if let Some(h) = synth_handles.first() {
                    println!("volume? (0.0-1.0)");
                    let mut v = String::new();
                    if io::stdin().read_line(&mut v).is_err() {
                        println!("Error reading input.");
                        continue;
                    }
                    if let Ok(vol) = v.trim().parse::<f32>() {
                        synth_set_vol(*h, vol);
                    }
                }
            }
            "input start" => {
                if start_input(pipeline_handle) == 0 {
                    println!("Input started.");
                } else {
                    eprintln!("Failed to start input");
                }
            }
            "metronome pattern" => {
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
            "metronome volume" => {
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
            "player start" => {
                let h = player_create(pipeline_handle);
                if h != 0 {
                    println!("🎧 Player started: {}", h);
                    player_handles.push(h);
                } else {
                    eprintln!("Failed to spawn player");
                }
            }
            "player load" => {
                if let Some(&h) = player_handles.last() {
                    println!("file path?");
                    let mut path = String::new();
                    if io::stdin().read_line(&mut path).is_err() {
                        println!("Error reading input.");
                        continue;
                    }
                    let path = path.trim();
                    let c_path = CString::new(path).unwrap();
                    if player_load_track(h, c_path.as_ptr()) == 0 {
                        println!("Loaded {}", path);
                    } else {
                        eprintln!("Failed to load {}", path);
                    }
                } else {
                    println!("No player running.");
                }
            }
            "player play" => {
                if let Some(&h) = player_handles.last() {
                    player_play(h);
                    println!("Player playing.");
                }
            }
            "player pause" => {
                if let Some(&h) = player_handles.last() {
                    player_pause(h);
                    println!("Player paused.");
                }
            }
            "player stop" => {
                if let Some(&h) = player_handles.last() {
                    player_stop(h);
                    println!("Player stopped.");
                }
            }
            "player seek" => {
                if let Some(&h) = player_handles.last() {
                    println!("seconds?");
                    let mut s = String::new();
                    if io::stdin().read_line(&mut s).is_err() {
                        continue;
                    }
                    if let Ok(sec) = s.trim().parse::<f64>() {
                        player_seek(h, sec);
                        println!("Seeked to {}s", sec);
                    }
                }
            }
            "player destroy" => {
                if let Some(h) = player_handles.pop() {
                    player_destroy(h);
                }
            }
            "onset start" => {
                if !workers.contains_key("onset") {
                    let handle = onset_start(pipeline_handle);

                    if handle != 0 {
                        workers.insert(String::from("onset"), handle);
                        worker_start(handle);
                    } else {
                        eprintln!("Error spawning recorder");
                    }
                }
            }
            "onset stop" => {
                if let Some(handle) = workers.remove("onset") {
                    worker_stop(handle);
                    println!("\nOnset stopped.");
                } else {
                    println!("Onset is not running.");
                }
            }
            "onset pause" => {
                if let Some(handle) = workers.get("onset") {
                    worker_pause(*handle);
                    println!("\nOnset paused.");
                } else {
                    println!("Onset is not running.");
                }
            }
            "onset resume" => {
                if let Some(handle) = workers.get("onset") {
                    worker_start(*handle);
                    println!("\nOnset resumed.");
                } else {
                    println!("Onset is not running.");
                }
            }
            "all stop" => {
                println!("Stopping all workers...");
                for (name, handle) in workers.drain() {
                    println!("Released {}", name);
                    worker_stop(handle);
                }
                for h in met_handles.drain(..) {
                    metronome_stop(h);
                }
                for h in synth_handles.drain(..) {
                    synth_stop(h);
                }
                for h in player_handles.drain(..) {
                    player_destroy(h);
                }
            }
            "all exit" => {
                println!("Exiting simulation.");
                for (_, handle) in workers.drain() {
                    worker_stop(handle);
                }
                for h in met_handles {
                    metronome_stop(h);
                }
                for h in synth_handles {
                    synth_stop(h);
                }
                for h in player_handles {
                    player_destroy(h);
                }
                shutdown_pipeline(pipeline_handle);
                break;
            }
            _ => {
                println!(
                    "Unknown command. Try: audio play, audio pause, recorder start, recorder stop, tuner start, tuner stop, metronome start, synth start, player start, input start, all stop, all exit"
                )
            }
        }
    }
}
