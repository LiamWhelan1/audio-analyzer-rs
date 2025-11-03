use crate::audio_io::AudioPipeline;
use crate::generators::metronome::BeatStrength;
use crate::traits::Worker;
use crate::{
    drone, metronome, pause, pause_worker, play, record, start_worker, stop_worker, tune, upload,
};
use std::collections::HashMap;
use std::io::{self, Write};
pub fn run_cli_simulation() {
    let mut workers: HashMap<String, Box<dyn Worker>> = HashMap::with_capacity(u8::MAX as usize);
    let mut builder = AudioPipeline::new().ok();
    let mut n = 0;
    println!("\n🎼  Cadenza CLI Simulation Layer");
    println!(
        "Type a command: play | pause | record | tune | metronome | upload | drone | stop | exit\n"
    );

    loop {
        print!("> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            println!("Error reading input. Try again.");
            continue;
        }

        let command = input.trim().to_lowercase();

        match command.as_str() {
            "play" => {
                let b = if let Some(b) = builder.as_mut() {
                    b
                } else {
                    builder = Some(AudioPipeline::new().unwrap());
                    builder.as_mut().unwrap()
                };
                if let Err(e) = play(b) {
                    eprintln!("Error: {}", e);
                }
            }
            "pause" => {
                if let Err(e) = pause() {
                    eprintln!("Error: {}", e);
                }
            }
            "record" => {
                let b = if let Some(b) = builder.as_mut() {
                    b
                } else {
                    builder = Some(AudioPipeline::new().unwrap());
                    builder.as_mut().unwrap()
                };
                let recorder = record(b, &format!("test{n}.wav"));
                if let Err(e) = recorder {
                    eprintln!("Error: {}", e);
                } else {
                    workers.insert(format!("rec{n}"), Box::new(recorder.unwrap()));
                }
                n += 1;
            }
            "tune" => {
                let b = if let Some(b) = builder.as_mut() {
                    b
                } else {
                    builder = Some(AudioPipeline::new().unwrap());
                    builder.as_mut().unwrap()
                };
                let tuner = tune(crate::analysis::TuningSystem::EqualTemperament, b);
                if let Err(e) = tuner {
                    eprintln!("Error: {}", e);
                } else {
                    workers.insert("tune".to_string(), Box::new(tuner.unwrap()));
                }
            }
            "metronome" => {
                let metronome = metronome(
                    120.0,
                    4,
                    4,
                    Some(vec![2]),
                    vec![
                        BeatStrength::Strong,
                        BeatStrength::Weak,
                        BeatStrength::Medium,
                        BeatStrength::Weak,
                    ],
                );
                if let Err(e) = metronome {
                    eprintln!("Error: {}", e);
                } else {
                    // workers.push(metronome.unwrap());
                }
            }
            "upload" => {
                if let Err(e) = upload("picture") {
                    eprintln!("Error: {}", e);
                }
            }
            "drone" => {
                if let Err(e) = drone("A4") {
                    eprintln!("Error: {}", e);
                }
            }
            "stop" => {
                if let Some(_) = builder.as_mut() {
                    let mut b = builder.take().unwrap();
                    for w in &mut workers.values_mut() {
                        stop_worker(w, &mut b);
                    }
                    workers.clear();
                    println!("All workers stopped");
                    b.shutdown();
                };
            }
            "stop record" => {
                let b = if let Some(b) = builder.as_mut() {
                    b
                } else {
                    builder = Some(AudioPipeline::new().unwrap());
                    builder.as_mut().unwrap()
                };
                for i in 0..n {
                    let name = &format!("rec{i}");
                    if let Some(r) = workers.get_mut(name) {
                        stop_worker(r, b);
                        workers.remove(name)
                    } else {
                        continue;
                    };
                }
            }
            "pause record" => {
                for i in 0..n {
                    let name = &format!("rec{i}");
                    if let Some(r) = workers.get_mut(name) {
                        pause_worker(r);
                    } else {
                        continue;
                    };
                }
            }
            "start record" => {
                for i in 0..n {
                    let name = &format!("rec{i}");
                    if let Some(r) = workers.get_mut(name) {
                        start_worker(r);
                    } else {
                        continue;
                    };
                }
            }
            "stop tune" => {
                let b = if let Some(b) = builder.as_mut() {
                    b
                } else {
                    builder = Some(AudioPipeline::new().unwrap());
                    builder.as_mut().unwrap()
                };
                if let Some(r) = workers.get_mut("tune") {
                    stop_worker(r, b);
                    workers.remove("tune")
                } else {
                    continue;
                };
            }
            "pause tune" => {
                if let Some(r) = workers.get_mut("tune") {
                    pause_worker(r);
                } else {
                    continue;
                };
            }
            "start tune" => {
                if let Some(r) = workers.get_mut("tune") {
                    start_worker(r);
                } else {
                    continue;
                };
            }
            "exit" => {
                println!("👋  Exiting simulation.");
                break;
            }
            _ => {
                println!(
                    "Unknown command. Try one of: play, pause, record, tune, metronome, upload, drone, stop, exit"
                )
            }
        }
    }
}
