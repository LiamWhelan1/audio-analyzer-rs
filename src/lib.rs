pub mod analysis;
pub mod audio_io;
pub mod dsp;
pub mod generators;
pub mod testing;
pub mod traits;

use std::ffi::CStr;
use std::os::raw::c_char;
use std::{panic, slice};

use crate::audio_io::AudioPipeline;
use crate::generators::player::PlayerController;
use crate::generators::{
    metronome::{BeatStrength, MetronomeCommand},
    synth::SynthCommand,
};
use crate::traits::Worker;

/// A concrete wrapper around the dynamic Worker trait with pipeline reference for cleanup.
pub struct WorkerHandle {
    pub worker: Box<dyn Worker>,
    pub pipeline: *mut AudioPipeline,
}

/// Wrap a worker into a generic FFI handle.
fn into_worker_handle<T: Worker + 'static>(worker: T, pipeline: *mut AudioPipeline) -> i64 {
    let boxed_worker: Box<dyn Worker> = Box::new(worker);
    let wrapper = Box::new(WorkerHandle {
        worker: boxed_worker,
        pipeline,
    });
    Box::into_raw(wrapper) as i64
}

/// Access the generic Worker from an FFI handle.
unsafe fn worker_from_handle<'a>(handle: i64) -> Option<&'a mut Box<dyn Worker>> {
    if handle == 0 {
        return None;
    }
    unsafe {
        let wrapper = &mut *(handle as *mut WorkerHandle);
        Some(&mut wrapper.worker)
    }
}

/// Drop a worker handle and free its memory.
unsafe fn drop_worker_handle(handle: i64) {
    if handle != 0 {
        unsafe {
            drop(Box::from_raw(handle as *mut WorkerHandle));
        }
    }
}

/// Wrap a pipeline into an FFI handle.
fn into_pipeline_handle(p: AudioPipeline) -> i64 {
    Box::into_raw(Box::new(p)) as i64
}

/// Access a pipeline from an FFI handle.
unsafe fn pipeline_from_handle<'a>(handle: i64) -> Option<&'a mut AudioPipeline> {
    if handle == 0 {
        return None;
    }
    unsafe { Some(&mut *(handle as *mut AudioPipeline)) }
}

/// Convert a C string pointer to a Rust string slice.
pub unsafe fn cstr_to_rust_str(c_ptr: *const c_char) -> Option<&'static str> {
    if c_ptr.is_null() {
        return None;
    }
    unsafe { CStr::from_ptr(c_ptr).to_str().ok() }
}

/// Handle to a running metronome instance.
pub struct MetronomeHandle {
    pub producer: rtrb::Producer<MetronomeCommand>,
}

/// Drop a metronome handle and free its memory.
unsafe fn drop_metronome_handle(handle: i64) {
    if handle != 0 {
        unsafe { drop(Box::from_raw(handle as *mut MetronomeHandle)) };
    }
}

/// Access a metronome handle from an FFI handle.
unsafe fn metronome_from_handle<'a>(handle: i64) -> Option<&'a mut MetronomeHandle> {
    if handle == 0 {
        return None;
    }
    unsafe { Some(&mut *(handle as *mut MetronomeHandle)) }
}

/// Initialize a new audio pipeline and return its handle.
#[unsafe(no_mangle)]
pub extern "C" fn init_pipeline() -> i64 {
    let result = panic::catch_unwind(|| match AudioPipeline::new() {
        Ok(p) => into_pipeline_handle(p),
        Err(e) => {
            eprintln!("Failed to init pipeline: {}", e);
            0
        }
    });

    match result {
        Ok(handle) => handle,
        Err(_) => {
            eprintln!("Rust panic caught during pipeline init");
            0
        }
    }
}

/// Shutdown the audio pipeline and free its resources.
#[unsafe(no_mangle)]
pub extern "C" fn shutdown_pipeline(handle: i64) {
    unsafe {
        if handle != 0 {
            drop(Box::from_raw(handle as *mut AudioPipeline));
        }
    }
    println!("Pipeline shut down.");
}

/// Start the audio input stream.
#[unsafe(no_mangle)]
pub extern "C" fn start_input(handle: i64) -> i32 {
    if let Some(builder) = unsafe { pipeline_from_handle(handle) } {
        match builder.start_input() {
            Ok(_) => 0,
            Err(e) => {
                eprintln!("Error starting input: {}", e);
                -1
            }
        }
    } else {
        -1
    }
}

/// Start audio playback (output stream).
#[unsafe(no_mangle)]
pub extern "C" fn play_start(handle: i64) -> i32 {
    if let Some(builder) = unsafe { pipeline_from_handle(handle) } {
        println!("Playback started.");
        match builder.start_output() {
            Ok(_) => 0,
            Err(e) => {
                eprintln!("Error starting output for playback: {}", e);
                -1
            }
        }
    } else {
        -1
    }
}

/// Pause the audio pipeline globally.
#[unsafe(no_mangle)]
pub extern "C" fn pause_command(_handle: i64) -> i32 {
    println!("Pipeline paused.");
    0
}

/// Start a worker thread.
#[unsafe(no_mangle)]
pub extern "C" fn worker_start(handle: i64) -> i32 {
    if let Some(worker) = unsafe { worker_from_handle(handle) } {
        worker.start();
        0
    } else {
        -1
    }
}

/// Pause a worker thread.
#[unsafe(no_mangle)]
pub extern "C" fn worker_pause(handle: i64) -> i32 {
    if let Some(worker) = unsafe { worker_from_handle(handle) } {
        worker.pause();
        0
    } else {
        -1
    }
}

/// Stop a worker thread and free its resources.
#[unsafe(no_mangle)]
pub extern "C" fn worker_stop(handle: i64) -> i32 {
    if handle == 0 {
        return -1;
    }
    let wrapper = unsafe { &mut *(handle as *mut WorkerHandle) };
    let consumer_id = wrapper.worker.stop();
    if !wrapper.pipeline.is_null() {
        unsafe {
            let pipeline = &mut *wrapper.pipeline;
            pipeline.remove_consumer(consumer_id);
        }
    }
    unsafe {
        drop_worker_handle(handle);
    }
    println!("Worker stopped.");
    0
}

/// Start audio recording to a file.
#[unsafe(no_mangle)]
pub extern "C" fn record_start(pipeline_handle: i64, temp_path_ptr: *const c_char) -> i64 {
    let builder = unsafe { pipeline_from_handle(pipeline_handle) };
    let path = unsafe { cstr_to_rust_str(temp_path_ptr) };
    if let (Some(b), Some(p)) = (builder, path) {
        if let Err(e) = b.start_input() {
            eprintln!("Failed to start input stream for recording: {}", e);
            return 0;
        }
        println!("Spawning Recorder to: {}", p);
        match b.spawn_recorder(p) {
            Ok(recorder) => into_worker_handle(recorder, pipeline_handle as *mut AudioPipeline),
            Err(e) => {
                eprintln!("Failed to spawn recorder struct: {}", e);
                0
            }
        }
    } else {
        0
    }
}

// pub type AnalysisCallback = extern "C" fn(
//     name: *const c_char,
//     accuracy: f32,
//     notes: *const *const c_char,
//     notes_len: usize,
//     accuracies: *const f32,
//     acc_len: usize,
// );

pub type AnalysisCallback = extern "C" fn(names: *const *const c_char, len: usize);

/// Start audio tuning analysis with a callback for results.
#[unsafe(no_mangle)]
pub extern "C" fn tune_start(pipeline_handle: i64, callback: AnalysisCallback) -> i64 {
    let builder = unsafe { pipeline_from_handle(pipeline_handle) };

    if let Some(b) = builder {
        if let Err(e) = b.start_input() {
            eprintln!("Failed to start input stream for tuning: {}", e);
            return 0;
        }

        println!("Tuner started.");
        match b.spawn_transformer(callback) {
            Ok(tuner) => into_worker_handle(tuner, pipeline_handle as *mut AudioPipeline),
            Err(e) => {
                eprintln!("Failed to spawn tuner struct: {}", e);
                0
            }
        }
    } else {
        0
    }
}

/// Start audio onset detection.
#[unsafe(no_mangle)]
pub extern "C" fn onset_start(pipeline_handle: i64) -> i64 {
    let builder = unsafe { pipeline_from_handle(pipeline_handle) };

    if let Some(b) = builder {
        println!("Spawning onset detector");
        match b.spawn_onset() {
            Ok(onset) => into_worker_handle(onset, pipeline_handle as *mut AudioPipeline),
            Err(e) => {
                eprintln!("Failed to spawn onset struct: {}", e);
                0
            }
        }
    } else {
        0
    }
}

/// Start a metronome with BPM, pattern, and polyrhythm settings.
#[unsafe(no_mangle)]
pub extern "C" fn metronome_start(
    pipeline_handle: i64,
    bpm: f32,
    pattern_ptr: *const usize,
    pattern_len: usize,
    polys_ptr: *const *const usize,
    polys_lens_ptr: *const usize,
    restart: bool,
) -> i64 {
    let builder = unsafe { pipeline_from_handle(pipeline_handle) };

    if let Some(b) = builder {
        if let Err(e) = b.start_output() {
            eprintln!("Failed to start output for metronome: {}", e);
            return 0;
        }

        let bpm = if bpm == 0.0 { None } else { Some(bpm) };

        let pattern: Option<Vec<BeatStrength>> = if pattern_ptr.is_null() || pattern_len == 0 {
            None
        } else {
            let raw = unsafe { slice::from_raw_parts(pattern_ptr, pattern_len) };
            Some(
                raw.iter()
                    .map(|&p| match p {
                        3 => BeatStrength::Strong,
                        2 => BeatStrength::Medium,
                        1 => BeatStrength::Weak,
                        _ => BeatStrength::None,
                    })
                    .collect(),
            )
        };

        let polys_lens: Option<Vec<usize>> = if polys_lens_ptr.is_null() || pattern_len == 0 {
            None
        } else {
            Some(unsafe { slice::from_raw_parts(polys_lens_ptr, pattern_len).to_vec() })
        };

        let polys: Option<Vec<Vec<usize>>> = if polys_ptr.is_null() || polys_lens.is_none() {
            None
        } else {
            let lens = polys_lens.as_ref().unwrap();

            let polys_raw = unsafe { slice::from_raw_parts(polys_ptr, lens.len()) };

            Some(
                polys_raw
                    .iter()
                    .enumerate()
                    .map(|(i, p)| unsafe { slice::from_raw_parts(*p, lens[i]).to_vec() })
                    .collect(),
            )
        };

        let producer = b.spawn_metronome(bpm, pattern, polys, restart);

        let handle = Box::new(MetronomeHandle { producer });
        Box::into_raw(handle) as i64
    } else {
        0
    }
}

/// Stop the metronome and free its resources.
#[unsafe(no_mangle)]
pub extern "C" fn metronome_stop(handle: i64) -> i32 {
    unsafe {
        if let Some(met) = metronome_from_handle(handle) {
            let _ = met.producer.push(MetronomeCommand::Stop);
            drop_metronome_handle(handle);
            println!("Metronome stopped.");
            0
        } else {
            -1
        }
    }
}

/// Set the metronome BPM.
#[unsafe(no_mangle)]
pub extern "C" fn metronome_set_bpm(handle: i64, bpm: f32) -> i32 {
    println!("Setting bpm to {bpm}");
    if let Some(met) = unsafe { metronome_from_handle(handle) } {
        if met.producer.push(MetronomeCommand::SetBpm(bpm)).is_ok() {
            return 0;
        }
    }
    -1
}

/// Set the metronome volume.
#[unsafe(no_mangle)]
pub extern "C" fn metronome_set_volume(handle: i64, volume: f32) -> i32 {
    if let Some(met) = unsafe { metronome_from_handle(handle) } {
        if met
            .producer
            .push(MetronomeCommand::SetVolume(volume))
            .is_ok()
        {
            return 0;
        }
    }
    -1
}

/// Set polyrhythmic subdivisions for a specific beat.
#[unsafe(no_mangle)]
pub extern "C" fn metronome_set_subdivisions(
    handle: i64,
    divs_ptr: *const usize,
    len: usize,
    beat_index: usize,
) -> i32 {
    if let Some(met) = unsafe { metronome_from_handle(handle) } {
        if divs_ptr.is_null() {
            if met
                .producer
                .push(MetronomeCommand::SetPolyrhythm((vec![], beat_index)))
                .is_ok()
            {
                return 0;
            }
            return -1;
        }

        let subdivisions = unsafe { slice::from_raw_parts(divs_ptr, len).to_vec() };

        if met
            .producer
            .push(MetronomeCommand::SetPolyrhythm((subdivisions, beat_index)))
            .is_ok()
        {
            return 0;
        }
    }
    -1
}

/// Set the metronome beat pattern.
#[unsafe(no_mangle)]
pub extern "C" fn metronome_set_pattern(handle: i64, pattern_ptr: *const i32, len: usize) -> i32 {
    if let Some(met) = unsafe { metronome_from_handle(handle) } {
        if pattern_ptr.is_null() || len == 0 {
            return -1;
        }

        let raw_pattern = unsafe { slice::from_raw_parts(pattern_ptr, len) };

        let pattern: Vec<BeatStrength> = raw_pattern
            .iter()
            .map(|&p| match p {
                3 => BeatStrength::Strong,
                2 => BeatStrength::Medium,
                1 => BeatStrength::Weak,
                _ => BeatStrength::None,
            })
            .collect();

        if met
            .producer
            .push(MetronomeCommand::SetPattern(pattern))
            .is_ok()
        {
            return 0;
        }
    }
    -1
}

/// Handle to a running synthesizer instance.
pub struct SynthHandle {
    pub producer: rtrb::Producer<SynthCommand>,
}

/// Drop a synth handle and free its memory.
unsafe fn drop_synth_handle(handle: i64) {
    if handle != 0 {
        unsafe { drop(Box::from_raw(handle as *mut SynthHandle)) };
    }
}

/// Access a synth handle from an FFI handle.
unsafe fn synth_from_handle<'a>(handle: i64) -> Option<&'a mut SynthHandle> {
    if handle == 0 {
        return None;
    }
    unsafe { Some(&mut *(handle as *mut SynthHandle)) }
}

/// Start a synthesizer and return its handle.
#[unsafe(no_mangle)]
pub extern "C" fn synth_start(pipeline_handle: i64) -> i64 {
    let builder = unsafe { pipeline_from_handle(pipeline_handle) };
    if let Some(b) = builder {
        if let Err(e) = b.start_output() {
            eprintln!("Failed to start output for synth: {}", e);
            return 0;
        }
        let producer = b.spawn_synthesizer();
        let handle = Box::new(SynthHandle { producer });
        println!("Synthesizer started.");
        Box::into_raw(handle) as i64
    } else {
        0
    }
}

/// Stop the synthesizer and free its resources.
#[unsafe(no_mangle)]
pub extern "C" fn synth_stop(handle: i64) -> i32 {
    unsafe {
        if let Some(s) = synth_from_handle(handle) {
            let _ = s.producer.push(SynthCommand::Stop);
            drop_synth_handle(handle);
            println!("Synthesizer stopped.");
            0
        } else {
            -1
        }
    }
}

/// Clear synthesizer state.
#[unsafe(no_mangle)]
pub extern "C" fn synth_clear(handle: i64) -> i32 {
    if let Some(s) = unsafe { synth_from_handle(handle) } {
        if s.producer.push(SynthCommand::Clear).is_ok() {
            return 0;
        }
    }
    -1
}

/// Play a live note on the synthesizer.
#[unsafe(no_mangle)]
pub extern "C" fn synth_play_live(handle: i64, freq: f32, velocity: f32) -> i32 {
    if let Some(s) = unsafe { synth_from_handle(handle) } {
        if velocity > 0.0 {
            if s.producer
                .push(SynthCommand::NoteOn {
                    freq,
                    velocity,
                    instrument: generators::Instrument::Violin,
                })
                .is_ok()
            {
                return 0;
            }
        } else {
            if s.producer.push(SynthCommand::NoteOff { freq }).is_ok() {
                return 0;
            }
        }
    }
    -1
}

/// Load a MIDI file into the synthesizer.
#[unsafe(no_mangle)]
pub extern "C" fn synth_load_file(handle: i64, midi_path_ptr: *const c_char) -> i32 {
    let path = unsafe { cstr_to_rust_str(midi_path_ptr) };
    let synth = unsafe { synth_from_handle(handle) };
    if let (Some(s), Some(p)) = (synth, path) {
        if s.producer
            .push(SynthCommand::LoadFile(
                p.to_string(),
                generators::Instrument::Piano,
            ))
            .is_ok()
        {
            println!("Sequence loaded: {}", p);
            return 0;
        }
    }
    -1
}

/// Start synthesizer playback from a specific measure.
#[unsafe(no_mangle)]
pub extern "C" fn synth_play_from(handle: i64, measure: usize) -> i32 {
    if let Some(s) = unsafe { synth_from_handle(handle) } {
        if s.producer
            .push(SynthCommand::Play {
                start_measure_idx: measure,
            })
            .is_ok()
        {
            return 0;
        }
    }
    -1
}

/// Pause synthesizer playback.
#[unsafe(no_mangle)]
pub extern "C" fn synth_pause(handle: i64) -> i32 {
    if let Some(s) = unsafe { synth_from_handle(handle) } {
        if s.producer.push(SynthCommand::Pause).is_ok() {
            return 0;
        }
    }
    -1
}

/// Resume synthesizer playback.
#[unsafe(no_mangle)]
pub extern "C" fn synth_resume(handle: i64) -> i32 {
    if let Some(s) = unsafe { synth_from_handle(handle) } {
        if s.producer.push(SynthCommand::Resume).is_ok() {
            return 0;
        }
    }
    -1
}

/// Set synthesizer volume.
#[unsafe(no_mangle)]
pub extern "C" fn synth_set_vol(handle: i64, volume: f32) -> i32 {
    if let Some(s) = unsafe { synth_from_handle(handle) } {
        if s.producer.push(SynthCommand::SetVolume(volume)).is_ok() {
            return 0;
        }
    }
    -1
}

/// Start a drone generator (not yet implemented).
#[unsafe(no_mangle)]
pub extern "C" fn drone_start(_pitch: f32) -> i64 {
    println!("TODO: Drone spawn logic.");
    0
}

/// Handle to a running audio player instance.
pub struct PlayerHandle {
    pub controller: PlayerController,
}

/// Drop a player handle and free its memory.
unsafe fn drop_player_handle(handle: i64) {
    if handle != 0 {
        unsafe { drop(Box::from_raw(handle as *mut PlayerHandle)) };
    }
}

/// Access a player handle from an FFI handle.
unsafe fn player_from_handle<'a>(handle: i64) -> Option<&'a mut PlayerHandle> {
    if handle == 0 {
        return None;
    }
    unsafe { Some(&mut *(handle as *mut PlayerHandle)) }
}

/// Create and start an audio player.
#[unsafe(no_mangle)]
pub extern "C" fn player_create(pipeline_handle: i64) -> i64 {
    let builder = unsafe { pipeline_from_handle(pipeline_handle) };
    if let Some(b) = builder {
        if let Err(e) = b.start_output() {
            eprintln!("Failed to start output for player: {}", e);
            return 0;
        }

        let controller = b.spawn_player();

        let handle = Box::new(PlayerHandle { controller });
        println!("Audio player created.");
        Box::into_raw(handle) as i64
    } else {
        0
    }
}

/// Load an audio file into the player.
#[unsafe(no_mangle)]
pub extern "C" fn player_load_track(handle: i64, path_ptr: *const c_char) -> i32 {
    let p_handle = unsafe { player_from_handle(handle) };
    let path = unsafe { cstr_to_rust_str(path_ptr) };

    if let (Some(ph), Some(p)) = (p_handle, path) {
        println!("Loading track: {}", p);
        if let Err(e) = ph.controller.load_file(p) {
            eprintln!("Failed to load track '{}': {}", p, e);
            return -1;
        }
        return 0;
    }
    -1
}

/// Start playback.
#[unsafe(no_mangle)]
pub extern "C" fn player_play(handle: i64) -> i32 {
    if let Some(ph) = unsafe { player_from_handle(handle) } {
        ph.controller.play();
        return 0;
    }
    -1
}

/// Pause playback.
#[unsafe(no_mangle)]
pub extern "C" fn player_pause(handle: i64) -> i32 {
    if let Some(ph) = unsafe { player_from_handle(handle) } {
        ph.controller.pause();
        return 0;
    }
    -1
}

/// Stop playback.
#[unsafe(no_mangle)]
pub extern "C" fn player_stop(handle: i64) -> i32 {
    if let Some(ph) = unsafe { player_from_handle(handle) } {
        ph.controller.stop();
        return 0;
    }
    -1
}

/// Seek to a position in the track.
#[unsafe(no_mangle)]
pub extern "C" fn player_seek(handle: i64, seconds: f64) -> i32 {
    if let Some(ph) = unsafe { player_from_handle(handle) } {
        ph.controller.seek(seconds);
        return 0;
    }
    -1
}

/// Destroy the player and free its resources.
#[unsafe(no_mangle)]
pub extern "C" fn player_destroy(handle: i64) -> i32 {
    unsafe {
        if handle != 0 {
            if let Some(ph) = player_from_handle(handle) {
                ph.controller.stop();
            }
            drop_player_handle(handle);
            println!("Player destroyed.");
            return 0;
        }
    }
    -1
}
