pub mod analysis;
pub mod audio_io;
pub mod dsp;
pub mod generators;
// pub mod pipeline;
pub mod resample;
pub mod testing;
pub mod traits;

use std::ffi::CStr;
use std::os::raw::c_char;
use std::{panic, slice};
// use std::sync::Arc;

use crate::audio_io::AudioPipeline;
use crate::generators::player::PlayerController;
use crate::generators::{
    SynthNote, Waveform,
    metronome::{BeatStrength, MetronomeCommand},
    synth::SynthCommand,
};
use crate::traits::Worker;

// ============================================================================
//  FFI HANDLE MANAGEMENT (The Wrapper)
// ============================================================================

/// A concrete wrapper around the dynamic Worker trait.
/// We wrap the Worker and a pointer to the parent Pipeline to handle cleanup.
pub struct WorkerHandle {
    pub worker: Box<dyn Worker>,
    pub pipeline: *mut AudioPipeline,
}

/// Helper: Wraps a specific worker into a generic FFI handle.
fn into_worker_handle<T: Worker + 'static>(worker: T, pipeline: *mut AudioPipeline) -> i64 {
    let boxed_worker: Box<dyn Worker> = Box::new(worker);
    let wrapper = Box::new(WorkerHandle {
        worker: boxed_worker,
        pipeline,
    });
    Box::into_raw(wrapper) as i64
}

/// Helper: Access the generic Worker from the FFI handle.
unsafe fn worker_from_handle<'a>(handle: i64) -> Option<&'a mut Box<dyn Worker>> {
    if handle == 0 {
        return None;
    }
    unsafe {
        let wrapper = &mut *(handle as *mut WorkerHandle);
        Some(&mut wrapper.worker)
    }
}

/// Helper: Consumes (drops) the generic Worker handle.
unsafe fn drop_worker_handle(handle: i64) {
    if handle != 0 {
        unsafe {
            drop(Box::from_raw(handle as *mut WorkerHandle));
        }
    }
}

/// Helper: Generic pipeline handle management
fn into_pipeline_handle(p: AudioPipeline) -> i64 {
    Box::into_raw(Box::new(p)) as i64
}

unsafe fn pipeline_from_handle<'a>(handle: i64) -> Option<&'a mut AudioPipeline> {
    if handle == 0 {
        return None;
    }
    unsafe { Some(&mut *(handle as *mut AudioPipeline)) }
}

pub unsafe fn cstr_to_rust_str(c_ptr: *const c_char) -> Option<&'static str> {
    if c_ptr.is_null() {
        return None;
    }
    unsafe { CStr::from_ptr(c_ptr).to_str().ok() }
}

pub struct MetronomeHandle {
    pub producer: rtrb::Producer<MetronomeCommand>,
}

unsafe fn drop_metronome_handle(handle: i64) {
    if handle != 0 {
        unsafe { drop(Box::from_raw(handle as *mut MetronomeHandle)) };
    }
}

unsafe fn metronome_from_handle<'a>(handle: i64) -> Option<&'a mut MetronomeHandle> {
    if handle == 0 {
        return None;
    }
    unsafe { Some(&mut *(handle as *mut MetronomeHandle)) }
}

// ============================================================================
//  PIPELINE LIFECYCLE
// ============================================================================

#[unsafe(no_mangle)]
pub extern "C" fn init_pipeline() -> i64 {
    // Safety: catch_unwind prevents Rust panics from crashing the C/Java/Swift runtime
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

#[unsafe(no_mangle)]
pub extern "C" fn shutdown_pipeline(handle: i64) {
    unsafe {
        if handle != 0 {
            // Dropping the AudioPipeline will trigger its Drop trait.
            // This should clean up input/output streams and cpal resources.
            drop(Box::from_raw(handle as *mut AudioPipeline));
        }
    }
    println!("Pipeline shut down and memory freed.");
}

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

#[unsafe(no_mangle)]
pub extern "C" fn play_start(handle: i64) -> i32 {
    if let Some(builder) = unsafe { pipeline_from_handle(handle) } {
        println!("▶️  Play command received.");
        // Note: play likely needs output stream, not input.
        // If play requires start_input() logic, keep as is.
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

#[unsafe(no_mangle)]
pub extern "C" fn pause_command(_handle: i64) -> i32 {
    // For global pipeline pause
    println!("⏸️  Global Pause command.");
    0
}

// ============================================================================
//  GENERIC WORKER CONTROL
// ============================================================================

#[unsafe(no_mangle)]
pub extern "C" fn worker_start(handle: i64) -> i32 {
    if let Some(worker) = unsafe { worker_from_handle(handle) } {
        worker.start();
        0
    } else {
        -1
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn worker_pause(handle: i64) -> i32 {
    if let Some(worker) = unsafe { worker_from_handle(handle) } {
        worker.pause();
        0
    } else {
        -1
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn worker_stop(handle: i64) -> i32 {
    if handle == 0 {
        return -1;
    }

    let wrapper = unsafe { &mut *(handle as *mut WorkerHandle) };

    // 1. Stop the worker logic (returns the consumer handle ID)
    let consumer_id = wrapper.worker.stop();

    // 2. Implicitly remove consumer from the pipeline
    if !wrapper.pipeline.is_null() {
        unsafe {
            let pipeline = &mut *wrapper.pipeline;

            // A. Remove the specific consumer for this worker
            pipeline.remove_consumer(consumer_id);

            // B. Check if we should stop the microphone/input stream
            pipeline.stop_input();
        }
    }

    // 3. Free the worker wrapper memory
    unsafe {
        drop_worker_handle(handle);
    }

    println!("Worker stopped, consumer removed, memory released.");
    0
}

// ============================================================================
//  SPECIFIC CREATORS (Return Generic Handles)
// ============================================================================

#[unsafe(no_mangle)]
pub extern "C" fn record_start(pipeline_handle: i64, temp_path_ptr: *const c_char) -> i64 {
    let builder = unsafe { pipeline_from_handle(pipeline_handle) };
    let path = unsafe { cstr_to_rust_str(temp_path_ptr) };

    if let (Some(b), Some(p)) = (builder, path) {
        // 1. Ensure Input Stream is Running
        // We must start the input before spawning the recorder so data is available.
        if let Err(e) = b.start_input() {
            eprintln!("Failed to start input stream for recording: {}", e);
            return 0;
        }

        println!("🎙️  Spawning Recorder to: {}", p);
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

pub type AnalysisCallback = extern "C" fn(name: *const c_char, cents: f32);

#[unsafe(no_mangle)]
pub extern "C" fn tune_start(pipeline_handle: i64, callback: AnalysisCallback) -> i64 {
    let builder = unsafe { pipeline_from_handle(pipeline_handle) };

    if let Some(b) = builder {
        // 1. Ensure Input Stream is Running
        if let Err(e) = b.start_input() {
            eprintln!("Failed to start input stream for tuning: {}", e);
            return 0;
        }

        println!("🎵  Spawning Tuner.");
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

// ============================================================================
//  METRONOME CONTROLS
// ============================================================================

#[unsafe(no_mangle)]
pub extern "C" fn metronome_start(pipeline_handle: i64, bpm: f32) -> i64 {
    let builder = unsafe { pipeline_from_handle(pipeline_handle) };

    if let Some(b) = builder {
        // Ensure audio output is running
        if let Err(e) = b.start_output() {
            eprintln!("Failed to start output for metronome: {}", e);
            return 0;
        }

        // Spawn and get the producer
        let producer = b.spawn_metronome(bpm);

        // Wrap in a handle
        let handle = Box::new(MetronomeHandle { producer });

        println!("⏱️  Metronome started at {} BPM", bpm);
        Box::into_raw(handle) as i64
    } else {
        0
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn metronome_stop(handle: i64) -> i32 {
    unsafe {
        if let Some(met) = metronome_from_handle(handle) {
            // Option A: Send explicit stop (Good for gentle cleanup)
            let _ = met.producer.push(MetronomeCommand::Stop);

            // Option B: Just drop the handle (Disconnects channel)
            drop_metronome_handle(handle);
            println!("⏱️  Metronome stopped.");
            0
        } else {
            -1
        }
    }
}

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

/// Sets active subdivisions using an array of integers.
///
/// `divs_ptr`: Pointer to an array of `usize` (e.g., [3, 4] for triplets + 16ths).
/// `len`: Number of elements in the array.
#[unsafe(no_mangle)]
pub extern "C" fn metronome_set_subdivisions(
    handle: i64,
    divs_ptr: *const usize,
    len: usize,
) -> i32 {
    if let Some(met) = unsafe { metronome_from_handle(handle) } {
        if divs_ptr.is_null() {
            // If null, assume clear subdivisions
            if met
                .producer
                .push(MetronomeCommand::SetSubdivisions(vec![]))
                .is_ok()
            {
                return 0;
            }
            return -1;
        }

        let subdivisions = unsafe { slice::from_raw_parts(divs_ptr, len).to_vec() };

        if met
            .producer
            .push(MetronomeCommand::SetSubdivisions(subdivisions))
            .is_ok()
        {
            return 0;
        }
    }
    -1
}

/// Sets the main beat pattern using an array of integers.
///
/// Mappings:
/// 0: None (Silence)
/// 1: Weak
/// 2: Medium
/// 3: Strong
///
/// `pattern_ptr`: Pointer to an array of `i32`.
/// `len`: Number of beats in the pattern.
#[unsafe(no_mangle)]
pub extern "C" fn metronome_set_pattern(handle: i64, pattern_ptr: *const i32, len: usize) -> i32 {
    if let Some(met) = unsafe { metronome_from_handle(handle) } {
        if pattern_ptr.is_null() || len == 0 {
            return -1;
        }

        // Convert raw C array to Rust slice
        let raw_pattern = unsafe { slice::from_raw_parts(pattern_ptr, len) };

        // Map integers to BeatStrength enum
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

// ============================================================================
//  SYNTHESIZER FFI
// ============================================================================

pub struct SynthHandle {
    pub producer: rtrb::Producer<SynthCommand>,
}

unsafe fn drop_synth_handle(handle: i64) {
    if handle != 0 {
        unsafe { drop(Box::from_raw(handle as *mut SynthHandle)) };
    }
}

unsafe fn synth_from_handle<'a>(handle: i64) -> Option<&'a mut SynthHandle> {
    if handle == 0 {
        return None;
    }
    unsafe { Some(&mut *(handle as *mut SynthHandle)) }
}

/// A C-compatible struct for passing note data.
#[repr(C)]
pub struct FFISynthNote {
    pub freq: f32,
    pub duration_beats: f32,
    pub onset_beats: f32,
    pub velocity: f32,
}

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
        println!("🎹 Synthesizer spawned.");
        Box::into_raw(handle) as i64
    } else {
        0
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn synth_stop(handle: i64) -> i32 {
    unsafe {
        if let Some(s) = synth_from_handle(handle) {
            let _ = s.producer.push(SynthCommand::Stop);
            drop_synth_handle(handle);
            println!("🎹 Synthesizer stopped.");
            0
        } else {
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn synth_play_live(handle: i64, freq: f32, velocity: f32) -> i32 {
    if let Some(s) = unsafe { synth_from_handle(handle) } {
        if velocity > 0.0 {
            if s.producer
                .push(SynthCommand::NoteOn {
                    freq,
                    velocity,
                    waveform: Waveform::Sine,
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

#[unsafe(no_mangle)]
pub extern "C" fn synth_load_sequence(
    handle: i64,
    notes_ptr: *const FFISynthNote,
    len: usize,
) -> i32 {
    if let Some(s) = unsafe { synth_from_handle(handle) } {
        if notes_ptr.is_null() {
            return -1;
        }

        let raw_notes = unsafe { slice::from_raw_parts(notes_ptr, len) };

        // Convert FFI structs to internal Rust structs
        let notes: Vec<SynthNote> = raw_notes
            .iter()
            .map(|n| SynthNote {
                freq: n.freq,
                duration_beats: n.duration_beats,
                onset_beats: n.onset_beats,
                velocity: n.velocity,
                waveform: Waveform::Triangle,
            })
            .collect();

        if s.producer.push(SynthCommand::LoadSequence(notes)).is_ok() {
            println!("🎹 Sequence loaded with {} notes", len);
            return 0;
        }
    }
    -1
}

#[unsafe(no_mangle)]
pub extern "C" fn synth_pause(handle: i64) -> i32 {
    if let Some(s) = unsafe { synth_from_handle(handle) } {
        if s.producer.push(SynthCommand::Pause).is_ok() {
            return 0;
        }
    }
    -1
}

#[unsafe(no_mangle)]
pub extern "C" fn synth_resume(handle: i64) -> i32 {
    if let Some(s) = unsafe { synth_from_handle(handle) } {
        if s.producer.push(SynthCommand::Resume).is_ok() {
            return 0;
        }
    }
    -1
}

#[unsafe(no_mangle)]
pub extern "C" fn synth_set_vol(handle: i64, volume: f32) -> i32 {
    if let Some(s) = unsafe { synth_from_handle(handle) } {
        if s.producer.push(SynthCommand::SetVolume(volume)).is_ok() {
            return 0;
        }
    }
    -1
}

#[unsafe(no_mangle)]
pub extern "C" fn drone_start(_pitch: f32) -> i64 {
    println!("TODO: Drone spawn logic.");
    0
}

// ============================================================================
//  PLAYER FFI (NEW)
// ============================================================================

pub struct PlayerHandle {
    pub controller: PlayerController,
}

unsafe fn drop_player_handle(handle: i64) {
    if handle != 0 {
        unsafe { drop(Box::from_raw(handle as *mut PlayerHandle)) };
    }
}

unsafe fn player_from_handle<'a>(handle: i64) -> Option<&'a mut PlayerHandle> {
    if handle == 0 {
        return None;
    }
    unsafe { Some(&mut *(handle as *mut PlayerHandle)) }
}

#[unsafe(no_mangle)]
pub extern "C" fn player_create(pipeline_handle: i64) -> i64 {
    let builder = unsafe { pipeline_from_handle(pipeline_handle) };
    if let Some(b) = builder {
        // Ensure audio output is running
        if let Err(e) = b.start_output() {
            eprintln!("Failed to start output for player: {}", e);
            return 0;
        }

        // IMPORTANT: You need to implement spawn_player() in your AudioPipeline struct
        let controller = b.spawn_player();

        let handle = Box::new(PlayerHandle { controller });
        println!("🎧 Audio Player spawned.");
        Box::into_raw(handle) as i64
    } else {
        0
    }
}

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

#[unsafe(no_mangle)]
pub extern "C" fn player_play(handle: i64) -> i32 {
    if let Some(ph) = unsafe { player_from_handle(handle) } {
        ph.controller.play();
        return 0;
    }
    -1
}

#[unsafe(no_mangle)]
pub extern "C" fn player_pause(handle: i64) -> i32 {
    if let Some(ph) = unsafe { player_from_handle(handle) } {
        ph.controller.pause();
        return 0;
    }
    -1
}

#[unsafe(no_mangle)]
pub extern "C" fn player_stop(handle: i64) -> i32 {
    if let Some(ph) = unsafe { player_from_handle(handle) } {
        ph.controller.stop();
        return 0;
    }
    -1
}

#[unsafe(no_mangle)]
pub extern "C" fn player_seek(handle: i64, seconds: f64) -> i32 {
    if let Some(ph) = unsafe { player_from_handle(handle) } {
        ph.controller.seek(seconds);
        return 0;
    }
    -1
}

#[unsafe(no_mangle)]
pub extern "C" fn player_destroy(handle: i64) -> i32 {
    unsafe {
        if handle != 0 {
            if let Some(ph) = player_from_handle(handle) {
                ph.controller.stop();
            }
            drop_player_handle(handle);
            println!("🎧 Player destroyed.");
            return 0;
        }
    }
    -1
}
