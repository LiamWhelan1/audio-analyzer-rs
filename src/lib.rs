pub mod analysis;
pub mod audio_io;
pub mod dsp;
pub mod generators;
// pub mod pipeline;
pub mod resample;
pub mod testing;
pub mod traits;

use anyhow::Result;
use std::ffi::CStr;
use std::os::raw::c_char;
use std::panic;
// use std::sync::Arc;

use crate::audio_io::AudioPipeline;
use crate::audio_io::recorder::Recorder;
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

unsafe fn cstr_to_rust_str(c_ptr: *const c_char) -> Option<&'static str> {
    if c_ptr.is_null() {
        return None;
    }
    unsafe { CStr::from_ptr(c_ptr).to_str().ok() }
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
        match builder.start_input() {
            Ok(_) => 0,
            Err(e) => {
                eprintln!("Error starting input for playback: {}", e);
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

pub type AnalysisCallback = extern "C" fn(pitch_hz: f32);

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

#[unsafe(no_mangle)]
pub extern "C" fn metronome_start(_bpm: f32) -> i64 {
    println!("TODO: Metronome spawn logic.");
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn drone_start(_pitch: f32) -> i64 {
    println!("TODO: Drone spawn logic.");
    0
}
