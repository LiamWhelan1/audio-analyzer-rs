use rtrb::{Consumer, Producer, RingBuffer};
use serde::Serialize;
use std::{sync::Arc, thread, time::Duration};

use crate::{analysis::theory::Interval, generators::Note};

// ─── Tuner enums ────────────────────────────────────────────────────

#[derive(PartialEq, Copy, Clone, Debug, Serialize, Default)]
pub enum TuningSystem {
    #[default]
    EqualTemperament,
    JustIntonation,
}

#[derive(PartialEq, Clone, Copy, Debug, Serialize, Default)]
pub enum TunerMode {
    #[default]
    MultiPitch,
    SinglePitch,
}

// ─── Commands: sent from any thread (UI, FFI, etc.) into the Tuner ──

pub enum TunerCommand {
    SetKey(String),
    SetBaseFreq(f32),
    SetMode(TunerMode),
    SetSystem(TuningSystem),
}

// ─── Output payload: what the React frontend reads ──────────────────
//
// This is the data that leaves the audio thread. We wrap it in an
// `Arc<parking_lot::RwLock<_>>` so the frontend can read it at its own
// pace (requestAnimationFrame / setInterval) without blocking audio.
//
// Why RwLock instead of Mutex?
//   • The audio thread is the *only* writer and writes infrequently
//     (once per analysis hop — ~23 ms at 2048/44100).
//   • Multiple JS polling intervals or React render cycles may read
//     concurrently.  RwLock lets them all proceed without contention.
//
// Why parking_lot?
//   • No poisoning (audio thread panics don't wedge the UI).
//   • try_write / try_read are wait-free when uncontested, which keeps
//     the audio thread deterministic.

#[derive(Clone, Debug, Serialize, Default)]
pub struct TunerOutput {
    /// The primary display string — e.g. "C#4", "Am7", "P5"
    pub label: String,
    /// Cents offset from the nearest target pitch (-50..+50)
    pub cents: f32,
    /// Individual note names detected this frame
    pub notes: Vec<String>,
    /// Per-note cent accuracy
    pub accuracies: Vec<f32>,
    /// Current tuner configuration echoed back for the UI
    pub mode: TunerMode,
    pub system: TuningSystem,
    pub base_freq: f32,
    pub key: String,
}

// ─── Tuner ──────────────────────────────────────────────────────────

pub struct Tuner {
    key: String, // e.g. "C major"
    base: f32,
    mode: TunerMode,
    system: TuningSystem,

    // Channel for inbound parameter changes (like MetronomeCommand)
    command_rx: Consumer<TunerCommand>,

    // Channel carrying raw detected (freq, hit_count) from STFT
    note_rx: Consumer<Vec<(f32, i32)>>,

    // Shared output that any reader (React, FFI callback, etc.) can poll
    output: Arc<parking_lot::RwLock<TunerOutput>>,
}

impl Tuner {
    /// Create a new Tuner.
    ///
    /// Returns `(Tuner, Producer<TunerCommand>, Arc<RwLock<TunerOutput>>)`.
    ///
    /// * The `Producer` is held by whatever thread dispatches UI actions
    ///   (e.g. an FFI bridge or a wasm-bindgen callback).
    /// * The `Arc<RwLock<TunerOutput>>` is cloned into the React bridge
    ///   so JS can poll it via `requestAnimationFrame`.
    pub fn new(
        note_rx: Consumer<Vec<(f32, i32)>>,
    ) -> (
        Self,
        Producer<TunerCommand>,
        Arc<parking_lot::RwLock<TunerOutput>>,
    ) {
        let (cmd_tx, cmd_rx) = RingBuffer::new(64);
        let output = Arc::new(parking_lot::RwLock::new(TunerOutput::default()));

        let tuner = Self {
            key: "C major".into(),
            base: 440.0,
            mode: TunerMode::MultiPitch,
            system: TuningSystem::EqualTemperament,
            command_rx: cmd_rx,
            note_rx,
            output: Arc::clone(&output),
        };

        (tuner, cmd_tx, output)
    }

    /// Drain and apply every pending command — identical pattern to
    /// `Metronome::handle_commands`.
    fn handle_commands(&mut self) {
        while let Ok(cmd) = self.command_rx.pop() {
            match cmd {
                TunerCommand::SetBaseFreq(b) => self.base = b.clamp(220.0, 880.0),
                TunerCommand::SetKey(k) => self.key = k,
                TunerCommand::SetMode(m) => self.mode = m,
                TunerCommand::SetSystem(s) => self.system = s,
            }
        }
    }

    /// Run the tuner loop on a dedicated thread.
    ///
    /// This consumes `self` — the thread owns all mutable state.
    /// Other threads interact only through the command channel and
    /// the shared `TunerOutput`.
    pub fn run(mut self) {
        thread::spawn(move || {
            while !self.note_rx.is_abandoned() {
                // 1. Apply any parameter changes from the UI thread
                self.handle_commands();

                // 2. Process incoming note data from STFT
                let notes_data = match self.note_rx.pop() {
                    Ok(n) => n,
                    Err(_) => {
                        thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                };

                if notes_data.is_empty() {
                    continue;
                }

                // 3. Build the output depending on mode
                let label: String;
                let mut cents: f32 = 0.0;
                let mut note_names: Vec<String> = Vec::new();
                let mut accuracies: Vec<f32> = Vec::new();

                if notes_data.len() == 1 || self.mode == TunerMode::SinglePitch {
                    // Pick the note with the most hits
                    let best = notes_data.iter().max_by(|a, b| a.1.cmp(&b.1)).unwrap();
                    let note = Note::from_freq(best.0, Some(self.base));
                    label = note.get_name();
                    cents = note.get_cents();
                    note_names.push(note.get_name());
                    accuracies.push(note.get_cents());
                } else if notes_data.len() == 2 {
                    let mut freqs: Vec<f32> = notes_data.iter().map(|(f, _)| *f).collect();
                    freqs.sort_by(|a, b| a.total_cmp(&b));
                    // Interval detection (uses tuning system)
                    let interval = Interval::new(&freqs, Some(self.system));
                    let notes: Vec<Note> = freqs
                        .iter()
                        .map(|f| Note::from_freq(*f, Some(self.base)))
                        .collect();
                    for note in notes {
                        note_names.push(note.get_name());
                        accuracies.push(note.get_cents());
                    }
                    label = interval.get_name();
                    cents = interval.get_accuracy();
                } else {
                    // 3+ notes → chord identification (placeholder)
                    for (freq, _) in &notes_data {
                        let note = Note::from_freq(*freq, Some(self.base));
                        note_names.push(note.get_name());
                        accuracies.push(note.get_cents());
                    }
                    label = note_names.join(" ");
                }

                // 4. Publish to the shared output — non-blocking write
                //    try_write avoids stalling the audio pipeline if a
                //    reader is mid-read (extremely rare with parking_lot).
                if let Some(mut out) = self.output.try_write() {
                    out.label = label;
                    out.cents = cents;
                    out.notes = note_names;
                    out.accuracies = accuracies;
                    out.mode = self.mode;
                    out.system = self.system;
                    out.base_freq = self.base;
                    out.key = self.key.clone();
                }
            }
        });
    }
}

// ─── Example: wiring it all together ────────────────────────────────
//
// In main / lib init:
//
//   let (note_tx, note_rx) = RingBuffer::<Vec<(f32, i32)>>::new(64);
//
//   // STFT writes confirmed_candidates into note_tx instead of
//   // building CStrings and calling the FFI callback directly.
//
//   let (tuner, cmd_tx, tuner_output) = Tuner::new(note_rx);
//   tuner.run();   // spawns its own thread
//
//   // Hand `cmd_tx` to the FFI / wasm-bindgen layer that talks to JS.
//   // Hand `tuner_output` to the polling bridge (see below).
//
// ─── React bridge (FFI or wasm-bindgen) ─────────────────────────────
//
// ### Option A — FFI callback at ~60 Hz (matches requestAnimationFrame)
//
//   A dedicated timer thread reads `tuner_output` and invokes a C
//   callback that the React native bridge registered:
//
//     fn start_ui_pump(
//         output: Arc<parking_lot::RwLock<TunerOutput>>,
//         cb: extern "C" fn(*const c_char),  // receives JSON
//     ) {
//         thread::spawn(move || loop {
//             {
//                 let snap = output.read();
//                 let json = serde_json::to_string(&*snap).unwrap();
//                 let c = CString::new(json).unwrap();
//                 cb(c.as_ptr());
//             }
//             thread::sleep(Duration::from_millis(16)); // ~60 fps
//         });
//     }
//
// ### Option B — wasm-bindgen (WebAssembly in the browser)
//
//   Expose a `poll_tuner()` function that JS calls every frame:
//
//     #[wasm_bindgen]
//     pub fn poll_tuner() -> JsValue {
//         let snap = TUNER_OUTPUT.read();        // global Arc
//         serde_wasm_bindgen::to_value(&*snap).unwrap()
//     }
//
//     #[wasm_bindgen]
//     pub fn set_tuner_base_freq(freq: f32) {
//         CMD_TX.lock().push(TunerCommand::SetBaseFreq(freq)).ok();
//     }
//
//   React side:
//
//     useEffect(() => {
//       let id = requestAnimationFrame(function tick() {
//         const data = wasm.poll_tuner();
//         setTunerData(data);
//         id = requestAnimationFrame(tick);
//       });
//       return () => cancelAnimationFrame(id);
//     }, []);
