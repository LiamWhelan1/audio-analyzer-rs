# Debug Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ASCII stderr debug output in `src/audio_io/stft.rs` with a Rerun live spectrum viewer and a `plotters`-based static three-panel PNG exporter.

**Architecture:** The `#[cfg(debug_assertions)]` gate and all ASCII rendering code are deleted. All new visualization code lives under `#[cfg(feature = "dev-tools")]`. The Rerun `RecordingStream` is initialized once at thread start; `dbg_log_rerun` is called every FFT frame. A separate `png_frame` counter fires `dbg_export_png` every 2500 frames; raw/windowed sample buffers are only captured on those frames to avoid per-frame allocations.

**Tech Stack:** Rust, `rerun 0.30.1` (already in deps), `plotters 0.3` (new optional dep), both gated on `dev-tools` feature.

---

## File Map

| File | Change |
|---|---|
| `Cargo.toml` | Add `plotters = { version = "0.3", optional = true }` to `[dependencies]`; add `"plotters"` to `dev-tools` feature |
| `src/audio_io/stft.rs` | Add `freq_to_note_label`; replace all `#[cfg(debug_assertions)]` blocks with `#[cfg(feature = "dev-tools")]` blocks; add `dbg_log_rerun` and `dbg_export_png` functions; delete ASCII helpers |

---

## Task 1: Add `plotters` dependency and verify both crates compile

**Files:**
- Modify: `Cargo.toml`

- [ ] **Step 1: Add plotters to Cargo.toml**

  Open `Cargo.toml`. Change the `dev-tools` feature line and add the plotters dep:

  ```toml
  [features]
  dev-tools = ["rerun", "plotters"]

  [dependencies]
  # existing deps stay unchanged; add this line:
  plotters = { version = "0.3", optional = true }
  ```

- [ ] **Step 2: Verify both crates resolve**

  ```
  cargo build --features dev-tools 2>&1 | head -30
  ```

  Expected: compiles (or fails only due to unused import warnings, not missing crates). If `plotters` fails to resolve, run `cargo update` first.

- [ ] **Step 3: Commit**

  ```bash
  git add Cargo.toml Cargo.lock
  git commit -m "feat: add plotters optional dep under dev-tools"
  ```

---

## Task 2: Extract `freq_to_note_label` as an ungated helper

The existing `dbg_freq_to_note` lives inside `#[cfg(debug_assertions)] impl STFT`. It needs to be available to both the Rerun and PNG code paths. Move it out of the cfg block and rename it.

**Files:**
- Modify: `src/audio_io/stft.rs` lines 810–828 (inside `#[cfg(debug_assertions)] impl STFT`)

- [ ] **Step 1: Add `freq_to_note_label` to the main `impl STFT` block**

  Find the closing `}` of the main `impl STFT` block (currently around line 648, before the `// ─── Debug` comment). Add this function there:

  ```rust
  /// Convert a frequency in Hz to the nearest MIDI note name + cents deviation.
  fn freq_to_note_label(freq: f32) -> String {
      if freq <= 0.0 {
          return "?".to_string();
      }
      let midi = 69.0 + 12.0 * (freq / 440.0).log2();
      let midi_round = midi.round() as i32;
      let names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
      let name = names[((midi_round % 12 + 12) % 12) as usize];
      let octave = midi_round / 12 - 1;
      let cents = ((midi - midi.round()) * 100.0) as i32;
      if cents == 0 {
          format!("{name}{octave}")
      } else {
          format!("{name}{octave} {cents:+}¢")
      }
  }
  ```

- [ ] **Step 2: Build to confirm it compiles (debug_assertions still present, no breakage)**

  ```
  cargo build 2>&1 | grep "^error" | head -10
  ```

  Expected: no errors.

---

## Task 3: Wire up `dev-tools` frame counters and Rerun `RecordingStream` initialization

Replace the single `#[cfg(debug_assertions)] let mut dbg_fft_frame` counter with two `dev-tools`-gated counters plus a `RecordingStream`.

**Files:**
- Modify: `src/audio_io/stft.rs` lines 171–184 (thread spawn setup)

- [ ] **Step 1: Replace the debug_assertions frame counter**

  Find and delete:
  ```rust
  #[cfg(debug_assertions)]
  let mut dbg_fft_frame: usize = 0;
  ```

  Replace with:
  ```rust
  #[cfg(feature = "dev-tools")]
  let mut rerun_frame: usize = 0;
  #[cfg(feature = "dev-tools")]
  let mut png_frame: usize = 0usize;
  ```

- [ ] **Step 2: Initialize RecordingStream and create `debug_frames/` directory**

  Immediately after the two counters (still before the `while` loop), add:

  ```rust
  #[cfg(feature = "dev-tools")]
  let rec = rerun::RecordingStreamBuilder::new("audio_analyzer")
      .spawn()
      .expect("failed to spawn Rerun viewer");

  #[cfg(feature = "dev-tools")]
  std::fs::create_dir_all("debug_frames")
      .expect("failed to create debug_frames/ directory");
  ```

- [ ] **Step 3: Build with dev-tools to confirm**

  ```
  cargo build --features dev-tools 2>&1 | grep "^error" | head -10
  ```

  Expected: no errors. Rerun viewer binary may print a message but compilation succeeds.

- [ ] **Step 4: Commit**

  ```bash
  git add src/audio_io/stft.rs
  git commit -m "feat: add dev-tools frame counters and Rerun stream init"
  ```

---

## Task 4: Implement `dbg_log_rerun`

Add a new `#[cfg(feature = "dev-tools")] impl STFT` block at the bottom of the file with `dbg_log_rerun`.

**Files:**
- Modify: `src/audio_io/stft.rs` — append new impl block after line 928

- [ ] **Step 1: Append the new impl block**

  Add after the final closing `}` of the existing `#[cfg(debug_assertions)] impl STFT` block (line 928):

  ```rust
  // ─── dev-tools visualization helpers ─────────────────────────────────────────

  #[cfg(feature = "dev-tools")]
  impl STFT {
      /// Log one FFT frame to the Rerun live viewer.
      ///
      /// Three entities are logged:
      /// - `spectrum/line`        — full magnitude spectrum as a line strip (log10-freq x-axis)
      /// - `spectrum/noise_floor` — horizontal line at the noise floor magnitude
      /// - `spectrum/pitches`     — labeled points at each hysteresis-stable pitch
      fn dbg_log_rerun(
          rec: &rerun::RecordingStream,
          frame: usize,
          mags: &[f32],
          bin_width: f32,
          min_freq: f32,
          max_freq: f32,
          noise_floor: f32,
          stable: &[(f32, f32)],
      ) {
          rec.set_time_sequence("frame", frame as i64);

          let min_bin = ((min_freq / bin_width).ceil() as usize).max(1);
          let max_bin =
              ((max_freq / bin_width).floor() as usize).min(mags.len().saturating_sub(1));

          // Spectrum line — (log10(freq), magnitude) per bin.
          let spectrum_strip: Vec<[f32; 2]> = (min_bin..=max_bin)
              .map(|b| [(b as f32 * bin_width).log10(), mags[b]])
              .collect();
          let _ = rec.log(
              "spectrum/line",
              &rerun::LineStrips2D::new([spectrum_strip])
                  .with_colors([rerun::Color::from_rgb(100, 200, 255)]),
          );

          // Noise floor — horizontal line across the full frequency range.
          let x_lo = min_freq.log10();
          let x_hi = max_freq.log10();
          let _ = rec.log(
              "spectrum/noise_floor",
              &rerun::LineStrips2D::new([vec![[x_lo, noise_floor], [x_hi, noise_floor]]])
                  .with_colors([rerun::Color::from_rgb(255, 80, 80)]),
          );

          // Stable pitches — labeled points on the spectrum line.
          if !stable.is_empty() {
              let positions: Vec<[f32; 2]> = stable
                  .iter()
                  .map(|&(freq, _)| {
                      let bin = (freq / bin_width).round() as usize;
                      let mag = mags[bin.min(mags.len().saturating_sub(1))];
                      [freq.log10(), mag]
                  })
                  .collect();
              let labels: Vec<String> = stable
                  .iter()
                  .map(|&(freq, score)| {
                      format!("{} {:.1}", Self::freq_to_note_label(freq), score)
                  })
                  .collect();
              let _ = rec.log(
                  "spectrum/pitches",
                  &rerun::Points2D::new(positions)
                      .with_labels(labels)
                      .with_colors([rerun::Color::from_rgb(255, 220, 50)])
                      .with_radii([rerun::Radius::new_scene_units(0.003)]),
              );
          }
      }
  }
  ```

- [ ] **Step 2: Build**

  ```
  cargo build --features dev-tools 2>&1 | grep "^error" | head -20
  ```

  Expected: no errors. If `Radius::new_scene_units` doesn't exist on this version, replace with `rerun::components::Radius(0.003)` or remove the `.with_radii(...)` call entirely.

- [ ] **Step 3: Commit**

  ```bash
  git add src/audio_io/stft.rs
  git commit -m "feat: add dbg_log_rerun for Rerun live spectrum viewer"
  ```

---

## Task 5: Hook `dbg_log_rerun` into the per-frame processing loop

The Rerun call needs `mags`, `bin_width`, `noise_floor`, and `stable_pitches` — all available after the main FFT block. Hook it in without touching the silence path.

**Files:**
- Modify: `src/audio_io/stft.rs` — inside `while available_samples >= window_size` loop

- [ ] **Step 1: Capture frame data for the Rerun call**

  The current code computes `magnitudes`, `bin_width`, and `noise_floor` inside the `else` branch of `if is_silence`. To make them available to the `#[cfg(feature = "dev-tools")]` block after the branch, add an `Option` capture. 

  Before the `let raw_pitches = if is_silence {` line, add:

  ```rust
  #[cfg(feature = "dev-tools")]
  let mut dev_rerun_data: Option<(Vec<f32>, f32, f32)> = None;
  ```

  Inside the `else` branch, after the `let noise_floor = ...` computation (currently around line 302–305) but **before** the `Self::extract_pitches(...)` call, add:

  ```rust
  #[cfg(feature = "dev-tools")]
  {
      dev_rerun_data = Some((magnitudes.clone(), bin_width, noise_floor));
  }
  ```

- [ ] **Step 2: Call `dbg_log_rerun` after `pitch_tracker.process`**

  Find the line `let stable_pitches = pitch_tracker.process(raw_pitches);` (currently around line 317). Immediately after it, add:

  ```rust
  #[cfg(feature = "dev-tools")]
  {
      if let Some((ref mags, bw, nf)) = dev_rerun_data {
          Self::dbg_log_rerun(
              &rec,
              rerun_frame,
              mags,
              bw,
              MIN_FREQ,
              MAX_FREQ,
              nf,
              &stable_pitches,
          );
      }
      rerun_frame += 1;
  }
  ```

- [ ] **Step 3: Build**

  ```
  cargo build --features dev-tools 2>&1 | grep "^error" | head -20
  ```

  Expected: no errors.

- [ ] **Step 4: Commit**

  ```bash
  git add src/audio_io/stft.rs
  git commit -m "feat: call dbg_log_rerun every FFT frame under dev-tools"
  ```

---

## Task 6: Implement `dbg_export_png`

Add `dbg_export_png` to the existing `#[cfg(feature = "dev-tools")] impl STFT` block.

**Files:**
- Modify: `src/audio_io/stft.rs` — inside the `#[cfg(feature = "dev-tools")] impl STFT` block added in Task 4

- [ ] **Step 1: Add the function**

  Inside the `#[cfg(feature = "dev-tools")] impl STFT` block (after `dbg_log_rerun`), add:

  ```rust
  /// Export a three-panel PNG to `debug_frames/frame_{n}.png`.
  ///
  /// Panel 1 — Raw Signal (time domain)
  /// Panel 2 — Hann-Windowed Signal (same y-scale as panel 1)
  /// Panel 3 — Log-scale FFT spectrum with noise floor + pitch labels
  fn dbg_export_png(
      frame: usize,
      raw: &[f32],
      windowed: &[f32],
      mags: &[f32],
      bin_width: f32,
      min_freq: f32,
      max_freq: f32,
      noise_floor: f32,
      stable: &[(f32, f32)],
  ) -> Result<(), Box<dyn std::error::Error>> {
      use plotters::prelude::*;

      let path = format!("debug_frames/frame_{frame}.png");
      let root = BitMapBackend::new(&path, (1200, 1800)).into_drawing_area();
      root.fill(&RGBColor(20, 20, 30))?;

      let areas = root.split_evenly((3, 1));
      let (area1, area2, area3) = (&areas[0], &areas[1], &areas[2]);

      // ── Panel 1: Raw Signal ──────────────────────────────────────────────────
      let raw_min = raw.iter().cloned().fold(f32::INFINITY, f32::min);
      let raw_max = raw.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
      let y_lo = raw_min - (raw_max - raw_min).abs() * 0.05;
      let y_hi = raw_max + (raw_max - raw_min).abs() * 0.05;
      let y_lo = if (y_hi - y_lo).abs() < 1e-10 { y_lo - 1.0 } else { y_lo };
      let y_hi = if (y_hi - y_lo).abs() < 1e-10 { y_hi + 1.0 } else { y_hi };

      {
          let mut chart = ChartBuilder::on(area1)
              .caption(
                  format!("Raw Signal — Frame {frame}"),
                  ("sans-serif", 18).into_font().color(&WHITE),
              )
              .margin(15)
              .x_label_area_size(30)
              .y_label_area_size(70)
              .build_cartesian_2d(0usize..raw.len(), y_lo..y_hi)?;

          chart
              .configure_mesh()
              .x_labels(5)
              .y_labels(5)
              .label_style(("sans-serif", 11).into_font().color(&WHITE))
              .axis_style(RGBColor(100, 100, 120))
              .draw()?;

          chart.draw_series(LineSeries::new(
              raw.iter().enumerate().map(|(i, &v)| (i, v)),
              WHITE.stroke_width(1),
          ))?;
      }

      // ── Panel 2: Hann-Windowed Signal (shared y-range with Panel 1) ─────────
      {
          let mut chart = ChartBuilder::on(area2)
              .caption(
                  "Hann-Windowed Signal",
                  ("sans-serif", 18).into_font().color(&WHITE),
              )
              .margin(15)
              .x_label_area_size(30)
              .y_label_area_size(70)
              .build_cartesian_2d(0usize..windowed.len(), y_lo..y_hi)?;

          chart
              .configure_mesh()
              .x_labels(5)
              .y_labels(5)
              .label_style(("sans-serif", 11).into_font().color(&WHITE))
              .axis_style(RGBColor(100, 100, 120))
              .draw()?;

          chart.draw_series(LineSeries::new(
              windowed.iter().enumerate().map(|(i, &v)| (i, v)),
              RGBColor(100, 200, 255).stroke_width(1),
          ))?;
      }

      // ── Panel 3: Log-scale FFT Spectrum ──────────────────────────────────────
      let min_bin = ((min_freq / bin_width).ceil() as usize).max(1);
      let max_bin =
          ((max_freq / bin_width).floor() as usize).min(mags.len().saturating_sub(1));
      let x_lo = min_freq.log10();
      let x_hi = max_freq.log10();
      let spec_max = mags[min_bin..=max_bin]
          .iter()
          .cloned()
          .fold(0.0f32, f32::max)
          .max(noise_floor * 2.0);

      {
          let mut chart = ChartBuilder::on(area3)
              .caption(
                  "FFT Spectrum — Detected Pitches",
                  ("sans-serif", 18).into_font().color(&WHITE),
              )
              .margin(15)
              .x_label_area_size(40)
              .y_label_area_size(70)
              .build_cartesian_2d(x_lo..x_hi, 0f32..spec_max)?;

          let tick_freqs: &[f32] =
              &[50.0, 100.0, 200.0, 500.0, 1_000.0, 2_000.0, 4_000.0, 8_000.0];

          chart
              .configure_mesh()
              .x_labels(tick_freqs.len())
              .x_label_formatter(&|x| {
                  let hz = 10f32.powf(*x);
                  if hz >= 1_000.0 {
                      format!("{:.0}k", hz / 1_000.0)
                  } else {
                      format!("{hz:.0}")
                  }
              })
              .y_labels(5)
              .label_style(("sans-serif", 11).into_font().color(&WHITE))
              .axis_style(RGBColor(100, 100, 120))
              .draw()?;

          // Spectrum line.
          chart.draw_series(LineSeries::new(
              (min_bin..=max_bin)
                  .map(|b| ((b as f32 * bin_width).log10(), mags[b])),
              RGBColor(100, 200, 255).stroke_width(1),
          ))?;

          // Noise floor — solid red line.
          chart
              .draw_series(LineSeries::new(
                  vec![(x_lo, noise_floor), (x_hi, noise_floor)],
                  RED.stroke_width(1),
              ))?
              .label("noise floor")
              .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 15, y)], RED));

          // Pitch labels and circles.
          for &(freq, score) in stable {
              if freq < min_freq || freq > max_freq {
                  continue;
              }
              let x = freq.log10();
              let bin = (freq / bin_width).round() as usize;
              let y = mags[bin.min(mags.len().saturating_sub(1))];
              let y_label = y + spec_max * 0.04;

              chart.draw_series(std::iter::once(Circle::new(
                  (x, y),
                  5i32,
                  YELLOW.filled(),
              )))?;
              chart.draw_series(std::iter::once(Text::new(
                  format!("{} {score:.1}", Self::freq_to_note_label(freq)),
                  (x, y_label),
                  ("sans-serif", 11).into_font().color(&YELLOW),
              )))?;
          }

          chart
              .configure_series_labels()
              .label_font(("sans-serif", 11).into_font().color(&WHITE))
              .draw()?;
      }

      root.present()?;
      Ok(())
  }
  ```

- [ ] **Step 2: Build**

  ```
  cargo build --features dev-tools 2>&1 | grep "^error" | head -20
  ```

  Expected: no errors. If `stroke_width` is not available directly on a color (e.g., `WHITE.stroke_width(1)` fails), use `ShapeStyle::from(&WHITE).stroke_width(1)` instead.

- [ ] **Step 3: Commit**

  ```bash
  git add src/audio_io/stft.rs
  git commit -m "feat: add dbg_export_png three-panel plotters exporter"
  ```

---

## Task 7: Hook `dbg_export_png` into the processing loop

**Files:**
- Modify: `src/audio_io/stft.rs` — inside `while available_samples >= window_size`, near the Rerun hook added in Task 5

- [ ] **Step 1: Add PNG trigger counter check and raw/windowed capture**

  Add a `dev_png_data` Option before the `let raw_pitches = if is_silence` line (alongside `dev_rerun_data`):

  ```rust
  #[cfg(feature = "dev-tools")]
  let mut dev_png_data: Option<(Vec<f32>, Vec<f32>)> = None;
  ```

  Inside the `else` branch of `if is_silence`, **before** the line that applies the Hann window (`for i in 0..window_size { ... ring_buffer[idx] * hann[i] }`), capture the raw signal:

  ```rust
  #[cfg(feature = "dev-tools")]
  let png_raw: Option<Vec<f32>> = if png_frame % 2500 == 0 {
      Some(
          (0..window_size)
              .map(|i| ring_buffer[(ring_read_pos + i) % ring_buffer_len])
              .collect(),
      )
  } else {
      None
  };
  ```

  Then, **after** the Hann window loop but **before** calling `fft_processor.process_forward(...)`, capture the windowed signal:

  ```rust
  #[cfg(feature = "dev-tools")]
  if let Some(raw_captured) = png_raw {
      dev_png_data = Some((raw_captured, time_domain_window.clone()));
  }
  ```

- [ ] **Step 2: Call `dbg_export_png` in the dev-tools block after `pitch_tracker.process`**

  Find the `#[cfg(feature = "dev-tools")]` block added in Task 5 (the one that calls `dbg_log_rerun`). Inside that block, after `rerun_frame += 1;`, add:

  ```rust
  if let Some((ref raw, ref windowed)) = dev_png_data {
      if let Some((ref mags, bw, nf)) = dev_rerun_data {
          if let Err(e) = Self::dbg_export_png(
              png_frame,
              raw,
              windowed,
              mags,
              bw,
              MIN_FREQ,
              MAX_FREQ,
              nf,
              &stable_pitches,
          ) {
              eprintln!("PNG export error (frame {png_frame}): {e}");
          }
      }
  }
  png_frame += 1;
  ```

- [ ] **Step 3: Build**

  ```
  cargo build --features dev-tools 2>&1 | grep "^error" | head -20
  ```

  Expected: no errors.

- [ ] **Step 4: Commit**

  ```bash
  git add src/audio_io/stft.rs
  git commit -m "feat: hook dbg_export_png every 2500 frames under dev-tools"
  ```

---

## Task 8: Delete all ASCII debug code

Now that both visualization paths are wired up, remove the old debug infrastructure.

**Files:**
- Modify: `src/audio_io/stft.rs`

- [ ] **Step 1: Delete the `#[cfg(debug_assertions)]` trigger and capture blocks**

  Delete the following `#[cfg(debug_assertions)]` blocks from inside `while available_samples >= window_size`:

  1. The trigger block (`dbg_trigger = ...`) — currently lines ~236–241
  2. The `dbg_capture: Option<...>` declaration — lines ~244–250
  3. The raw capture block (`let dbg_raw: Vec<f32> = if dbg_trigger ...`) — lines ~257–264
  4. The windowed capture block (`let dbg_windowed: Vec<f32> = if dbg_trigger ...`) — lines ~272–277
  5. The `if dbg_trigger { dbg_capture = Some(...) }` block — lines ~296–300
  6. The entire emit block (`#[cfg(debug_assertions)] if dbg_trigger { ... }`) — lines ~320–458

- [ ] **Step 2: Delete the ASCII impl block at the bottom of the file**

  Delete the entire block:
  ```rust
  // ─── Debug visualisation helpers (compiled only in debug builds) ───────────────

  #[cfg(debug_assertions)]
  impl STFT {
      fn dbg_plot_signal(...) { ... }
      fn dbg_plot_spectrum(...) { ... }
      fn dbg_freq_to_note(...) { ... }   // now replaced by freq_to_note_label
      fn dbg_emit_frame(...) { ... }
  }
  ```
  This is lines ~650–928.

- [ ] **Step 3: Build without feature (baseline check)**

  ```
  cargo build 2>&1 | grep "^error" | head -10
  ```

  Expected: no errors, no debug visualization code compiled in.

- [ ] **Step 4: Build with feature**

  ```
  cargo build --features dev-tools 2>&1 | grep "^error" | head -10
  ```

  Expected: no errors.

- [ ] **Step 5: Commit**

  ```bash
  git add src/audio_io/stft.rs
  git commit -m "refactor: remove ASCII debug output; dev-tools visualization only"
  ```

---

## Task 9: End-to-end verification

- [ ] **Step 1: Run without dev-tools — confirm no performance overhead**

  ```
  cargo build --release 2>&1 | grep "^error" | head -5
  ```

  Expected: builds cleanly. Zero visualization code in the release binary.

- [ ] **Step 2: Run with dev-tools and check Rerun viewer**

  Build and launch:
  ```
  cargo run --features dev-tools -- <your-audio-file-or-args>
  ```

  Expected: Rerun viewer window opens automatically. Navigate to `spectrum/line` — spectrum updates every frame. `spectrum/noise_floor` appears as a red horizontal line. `spectrum/pitches` shows labeled points when notes are detected.

- [ ] **Step 3: Check PNG output**

  After 2500 frames, inspect:
  ```
  ls -lh debug_frames/
  ```

  Expected: `frame_0.png` exists (first PNG fires at frame 0), then `frame_2500.png` etc.

  Open `debug_frames/frame_0.png` in an image viewer:
  - Panel 1: white waveform on dark background, labelled "Raw Signal"
  - Panel 2: cyan waveform at same y-scale, labelled "Hann-Windowed Signal"
  - Panel 3: cyan spectrum line, red noise floor line with legend, yellow circles with note labels at pitch locations, x-axis labelled in Hz

- [ ] **Step 4: Commit if any minor fixes were made during verification**

  ```bash
  git add -p src/audio_io/stft.rs
  git commit -m "fix: minor visualization adjustments after end-to-end testing"
  ```
