# Debug Visualization: Rerun Live Viewer + Plotters Static PNG

**Date:** 2026-04-14  
**File:** `src/audio_io/stft.rs`  
**Feature gate:** `dev-tools` (already exists; gains `plotters` alongside `rerun`)

---

## Goals

Replace the existing ASCII stderr debug output with two professional visualization outputs:

1. **Rerun live viewer** — streams the frequency spectrum, noise floor, and detected pitches every frame while the STFT thread is running.
2. **Static poster PNG** — saves a three-panel flowchart image (raw waveform → windowed waveform → annotated spectrum) to disk every ~2500 frames.

---

## Cargo.toml Changes

```toml
[features]
dev-tools = ["rerun", "plotters"]

[dependencies]
plotters = { version = "0.3", optional = true }
# rerun already present at 0.30.0
```

---

## Architecture

The `#[cfg(debug_assertions)]` gate and all ASCII rendering functions are removed entirely. All new visualization code is gated on `#[cfg(feature = "dev-tools")]`.

Two independent output paths share the same frame data:

```
STFT thread
  │
  ├─ every frame ──────────────► dbg_log_rerun(stream, mags, noise_floor, stable, ...)
  │
  └─ every ~2500th frame ──────► dbg_export_png(frame, raw, windowed, mags, noise_floor, stable, ...)
```

The `RecordingStream` is created once when the STFT thread starts and moved into the closure. Raw and windowed sample buffers are only captured when the PNG counter fires (avoids a large allocation every frame).

Two separate function counters replace the single `dbg_fft_frame`:
- `rerun_frame: usize` — incremented every FFT frame; Rerun logs every frame.
- `png_frame: usize` — fires every `PNG_INTERVAL = 2500` frames.

---

## Rerun Live Viewer

### Initialization

```rust
#[cfg(feature = "dev-tools")]
let rec = rerun::RecordingStreamBuilder::new("audio_analyzer")
    .spawn()
    .expect("failed to start Rerun viewer");
```

Called once at thread start, before the processing loop.

### Per-frame logging (`dbg_log_rerun`)

Three entities logged each frame using `rec.set_time_sequence("frame", rerun_frame)`:

| Entity path | Rerun type | Content |
|---|---|---|
| `spectrum/line` | `LineStrips2D` | One point per FFT bin: `(log10(bin * bin_width), magnitude)`. Covers `min_freq`–`max_freq`. |
| `spectrum/noise_floor` | `LineStrips2D` | Two points spanning the full x range at `y = noise_floor`. Styled as a distinct colour (e.g. red). |
| `spectrum/pitches` | `Points2D` + `TextBatch` | One point per stable pitch at `(log10(freq), mags[(freq/bin_width).round() as usize])`, with a text label `"{note} {score:.1}"` offset above. |

The `log10` transform is applied in Rust before logging — Rerun's axis labels won't show Hz directly, so the x-axis is in log10(Hz) units. Tick labels are not available in Rerun's 2D view, so the axis values are left as-is; the log scale is visually apparent from the spacing.

Rerun's timeline handles frame-to-frame animation natively — no interpolation code needed, no performance cost.

---

## Static PNG Export (`dbg_export_png`)

### Trigger

Fires when `png_frame % PNG_INTERVAL == 0` where `PNG_INTERVAL = 2500`.

### Output path

`debug_frames/frame_{n}.png` — directory created with `std::fs::create_dir_all` at thread start.

### Layout

**Size:** 1200 × 1800 px  
**Background:** Dark (`RGBColor(20, 20, 30)`)  
**Panels:** Three equal-height regions (~500 px each), stacked vertically with 50 px margins between.

#### Panel 1 — Raw Signal
- X: sample index (0 to window_size)
- Y: amplitude, auto-scaled to frame min/max
- Line plot, white foreground
- Title: `"Raw Signal — Frame {n}"`

#### Panel 2 — Hann-Windowed Signal
- Same axes and scale as Panel 1 (y-range computed from Panel 1's raw min/max, reused here for direct comparison)
- Title: `"Hann-Windowed Signal"`

#### Panel 3 — Log-Scale Spectrum
- X: log10(frequency), with custom tick labels at 50, 100, 200, 500, 1k, 2k, 4k, 8k Hz
- Y: magnitude (linear), 0 to frame max
- **Spectrum line:** one `plotters` `LineSeries` point per bin — adjacent bins connected, no interpolation beyond that
- **Noise floor:** horizontal dashed `LineSeries` at `y = noise_floor`, labelled `"noise floor"`
- **Stable pitches:** filled circle at `(log10(freq), mag)` per detected pitch, with text label `"{note}\n{score:.1}"` drawn above
- Title: `"FFT Spectrum — Detected Pitches"`

### Note name helper

The existing `dbg_freq_to_note` function is kept (renamed to `freq_to_note_label`) and used by both the Rerun and PNG paths for pitch labels.

---

## stft.rs Changes Summary

| What | Action |
|---|---|
| `#[cfg(debug_assertions)]` frame counter and trigger block | Removed; replaced with `#[cfg(feature = "dev-tools")]` dual counters |
| `dbg_emit_frame` function | Replaced by `dbg_log_rerun` + `dbg_export_png` |
| `dbg_plot_spectrum` (ASCII bar chart) | Removed |
| `dbg_plot_signal` (ASCII waveform) | Removed |
| `dbg_freq_to_note` helper | Kept, renamed `freq_to_note_label`, made non-`cfg`-gated |
| `RecordingStream` init | Added at thread start under `#[cfg(feature = "dev-tools")]` |
| `debug_frames/` dir creation | Added at thread start under `#[cfg(feature = "dev-tools")]` |
| Raw/windowed buffer capture | Only captured when `png_frame % PNG_INTERVAL == 0` |

---

## Verification

1. `cargo build` — no errors (baseline, no feature)
2. `cargo build --features dev-tools` — compiles with Rerun + plotters
3. Run with `--features dev-tools`: Rerun viewer opens, spectrum animates per frame
4. After 2500 frames: `debug_frames/frame_2500.png` exists, opens in image viewer
5. PNG panel 3 shows noise floor line and pitch labels at correct frequencies
6. Removing `--features dev-tools`: no visualization code compiled in, no performance impact
