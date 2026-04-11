"""
Dataset preparation for the polyphonic pitch detector.

Generates synthetic polyphonic training samples by mixing isolated NSynth notes,
then extracts log-mel features matching the Rust STFT parameters exactly.

Usage
-----
    python prepare_dataset.py --nsynth_dir /data/nsynth-train --out_dir /data/pitch_dataset

    # Ctrl+C saves progress; resume later with:
    python prepare_dataset.py --nsynth_dir /data/nsynth-train --out_dir /data/pitch_dataset --resume

    # If generation is disk-speed limited, reduce concurrent I/O (not cache):
    python prepare_dataset.py ... --workers 4

Rust STFT parameters replicated here:
    sample_rate  = 44100
    fft_size     = 2048
    n_mels       = 128
    fmin         = 30 Hz
    fmax         = 8000 Hz
    window       = Hann

Bottleneck guide
----------------
  CPU-limited  → increase --workers (default: all cores)
  Disk-limited → decrease --workers to 2–4; adding more workers only increases
                 concurrent random I/O without improving throughput
  RAM-limited  → decrease --cache_limit; total cached audio RAM ≈ workers ×
                 cache_limit × 700 KB. Default is 100 (~70 MB per worker).

  tqdm note: the progress bar shows "?" for ~30 s on Windows while workers
  spawn and import libraries.  This is normal — generation starts after.

Augmentation pipeline (per sample)
-----------------------------------
  Per note (before mixing):
    1. Random sustain-phase window from the 4-second NSynth clip
    2. Pitch deviation = intonation (±50 cents) + vibrato phase sample
       (depth 0–50 cents, random phase) — combined shift up to ~±100 cents
    3. Velocity: per-note gain −12 to 0 dB

  On the polyphonic mix:
    4. Global mix shift ±25 cents — prevents overfitting to exact-semitone
       spectral patterns; all Gaussian label centres shift identically
    5. Reverb: synthetic RIR (Sabine, RT60 0.1–0.5 s) with p=0.5
    6. Global gain ±6 dB
    7. White noise at SNR 20–45 dB

Labels
------
  Gaussian soft targets centred on each active note's 20-cent bin (σ=25 cents).
  Only the ±5-bin window is computed (sparse); values beyond are < 3×10⁻⁴.
  Multiple notes are merged via element-wise maximum.

Resume / graceful shutdown
--------------------------
  Features and labels are written directly to memory-mapped .npy files as
  batches complete, so no progress is lost on Ctrl+C or a crash.
  A .progress.json sidecar records which batches finished.  Re-run with
  --resume to skip completed batches and continue from where it stopped.
"""

import argparse
import json
import math
import multiprocessing
import os
import random
import signal
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly
from tqdm import tqdm

# ── constants matching Rust stft.rs ──────────────────────────────────────────
SAMPLE_RATE       = 44_100
FFT_SIZE          = 2_048
N_MELS            = 128
FMIN              = 30.0   # covers C1 fundamental (32.7 Hz); must match pitch_detector.rs
FMAX              = 8_000.0
MIDI_MIN          = 24     # C1
MIDI_MAX          = 108     # C8
NUM_SEMITONES     = MIDI_MAX - MIDI_MIN + 1   # 85
BINS_PER_SEMITONE = 5      # 20-cent resolution — must match model.py
NUM_BINS          = NUM_SEMITONES * BINS_PER_SEMITONE  # 425

N_TRAIN = 450_000
N_VAL   = 50_000

# ── Gaussian label constants ──────────────────────────────────────────────────
_GAUSS_SIGMA = 1.25   # bins (25 cents at 20-cent resolution)
_GAUSS_HALF  = 5      # sparse half-width; exp(-0.5*(5/1.25)²) ≈ 3×10⁻⁴


# ── mel filterbank (mirrors src/dsp/spectrogram.rs) ──────────────────────────

def hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_mel_filterbank(
    sample_rate: int = SAMPLE_RATE,
    n_fft: int       = FFT_SIZE,
    n_mels: int      = N_MELS,
    fmin: float      = FMIN,
    fmax: float      = FMAX,
) -> np.ndarray:
    """Returns shape (n_mels, n_fft//2 + 1) filterbank matrix."""
    mel_min   = hz_to_mel(fmin)
    mel_max   = hz_to_mel(fmax)
    mel_pts   = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts    = np.array([mel_to_hz(m) for m in mel_pts])
    fft_freqs = np.arange(n_fft // 2 + 1) * (sample_rate / n_fft)

    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(n_mels):
        f_left, f_center, f_right = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
        up   = (fft_freqs >= f_left)   & (fft_freqs <= f_center)
        down = (fft_freqs >= f_center) & (fft_freqs <= f_right)
        filters[m, up]   = (fft_freqs[up]   - f_left)   / (f_center - f_left)
        filters[m, down] = (f_right - fft_freqs[down])   / (f_right  - f_center)
        norm = filters[m].sum()
        if norm > 0:
            filters[m] /= norm
    return filters


# ── feature extraction ────────────────────────────────────────────────────────

HANN: np.ndarray               = np.hanning(FFT_SIZE).astype(np.float32)
MEL_FILTERS: Optional[np.ndarray] = None


def get_mel_filters() -> np.ndarray:
    global MEL_FILTERS
    if MEL_FILTERS is None:
        MEL_FILTERS = build_mel_filterbank()
    return MEL_FILTERS


def extract_features(audio: np.ndarray) -> np.ndarray:
    """2048-sample mono → (128,) log-mel feature vector matching Rust preprocessing."""
    windowed  = audio * HANN
    spectrum  = np.fft.rfft(windowed)
    power     = (np.abs(spectrum) ** 2).astype(np.float32)
    mel_power = get_mel_filters() @ power
    return np.log(mel_power + 1e-6)


# ── audio helpers ─────────────────────────────────────────────────────────────

def resample_to_target(audio: np.ndarray, src_sr: int, tgt_sr: int = SAMPLE_RATE) -> np.ndarray:
    if src_sr == tgt_sr:
        return audio
    gcd = math.gcd(src_sr, tgt_sr)
    return resample_poly(audio, tgt_sr // gcd, src_sr // gcd).astype(np.float32)


def load_and_resample(path: str) -> Optional[np.ndarray]:
    """Load audio file, convert to mono float32 at SAMPLE_RATE."""
    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return resample_to_target(audio, sr)
    except Exception:
        return None


# ── augmentation ──────────────────────────────────────────────────────────────

def _apply_gaussian(label: np.ndarray, center: float) -> None:
    """Sparse in-place max-merge of a Gaussian at `center` into `label`.

    Only the ±_GAUSS_HALF bin neighbourhood is updated (~11 elements vs 360),
    cutting per-Gaussian cost by ~35×.  Values outside this window are < 3×10⁻⁴.
    """
    lo = max(0, int(center) - _GAUSS_HALF)
    hi = min(NUM_BINS, int(center) + _GAUSS_HALF + 1)
    if lo >= hi:
        return
    offsets = np.arange(lo, hi, dtype=np.float32) - center
    vals    = np.exp(-0.5 * (offsets / _GAUSS_SIGMA) ** 2)
    np.maximum(label[lo:hi], vals, out=label[lo:hi])


def sample_pitch_deviation(rng: random.Random) -> float:
    """Intonation error (±50 cents) + vibrato phase sample (depth 0–50 cents).

    Samples a random point in the vibrato cycle, so non-vibrato instruments
    contribute near-zero depth and vibrato instruments vary continuously.
    Combined shift can reach ~±100 cents; the label centre tracks it exactly.
    """
    intonation = rng.uniform(-0.5, 0.5)
    depth      = rng.uniform(0.0, 0.5)
    phase      = rng.uniform(0.0, math.tau)
    return intonation + depth * math.sin(phase)


def pitch_shift_semitones(audio: np.ndarray, semitones: float) -> np.ndarray:
    """Pitch shift via rational resampling."""
    ratio         = 2.0 ** (semitones / 12.0)
    stretched_len = int(len(audio) / ratio)
    if stretched_len < 1:
        return audio
    stretched = resample_poly(audio, len(audio), max(1, stretched_len))
    if len(stretched) >= len(audio):
        return stretched[:len(audio)]
    return np.concatenate([stretched, np.zeros(len(audio) - len(stretched), dtype=np.float32)])


def add_reverb(audio: np.ndarray, rng: random.Random, p: float = 0.5) -> np.ndarray:
    """Synthetic RIR (Sabine exponential decay) applied to the mix, p=0.5."""
    if rng.random() >= p:
        return audio
    rt60   = rng.uniform(0.1, 0.5)
    n_rir  = int(SAMPLE_RATE * rt60)
    t      = np.arange(n_rir, dtype=np.float32) / SAMPLE_RATE
    np_rng = np.random.default_rng(rng.randint(0, 0xFFFFFFFF))
    rir    = np_rng.standard_normal(n_rir).astype(np.float32) * np.exp(-6.9 * t / rt60)
    rir   /= np.abs(rir).max() + 1e-12
    return fftconvolve(audio, rir)[:len(audio)].astype(np.float32)


def add_noise(audio: np.ndarray, rng: random.Random) -> np.ndarray:
    snr_db       = rng.uniform(20.0, 45.0)
    signal_power = float(np.mean(audio ** 2)) + 1e-12
    noise_std    = math.sqrt(signal_power / (10.0 ** (snr_db / 10.0)))
    np_rng       = np.random.default_rng(rng.randint(0, 0xFFFFFFFF))
    return audio + np_rng.standard_normal(len(audio)).astype(np.float32) * noise_std


# ── NSynth helpers ────────────────────────────────────────────────────────────

def load_nsynth_index(nsynth_dir: str) -> list[dict]:
    """Parse examples.json → list of {name, pitch, dir} within the MIDI range."""
    json_path = Path(nsynth_dir) / "examples.json"
    with open(json_path) as f:
        examples = json.load(f)
    entries = [
        {"name": name, "pitch": meta["pitch"], "dir": nsynth_dir}
        for name, meta in examples.items()
        if MIDI_MIN <= meta.get("pitch", -1) <= MIDI_MAX
    ]
    print(f"Found {len(entries)} NSynth notes in MIDI {MIDI_MIN}–{MIDI_MAX}")
    return entries


def get_audio_path(entry: dict) -> str:
    return str(Path(entry["dir"]) / "audio" / f"{entry['name']}.wav")


def _write_compact_index(entries: list[dict], path: Path) -> None:
    """Write a compact [[name, pitch], …] index for fast worker startup.

    Loading this ~12 MB file takes < 1 s per worker.  Loading the full
    examples.json (200–500 MB with all metadata for 289 K notes) takes
    tens of seconds and hundreds of MB of RAM *per worker*, which is the
    root cause of the multi-minute hang and RAM/disk exhaustion on startup.
    """
    with open(path, "w") as f:
        json.dump([[e["name"], e["pitch"]] for e in entries], f, separators=(',', ':'))


# ── sample window from note audio ─────────────────────────────────────────────

_TRANSIENT_SAMPLES = int(0.1 * SAMPLE_RATE)   # skip attack and release (0.1 s each)


def get_window(audio: np.ndarray, rng: random.Random) -> np.ndarray:
    """Random FFT_SIZE window from the sustain portion of a note."""
    usable_start = _TRANSIENT_SAMPLES
    usable_end   = len(audio) - _TRANSIENT_SAMPLES - FFT_SIZE
    start = rng.randint(usable_start, usable_end) if usable_end > usable_start else 0
    chunk = audio[start : start + FFT_SIZE]
    if len(chunk) < FFT_SIZE:
        chunk = np.pad(chunk, (0, FFT_SIZE - len(chunk)))
    return chunk.astype(np.float32)


# ── sample generation ─────────────────────────────────────────────────────────

def make_sample(
    entries:     list[dict],
    audio_cache: dict,
    rng:         random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (features [128], label [360]) for one polyphonic training sample."""
    n_notes  = rng.choices([1, 2, 3, 4], weights=[30, 40, 20, 10])[0]
    selected = rng.sample(entries, min(n_notes, len(entries)))

    mix          = np.zeros(FFT_SIZE, dtype=np.float32)
    note_centers: list[float] = []

    for entry in selected:
        path = get_audio_path(entry)
        if path not in audio_cache:
            audio_cache[path] = load_and_resample(path)
        audio = audio_cache[path]
        if audio is None:
            continue

        window = get_window(audio, rng)

        # Per-note pitch deviation (intonation + vibrato phase sample)
        shift = sample_pitch_deviation(rng)
        if abs(shift) > 0.02:
            window = pitch_shift_semitones(window, shift)

        window *= 10.0 ** (rng.uniform(-12.0, 0.0) / 20.0)   # velocity
        mix    += window

        nominal = (entry["pitch"] - MIDI_MIN) * BINS_PER_SEMITONE + BINS_PER_SEMITONE // 2
        note_centers.append(float(nominal) + shift * BINS_PER_SEMITONE)

    # ── global mix pitch shift ±25 cents ─────────────────────────────────────
    # All notes shift identically so polyphonic intervals are preserved.
    # Prevents the model from memorising exact-semitone spectral patterns.
    global_shift      = rng.uniform(-0.25, 0.25)
    global_shift_bins = global_shift * BINS_PER_SEMITONE
    if abs(global_shift) > 0.02 and mix.any():
        mix = pitch_shift_semitones(mix, global_shift)

    # ── Gaussian soft labels (sparse) ────────────────────────────────────────
    label = np.zeros(NUM_BINS, dtype=np.float32)
    for c in note_centers:
        _apply_gaussian(label, max(0.0, min(float(NUM_BINS - 1), c + global_shift_bins)))

    # ── post-mix augmentation ─────────────────────────────────────────────────
    mix = add_reverb(mix, rng)                          # room acoustics, p=0.5
    mix *= 10.0 ** (rng.uniform(-6.0, 0.0) / 20.0)     # global gain ±6 dB
    mix  = add_noise(mix, rng)
    mix  = np.clip(mix, -1.0, 1.0)

    return extract_features(mix), label


# ── normalization ─────────────────────────────────────────────────────────────

def compute_normalization(features_arr: np.ndarray) -> tuple[float, float]:
    return float(features_arr.mean()), float(features_arr.std())


# ── multiprocessing worker ────────────────────────────────────────────────────
# Module-level state persists across batch calls within the same worker process,
# so the audio cache warms up naturally without being re-created per task.
#
# Workers receive index_path — a path to the compact [[name, pitch], …] JSON
# written once by the main process.  Each worker loads only that small file
# (~12 MB) rather than parsing the full examples.json (200–500 MB with all
# metadata fields).  This reduces per-worker startup from minutes to < 5 s and
# cuts per-worker RAM by hundreds of MB, preventing the RAM/disk exhaustion
# seen when all workers initialise simultaneously on Windows (spawn mode).

_mp_entries:     list = []
_mp_cache:       dict = {}
_mp_cache_limit: int  = 100


def _mp_load_entries(nsynth_dir: str, index_path: str) -> list[dict]:
    """Load the compact index written by the main process."""
    with open(index_path) as f:
        data = json.load(f)
    return [{"name": name, "pitch": pitch, "dir": nsynth_dir} for name, pitch in data]


def _mp_init(nsynth_dir: str, index_path: str, cache_limit: int) -> None:
    """Runs once per worker at pool startup."""
    global _mp_entries, _mp_cache, _mp_cache_limit
    _mp_entries     = _mp_load_entries(nsynth_dir, index_path)
    _mp_cache       = {}
    _mp_cache_limit = cache_limit
    get_mel_filters()                             # pre-warm: first sample won't pay this cost
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # main process owns Ctrl+C


def _mp_batch(args: tuple) -> tuple:
    """(task_idx, seed, batch_size) → (task_idx, features [B×128], labels [B×360])"""
    task_idx, seed, batch_size = args
    rng    = random.Random(seed)
    feats  = np.empty((batch_size, N_MELS),  dtype=np.float32)
    labels = np.empty((batch_size, NUM_BINS), dtype=np.float32)
    for i in range(batch_size):
        feats[i], labels[i] = make_sample(_mp_entries, _mp_cache, rng)
        while len(_mp_cache) > _mp_cache_limit:
            _mp_cache.pop(next(iter(_mp_cache)))   # evict until at limit
    return task_idx, feats, labels


# ── memmap helper ─────────────────────────────────────────────────────────────

def _open_memmap(path: Path, shape: tuple[int, ...], try_resume: bool) -> np.ndarray:
    """Open an existing .npy memmap for in-place writing, or create a fresh one."""
    if try_resume and path.exists():
        try:
            mm = np.lib.format.open_memmap(path, mode="r+")
            if mm.shape == shape:
                return mm
            print(f"  WARNING: {path.name} shape {mm.shape} != {shape} — recreating.")
        except Exception as exc:
            print(f"  WARNING: cannot open {path.name} for resume ({exc}) — recreating.")
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)


# ── split generation ──────────────────────────────────────────────────────────

def generate_split(
    nsynth_dir:        str,
    n_samples:         int,
    out_path:          Path,
    seed:              int,
    audio_cache_limit: int  = 2000,
    n_workers:         int  = 0,
    resume:            bool = False,
    index_path:        str  = "",
) -> np.ndarray:
    """Generate and save one dataset split, parallelised across CPU cores.

    Features and labels are written directly to memory-mapped .npy files as
    batches arrive, so a Ctrl+C (or crash) can be resumed with --resume.

    Bottleneck tuning
    -----------------
    Disk-limited: lower --workers (fewer concurrent random reads).
                  Do NOT lower --cache_limit — that only increases cache misses.
    RAM-limited:  lower --cache_limit.  Total RAM ≈ workers × cache_limit × 700 KB.
                  Default is 100 (~70 MB/worker). Hit rate is low with 280 K
                  unique files, so large values waste RAM without helping.
    CPU-limited:  raise --workers (up to os.cpu_count()).
    """
    if n_workers <= 0:
        n_workers = max(1, os.cpu_count() or 4)

    progress_path = out_path / ".progress.json"
    feat_path     = out_path / "features.npy"
    label_path    = out_path / "labels.npy"

    # ── restore or compute n_batches ─────────────────────────────────────────
    # n_batches must be read from the progress file before building the task
    # list.  task_start[i] is i*per_batch — if n_batches changes between runs
    # (because --workers changed) every task offset shifts, so completed-task
    # indices from a previous run map to wrong positions in the memmaps.
    # Freezing n_batches in the progress file makes it safe to resume with any
    # --workers value.
    n_batches_default = max(n_workers * 100, 500)
    completed: set[int] = set()

    if resume and progress_path.exists():
        with open(progress_path) as f:
            prog = json.load(f)
        if prog.get("n_samples") == n_samples and prog.get("seed") == seed:
            if "n_batches" not in prog:
                print("  WARNING: progress file predates n_batches tracking — "
                      "cannot safely resume. Delete .progress.json and restart.")
                completed = set()
                n_batches = n_batches_default
            else:
                n_batches = prog["n_batches"]
                completed = set(prog.get("completed_tasks", []))
        else:
            n_batches = n_batches_default
            print("  WARNING: progress file mismatch (seed or n_samples changed) — starting fresh.")
    else:
        n_batches = n_batches_default

    # ── build deterministic task list ────────────────────────────────────────
    # Many small batches → frequent progress-bar updates + fine-grained resume.
    per_batch = max(1, n_samples // n_batches)

    tasks:      list[tuple[int, int, int]] = []   # (task_idx, batch_seed, batch_size)
    task_start: dict[int, int]             = {}   # task_idx → write offset

    pos, rem = 0, n_samples
    for i in range(n_batches):
        bs = min(per_batch, rem)
        if bs <= 0:
            break
        tasks.append((i, seed + i * 997, bs))
        task_start[i] = pos
        pos += bs
        rem -= bs
    if rem > 0:                              # absorb rounding remainder into last task
        idx, s, bs = tasks[-1]
        tasks[-1] = (idx, s, bs + rem)

    if completed:
        n_done = sum(tasks[i][2] for i in completed if i < len(tasks))
        print(f"  Resuming: {len(completed)}/{len(tasks)} batches done "
              f"({n_done:,}/{n_samples:,} samples).")

    # ── open (or create) memory-mapped output files ───────────────────────────
    has_prior = resume and bool(completed)
    feats_mm  = _open_memmap(feat_path,  (n_samples, N_MELS),  has_prior)
    labels_mm = _open_memmap(label_path, (n_samples, NUM_BINS), has_prior)

    pending   = [t for t in tasks if t[0] not in completed]
    n_initial = sum(tasks[i][2] for i in completed if i < len(tasks))

    if not pending:
        print(f"  All {len(tasks)} batches already complete — nothing to do.")
        del feats_mm, labels_mm
        return np.lib.format.open_memmap(feat_path, mode="r")

    stop = False

    def _save_progress() -> None:
        with open(progress_path, "w") as f:
            json.dump({"n_samples": n_samples, "seed": seed,
                       "n_batches": n_batches,
                       "completed_tasks": sorted(completed)}, f)

    # ── parallel dispatch ─────────────────────────────────────────────────────
    # imap_unordered: results arrive as workers finish (no ordering stall),
    # giving smoother progress updates and better CPU utilisation.
    # task_start[] maps each task_idx to its fixed write position so out-of-order
    # results go to the correct location in the memmap.
    ctx = multiprocessing.get_context("spawn")
    print(f"  Launching {n_workers} workers "
          f"(first result may take ~30 s while workers import libraries)…")

    try:
        with ctx.Pool(n_workers, initializer=_mp_init,
                      initargs=(nsynth_dir, index_path, audio_cache_limit)) as pool:
            with tqdm(
                total         = n_samples,
                initial       = n_initial,
                desc          = out_path.name,
                unit          = "sample",
                dynamic_ncols = True,
                smoothing     = 0.15,
                mininterval   = 1.0,
            ) as pbar:
                for task_idx, feats, lbls in pool.imap_unordered(_mp_batch, pending):
                    s, n = task_start[task_idx], len(feats)
                    feats_mm [s : s + n] = feats
                    labels_mm[s : s + n] = lbls
                    completed.add(task_idx)
                    pbar.update(n)
    except (KeyboardInterrupt, multiprocessing.ProcessError):
        # KeyboardInterrupt  — user pressed Ctrl+C.
        # ProcessError       — a worker was killed (e.g. by the same Ctrl+C signal
        #                      reaching the process group before pool.terminate() ran).
        # Both are non-fatal: save progress and let the user --resume.
        stop = True
        print("\nCtrl+C — saving progress. Re-run with --resume to continue.")
    finally:
        del feats_mm, labels_mm   # close + flush OS write-back cache to disk
        _save_progress()

    if stop:
        n_remaining = sum(1 for t in tasks if t[0] not in completed)
        print(f"  Stopped with {n_remaining}/{len(tasks)} batches remaining.")
        raise SystemExit(0)

    progress_path.unlink(missing_ok=True)   # clean up sidecar on clean completion
    print(f"  Saved {n_samples} samples to {out_path}")
    return np.lib.format.open_memmap(feat_path, mode="r")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsynth_dir",   required=True,
                        help="Path to extracted nsynth-train directory "
                             "(contains examples.json and audio/)")
    parser.add_argument("--out_dir",      required=True,
                        help="Output directory for dataset files")
    parser.add_argument("--n_train",      type=int, default=N_TRAIN)
    parser.add_argument("--n_val",        type=int, default=N_VAL)
    parser.add_argument("--workers",      type=int, default=0,
                        help="Worker processes (0 = all CPUs). "
                             "If disk-speed limited, lower this (e.g. --workers 4) "
                             "rather than --cache_limit. Default: 0")
    parser.add_argument("--cache_limit",  type=int, default=100,
                        help="Max WAV files cached per worker. "
                             "Total RAM ≈ workers × cache_limit × 700 KB. "
                             "With 280 K unique files and random sampling the cache "
                             "hit rate is low, so large values waste RAM without "
                             "improving throughput. Default: 100")
    parser.add_argument("--resume",       action="store_true",
                        help="Continue an interrupted run from saved progress. "
                             "Skips already-completed batches.")
    args = parser.parse_args()

    out = Path(args.out_dir)
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "val").mkdir(parents=True, exist_ok=True)

    # Validate the index exists and has entries before launching workers.
    entries = load_nsynth_index(args.nsynth_dir)
    if not entries:
        raise RuntimeError("No valid NSynth entries found — check --nsynth_dir")
    print(f"  Validation OK — workers will load entries independently from {args.nsynth_dir}")

    # Write a compact (name, pitch) index once so workers don't each have to
    # parse the full examples.json (200–500 MB), which caused the multi-minute
    # hang and RAM/disk exhaustion seen previously.
    compact_index_path = out / "nsynth_index.json"
    _write_compact_index(entries, compact_index_path)

    kw = dict(audio_cache_limit=args.cache_limit, n_workers=args.workers, resume=args.resume,
              index_path=str(compact_index_path))

    print("Generating training split…")
    train_features = generate_split(args.nsynth_dir, args.n_train, out / "train", seed=42,   **kw)

    print("Generating validation split…")
    generate_split(args.nsynth_dir, args.n_val, out / "val", seed=1337, **kw)

    mean, std = compute_normalization(train_features)
    norm = {"mean": mean, "std": std if std > 1e-6 else 1.0}
    with open(out / "normalization.json", "w") as f:
        json.dump(norm, f, indent=2)
    print(f"Normalization — mean: {mean:.4f}, std: {std:.4f}")
    print("Dataset preparation complete.")


if __name__ == "__main__":
    multiprocessing.freeze_support()   # required on Windows for spawn mode
    main()
