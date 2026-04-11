"""
Training script for the two-head polyphonic pitch detector with curriculum learning.

Usage
-----
    # Start training
    python train.py --dataset_dir data/pitch_dataset --output_dir ./checkpoints

    # Resume after pausing (Ctrl+C) or a crash
    python train.py --dataset_dir data/pitch_dataset --output_dir ./checkpoints --resume

Pausing
-------
Press Ctrl+C at any time.  The current epoch completes, a full checkpoint is
written, and the script exits cleanly.  Re-run with --resume to continue.
A checkpoint is also written at the end of every epoch, so a hard kill (power
loss, OOM) only loses the work done in that epoch.

CPU speedups (16-core machine)
-------------------------------
  UpdatableSampler
      DataLoaders are created once and reused across all epochs.  Without this,
      a new DataLoader is constructed every epoch, which respawns all worker
      processes each time (~1-2s overhead per epoch and no benefit from
      persistent_workers).

  persistent_workers + prefetch_factor
      Worker processes stay alive between epochs and pre-fetch batches while
      the main process is computing gradients.  On a CPU-only machine this
      overlaps data loading with compute.

  torch.set_num_threads / set_num_interop_threads
      Splits cores explicitly between DataLoader workers (data loading) and
      PyTorch's internal thread pool (matrix ops).  Default: 8+8 for 16-core.

  pin_memory=False on CPU
      pin_memory is for GPU DMA transfers and adds overhead on CPU-only.

  batch_size=1024
      Larger batches vectorize better on CPU and reduce Python loop overhead.
      Memory cost on CPU is negligible (the full dataset is already in RAM).

  Vectorised note_f1_topk
      Replaced the O(N) Python for-loop over samples with a batched scatter
      operation, cutting metric computation time by ~20x for 450K samples.

Curriculum schedule
-------------------
  Phase 1  (epochs 1–phase1_end):
      100% single-note samples.  pos_weight = 71.

  Phase 2  (epochs phase1_end–phase2_end):
      35% single-note + 65% two-note.  pos_weight = 43.

  Phase 3  (epochs phase2_end–end):
      Full distribution (20/35/30/15 for 1/2/3/4-note).  pos_weight = 29.

Label smoothing
---------------
  Smooth positives only: target 1 -> (1-eps), target 0 stays at 0.
  The symmetric formula (target*(1-eps) + 0.5*eps) moves negatives to 0.025,
  which combined with a large pos_weight pushes all note outputs above 0.5.
"""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler

from model import ImprovedPitchDetector, NUM_NOTES, NUM_SEMITONES, BINS_PER_SEMITONE, MAX_NOTES


def _worker_init_fn(worker_id: int) -> None:
    """Ignore SIGINT in DataLoader workers so Ctrl+C is handled only by the main process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


# ── curriculum schedule ───────────────────────────────────────────────────────

CURRICULUM_PHASES = [
    ({1: 1.0},                          1.0),   # Phase 1: single-note only
    ({1: 0.35, 2: 0.65},               1.65),   # Phase 2: single + two-note
    ({1: 0.20, 2: 0.35, 3: 0.30, 4: 0.15}, 2.40),  # Phase 3: full distribution
]

# Validation always uses Phase 3 proportions so val_f1 is comparable across
# all phases — Phase 1 validation is NOT single-note-only.
VAL_COUNT_WEIGHTS = CURRICULUM_PHASES[2][0]   # {1:0.20, 2:0.35, 3:0.30, 4:0.15}


def get_phase(epoch: int, phase1_end: int, phase2_end: int) -> int:
    if epoch <= phase1_end:
        return 0
    if epoch <= phase2_end:
        return 1
    return 2


def pos_weight_for_avg_notes(avg_notes: float) -> float:
    # With 360 bins and BINS_PER_SEMITONE active bins per note, the neg/pos
    # ratio is (360 - 5*avg) / (5*avg) = (72 - avg) / avg — numerically
    # identical to the old 72-bin formula.
    pos_bins = avg_notes * BINS_PER_SEMITONE
    return (NUM_NOTES - pos_bins) / pos_bins


# ── dataset ──────────────────────────────────────────────────────────────────

class PitchDataset(Dataset):
    def __init__(self, split_dir: Path, norm_mean: float, norm_std: float):
        self.features    = np.load(split_dir / "features.npy")
        self.labels      = np.load(split_dir / "labels.npy")
        # Gaussian soft labels: raw sum is ~3.1 per note, not 1.  Binarise at
        # the semitone level first so curriculum sampling gets integer counts.
        sem = self.labels.reshape(len(self.labels), -1, BINS_PER_SEMITONE).max(axis=-1)
        self.note_counts = (sem > 0.5).sum(axis=1).astype(np.int32)
        self.mean        = norm_mean
        self.std         = norm_std

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        feat  = (self.features[idx] - self.mean) / self.std
        label = self.labels[idx]
        count = self.note_counts[idx]
        return (
            torch.from_numpy(feat).unsqueeze(0),
            torch.from_numpy(label),
            torch.tensor(count, dtype=torch.long),
        )


class UpdatableSampler(Sampler):
    """
    A sampler whose index array can be replaced between epochs without
    destroying and recreating the DataLoader.  This lets persistent_workers
    stay alive across curriculum transitions.
    """

    def __init__(self, indices: np.ndarray):
        self._indices = indices

    def set_indices(self, indices: np.ndarray) -> None:
        self._indices = indices

    def __iter__(self):
        return iter(self._indices.tolist())

    def __len__(self) -> int:
        return len(self._indices)


def build_curriculum_indices(
    note_counts:   np.ndarray,
    count_weights: dict,
    target_size:   int,
    rng:           np.random.Generator,
) -> np.ndarray:
    total_weight = sum(count_weights.values())
    norm_weights = {k: v / total_weight for k, v in count_weights.items()}
    parts = []
    for count, weight in norm_weights.items():
        idxs     = np.where(note_counts == count)[0]
        if len(idxs) == 0:
            continue
        n_needed = max(1, int(weight * target_size))
        replace  = len(idxs) < n_needed
        parts.append(rng.choice(idxs, size=n_needed, replace=replace))
    combined = np.concatenate(parts)
    rng.shuffle(combined)
    return combined


# ── metrics ───────────────────────────────────────────────────────────────────

def bins_to_semitones(x: torch.Tensor) -> torch.Tensor:
    """Max-pool 360-bin tensor → 72-semitone tensor.  [N, 360] → [N, 72]."""
    N = x.shape[0]
    return x.view(N, NUM_SEMITONES, BINS_PER_SEMITONE).amax(dim=-1)


def note_f1_threshold(note_probs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    sem_probs   = bins_to_semitones(note_probs)
    gt          = (bins_to_semitones(targets) > 0.5).float()
    pred_bin    = (sem_probs > threshold).float()
    tp = (pred_bin * gt).sum().item()
    fp = (pred_bin * (1 - gt)).sum().item()
    fn = ((1 - pred_bin) * gt).sum().item()
    p  = tp / (tp + fp + 1e-8)
    r  = tp / (tp + fn + 1e-8)
    return 2 * p * r / (p + r + 1e-8)


def note_f1_topk(
    note_probs:   torch.Tensor,
    count_logits: torch.Tensor,
    targets:      torch.Tensor,
) -> float:
    """
    Count-guided F1 at semitone resolution — fully vectorised, no Python loop.

    Bins are max-pooled to semitone level so the 5 bins belonging to the same
    semitone vote together rather than competing for top-K slots.
    """
    sem_probs = bins_to_semitones(note_probs)                   # [N, 72]
    gt        = (bins_to_semitones(targets) > 0.5).float()      # [N, 72]

    counts = count_logits.argmax(dim=1).clamp(min=1)            # [N]
    max_k  = int(counts.max().item())

    _, top_idx = torch.topk(sem_probs, max_k, dim=1)            # [N, max_k]
    valid    = torch.arange(max_k).unsqueeze(0) < counts.unsqueeze(1)
    pred_bin = torch.zeros_like(gt)
    pred_bin.scatter_(1, top_idx, valid.float())

    tp = (pred_bin * gt).sum().item()
    fp = (pred_bin * (1 - gt)).sum().item()
    fn = ((1 - pred_bin) * gt).sum().item()
    p  = tp / (tp + fp + 1e-8)
    r  = tp / (tp + fn + 1e-8)
    return 2 * p * r / (p + r + 1e-8)


def count_exact_accuracy(count_logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred  = count_logits.argmax(dim=1)
    # Binarise at semitone level: raw bin sum is fractional with Gaussian labels.
    truth = (bins_to_semitones(targets) > 0.5).sum(dim=1).long().clamp(0, MAX_NOTES)
    return (pred == truth).float().mean().item()


# ── loss ──────────────────────────────────────────────────────────────────────

def make_count_criterion(count_weights: dict, device: torch.device) -> nn.Module:
    """
    Weighted CrossEntropyLoss for the count head.

    Without class weights, the model minimises CE by predicting the majority class
    (e.g. "always 2" in Phase 2 = 65% correct) and never learning minority classes.
    Inverse-frequency weights make every class equally important regardless of how
    often it appears in the current curriculum phase.
    """
    class_weight = torch.ones(MAX_NOTES + 1, device=device)  # default=1 for count=0
    total = sum(count_weights.values())
    n_classes = len(count_weights)
    for k, w in count_weights.items():
        if 0 < k <= MAX_NOTES:
            class_weight[k] = total / (n_classes * w)  # inverse-frequency
    return nn.CrossEntropyLoss(weight=class_weight)


def make_note_criterion(avg_notes: float, label_smoothing: float, device: torch.device):
    pw_scalar  = pos_weight_for_avg_notes(avg_notes)
    pos_weight = torch.full((NUM_NOTES,), pw_scalar, device=device)

    if label_smoothing > 0.0:
        class SmoothedBCELoss(nn.Module):
            def __init__(self, pos_weight, eps):
                super().__init__()
                self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                self.eps = eps

            def forward(self, logits, targets):
                # Smooth positives only: 1 -> (1-eps), negatives stay at 0.
                # The symmetric formula targets*(1-eps)+0.5*eps pushes negatives
                # to 0.025, which with a large pos_weight drives all sigmoid
                # outputs above 0.5 — defeating the threshold metric entirely.
                smooth = targets.clamp(0.0, 1.0 - self.eps)
                return self.bce(logits, smooth)

        return SmoothedBCELoss(pos_weight, label_smoothing).to(device)
    else:
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)


# ── one epoch ─────────────────────────────────────────────────────────────────

def run_epoch(model, loader, note_crit, count_crit, count_weight, device, optimizer=None) -> dict:
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss        = 0.0
    all_note_probs    = []
    all_count_logits  = []
    all_targets       = []
    all_count_targets = []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for features, labels, count_targets in loader:
            features      = features.to(device)
            labels        = labels.to(device)
            count_targets = count_targets.clamp(0, MAX_NOTES).to(device)

            count_logits, note_logits = model(features)

            # Convert Gaussian soft labels → hard semitone-block labels for BCE.
            # A semitone is active if its 5-bin peak exceeds 0.5; all 5 bins of an
            # active semitone are set to 1, all bins of inactive semitones to 0.
            # This aligns the loss directly with the semitone-level F1 metric so
            # the model cannot minimize loss by spreading probability across bins.
            B = labels.shape[0]
            sem_active = labels.view(B, NUM_SEMITONES, BINS_PER_SEMITONE).amax(-1) > 0.5
            hard_labels = (sem_active.unsqueeze(-1)
                           .expand(-1, -1, BINS_PER_SEMITONE)
                           .reshape(B, NUM_NOTES)
                           .float())

            note_loss  = note_crit(note_logits, hard_labels)
            count_loss = count_crit(count_logits, count_targets)
            loss       = note_loss + count_weight * count_loss

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss        += loss.item() * len(features)
            all_note_probs.append(torch.sigmoid(note_logits).detach().cpu())
            all_count_logits.append(count_logits.detach().cpu())
            all_targets.append(labels.detach().cpu())
            all_count_targets.append(count_targets.detach().cpu())

    note_probs    = torch.cat(all_note_probs)
    count_logits  = torch.cat(all_count_logits)
    targets       = torch.cat(all_targets)
    count_truth   = torch.cat(all_count_targets)   # correct integer counts from dataset

    pred_counts = count_logits.argmax(dim=1)
    per_count_acc = {}
    for k in range(1, MAX_NOTES + 1):
        mask = count_truth == k
        if mask.sum() > 0:
            per_count_acc[k] = (pred_counts[mask] == k).float().mean().item()

    return {
        "loss":          total_loss / len(loader.dataset),
        "f1_thresh":     note_f1_threshold(note_probs, targets),
        "f1_topk":       note_f1_topk(note_probs, count_logits, targets),
        "count_acc":     (pred_counts == count_truth).float().mean().item(),
        "per_count_acc": per_count_acc,
    }


# ── checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(path: Path, epoch: int, model, optimizer, scheduler,
                    rng, rng_val, best_val_f1, patience_count, history, args):
    torch.save({
        "epoch":         epoch,
        "model":         model.state_dict(),
        "optimizer":     optimizer.state_dict(),
        "scheduler":     scheduler.state_dict(),
        "rng":           rng.bit_generator.state,
        "rng_val":       rng_val.bit_generator.state,
        "best_val_f1":   best_val_f1,
        "patience_count":patience_count,
        "history":       history,
        "args":          vars(args),
    }, path)


def load_checkpoint(path: Path, model, optimizer, scheduler, rng, rng_val):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    rng.bit_generator.state     = ckpt["rng"]
    rng_val.bit_generator.state = ckpt["rng_val"]
    print(f"Resumed from epoch {ckpt['epoch']}  "
          f"(best val f1_topk so far: {ckpt['best_val_f1']:.4f})")
    print(f"Original args: {ckpt['args']}")
    return (
        ckpt["epoch"],
        ckpt["best_val_f1"],
        ckpt["patience_count"],
        ckpt["history"],
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    default_workers = max(1, (os.cpu_count() or 16) // 2)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir",   required=True)
    parser.add_argument("--output_dir",    default="./checkpoints")
    parser.add_argument("--epochs",        type=int,   default=100)
    parser.add_argument("--batch_size",    type=int,   default=1024,
                        help="Larger batches vectorise better on CPU. Default 1024.")
    parser.add_argument("--lr",            type=float, default=5e-4)
    parser.add_argument("--count_weight",  type=float, default=1.0)
    parser.add_argument("--label_smooth",  type=float, default=0.05)
    parser.add_argument("--phase1_end",    type=int,   default=20)
    parser.add_argument("--phase2_end",    type=int,   default=45)
    parser.add_argument("--train_size",    type=int,   default=450_000)
    parser.add_argument("--patience",      type=int,   default=12)
    parser.add_argument("--workers",       type=int,   default=default_workers,
                        help=f"DataLoader worker processes. Default {default_workers} "
                             f"(half of CPU count). Remaining cores go to PyTorch threads.")
    parser.add_argument("--resume",        action="store_true",
                        help="Resume from checkpoint.pt in --output_dir.")
    parser.add_argument("--resume_path",   type=str,   default=None,
                        help="Explicit checkpoint path (overrides --resume default location).")
    args = parser.parse_args()

    # ── CPU thread allocation ────────────────────────────────────────────────
    # Split cores between DataLoader workers (I/O + preprocessing) and
    # PyTorch's internal BLAS/OpenMP thread pool (matrix ops).
    total_cores    = os.cpu_count() or 16
    compute_threads = max(1, total_cores - args.workers)
    torch.set_num_threads(compute_threads)
    torch.set_num_interop_threads(max(1, compute_threads // 2))
    print(f"CPU cores: {total_cores}  |  "
          f"DataLoader workers: {args.workers}  |  "
          f"PyTorch compute threads: {compute_threads}")

    dataset_dir = Path(args.dataset_dir)
    out_dir     = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(dataset_dir / "normalization.json") as f:
        norm = json.load(f)
    mean, std = norm["mean"], norm["std"]
    print(f"Normalization — mean: {mean:.4f}, std: {std:.4f}")

    train_ds = PitchDataset(dataset_dir / "train", mean, std)
    val_ds   = PitchDataset(dataset_dir / "val",   mean, std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    use_pin_memory = device.type == "cuda"  # pin_memory adds overhead on CPU

    # ── build DataLoaders once with UpdatableSampler ─────────────────────────
    # Reusing the same DataLoader object across epochs keeps worker processes
    # alive (persistent_workers=True) and avoids ~1s respawn overhead per epoch.
    initial_train_idxs = np.arange(min(args.train_size, len(train_ds)))
    initial_val_idxs   = np.arange(min(20_000, len(val_ds)))

    train_sampler = UpdatableSampler(initial_train_idxs)
    val_sampler   = UpdatableSampler(initial_val_idxs)

    # worker_init_fn makes worker processes ignore SIGINT so that Ctrl+C is
    # handled only by the main process, preventing forrtl/MKL crash noise on Windows.
    loader_kwargs = dict(
        batch_size        = args.batch_size,
        num_workers       = args.workers,
        pin_memory        = use_pin_memory,
        persistent_workers= args.workers > 0,
        prefetch_factor   = 4 if args.workers > 0 else None,
        worker_init_fn    = _worker_init_fn if args.workers > 0 else None,
    )
    train_dl = DataLoader(train_ds, sampler=train_sampler, **loader_kwargs)
    val_dl   = DataLoader(val_ds,   sampler=val_sampler,   **loader_kwargs)

    # ── model + optimiser ────────────────────────────────────────────────────
    model     = ImprovedPitchDetector().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup     = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)
    cosine     = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - 5)
    scheduler  = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[5]
    )

    rng     = np.random.default_rng(seed=42)
    rng_val = np.random.default_rng(seed=99)

    start_epoch    = 1
    best_val_f1    = -1.0
    patience_count = 0
    history        = []
    current_phase  = -1
    note_crit      = None
    count_crit     = None

    # ── resume ───────────────────────────────────────────────────────────────
    ckpt_path = Path(args.resume_path) if args.resume_path else out_dir / "checkpoint.pt"
    if args.resume or (args.resume_path and ckpt_path.exists()):
        if not ckpt_path.exists():
            print(f"No checkpoint found at {ckpt_path} — starting from scratch.")
        else:
            start_epoch, best_val_f1, patience_count, history = load_checkpoint(
                ckpt_path, model, optimizer, scheduler, rng, rng_val
            )
            start_epoch += 1   # resume at the epoch after the last completed one

    # ── Ctrl+C handler ───────────────────────────────────────────────────────
    # Sets a flag so the training loop can finish the current epoch cleanly
    # before saving and exiting (avoids a half-written checkpoint).
    stop_requested = False

    def _sigint_handler(*_):
        nonlocal stop_requested
        if stop_requested:
            print("\nForce-quitting.")
            sys.exit(1)
        print("\nCtrl+C received — finishing current epoch then saving checkpoint.")
        print("Press Ctrl+C again to force-quit immediately.")
        stop_requested = True

    signal.signal(signal.SIGINT, _sigint_handler)

    # ── training loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        phase_idx = get_phase(epoch, args.phase1_end, args.phase2_end)
        count_weights, avg_notes = CURRICULUM_PHASES[phase_idx]

        if phase_idx != current_phase:
            current_phase  = phase_idx
            patience_count = 0   # each phase gets a fresh patience budget
            # Partial LR restart: floor the LR at 50% of initial so the model
            # has enough gradient signal to adapt to the harder distribution.
            if phase_idx > 0:
                restart_lr = args.lr * 0.5
                for pg in optimizer.param_groups:
                    pg["lr"] = max(pg["lr"], restart_lr)
            note_crit  = make_note_criterion(avg_notes, args.label_smooth, device)
            count_crit = make_count_criterion(count_weights, device)
            pw         = pos_weight_for_avg_notes(avg_notes)
            phase_names = ["Phase 1 (1-note)", "Phase 2 (1+2-note)", "Phase 3 (full)"]
            print(f"\n-- {phase_names[phase_idx]}  avg_notes={avg_notes:.2f}  pos_weight={pw:.1f} --")

        # Update sampler indices (no DataLoader rebuild needed)
        train_idxs = build_curriculum_indices(
            train_ds.note_counts, count_weights, args.train_size, rng
        )
        val_idxs = build_curriculum_indices(
            val_ds.note_counts, VAL_COUNT_WEIGHTS, min(20_000, len(val_ds)), rng_val
        )
        train_sampler.set_indices(train_idxs)
        val_sampler.set_indices(val_idxs)

        t0 = time.time()
        tr = run_epoch(model, train_dl, note_crit, count_crit,
                       args.count_weight, device, optimizer=optimizer)
        va = run_epoch(model, val_dl,   note_crit, count_crit,
                       args.count_weight, device)
        scheduler.step()

        elapsed = round(time.time() - t0, 1)
        row = {
            "epoch":         epoch,
            "phase":         phase_idx + 1,
            "lr":            round(optimizer.param_groups[0]["lr"], 6),
            "train_loss":    round(tr["loss"],      4),
            "train_f1_topk": round(tr["f1_topk"],   4),
            "train_count":   round(tr["count_acc"], 4),
            "val_loss":      round(va["loss"],       4),
            "val_f1_thresh": round(va["f1_thresh"],  4),
            "val_f1_topk":   round(va["f1_topk"],    4),
            "val_count_acc": round(va["count_acc"],  4),
            "val_per_count": {str(k): round(v, 4) for k, v in va["per_count_acc"].items()},
            "elapsed_s":     elapsed,
        }
        history.append(row)

        per_cnt_str = "  ".join(
            f"cnt{k}={v:.2f}" for k, v in sorted(va["per_count_acc"].items())
        )
        print(
            f"[{epoch:3d}/{args.epochs}|P{phase_idx+1}] "
            f"loss {tr['loss']:.3f}/{va['loss']:.3f}  "
            f"f1(topk) {tr['f1_topk']:.3f}/{va['f1_topk']:.3f}  "
            f"f1(thr) {va['f1_thresh']:.3f}  "
            f"cnt {va['count_acc']:.3f} [{per_cnt_str}]  "
            f"lr {row['lr']}  ({elapsed}s)"
        )

        # Save full checkpoint every epoch (for resume)
        save_checkpoint(
            ckpt_path, epoch, model, optimizer, scheduler,
            rng, rng_val, best_val_f1, patience_count, history, args,
        )

        if va["f1_topk"] > best_val_f1:
            best_val_f1    = va["f1_topk"]
            patience_count = 0
            # best.pt is model weights only — used by export.py
            torch.save(model.state_dict(), out_dir / "best.pt")
            print(f"  New best val f1_topk: {best_val_f1:.4f}")
        else:
            patience_count += 1
            if phase_idx == 2 and patience_count >= args.patience:
                print(f"Early stopping (Phase 3) after {epoch} epochs.")
                break

        if stop_requested:
            print(f"Checkpoint saved to {ckpt_path}. Resume with --resume.")
            break

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Done. Best val f1_topk: {best_val_f1:.4f}")
    print(f"Resume with: python train.py --dataset_dir {args.dataset_dir} "
          f"--output_dir {args.output_dir} --resume")


if __name__ == "__main__":
    main()
