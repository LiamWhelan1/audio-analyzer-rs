"""
Two-head depthwise-separable CNN for polyphonic pitch detection.

Architecture
------------
Shared backbone → two branches:
  • Count head  : predicts how many notes are simultaneously active (0–4)
  • Note head   : predicts which of the 425 frequency bins are active
                  (85 semitones × 5 bins/semitone, 20-cent resolution)

Separating count from identity lets each head specialise.  At inference the
count head controls how many notes are selected from the note head's ranked
probability list (top-K), eliminating the need for a fixed 0.5 threshold.

Input : [batch, 1, 128]   – 128 log-mel bins (50 Hz – 8 kHz), z-score normalised
Outputs: (count_logits [batch, 5], note_logits [batch, 425])
"""

import torch
import torch.nn as nn

# MIDI range covered by the note head (inclusive)
MIDI_MIN          = 24    # C1  ~  32.7 Hz
MIDI_MAX          = 108    # C8 ~ 4186 Hz
NUM_SEMITONES     = MIDI_MAX - MIDI_MIN + 1  # 85
BINS_PER_SEMITONE = 5     # 20-cent resolution (CREPE-style)
NUM_NOTES         = NUM_SEMITONES * BINS_PER_SEMITONE  # 425

# Maximum number of simultaneous notes the count head can predict
MAX_NOTES = 4


class ResidualDSBlock(nn.Module):
    """
    Depthwise-separable conv block with a residual skip connection.

    dilation > 1 broadens the receptive field without extra parameters,
    useful for capturing harmonic overtone patterns that span many mel bins.
    GELU activations tend to outperform ReLU on classification tasks.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int,
        dilation:     int = 1,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Sequential(
            # depthwise: each channel filtered independently
            nn.Conv1d(
                in_channels, in_channels, kernel_size,
                padding=padding, dilation=dilation, groups=in_channels, bias=False,
            ),
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
            # pointwise: mix channels
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

        # 1×1 projection so residual dimensions match when channels change
        self.skip = (
            nn.Conv1d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.skip(x)


class ImprovedPitchDetector(nn.Module):
    """
    Two-head pitch detector with a wider residual depthwise-separable backbone.

    Backbone width progression: 1 → 32 → 64 → 128 → 256 → 256 → 256, spatial pool → 8
    Dilated convolutions in blocks 3 and 5 capture harmonic patterns that
    span many mel bins without proportionally increasing parameter count.

    Loss during training
    --------------------
    note_loss  = BCEWithLogitsLoss(pos_weight, label_smoothing) on note_logits
    count_loss = CrossEntropyLoss on count_logits
    total      = note_loss + count_weight * count_loss

    Inference
    ---------
    count = argmax(count_logits).clamp(min=1)
    notes = top-count(sigmoid(note_logits))
    """

    def __init__(self, num_notes: int = NUM_NOTES, max_notes: int = MAX_NOTES):
        super().__init__()

        # ── shared backbone ──────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),      # [B, 32, 64]
        )

        self.block1 = ResidualDSBlock(32,  64,  kernel_size=5)              # [B, 64,  64]
        self.pool1  = nn.MaxPool1d(2)                                        # [B, 64,  32]
        self.block2 = ResidualDSBlock(64,  128, kernel_size=3)              # [B, 128, 32]
        self.block3 = ResidualDSBlock(128, 128, kernel_size=3, dilation=2)  # [B, 128, 32]
        self.block4 = ResidualDSBlock(128, 256, kernel_size=3)              # [B, 256, 32]
        self.block5 = ResidualDSBlock(256, 256, kernel_size=3, dilation=2)  # [B, 256, 32]

        # Pool to 8 frequency half-octave bands then flatten — preserves which part
        # of the spectrum is active (critical for pitch), unlike GlobalAvgPool which
        # dilutes the positional signal away entirely.  8 bands (vs 4) gives enough
        # resolution to distinguish notes within the same octave.
        self.spatial_pool = nn.AdaptiveAvgPool1d(8)   # [B, 256, 8]
        _flat = 256 * 8                                # 2048

        # ── count head ───────────────────────────────────────────────────────
        self.count_head = nn.Sequential(
            nn.Flatten(),                  # [B, 256, 4] → [B, 1024]
            nn.Linear(_flat, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, max_notes + 1),
        )

        # ── note head ────────────────────────────────────────────────────────
        # One logit per 20-cent bin (360 = 72 semitones × 5).
        self.note_head = nn.Sequential(
            nn.Flatten(),                  # [B, 256, 4] → [B, 1024]
            nn.Linear(_flat, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_notes),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, 1, 128]
        Returns:
            count_logits: [batch, 5]    (no softmax – use CrossEntropyLoss)
            note_logits:  [batch, 425]  (no sigmoid – use BCEWithLogitsLoss)
        """
        x = self.stem(x)
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.spatial_pool(x)      # [B, 256, 8]

        return self.count_head(x), self.note_head(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = ImprovedPitchDetector()
    dummy = torch.randn(4, 1, 128)
    count_logits, note_logits = model(dummy)
    print(f"count_logits shape : {count_logits.shape}")
    print(f"note_logits  shape : {note_logits.shape}")
    print(f"Parameters         : {count_parameters(model):,}")
