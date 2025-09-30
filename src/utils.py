"""
utils.py — Utilities: metrics & helpers.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn


@torch.no_grad()
def precision_recall_at_threshold(
    model: nn.Module,
    loader,
    device: torch.device,
    thr: float = 0.5,
    max_batches: int = 50
) -> Tuple[float, float, float]:
    """Compute Precision/Recall/F1 on a subset of validation batches."""
    model.eval()
    TP = FP = FN = 0.0
    n = 0
    amp = torch.cuda.is_available()
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        with torch.amp.autocast('cuda', enabled=amp):
            logits = model(xb)
        preds = (torch.sigmoid(logits) >= thr).float()
        TP += (preds * yb).sum().item()
        FP += (preds * (1 - yb)).sum().item()
        FN += ((1 - preds) * yb).sum().item()
        n += 1
        if n >= max_batches:
            break
    precision = TP / max(1.0, (TP + FP))
    recall    = TP / max(1.0, (TP + FN))
    f1        = 2 * precision * recall / max(1e-8, (precision + recall))
    return float(precision), float(recall), float(f1)


def density(frames: np.ndarray) -> tuple[float, float]:
    """Return mean/std of per-frame active-note ratio for (T,88) {0,1} rolls."""
    per_frame = frames.mean(axis=1) if frames.size else np.array([0.0])
    return float(per_frame.mean()), float(per_frame.std())


def pitch_hist(frames: np.ndarray) -> np.ndarray:
    """Return normalized pitch usage histogram for (T,88) {0,1} rolls."""
    T = max(1, frames.shape[0])
    return frames.sum(axis=0) / T
