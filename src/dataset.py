"""
dataset.py — Data loading and dataset builders for Nottingham (.mat) piano-rolls.
- Loads MATLAB .mat into Python-friendly structures
- Converts to binary float32 (T, 88) piano-rolls
- Provides PyTorch Dataset for sliding windows (next-frame prediction)
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset, DataLoader

PITCH_MIN = 21   # MIDI A0
PITCH_MAX = 108  # MIDI C8
NUM_PITCH = PITCH_MAX - PITCH_MIN + 1  # 88


def load_nottingham(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Nottingham .mat and return (train_raw, valid_raw, test_raw)
    as 1D object arrays. Handles MATLAB singleton dims robustly.
    """
    mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)

    def _get(key: str) -> np.ndarray:
        arr = np.asarray(mat[key]).squeeze()
        # In rare cases squeezing yields a scalar; wrap it back
        if arr.ndim == 0:
            arr = np.array([arr.item()], dtype=object)
        return arr

    return _get('traindata'), _get('validdata'), _get('testdata')


def as_binary_float32_list(obj_arr: np.ndarray) -> List[np.ndarray]:
    """
    Convert MATLAB-style object array into a list of (T,88) float32 arrays
    with values in {0.0, 1.0}.
    """
    out: List[np.ndarray] = []
    for i in range(obj_arr.shape[0]):
        R = np.asarray(obj_arr[i])
        # Threshold >0 to binary, then cast to float32
        R = (R > 0).astype(np.float32)
        out.append(R)
    return out


class MultiHotNextStepDataset(Dataset):
    """
    Creates overlapping (X, Y) windows from a list of piano-rolls.
    X: [t .. t+L-1],  Y: [t+1 .. t+L]
    where each frame is an 88-dim multi-hot vector (float32 in {0,1}).
    """
    def __init__(self, rolls: List[np.ndarray], seq_len: int):
        self.seq_len = seq_len
        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []
        for R in rolls:
            T = int(R.shape[0])
            # Skip too-short sequences
            if T <= seq_len:
                continue
            # Sliding windows
            for t in range(0, T - seq_len):
                X = R[t      : t + seq_len]   # (L, 88)
                Y = R[t + 1  : t + seq_len + 1]
                self.samples.append((X, Y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X, Y = self.samples[idx]
        return torch.from_numpy(X), torch.from_numpy(Y)


def build_dataloaders(
    train_rolls: List[np.ndarray],
    valid_rolls: List[np.ndarray],
    seq_len: int,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Construct PyTorch DataLoaders for training and validation.
    """
    train_ds = MultiHotNextStepDataset(train_rolls, seq_len)
    valid_ds = MultiHotNextStepDataset(valid_rolls, seq_len)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    valid_dl = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_dl, valid_dl
