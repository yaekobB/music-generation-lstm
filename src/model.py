"""
model.py — LSTM-based next-frame predictor for piano-rolls.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class MusicLSTM(nn.Module):
    """
    Multi-label next-step predictor over 88 pitches.
    Input  : (B, L, 88)  multi-hot frames
    Output : (B, L, 88)  logits for next-step frame at each position
    """
    def __init__(self, input_size: int = 88, hidden_size: int = 192, num_layers: int = 2, dropout: float = 0.4):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)   # (B,L,88) -> (B,L,H)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, input_size)          # (B,L,H) -> (B,L,88)

        # Xavier init for stability
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.constant_(self.input_proj.bias, 0.)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.constant_(self.out.bias, 0.)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h, _ = self.lstm(h)
        h = self.dropout(h)
        logits = self.out(h)  # (B,L,88)
        return logits


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
