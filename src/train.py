"""
train.py — Train MusicLSTM on Nottingham.
Usage:
  python src/train.py --data data/Nottingham.mat --epochs 20 --seq-len 128 --batch-size 32 --lr 1e-5 --save models/lstm_nottingham.pt
"""

from __future__ import annotations
import argparse
from time import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from src.dataset import load_nottingham, as_binary_float32_list, build_dataloaders
from src.model import MusicLSTM, count_parameters


def estimate_pos_weight(dataloader, device: torch.device, max_batches: int = 20) -> torch.Tensor:
    """Estimate class imbalance pos_weight for BCEWithLogitsLoss."""
    pos = 0.0
    total = 0.0
    n = 0
    for _, yb in dataloader:
        yb = yb.to(device)
        pos += yb.sum().item()
        total += yb.numel()
        n += 1
        if n >= max_batches:
            break
    pos_rate = max(pos / max(1.0, total), 1e-6)
    neg_rate = 1.0 - pos_rate
    return torch.tensor([neg_rate / pos_rate], dtype=torch.float32, device=device)


def run_epoch(model: nn.Module, loader, device: torch.device, criterion, optimizer=None, clip_grad: float | None = 1.0) -> float:
    """Run one epoch; if optimizer is None → eval mode."""
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_elems = 0
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            logits = model(xb)
            loss = criterion(logits, yb)

        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if clip_grad is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()

        batch_elems = yb.numel()
        total_loss += loss.detach().item() * batch_elems
        total_elems += batch_elems
    return total_loss / max(1, total_elems)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to Nottingham .mat")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--save", default="models/lstm_nottingham.pt")
    ap.add_argument("--patience", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load data & prepare rolls
    train_raw, valid_raw, _ = load_nottingham(args.data)
    train_rolls = as_binary_float32_list(train_raw)
    valid_rolls = as_binary_float32_list(valid_raw)

    # Dataloaders
    train_dl, valid_dl = build_dataloaders(
        train_rolls, valid_rolls, seq_len=args.seq_len, batch_size=args.batch_size
    )

    # Model
    model = MusicLSTM(input_size=88, hidden_size=args.hidden, num_layers=args.layers, dropout=args.dropout).to(device)
    print(f"Parameters: {count_parameters(model):,}")

    # Loss with class imbalance correction
    pos_weight = estimate_pos_weight(train_dl, device=device, max_batches=20)
    print("Estimated pos_weight:", float(pos_weight), "\n")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = Adam(model.parameters(), lr=args.lr)

    best_val = float('inf')
    best_path = args.save
    patience = args.patience
    no_improve = 0
    
    print("\n Starting training... \n")

    for epoch in range(1, args.epochs + 1):
        t0 = time()
        train_loss = run_epoch(model, train_dl, device, criterion, optimizer, clip_grad=1.0)
        val_loss   = run_epoch(model, valid_dl, device, criterion, optimizer=None, clip_grad=None)
        dt = time() - t0
        print(f"[{epoch:02d}/{args.epochs}] train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  ({dt:.1f}s)")

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            no_improve = 0
            torch.save(model.state_dict(), best_path)  # weights only
        else:
            no_improve += 1
            if no_improve >= patience:
                print("  ↳ Early stopping triggered.")
                break

    # Load best weights back
    state = torch.load(best_path, map_location=device)
    model.load_state_dict(state)
    print("Loaded best model. Best val_loss:", best_val)
    print(f"  ↳ Saved best weights to {best_path}")


if __name__ == "__main__":
    main()
