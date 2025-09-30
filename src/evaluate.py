
"""
evaluate.py — Load model, (optionally) generate a sample, and run quick quantitative checks:
- Precision/Recall/F1 on validation (subset)
- Density (real vs generated)
- Pitch histogram plot (real vs generated)

Usage:
  python -m src.evaluate \    --weights models/lstm_nottingham.pt \    --data data/Nottingham.mat \    --seq-len 128 \    --batch-size 32 \    --seed-index 0 \    --seed-len 192 \    --steps 512 \    --temperature 1.0 \    --threshold 0.50 \    --plot samples/evaluation.png
"""

from __future__ import annotations
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from src.dataset import load_nottingham, as_binary_float32_list, build_dataloaders
from src.model import MusicLSTM
from src.generate import generate_roll
from src import utils


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to trained state_dict (.pt)")
    ap.add_argument("--data", required=True, help="Path to Nottingham .mat")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)

    # generation controls
    ap.add_argument("--seed-index", type=int, default=0)
    ap.add_argument("--seed-len", type=int, default=192)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=0.50)

    # model hyperparams (must match training)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.4)

    ap.add_argument("--plot", default="samples/evaluation.png", help="Where to save the pitch-hist plot (PNG).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ---- Data ----
    train_raw, valid_raw, _ = load_nottingham(args.data)
    train_rolls = as_binary_float32_list(train_raw)
    valid_rolls = as_binary_float32_list(valid_raw)
    train_dl, valid_dl = build_dataloaders(
        train_rolls, valid_rolls, seq_len=args.seq_len, batch_size=args.batch_size
    )

    # ---- Model ----
    model = MusicLSTM(input_size=88, hidden_size=args.hidden, num_layers=args.layers, dropout=args.dropout).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print("\nLoaded weights from:", args.weights)

    # ---- (a) Precision/Recall/F1 on validation ----
    p, r, f1 = utils.precision_recall_at_threshold(
        model, valid_dl, device=device, thr=args.threshold, max_batches=50
    )
    print(f"\nValidation metrics @{args.threshold:.2f}  —  Precision: {p:.3f}  Recall: {r:.3f}  F1: {f1:.3f}")

    # ---- (b) Generate one sample & density ----
    seed_full = valid_rolls[args.seed_index % len(valid_rolls)]
    seed = seed_full[: min(args.seed_len, seed_full.shape[0])]
    gen = generate_roll(
        model, device, seed_roll=seed, steps=args.steps,
        temperature=args.temperature, prob_threshold=args.threshold
    )
    print("Generated roll shape:", gen.shape)

    real_mean, real_std = utils.density(seed[:512])
    gen_mean,  gen_std  = utils.density(gen[-512:])
    print("Density (active-note ratio per frame):")
    print(f"  Real  mean={real_mean:.4f}, std={real_std:.4f}")
    print(f"  Gen   mean={gen_mean:.4f},  std={gen_std:.4f}")

    # ---- (c) Pitch histogram plot ----
    real_hist = utils.pitch_hist(seed[:512])
    gen_hist  = utils.pitch_hist(gen[-512:])

    plt.figure(figsize=(9, 3))
    plt.plot(real_hist, label='Real (seed 512)')
    plt.plot(gen_hist,  label='Generated (last 512)')
    plt.title('Pitch usage distribution (normalized)')
    plt.xlabel('Pitch bin (21..108 → 0..87)')
    plt.ylabel('Avg ON per frame')
    plt.legend(); plt.tight_layout()
    out_plot = args.plot
    Path(out_plot).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_plot, dpi=150)
    print('Saved plot to:', out_plot)


if __name__ == "__main__":
    main()
