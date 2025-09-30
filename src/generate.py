"""
generate.py — Inference: seed → autoregressive generation → MIDI export.

Example:
  python src/generate.py \
    --weights models/lstm_nottingham.pt \
    --data data/Nottingham.mat \
    --seed-index 0 --seed-len 192 \
    --steps 512 --temperature 1.00 --threshold 0.50 \
    --out samples/lstm_sample.mid
"""

from __future__ import annotations
import argparse
import numpy as np
import torch
import pretty_midi

from src.dataset import load_nottingham, as_binary_float32_list
from src.model import MusicLSTM


@torch.no_grad()
def generate_roll(model, device, seed_roll: np.ndarray, steps: int = 512, temperature: float = 1.0, prob_threshold: float = 0.5) -> np.ndarray:
    """
    Autoregressively extend a seed piano-roll by `steps` frames.
    Returns (L+steps, 88) float32 in {0,1}.
    """
    L = seed_roll.shape[0]
    ctx = torch.from_numpy(seed_roll.astype(np.float32)).unsqueeze(0).to(device)  # (1,L,88)
    out = [seed_roll.copy()]
    amp = torch.cuda.is_available()

    for _ in range(steps):
        with torch.amp.autocast('cuda', enabled=amp):
            logits = model(ctx)[:, -1, :]                         # (1,88)
        probs  = torch.sigmoid(logits / max(1e-6, temperature))   # temperature scaling
        next_f = (probs >= prob_threshold).float()                # deterministic thresholding
        out.append(next_f.squeeze(0).cpu().numpy())

        # Slide context to keep length L
        ctx = torch.cat([ctx, next_f.unsqueeze(1)], dim=1)
        if ctx.size(1) > L:
            ctx = ctx[:, -L:, :]

    return np.vstack(out)


def roll_to_midi(roll: np.ndarray, fs: int = 4, velocity: int = 80, program: int = 0) -> pretty_midi.PrettyMIDI:
    """
    Convert (T,88) binary piano-roll to a single-track MIDI.
    fs=4 => 4 frames/quarter-note (16th-note grid).
    """
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program)  # 0 = Acoustic Grand Piano
    PITCH_MIN = 21
    t_per_frame = 1.0 / fs

    active_start = [None] * 88
    T = roll.shape[0]
    for t in range(T):
        on = roll[t] > 0.5
        for p in range(88):
            if on[p] and active_start[p] is None:
                active_start[p] = t
            if (not on[p]) and (active_start[p] is not None):
                inst.notes.append(pretty_midi.Note(
                    velocity=velocity, pitch=PITCH_MIN + p,
                    start=active_start[p] * t_per_frame,
                    end=t * t_per_frame
                ))
                active_start[p] = None

    # Close lingering notes at the end
    for p, st in enumerate(active_start):
        if st is not None:
            inst.notes.append(pretty_midi.Note(
                velocity=velocity, pitch=PITCH_MIN + p,
                start=st * t_per_frame, end=T * t_per_frame
            ))

    pm.instruments.append(inst)
    return pm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to trained state_dict (.pt)")
    ap.add_argument("--data", required=True, help="Path to Nottingham .mat (to fetch validation seed)")
    ap.add_argument("--seed-index", type=int, default=0)
    ap.add_argument("--seed-len", type=int, default=192)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--out", default="samples/lstm_sample.mid")
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n Device:", device)

    # Rebuild model (must match training hyperparams), then load weights
    model = MusicLSTM(input_size=88, hidden_size=args.hidden, num_layers=args.layers, dropout=args.dropout).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print("\n Loaded weights from:", args.weights)

    # Get a real seed from validation split
    _, valid_raw, _ = load_nottingham(args.data)
    valid_rolls = as_binary_float32_list(valid_raw)
    print("\nValidation seeds available:", len(valid_rolls))
    seed_full = valid_rolls[args.seed_index % len(valid_rolls)]
    seed = seed_full[: min(args.seed_len, seed_full.shape[0])]

    # Generate
    gen = generate_roll(model, device, seed, steps=args.steps, temperature=args.temperature, prob_threshold=args.threshold)
    print("Generated roll shape:", gen.shape)

    # Export to MIDI
    pm = roll_to_midi(gen, fs=4, velocity=80, program=0)
    out_path = args.out
    pm.write(out_path)
    print("Wrote MIDI to:", out_path)


if __name__ == "__main__":
    main()
