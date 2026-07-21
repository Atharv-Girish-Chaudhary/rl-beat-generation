#!/usr/bin/env python3
"""
evaluate_discriminator.py — Evaluation-only validation of a trained discriminator checkpoint.

Pins the discriminator's validation accuracy to a committed, reproducible artifact instead
of a text-only claim. No training occurs and no checkpoint is modified.

Methodology mirrors scripts/train_discriminator.py exactly by reusing its dataset class:
  • load data/processed/groove_grids.npy, slice to 4 instrument rows (Phase 1)
  • drop all-zero grids from the positives
  • 50% real / 50% synthetic negatives (same mix as training)
  • 80/20 train/val split; accuracy is reported on the val split only
The dataset draws negatives at access time, so the run is seeded (numpy + torch, plus the
split generator) for reproducibility.

Usage:
  python evaluation/evaluate_discriminator.py                # Phase 1, seed 7
  python evaluation/evaluate_discriminator.py --seed 7 --output outputs/discriminator_phase1_eval.json
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from beat_rl.models import BeatDiscriminator


def _load_phase1_dataset_class():
    """Import _Phase1Dataset from scripts/train_discriminator.py so the eval uses the
    identical negative-sampling methodology as training (no duplicated logic to drift)."""
    spec = importlib.util.spec_from_file_location(
        "train_discriminator", REPO_ROOT / "scripts" / "train_discriminator.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._Phase1Dataset


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained discriminator checkpoint (no training).")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data", default=str(REPO_ROOT / "data" / "processed" / "groove_grids.npy"))
    parser.add_argument("--checkpoint", default=str(REPO_ROOT / "outputs" / "checkpoints" / "discriminator_phase1_v2.pt"))
    parser.add_argument("--output", default=str(REPO_ROOT / "outputs" / "discriminator_phase1_eval.json"))
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cpu"  # deterministic; the model is tiny

    # ── Data: identical preprocessing to training ─────────────────────────────
    raw = np.load(args.data)
    real_grids = raw[:, :4, :]                                # Phase 1 slice
    n_raw = len(real_grids)
    real_grids = real_grids[real_grids.sum(axis=(1, 2)) > 0]  # drop silent positives
    n_removed = n_raw - len(real_grids)

    Phase1Dataset = _load_phase1_dataset_class()
    dataset = Phase1Dataset(real_grids=real_grids, num_samples=len(real_grids) * 2)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # ── Model: load frozen checkpoint ─────────────────────────────────────────
    model = BeatDiscriminator(
        num_instruments=4, num_steps=16, d_model=64, num_heads=4, num_blocks=2, d_ff=128
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    criterion = nn.BCEWithLogitsLoss(reduction="sum")

    # ── Validation pass ───────────────────────────────────────────────────────
    correct = total = 0
    real_correct = real_total = fake_correct = fake_total = 0
    loss_sum = 0.0

    with torch.no_grad():
        for grids, labels in val_loader:
            grids = grids.to(device)
            labels = labels.to(device).view(-1, 1).float()
            logits, _ = model(grids)
            loss_sum += criterion(logits, labels).item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            hit = (preds == labels)
            correct += hit.sum().item()
            total += labels.size(0)
            is_real = labels == 1.0
            real_correct += hit[is_real].sum().item()
            real_total += is_real.sum().item()
            fake_correct += hit[~is_real].sum().item()
            fake_total += (~is_real).sum().item()

    val_acc = correct / total
    report = {
        "config": {
            "checkpoint":       str(Path(args.checkpoint).relative_to(REPO_ROOT)),
            "data":             str(Path(args.data).relative_to(REPO_ROOT)),
            "phase":            1,
            "seed":             args.seed,
            "batch_size":       args.batch_size,
            "split":            "80/20 train/val (val only), torch.random_split seeded",
            "n_raw_grids":      n_raw,
            "n_silent_removed": n_removed,
            "n_real_grids":     len(real_grids),
            "dataset_size":     len(dataset),
            "val_size":         val_size,
            "negative_mix":     "random/shuffled/density/silent/agent-fallback (same as training)",
        },
        "results": {
            "val_accuracy":      round(val_acc, 4),
            "val_bce_loss":      round(loss_sum / total, 4),
            "real_accuracy":     round(real_correct / real_total, 4),
            "fake_accuracy":     round(fake_correct / fake_total, 4),
            "n_correct":         int(correct),
            "n_total":           int(total),
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Val size   : {val_size}  (seed {args.seed})")
    print(f"Val acc    : {val_acc:.4f}   (real {report['results']['real_accuracy']:.4f} / fake {report['results']['fake_accuracy']:.4f})")
    print(f"Report     → {out_path}")


if __name__ == "__main__":
    main()
