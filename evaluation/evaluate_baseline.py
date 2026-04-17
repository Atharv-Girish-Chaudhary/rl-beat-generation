
"""
evaluate_baseline.py — Evaluate a random agent as a baseline.

Supports Phase 1 (4-layer, 4×16 grid) and Phase 2 (8-layer, 8×16 grid).
Computes evaluation metrics and generates a side-by-side comparison plot.

Outputs (Phase 1):
  • outputs/random_baseline_report.json
  • outputs/plots/baseline_comparison.png

Outputs (Phase 2):
  • outputs/random_baseline_report_phase2.json
  • outputs/plots/baseline_comparison_phase2.png

Usage:
  python evaluation/evaluate_baseline.py
  python evaluation/evaluate_baseline.py --phase 2
  python evaluation/evaluate_baseline.py --phase 2 --n_episodes 50
"""

import os
import json
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from beat_rl.env.beat_env import BeatGridEnv
from beat_rl.env.reward import compute_reward
from beat_rl.models.discriminator import BeatDiscriminator

# ── Phase 1 constants ─────────────────────────────────────────────────────────
_PHASE1_L, _PHASE1_T, _PHASE1_S = 4, 16, 15
_PHASE1_LAYER_NAMES = ["kick", "snare", "hihat", "clap"]

# ── Phase 2 constants ─────────────────────────────────────────────────────────
_PHASE2_L, _PHASE2_T, _PHASE2_S = 8, 16, 15
_PHASE2_LAYER_NAMES = ["kick", "snare", "hihat", "clap", "bass", "melody", "pad", "fx"]

# Module-level phase state (set during main())
LAYER_NAMES: list = _PHASE1_LAYER_NAMES
L: int = _PHASE1_L
T: int = _PHASE1_T
S: int = _PHASE1_S


def _dummy_reward(grid, final, action_coord):
    return 0.0


def metric_disc_score(grid: np.ndarray, disc: BeatDiscriminator) -> float:
    return compute_reward(
        grid, final=True, action_coord=None,
        phase=1, discriminator=disc, alpha=0.0, beta=1.0
    )


def metric_rule_reward(grid: np.ndarray) -> float:
    return compute_reward(
        grid, final=True, action_coord=None,
        phase=1, discriminator=None, alpha=1.0, beta=0.0
    )


def metric_beat_density(grid: np.ndarray) -> float:
    return float(grid.astype(bool).mean())


def metric_groove_consistency(grid: np.ndarray) -> float:
    half1 = grid[:, :8].astype(bool)
    half2 = grid[:, 8:].astype(bool)
    intersection = np.logical_and(half1, half2).sum()
    union = np.logical_or(half1, half2).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def metric_per_layer_density(grid: np.ndarray) -> dict:
    """Fraction of active steps per instrument layer (uses active LAYER_NAMES)."""
    return {
        name: float(grid[i].astype(bool).mean())
        for i, name in enumerate(LAYER_NAMES)
    }


def _agg(values: list) -> dict:
    arr = np.array(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std())}


def aggregate(episode_results: list) -> dict:
    scalar_keys = ["discriminator_score", "rule_reward", "beat_density", "groove_consistency"]
    summary = {k: _agg([ep[k] for ep in episode_results]) for k in scalar_keys}

    summary["per_layer_density"] = {
        name: _agg([ep["per_layer_density"][name] for ep in episode_results])
        for name in LAYER_NAMES
    }
    return summary


def main():
    global LAYER_NAMES, L, T, S  # allow metric helpers to see phase-specific values

    parser = argparse.ArgumentParser(
        description="Evaluate a random baseline agent (Phase 1 or 2)."
    )
    parser.add_argument(
        "--phase", type=int, default=1, choices=[1, 2],
        help="Which training phase to baseline (default: 1)",
    )
    parser.add_argument("--n_episodes", type=int, default=20)
    args = parser.parse_args()

    # ── Configure phase-specific constants ───────────────────────────────────
    if args.phase == 2:
        LAYER_NAMES = _PHASE2_LAYER_NAMES
        L = _PHASE2_L
        T = _PHASE2_T
        S = _PHASE2_S
        disc_filename       = "discriminator_phase2.pt"
        disc_d_ff           = 128
        report_filename     = "random_baseline_report_phase2.json"
        ppo_report_filename = "evaluation_report_phase2.json"
        plot_filename       = "baseline_comparison_phase2.png"
        print("Phase 2 random baseline (8-layer, 8×16 grid)")
    else:
        LAYER_NAMES = _PHASE1_LAYER_NAMES
        L = _PHASE1_L
        T = _PHASE1_T
        S = _PHASE1_S
        disc_filename       = "discriminator_phase1_v2.pt"
        disc_d_ff           = 128
        report_filename     = "random_baseline_report.json"
        ppo_report_filename = "evaluation_report.json"
        plot_filename       = "baseline_comparison.png"
        print("Phase 1 random baseline (4-layer, 4×16 grid)")

    LAYER_TO_SAMPLES = {i: list(range(1, S + 1)) for i in range(L)}

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # ── Load Discriminator ────────────────────────────────────────────────────
    disc_path = REPO_ROOT / "outputs" / "checkpoints" / disc_filename
    if not disc_path.exists():
        print(f"ERROR: Discriminator missing at {disc_path}")
        sys.exit(1)

    disc = BeatDiscriminator(L, T, d_model=64, num_heads=4, num_blocks=2, d_ff=disc_d_ff).to(device)
    disc.load_state_dict(torch.load(disc_path, map_location=device))
    disc.eval()
    print(f"Disc   : {disc_path}")

    # ── Init Env ──────────────────────────────────────────────────────────────
    env = BeatGridEnv(
        L=L, T=T, S=S,
        phase=args.phase,
        layer_to_samples=LAYER_TO_SAMPLES,
        reward_fn=_dummy_reward,
    )

    # ── Run random episodes ───────────────────────────────────────────────────
    episode_results = []
    print(f"Running {args.n_episodes} Random Baseline Episodes...")

    for ep in range(args.n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            l_idx, t_idx = random.choice(env.empty_cells)
            mask = env.get_action_mask(l_idx)
            valid_samples = np.where(mask)[0]
            s_idx = random.choice(valid_samples)
            action = (l_idx * env.T + t_idx) * (env.S + 1) + s_idx

            obs, _, done, _, _ = env.step(action)

        grid = env.grid.copy()

        metrics = {
            "discriminator_score": metric_disc_score(grid, disc),
            "rule_reward":         metric_rule_reward(grid),
            "beat_density":        metric_beat_density(grid),
            "groove_consistency":  metric_groove_consistency(grid),
            "per_layer_density":   metric_per_layer_density(grid),
        }
        episode_results.append(metrics)

        print(f"  Ep {ep+1:02d} | "
              f"Rule: {metrics['rule_reward']:.3f} | "
              f"Disc: {metrics['discriminator_score']:.5f} | "
              f"Dens: {metrics['beat_density']:.3f} | "
              f"Groove: {metrics['groove_consistency']:.3f}")

    summary = aggregate(episode_results)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n--- Baseline Summary ---")
    for k in ["rule_reward", "discriminator_score", "beat_density", "groove_consistency"]:
        print(f"{k:<20}: {summary[k]['mean']:.4f} ± {summary[k]['std']:.4f}")

    # ── Save Report ───────────────────────────────────────────────────────────
    report = {
        "config": {
            "phase":      args.phase,
            "n_episodes": args.n_episodes,
            "agent":      "random",
        },
        "summary":  summary,
        "episodes": episode_results,
    }

    report_out = REPO_ROOT / "outputs" / report_filename
    report_out.parent.mkdir(parents=True, exist_ok=True)
    with open(report_out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report to {report_out}")

    # ── Compare against PPO Agent (if report exists) ─────────────────────────
    ppo_report_path = REPO_ROOT / "outputs" / ppo_report_filename
    if ppo_report_path.exists():
        with open(ppo_report_path, "r") as f:
            ppo_data = json.load(f)

        ppo_rule      = ppo_data["summary"]["rule_reward"]["mean"]
        ppo_rule_std  = ppo_data["summary"]["rule_reward"]["std"]
        ppo_disc      = ppo_data["summary"]["discriminator_score"]["mean"]
        ppo_disc_std  = ppo_data["summary"]["discriminator_score"]["std"]

        rand_rule     = summary["rule_reward"]["mean"]
        rand_rule_std = summary["rule_reward"]["std"]
        rand_disc     = summary["discriminator_score"]["mean"]
        rand_disc_std = summary["discriminator_score"]["std"]

        labels    = ["Rule Reward", "Discriminator Score"]
        ppo_means = [ppo_rule, ppo_disc]
        ppo_stds  = [ppo_rule_std, ppo_disc_std]
        rand_means = [rand_rule, rand_disc]
        rand_stds  = [rand_rule_std, rand_disc_std]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.bar(x - width/2, ppo_means,  width, yerr=ppo_stds,  label='PPO Agent',    capsize=5, alpha=0.9, color='royalblue')
        ax.bar(x + width/2, rand_means, width, yerr=rand_stds, label='Random Agent', capsize=5, alpha=0.9, color='lightcoral')

        ax.set_ylabel('Score')
        ax.set_title(f'Performance Comparison: Random vs PPO Agent (Phase {args.phase})')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        for i, v in enumerate(ppo_means):
            ax.text(i - width/2, v + ppo_stds[i] + 0.02, f'{v:.4f}', ha='center', va='bottom', fontsize=9, rotation=90)
        for i, v in enumerate(rand_means):
            ax.text(i + width/2, v + rand_stds[i] + 0.02, f'{v:.4f}', ha='center', va='bottom', fontsize=9, rotation=90)

        fig.tight_layout()

        plot_dir = REPO_ROOT / "outputs" / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_out = plot_dir / plot_filename
        plt.savefig(plot_out, dpi=300)
        print(f"Saved comparison plot to {plot_out}")
    else:
        print(f"(No PPO report found at {ppo_report_path} — skipping comparison plot)")


if __name__ == "__main__":
    main()
