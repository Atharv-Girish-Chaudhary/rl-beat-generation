#!/usr/bin/env python3
"""
evaluate.py — Evaluate the trained beat-generation agent across N episodes.

Supports Phase 1 (4-layer, 4×16 grid) and Phase 2 (8-layer, 8×16 grid).

Metrics computed per episode:
  discriminator_score  — P(real) from BeatDiscriminator.sigmoid(logit)
  rule_reward          — rule-based score from compute_reward (no discriminator)
  beat_density         — fraction of non-silent cells across the full L×16 grid
  groove_consistency   — fraction of all hits that land on strong beats (steps 0,4,8,12)
  per_layer_density    — beat density broken down by instrument layer

Outputs:
  • Console: aligned summary table
  • File (Phase 1): outputs/evaluation_report.json
  • File (Phase 2): outputs/evaluation_report_phase2.json

Usage:
  python evaluation/evaluate.py
  python evaluation/evaluate.py --phase 2
  python evaluation/evaluate.py --n_episodes 50 --seed 7
  python evaluation/evaluate.py --phase 2 --checkpoint_dir outputs/checkpoints/ --n_episodes 10
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# ── Repo root on sys.path ─────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from beat_rl.env.beat_env import BeatGridEnv
from beat_rl.env.reward import compute_reward
from beat_rl.models.actor import CNNLayerStepSampleActor
from beat_rl.models.discriminator import BeatDiscriminator

# ── Phase 1 constants (must match train_ppo.py) ───────────────────────────────
_PHASE1_LAYER_NAMES = ["kick", "snare", "hihat", "clap"]
_PHASE1_L, _PHASE1_T, _PHASE1_S = 4, 16, 15

# ── Phase 2 constants ─────────────────────────────────────────────────────────
_PHASE2_LAYER_NAMES = ["kick", "snare", "hihat", "clap", "bass", "melody", "pad", "fx"]
_PHASE2_L, _PHASE2_T, _PHASE2_S = 8, 16, 15

# Strong-beat steps in a 16-step bar (quarter notes in 4/4 time)
STRONG_BEATS = {0, 4, 8, 12}

# Default output paths
_DEFAULT_REPORT_P1 = REPO_ROOT / "outputs" / "evaluation_report.json"
_DEFAULT_REPORT_P2 = REPO_ROOT / "outputs" / "evaluation_report_phase2.json"


# ── Module-level phase state (set during main()) ──────────────────────────────
# These are set once at startup so metric helpers don't need extra arguments.
LAYER_NAMES: list = _PHASE1_LAYER_NAMES
L: int = _PHASE1_L
T: int = _PHASE1_T
S: int = _PHASE1_S


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dummy_reward(grid, final, action_coord):
    """No-op reward fn — satisfies BeatGridEnv constructor during inference."""
    return 0.0


def _run_episode(actor: CNNLayerStepSampleActor, env: BeatGridEnv, device: str) -> np.ndarray:
    """Roll out one episode. Returns completed (L, T) grid."""
    obs, _ = env.reset()
    done = False
    with torch.no_grad():
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            action, _ = actor.act(obs_t)
            obs, _, done, _, _ = env.step(int(action))
    return env.grid.copy()


# ── Metric functions ──────────────────────────────────────────────────────────

def metric_disc_score(grid: np.ndarray, disc: BeatDiscriminator) -> float:
    """Discriminator P(real): reuse reward.py's existing _get_discriminator_score path."""
    return compute_reward(
        grid, final=True, action_coord=None,
        phase=1, discriminator=disc, alpha=0.0, beta=1.0
    )


def metric_rule_reward(grid: np.ndarray) -> float:
    """Rule-based score only (discriminator excluded)."""
    return compute_reward(
        grid, final=True, action_coord=None,
        phase=1, discriminator=None, alpha=1.0, beta=0.0
    )


def metric_beat_density(grid: np.ndarray) -> float:
    """Fraction of cells with a non-silent hit across the full grid."""
    return float(np.sum(grid > 0) / (L * T))


def metric_groove_consistency(grid: np.ndarray) -> float:
    """
    Fraction of total hits that fall on a strong beat (step ∈ {0,4,8,12}).
    Returns 0.0 if the grid is completely silent.
    """
    total_hits = int(np.sum(grid > 0))
    if total_hits == 0:
        return 0.0
    strong_hits = sum(
        int(np.sum(grid[:, step] > 0)) for step in STRONG_BEATS
    )
    return float(strong_hits / total_hits)


def metric_per_layer_density(grid: np.ndarray) -> dict:
    """Fraction of active steps per instrument layer (uses active LAYER_NAMES)."""
    return {
        name: float(np.sum(grid[i] > 0) / T)
        for i, name in enumerate(LAYER_NAMES)
    }


def evaluate_episode(grid: np.ndarray, disc: BeatDiscriminator) -> dict:
    """Compute all metrics for a single completed grid."""
    return {
        "discriminator_score": metric_disc_score(grid, disc),
        "rule_reward":         metric_rule_reward(grid),
        "beat_density":        metric_beat_density(grid),
        "groove_consistency":  metric_groove_consistency(grid),
        "per_layer_density":   metric_per_layer_density(grid),
    }


# ── Aggregation ───────────────────────────────────────────────────────────────

def _agg(values: list) -> dict:
    """Return mean + std for a list of floats, rounded to 4 dp."""
    arr = np.array(values, dtype=float)
    return {"mean": round(float(arr.mean()), 4), "std": round(float(arr.std()), 4)}


def aggregate(episode_results: list) -> dict:
    """Collapse per-episode dicts into summary statistics."""
    scalar_keys = ["discriminator_score", "rule_reward", "beat_density", "groove_consistency"]
    summary = {k: _agg([ep[k] for ep in episode_results]) for k in scalar_keys}

    # Per-layer density aggregated per instrument
    summary["per_layer_density"] = {
        name: _agg([ep["per_layer_density"][name] for ep in episode_results])
        for name in LAYER_NAMES
    }
    return summary


# ── Pretty-print ──────────────────────────────────────────────────────────────

def _bar(value: float, width: int = 20) -> str:
    """ASCII progress bar for a value in [0, 1]."""
    filled = round(value * width)
    return "█" * filled + "░" * (width - filled)


def print_summary(summary: dict, n_episodes: int, phase: int) -> None:
    W = 64
    print()
    print("╔" + "═" * (W - 2) + "╗")
    print(f"║{'RL Beat Agent — Evaluation Report':^{W-2}}║")
    print(f"║{f'Phase {phase}  ({n_episodes} episodes)':^{W-2}}║")
    print("╠" + "═" * (W - 2) + "╣")

    scalar_rows = [
        ("Discriminator score",  summary["discriminator_score"]),
        ("Rule reward",          summary["rule_reward"]),
        ("Beat density",         summary["beat_density"]),
        ("Groove consistency",   summary["groove_consistency"]),
    ]

    for label, stats in scalar_rows:
        mean, std = stats["mean"], stats["std"]
        bar = _bar(mean)
        print(f"║  {label:<22} {mean:.4f} ± {std:.4f}  {bar} ║")

    print("╠" + "═" * (W - 2) + "╣")
    print(f"║{'Per-layer density':^{W-2}}║")
    print("╠" + "═" * (W - 2) + "╣")

    for name in LAYER_NAMES:
        stats = summary["per_layer_density"][name]
        mean, std = stats["mean"], stats["std"]
        bar = _bar(mean)
        print(f"║  {name:<22} {mean:.4f} ± {std:.4f}  {bar} ║")

    print("╚" + "═" * (W - 2) + "╝")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global LAYER_NAMES, L, T, S  # allow metric helpers to see phase-specific values

    parser = argparse.ArgumentParser(
        description="Evaluate the trained beat agent across N episodes (Phase 1 or 2)."
    )
    parser.add_argument(
        "--phase", type=int, default=1, choices=[1, 2],
        help="Which training phase to evaluate (default: 1)",
    )
    parser.add_argument(
        "--n_episodes", type=int, default=20,
        help="Number of episodes to evaluate (default: 20)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--checkpoint_dir", default=str(REPO_ROOT / "outputs" / "checkpoints"),
        help="Directory containing actor/discriminator checkpoints",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to write JSON report (defaults to phase-appropriate path)",
    )
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)

    # ── Configure phase-specific constants ───────────────────────────────────
    if args.phase == 2:
        LAYER_NAMES = _PHASE2_LAYER_NAMES
        L = _PHASE2_L
        T = _PHASE2_T
        S = _PHASE2_S
        actor_filename      = "actor_phase2_best.pth"
        disc_filename       = "discriminator_phase2.pt"
        disc_d_ff           = 128
        default_report_path = _DEFAULT_REPORT_P2
        print("Phase 2 evaluation (8-layer, 8×16 grid)")
    else:
        LAYER_NAMES = _PHASE1_LAYER_NAMES
        L = _PHASE1_L
        T = _PHASE1_T
        S = _PHASE1_S
        actor_filename      = "actor_best.pth"
        disc_filename       = "discriminator_phase1_v2.pt"
        disc_d_ff           = 128
        default_report_path = _DEFAULT_REPORT_P1
        print("Phase 1 evaluation (4-layer, 4×16 grid)")

    LAYER_TO_SAMPLES = {i: list(range(1, S + 1)) for i in range(L)}
    out_path = Path(args.output) if args.output else default_report_path

    # ── Seed ─────────────────────────────────────────────────────────────────
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # ── Device ───────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # ── Load actor ───────────────────────────────────────────────────────────
    actor_path = ckpt_dir / actor_filename
    if not actor_path.exists():
        print(f"ERROR: actor checkpoint not found at {actor_path}")
        sys.exit(1)

    actor = CNNLayerStepSampleActor(
        L=L, T=T, S=S, env_layer_to_samples=LAYER_TO_SAMPLES
    ).to(device)
    actor.load_state_dict(torch.load(actor_path, map_location=device))
    actor.eval()
    print(f"Actor  : {actor_path}")

    # ── Load discriminator ────────────────────────────────────────────────────
    disc_path = ckpt_dir / disc_filename
    if not disc_path.exists():
        print(f"ERROR: discriminator checkpoint not found at {disc_path}")
        sys.exit(1)

    disc = BeatDiscriminator(
        num_instruments=L, num_steps=T,
        d_model=64, num_heads=4, num_blocks=2, d_ff=disc_d_ff
    ).to(device)
    disc.load_state_dict(torch.load(disc_path, map_location=device))
    disc.eval()
    print(f"Disc   : {disc_path}")

    # ── Build env ─────────────────────────────────────────────────────────────
    env = BeatGridEnv(
        L=L, T=T, S=S,
        reward_fn=_dummy_reward,
        layer_to_samples=LAYER_TO_SAMPLES,
        phase=args.phase,
    )

    # ── Run episodes ──────────────────────────────────────────────────────────
    print(f"\nRunning {args.n_episodes} episodes...\n")
    episode_results = []

    for ep in range(args.n_episodes):
        grid = _run_episode(actor, env, device)
        metrics = evaluate_episode(grid, disc)
        episode_results.append(metrics)

        # Progress line
        disc_s   = f"{metrics['discriminator_score']:.3f}"
        rule_s   = f"{metrics['rule_reward']:.3f}"
        dens_s   = f"{metrics['beat_density']:.3f}"
        groove_s = f"{metrics['groove_consistency']:.3f}"
        print(
            f"  ep {ep+1:>3}/{args.n_episodes}"
            f"  disc={disc_s}  rule={rule_s}"
            f"  density={dens_s}  groove={groove_s}"
        )

    # ── Aggregate & display ───────────────────────────────────────────────────
    summary = aggregate(episode_results)
    print_summary(summary, args.n_episodes, phase=args.phase)

    # ── Save JSON report ──────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "config": {
            "phase":           args.phase,
            "n_episodes":      args.n_episodes,
            "seed":            args.seed,
            "checkpoint_dir":  str(ckpt_dir),
            "actor":           str(actor_path),
            "discriminator":   str(disc_path),
        },
        "summary":  summary,
        "episodes": episode_results,
    }

    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"Report saved → {out_path}\n")


if __name__ == "__main__":
    main()
