#!/usr/bin/env python3
"""
generate_audio.py — Run the trained Phase 1 actor and render the beat grid to WAV.

Usage:
  python scripts/generate_audio.py
  python scripts/generate_audio.py --output outputs/my_beat.wav --bpm 90 --n_beats 8
  python scripts/generate_audio.py --seed 42          # reproducible result

Arguments:
  --output   Path to write the WAV file  (default: outputs/beat_sample.wav)
  --bpm      Tempo in beats per minute    (default: 120.0)
  --sr       Output sample rate in Hz     (default: 44100)
  --n_beats  How many times to loop the 1-bar grid (default: 4)
  --seed     Optional random seed for reproducibility
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import torch

# ── Repo root on sys.path so beat_rl package is importable ───────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from beat_rl.env.beat_env import BeatGridEnv
from beat_rl.models.actor import CNNLayerStepSampleActor

# ── Constants (must match training in scripts/train_ppo.py) ──────────────────

LAYER_NAMES = ["kick", "snare", "hihat", "clap"]   # Layer 0–3
SAMPLES_DIR = REPO_ROOT / "data" / "samples"
CHECKPOINT_PATH = REPO_ROOT / "outputs" / "checkpoints" / "actor_best.pth"

L, T, S = 4, 16, 15                                 # Phase 1 dims
LAYER_TO_SAMPLES = {i: list(range(1, S + 1)) for i in range(L)}


# ── Audio helpers ─────────────────────────────────────────────────────────────

def _load_metadata(layer_name: str) -> list:
    """Return the list of sample dicts from data/samples/{layer}/metadata.json."""
    path = SAMPLES_DIR / layer_name / "metadata.json"
    with open(path, "r") as fh:
        return json.load(fh)


def _load_sample(layer_name: str, sample_idx: int, target_sr: int) -> np.ndarray:
    """
    Load the audio file for a 1-based sample index from the given layer.

    sample_idx is the value stored in the beat grid (1–S). It maps to
    metadata[sample_idx - 1] (0-indexed). librosa resamples to target_sr
    and returns a mono float32 array.
    """
    metadata = _load_metadata(layer_name)
    entry = metadata[sample_idx - 1]          # 1-based → 0-based
    wav_path = SAMPLES_DIR / layer_name / entry["file"]
    audio, _ = librosa.load(str(wav_path), sr=target_sr, mono=True)
    return audio                               # float32, length varies


def render_grid(grid: np.ndarray, bpm: float = 120.0, sr: int = 44100) -> np.ndarray:
    """
    Render a (L, T) beat grid to a mono audio array at `sr` Hz.

    Grid values:
      0      → silence (skip)
      1–15   → 1-based index into data/samples/{layer}/metadata.json

    Timing:
      One 16th note = 60 / (bpm * 4) seconds
      At 120 BPM → ~31 ms per step, full bar ~0.5 s
      +1 s tail so the last hit can ring out naturally.
    """
    step_dur_s = 60.0 / (bpm * 4)
    step_samples = int(step_dur_s * sr)
    bar_samples = T * step_samples
    total_samples = bar_samples + sr           # bar + 1-second tail

    mix = np.zeros(total_samples, dtype=np.float32)

    for layer_idx, layer_name in enumerate(LAYER_NAMES):
        for step in range(T):
            sample_idx = int(grid[layer_idx, step])
            if sample_idx == 0:
                continue                       # silence — nothing to render

            audio = _load_sample(layer_name, sample_idx, target_sr=sr)
            onset = step * step_samples
            end = min(onset + len(audio), total_samples)
            mix[onset:end] += audio[: end - onset]

    # Peak-normalise to 0.9 FS to prevent clipping
    peak = np.abs(mix).max()
    if peak > 1e-6:
        mix = mix / peak * 0.9

    return mix, bar_samples


def _dummy_reward(grid, final, action_coord):
    """Placeholder reward fn — only used to satisfy BeatGridEnv's constructor."""
    return 0.0


# ── Inference ─────────────────────────────────────────────────────────────────

def generate_beat(
    actor: CNNLayerStepSampleActor,
    env: BeatGridEnv,
    device: str = "cpu",
) -> np.ndarray:
    """
    Run one full episode with the trained actor.
    Returns the completed (L, T) grid.
    """
    obs, _ = env.reset()
    done = False
    with torch.no_grad():
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            action, _ = actor.act(obs_t)
            obs, _, done, _, _ = env.step(int(action))
    return env.grid.copy()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate a beat with the trained RL agent and render to WAV."
    )
    parser.add_argument(
        "--output", default="outputs/beat_sample.wav",
        help="Output WAV path (default: outputs/beat_sample.wav)",
    )
    parser.add_argument(
        "--bpm", type=float, default=120.0,
        help="Tempo in BPM (default: 120)",
    )
    parser.add_argument(
        "--sr", type=int, default=44100,
        help="Output sample rate in Hz (default: 44100)",
    )
    parser.add_argument(
        "--n_beats", type=int, default=4,
        help="Number of times to loop the 1-bar grid (default: 4)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

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
    print(f"Using device: {device}")

    # ── Load actor ───────────────────────────────────────────────────────────
    actor = CNNLayerStepSampleActor(
        L=L, T=T, S=S, env_layer_to_samples=LAYER_TO_SAMPLES
    ).to(device)
    actor.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    actor.eval()
    print(f"Loaded actor from {CHECKPOINT_PATH}")

    # ── Build env (no real reward needed for inference) ───────────────────────
    env = BeatGridEnv(
        L=L, T=T, S=S,
        reward_fn=_dummy_reward,
        layer_to_samples=LAYER_TO_SAMPLES,
        phase=1,
    )

    # ── Generate beat grid ───────────────────────────────────────────────────
    print("\nRunning inference...")
    grid = generate_beat(actor, env, device=device)

    print(f"\nGenerated grid  (rows=layers, cols=16th-note steps):")
    layer_labels = ["kick ", "snare", "hihat", "clap "]
    for i, label in enumerate(layer_labels):
        row_str = " ".join(f"{v:2d}" for v in grid[i])
        print(f"  {label}: [{row_str}]")

    # ── Render one bar, then loop ─────────────────────────────────────────────
    print(f"\nRendering at {args.bpm} BPM, {args.sr} Hz...")
    one_bar_audio, bar_samples = render_grid(grid, bpm=args.bpm, sr=args.sr)

    # Repeat the pure bar section n_beats-1 times, then append the full
    # one_bar_audio (which includes the 1-second tail) at the end.
    pure_bar = one_bar_audio[:bar_samples]
    looped = np.concatenate([pure_bar] * (args.n_beats - 1) + [one_bar_audio])

    # ── Write WAV ────────────────────────────────────────────────────────────
    out_path = (
        REPO_ROOT / args.output
        if not Path(args.output).is_absolute()
        else Path(args.output)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), looped, args.sr)

    duration = len(looped) / args.sr
    print(f"\nSaved to: {out_path}")
    print(f"Duration: {duration:.2f}s  ({args.n_beats} bars at {args.bpm} BPM)")
    print("\nTo listen (macOS):  open " + str(out_path))


if __name__ == "__main__":
    main()
