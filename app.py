#!/usr/bin/env python3
"""
app.py — Streamlit UI for RL Beat Generation.

Run with:
  streamlit run app.py
"""

import sys
import tempfile
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st
import torch

# ── Repo root on sys.path so beat_rl package is importable ───────────────────
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from beat_rl.env.beat_env import BeatGridEnv
from beat_rl.env.reward import compute_reward
from beat_rl.models.actor import CNNLayerStepSampleActor
from scripts.generate_audio import generate_beat, render_grid

# ── Constants ─────────────────────────────────────────────────────────────────
L, T, S = 4, 16, 15
LAYER_TO_SAMPLES = {i: list(range(1, S + 1)) for i in range(L)}
LAYER_NAMES = ["Kick", "Snare", "Hihat", "Clap"]
CHECKPOINT_PATH = REPO_ROOT / "outputs" / "checkpoints" / "actor_best.pth"


def _dummy_reward(grid, final, action_coord):
    return 0.0


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@st.cache_resource
def load_actor():
    device = _detect_device()
    actor = CNNLayerStepSampleActor(
        L=L, T=T, S=S, env_layer_to_samples=LAYER_TO_SAMPLES
    ).to(device)
    actor.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    actor.eval()
    return actor, device


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="RL Beat Generation", layout="wide")
st.title("RL Beat Generation")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")
    bpm = st.slider("BPM", min_value=60, max_value=180, value=120, step=1)
    seed = st.number_input("Seed", value=42, step=1)
    n_bars = st.slider("N Bars", min_value=1, max_value=8, value=4, step=1)
    clicked = st.button("Generate Beat", use_container_width=True)

# ── Main area ─────────────────────────────────────────────────────────────────
if clicked:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    actor, device = load_actor()

    env = BeatGridEnv(
        L=L, T=T, S=S,
        reward_fn=_dummy_reward,
        layer_to_samples=LAYER_TO_SAMPLES,
        phase=1,
    )

    with st.spinner("Generating beat..."):
        grid = generate_beat(actor, env, device=device)

    # ── Beat grid heatmap ─────────────────────────────────────────────────────
    st.subheader("Beat Grid")

    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_under("white")

    fig, ax = plt.subplots(figsize=(10, 3.5))
    cax = ax.imshow(grid, cmap=cmap, aspect="auto", vmin=0.1, vmax=15)

    ax.set_xticks(np.arange(T))
    ax.set_xticklabels(np.arange(1, T + 1))
    ax.set_yticks(np.arange(L))
    ax.set_yticklabels(LAYER_NAMES)

    ax.set_xticks(np.arange(-0.5, T, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, L, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    for x in [3.5, 7.5, 11.5]:
        ax.axvline(x=x, color="red", linewidth=2, linestyle="--")

    ax.set_xlabel("16th Note Steps")
    cbar = fig.colorbar(cax, ticks=np.arange(1, 16))
    cbar.set_label("Sample ID")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Audio player ──────────────────────────────────────────────────────────
    st.subheader("Audio")

    with st.spinner("Rendering audio..."):
        one_bar, bar_samples = render_grid(grid, bpm=float(bpm), sr=44100)
        pure_bar = one_bar[:bar_samples]
        looped = np.concatenate([pure_bar] * (int(n_bars) - 1) + [one_bar])

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, looped, 44100)
        st.audio(tmp.name)

    # ── Evaluation metrics ────────────────────────────────────────────────────
    st.subheader("Evaluation Metrics")

    rule_reward = compute_reward(grid, final=True, phase=1)

    beat_density = float(np.sum(grid > 0)) / (L * T)

    first_half = grid[:, :8] > 0
    second_half = grid[:, 8:] > 0
    intersection = np.sum(first_half & second_half)
    union = np.sum(first_half | second_half)
    groove_consistency = float(intersection / union) if union > 0 else 0.0

    per_layer_density = [float(np.sum(grid[i] > 0)) / T for i in range(L)]

    col1, col2, col3 = st.columns(3)
    col1.metric("Rule Reward", f"{rule_reward:.3f}")
    col2.metric("Beat Density", f"{beat_density:.1%}")
    col3.metric("Groove Consistency", f"{groove_consistency:.3f}")

    fig2, ax2 = plt.subplots(figsize=(5, 3))
    ax2.bar(LAYER_NAMES, per_layer_density, color=["#4c72b0", "#dd8452", "#55a868", "#c44e52"])
    ax2.set_ylabel("Density (active steps / 16)")
    ax2.set_ylim(0, 1)
    ax2.set_title("Per-Layer Density")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

else:
    st.info("Configure the controls in the sidebar and click **Generate Beat** to start.")
