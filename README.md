# RL-Based Beat Generation

A two-level reinforcement learning system for automated beat generation. The agent learns to compose structurally coherent drum and melody arrangements by combining hand-crafted musical rules with a learned discriminator reward signal.

**Course:** CS 5180: Reinforcement Learning — Spring 2026, Northeastern University

**Team:** Atharv · Taha Ucar · Yixun Li
**Instructor:** Professor Yifan Hu

---

## Overview

This project frames music composition as a sequential decision-making problem. A PPO agent populates a beat grid one cell at a time, guided by a hybrid reward function that balances rule-based musical constraints with a transformer discriminator trained on real drum performances.

### Architecture

```text
┌─────────────────────────────────────────────────────┐
│                    PPO Agent                        │
│  ┌─────────────┐    ┌──────────────┐                │
│  │ CNN Actor   │    │ CNN Critic   │                │
│  │ (3 heads:   │    │ (V(s) →      │                │
│  │  layer,     │    │  scalar)     │                │
│  │  step,      │    └──────────────┘                │
│  │  sample)    │                                    │
│  └──────┬──────┘                                    │
│         │ action: (layer, step, sample)             │
└─────────┼───────────────────────────────────────────┘
          ▼
┌─────────────────────────────────────────────────────┐
│              Beat Grid Environment                  │
│  8×16 grid (L layers × T time steps)                │
│  One-hot encoded state → flattened observation      │
│  Episode: fill all cells → compute reward           │
└─────────┬───────────────────────────────────────────┘
          ▼
┌─────────────────────────────────────────────────────┐
│               Hybrid Reward                         │
│  R = α · R_rules + β · R_discriminator              │
│                                                     │
│  Rule-based:          Discriminator:                │
│  • Rhythmic structure  • Transformer encoder        │
│  • Density control     • Trained on Groove MIDI     │
│  • Repetition w/       • P(real) as reward signal   │
│    variation                                        │
└─────────────────────────────────────────────────────┘
```

### What the Agent Learns

The agent builds a beat grid step by step — like a producer filling in a drum machine. Each row is an instrument (kick, snare, hi-hat, etc.), each column is a 16th-note time step in one bar. At every step, the agent picks one cell to fill and which sound to place there.

A successful beat has:

- Kicks on strong beats (steps 0, 4, 8, 12)
- Snares on backbeats (steps 4, 12)
- Hi-hats filling in-between steps
- Appropriate density (30–60% of cells active)
- Repetition with variation between half-bars (Jaccard similarity 0.7–0.99)

---

## Curriculum Learning

Training uses a two-phase curriculum to manage complexity:

| | Phase 1 | Phase 2 |
| --- | --- | --- |
| **Grid** | 4×16 (drums only) | 8×16 (drums + bass, melody, pad, fx) |
| **Layers** | Kick, Snare, Hi-hat, Clap | + Bass, Melody, Pad, FX |
| **Reward weights** | α=0.9, β=0.1 (mostly rules) | α=0.5, β=0.5 (balanced) |
| **Transition** | — | Rolling mean rule score > 0.7 |
| **Timesteps** | 2M | 5M |

Phase 2 inherits Phase 1 weights with extra layer channels initialized to zero. 30% Phase 1 grids are included in discriminator batches to prevent catastrophic forgetting.

---

## Project Structure

```text
beat_gen/
├── data/
│   ├── download_samples.py          # Freesound API sample downloader
│   ├── process_groove.py            # Groove MIDI → grid converter
│   ├── raw/                         # Downloaded MIDI files
│   ├── processed/
│   │   └── groove_grids.npy         # (N, 4, 16) drum grids
│   └── samples/
│       ├── manifest.json            # Sample file → layer+index mapping
│       ├── kick/
│       ├── snare/
│       ├── hihat/
│       ├── clap/
│       ├── bass/
│       ├── melody/
│       ├── pad/
│       └── fx/
│
├── env/
│   ├── beat_env.py                  # Gymnasium environment
│   └── reward.py                    # Rule-based + discriminator rewards
│
├── models/
│   ├── actor.py                     # CNN policy network (3-head output)
│   ├── critic.py                    # CNN value network
│   └── discriminator.py             # Transformer encoder discriminator
│
├── training/
│   ├── pretrain_disc.py             # Pre-train discriminator on Groove MIDI
│   └── train_ppo.py                 # Main PPO training loop (SB3)
│
├── evaluation/
│   ├── evaluate.py                  # Quantitative metrics
│   ├── render_to_audio.py           # Grid → WAV audio rendering
│   └── check_phase1_ready.py        # Phase transition check
│
├── checkpoints/                     # Saved model weights
│   ├── discriminator_pretrained.pt
│   ├── phase1_best/
│   └── phase2_best/
│
├── logs/                            # TensorBoard logs
│   ├── phase1/
│   └── phase2/
│
└── configs/
    └── config.yaml                  # Hyperparameters
```

---

## Setup

### Requirements

- Python 3.10
- CUDA-compatible GPU (8+ GB VRAM recommended)
- Freesound API key

### Installation

```bash
# Create virtual environment
python3.10 -m venv beat_env
source beat_env/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install gymnasium numpy pretty_midi requests tqdm matplotlib tensorboard
pip install stable-baselines3[extra]

# Verify GPU access
python -c "import torch; print(torch.cuda.is_available())"

```

---

## Team Responsibilities

| Member | Area |
| --- | --- |
| **Atharv** | RL training pipeline, discriminator |
| **Taha** | Environment, reward system |
| **Yixun** | Data pipeline, audio processing |

---

## Compute

- **Google Colab Pro** — Primary GPU training environment
- **Northeastern Explorer HPC** — NVIDIA H200 GPUs for longer training runs

---

## References

---

## License

MIT
