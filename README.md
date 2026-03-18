<!-- markdownlint-disable MD013 -->
<!-- pymarkdown: disable MD013 -->

# RL-Based Beat Generation

A two-level reinforcement learning system for automated beat generation. The agent learns to compose structurally coherent drum and melody arrangements by combining hand-crafted musical rules with a learned discriminator reward signal.

## CS 5180: Reinforcement Learning — Spring 2026, Northeastern University

**Team:** Atharv · Taha Ucar · Yixun Li
**Instructor:** Professor Yifan Hu

---

## Overview

This project frames music composition as a sequential decision-making problem. A PPO agent populates a beat grid one cell at a time, guided by a hybrid reward function that balances rule-based musical constraints with a transformer discriminator trained on real drum performances.

### Architecture

```text
┌─────────────────────────────────────────────────────┐
│                    PPO Agent                         │
│  ┌─────────────┐    ┌──────────────┐                │
│  │ CNN Actor    │    │ CNN Critic   │                │
│  │ (3 heads:   │    │ (V(s) →      │                │
│  │  layer,     │    │  scalar)     │                │
│  │  step,      │    └──────────────┘                │
│  │  sample)    │                                     │
│  └──────┬──────┘                                     │
│         │ action: (layer, step, sample)              │
└─────────┼───────────────────────────────────────────┘
          ▼
┌─────────────────────────────────────────────────────┐
│              Beat Grid Environment                   │
│  8×16 grid (L layers × T time steps)                │
│  One-hot encoded state → flattened observation       │
│  Episode: fill all cells → compute reward            │
└─────────┬───────────────────────────────────────────┘
          ▼
┌─────────────────────────────────────────────────────┐
│               Hybrid Reward                          │
│  R = α · R_rules + β · R_discriminator              │
│                                                      │
│  Rule-based:          Discriminator:                 │
│  • Rhythmic structure  • Transformer encoder         │
│  • Density control     • Trained on Groove MIDI      │
│  • Repetition w/       • P(real) as reward signal    │
│    variation                                         │
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
- Freesound API key ([register here](https://freesound.org/apiv2/apply))

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

## Usage

### Step-by-step (run in this exact order)

#### 1. Download sound samples

```bash
# Set your Freesound API key in data/download_samples.py first
python data/download_samples.py
```

Downloads ~15 WAV files per layer × 8 layers = ~120 samples.

#### 2. Download and process Groove MIDI dataset

```bash
wget https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip
unzip groove-v1.0.0-midionly.zip -d data/raw/groove
python data/process_groove.py
```

Produces `groove_grids.npy` — approximately 1,000–1,100 real drum grids.

#### 3. Pre-train the discriminator

```bash
python training/pretrain_disc.py
```

Trains the transformer discriminator on real vs. random grids. **Do not proceed until validation accuracy > 0.75.**

#### 4. Train Phase 1 (drums only)

```bash
python -m training.train_ppo --phase 1
```

Monitor with TensorBoard:

```bash
tensorboard --logdir logs/
```

#### 5. Check Phase 1 readiness

```bash
python evaluation/check_phase1_ready.py
```

Transition when mean rule score > 0.7 over 100 episodes.

#### 6. Train Phase 2 (full grid)

```bash
python -m training.train_ppo --phase 2
```

#### 7. Evaluate

```bash
python evaluation/evaluate.py
python evaluation/render_to_audio.py
```

---

## Technical Details

### MDP Formulation

| Component | Definition |
| --- | --- |
| **State** | Partially-filled beat grid, one-hot encoded: shape (L, T, S+1) |
| **Action** | Tuple (layer, step, sample) — flat integer decoded at inference |
| **Reward** | Terminal: α·R_rules + β·R_disc; Intermediate: small compatibility hint |
| **Episode** | Start empty → fill all L×T cells → terminate |

**Action space:** L × T × (S+1) = up to 2,688 discrete actions per step. Sample selection is masked based on the chosen layer (kick samples can only go on the kick layer).

### Key Hyperparameters (Phase 1)

| Parameter | Value | Purpose |
| --- | --- | --- |
| `n_steps` | 2,048 | Steps per rollout buffer |
| `batch_size` | 64 | PPO mini-batch size |
| `learning_rate` | 3e-4 | Adam optimizer LR |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE bias-variance tradeoff |
| `clip_range` | 0.2 | PPO clipping ε |
| `ent_coef` | 0.01 | Entropy bonus |

### Discriminator

- Transformer encoder (d_model=64, 4 heads, 2 layers)
- Pre-trained on Groove MIDI dataset (binary cross-entropy, label smoothing ε=0.1)
- Updated every 100 episodes during RL training with a historical pool of 500 agent grids
- Three negative example types: random grids, shuffled layers, agent outputs

### Reward Sub-Components

| Sub-Reward | Target | Metric |
| --- | --- | --- |
| Rhythmic structure | Kicks on strong beats, snares on backbeats | Fraction of correct placements |
| Density control | 30–60% of cells active | Linear penalty outside range |
| Repetition w/ variation | Half-bar Jaccard similarity in [0.7, 0.99) | 1.0 in range, linear ramp outside |

---

## Evaluation Metrics

| Metric | Target | Description |
| --- | --- | --- |
| Discriminator score | > 0.70 | P(real) from held-out discriminator |
| Rule-based score | > 0.70 | Composite of rhythmic, density, repetition |
| Novelty (Hamming NN) | > 0.15 | Distance from nearest training example |
| Human listening test | > 3.5/5 | Blind Likert-scale ratings |

---

## Datasets

- **[Groove MIDI Dataset](https://magenta.tensorflow.org/datasets/groove)** — 13.6 hours, ~1,150 MIDI files, 22,000+ measures of professional drumming from 10 drummers. Used to train the discriminator.
- **[Freesound](https://freesound.org/)** — CC0-licensed one-shot WAV samples for the agent's sound library.

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

- Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- Gillick et al., "Learning to Groove with Inverse Sequence Transformations" (2019) — Groove MIDI Dataset
- Haarnoja et al., "Soft Actor-Critic" (2018)
- Stable Baselines3 — [Documentation](https://stable-baselines3.readthedocs.io/)

---

## License

MIT
