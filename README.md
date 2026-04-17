# RL Beat Generation

> A PPO agent that composes drum beats by filling an instrument × time grid cell-by-cell, guided by
> a hybrid reward of hand-crafted musical rules and a transformer discriminator trained on real
> performances.

**CS 5180 Reinforcement Learning · Northeastern University · Spring 2026**  
Atharv Chaudhary · Taha Ucar · Yixun Li

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/Tests-17%20passing-brightgreen)

---

## Demo

![First vs Best epoch comparison](outputs/plots/first_vs_best_comparison.png)

*Left: agent at epoch 0. Right: best Phase 1 checkpoint (epoch 486, reward 0.942). Blue = active
cell, number = sample ID chosen.*

### Streamlit App

Run the interactive demo locally:

```bash
conda activate rl-beats
pip install streamlit
streamlit run app.py
```

Opens at `localhost:8501`. Controls:

| Control | Description |
|---------|-------------|
| **Phase** | Switch between Phase 1 (4×16) and Phase 2 (8×16) |
| **BPM** | Tempo (60–180) |
| **Seed** | Reproducible beat generation |
| **N Bars** | Loop length (1–8) |

Click **Generate Beat** to produce a grid, hear the audio, and view evaluation metrics live.

---

## Architecture

### Agent & Environment

The system frames beat composition as a sequential MDP over an L×16 grid (L instrument layers × 16
16th-note time steps). The agent fills one cell per step until all L×16 cells are assigned.

```
┌─────────────────────────────────────────────────────────────────────┐
│                          PPO Agent                                   │
│                                                                       │
│   Observation: (L×T×(S+2),) float32                                  │
│     • Channels 0–14: one-hot sample index per cell                   │
│     • Channel 15:    silence flag                                     │
│     • Channel 16:    step_count / max_steps  (temporal progress)     │
│                                                                       │
│   ┌──────────────────────────────┐   ┌───────────────────────────┐   │
│   │   CNNLayerStepSampleActor    │   │       CNNBeatCritic        │   │
│   │                              │   │                             │   │
│   │  Conv2d(17,32) → ReLU        │   │  Conv2d(17,32) → ReLU      │   │
│   │  Conv2d(32,64) → ReLU        │   │  Conv2d(32,64) → ReLU      │   │
│   │  FC(64·L·T, 128) base        │   │  FC(64·L·T, 128) → ReLU    │   │
│   │                              │   │  FC(128, 1)  →  V(s)        │   │
│   │  ① layer_head  → L logits   │   └───────────────────────────┘   │
│   │  ② step_head   → T logits   │                                    │
│   │     (+ layer embedding)      │                                    │
│   │  ③ sample_head → S+1 logits │                                    │
│   │     (+ step embedding)       │                                    │
│   │                              │                                    │
│   │  Dynamic masking:            │                                    │
│   │  • Occupied cells blocked    │                                    │
│   │  • Wrong-instrument samples  │                                    │
│   │    blocked per layer         │                                    │
│   └──────────────────────────────┘                                   │
└─────────────────────────────────────────────────────────────────────┘
                              │ action: flat int → (layer, step, sample)
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│               BeatGridEnv (Phase 1: 4×16 / Phase 2: 8×16)           │
│  Grid: (L, 16) int64   −1=empty, 0=silence, 1–15=sample index       │
│  Episode: fill all L×16 cells, then terminate                        │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Hybrid Reward                                 │
│                                                                       │
│   R = α · R_rules  +  β · R_disc                                     │
│       (0.7)              (0.3)    ← Phase 1 weights (v3)             │
│                                                                       │
│   R_rules  (terminal, [0,1]):                                        │
│     +0.1  kick on step 0                                             │
│     +0.1  kick on step 8                                             │
│     +0.1  snare/clap on step 4                                       │
│     +0.1  snare/clap on step 12                                      │
│     +0.2  hi-hat count in [4, 12]                                    │
│     +0.4  Jaccard similarity (first vs second half) in [0.6, 0.95)  │
│     −0.01 per off-beat snare/clap hit                                │
│                                                                       │
│   R_disc  (terminal, [0,1]):                                         │
│     sigmoid( BeatDiscriminator(binary_grid) )                        │
│                                                                       │
│   R_intermediate:  +0.05 for anchor/backbeat hits as they are placed │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       BeatDiscriminator                               │
│  Input: (B, L, T) binary hit grid                                    │
│  token_embed: Linear(L, 64)   +   pos_embed: Embedding(T, 64)       │
│  2× EncoderBlock (MultiHeadAttention(4 heads) + LayerNorm + FFN)    │
│  Classifier: Linear(64, 32) → ReLU → Linear(32, 1) → logit          │
│  Pre-trained on Groove MIDI (real) vs synthetic negatives (fake)     │
└─────────────────────────────────────────────────────────────────────┘
```

**Action space factoring.** Instead of sampling from L·T·(S+1) actions flat, the actor decomposes
each decision into three sequential steps — layer → step → sample — reducing effective branching
and letting the architecture encode instrument hierarchy explicitly.

---

## Current Status

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 1** | ✅ Complete | PPO agent on 4×16 grid, audio generation, Streamlit demo |
| **Phase 2** | ✅ Complete | 8×16 grid expansion (Bass, Melody, Pad, FX layers added) |

---

## Results

### Phase 1 — v3 checkpoint

Evaluated over 20 episodes using `evaluation/evaluate.py` with `actor_phase1_best.pth` (v3) and
`discriminator_phase1_v2`.

| Metric | Value | Notes |
|--------|-------|-------|
| Best training reward | **0.942** | Epoch 486 / 500 |
| Rule reward (mean ± std) | **0.9585 ± 0.1330** | Across 20 eval episodes |
| Beat density | **0.4508 ± 0.0173** | Fraction of non-silent cells |
| Groove consistency | **0.3971 ± 0.0332** | Hits on strong beats (steps 0,4,8,12) / total hits |
| Discriminator score | **0.0005 ± 0.0003** | sigmoid(discriminator logit) |

**Per-layer density (Phase 1):**

| Instrument | Density | Interpretation |
|------------|---------|----------------|
| Kick | **0.9969 ± 0.0136** | Dense anchor — correct |
| Snare | **0.1844 ± 0.0503** | Learned restraint (was 0.97 in v1) |
| HiHat | **0.4844 ± 0.0762** | Balanced |
| Clap | **0.1375 ± 0.0250** | Very sparse — musically realistic |

**Random baseline** (20 episodes, `evaluation/evaluate_baseline.py`):

| Metric | Random Agent | PPO Agent (v3) |
|--------|-------------|----------------|
| Rule reward | ~0.10 | **0.9585** |
| Discriminator score | ~0.50 | **0.0005** |
| Beat density | ~0.50 | **0.4508** |
| Groove consistency | ~0.25 | **0.3971** |

**Training progression:**

| Run | Config | Best Reward |
|-----|--------|-------------|
| v1 | α=0.9, β=0.1, 250 ep, weak disc | 0.825 |
| v2 | α=0.6, β=0.4, 250 ep, weak disc | 0.779 |
| **v3 (final)** | **α=0.7, β=0.3, 500 ep, disc v2** | **0.942** |

### Phase 2 — 8×16 checkpoint

Evaluated over 20 episodes using `evaluation/evaluate.py --phase 2` with `actor_phase2_best.pth`
and `discriminator_phase2`.

| Metric | Phase 1 (v3) | Phase 2 | Notes |
|--------|-------------|---------|-------|
| Rule reward (mean ± std) | 0.9585 ± 0.1330 | **0.7215 ± 0.0781** | Phase 2 lower due to density spam |
| Beat density | 0.4508 ± 0.0173 | **0.8871 ± 0.0275** | Agent exploits density for reward |
| Groove consistency | 0.3971 ± 0.0332 | **0.2660 ± 0.0112** | Near random-noise floor (0.25) |
| Discriminator score | 0.0005 ± 0.0003 | **0.0008 ± 0.0002** | Effectively zero — disc not influencing agent |

**Phase 2 per-layer density:** Kick 0.978, Snare 0.903, HiHat **0.628** (learned throttle for hat
rule), Clap 0.834, Bass 0.922, Melody 0.944, Pad 0.972, FX 0.916.

> ⚠️ **Known issue:** The Phase 2 agent is stuck in a density-spam local optimum.
> See `docs/phase2_diagnostic.md` for root-cause analysis and proposed fixes.

---

## Project Structure

```
rl-beat-generation/
├── app.py                                # Streamlit interactive demo (Phase 1 + 2)
│
├── beat_rl/                              # Installable package
│   ├── env/
│   │   ├── beat_env.py                   # BeatGridEnv (Gymnasium, Phase 1 & 2)
│   │   ├── reward.py                     # compute_reward() — rules + discriminator
│   │   └── visualize_env.py              # Matplotlib grid heatmap
│   └── models/
│       ├── actor.py                      # CNNLayerStepSampleActor (3-head autoregressive)
│       ├── critic.py                     # CNNBeatCritic (V(s) → scalar)
│       └── discriminator.py              # BeatDiscriminator (transformer encoder)
│
├── scripts/
│   ├── train_ppo.py                      # PPO training loop (Phase 1 & 2 via --phase flag)
│   ├── train_discriminator.py            # Discriminator pre-training (Phase 1 & 2 via --phase flag)
│   ├── process_groove.py                 # Groove MIDI → (N, L, T) binary grids
│   ├── download_samples.py               # Freesound API sample downloader
│   └── generate_audio.py                 # Actor inference → WAV rendering
│
├── evaluation/
│   ├── evaluate.py                       # N-episode eval: rule reward, density, groove (--phase 1|2)
│   └── evaluate_baseline.py              # Random agent baseline (--phase 1|2)
│
├── configs/
│   ├── ppo_phase1.yaml                   # Phase 1 PPO hyperparameters (4×16)
│   ├── ppo_phase2.yaml                   # Phase 2 PPO hyperparameters (8×16)
│   ├── sac.yaml                          # SAC hyperparameters (reserved for future use)
│   └── discriminator.yaml                # Discriminator training config
│
├── hpc/                                  # HPC cluster scripts (SLURM / SSH)
│   ├── env.sh.example                    # Template — copy to env.sh and fill values
│   ├── env.sh                            # ⚠ Local only — gitignored
│   ├── submit_jobs.sh                    # Submit Phase 1 jobs
│   ├── submit_jobs_phase2.sh             # Submit Phase 2 jobs
│   ├── cancel_jobs.sh                    # Cancel running jobs
│   ├── setup_env.sh                      # Remote env setup
│   ├── sync_to_hpc.sh                    # Push code to cluster
│   ├── sync_from_hpc.sh                  # Pull outputs from cluster
│   ├── train_ppo.sbatch                  # SLURM batch script — PPO Phase 1
│   ├── train_ppo_phase2.sbatch           # SLURM batch script — PPO Phase 2
│   ├── train_discriminator.sbatch        # SLURM batch script — Discriminator Phase 1
│   └── train_discriminator_phase2.sbatch # SLURM batch script — Discriminator Phase 2
│
├── notebooks/
│   ├── train_ppo_colab.ipynb             # Colab PPO training notebook (T4/A100)
│   ├── train_discriminator_colab.ipynb   # Colab discriminator training notebook
│   ├── discriminator_model.ipynb         # Discriminator architecture exploration
│   └── discriminator_notes.ipynb         # Discriminator research notes
│
├── data/
│   ├── processed/groove_grids.npy        # Pre-processed Groove MIDI grids (gitignored)
│   ├── raw/groove/                       # Groove MIDI dataset (gitignored)
│   └── samples/                          # Freesound WAV samples (gitignored)
│       ├── kick/   snare/  hihat/  clap/
│       ├── bass/   melody/ pad/    fx/
│       └── {layer}/metadata.json         # ID → filename mapping
│
├── outputs/
│   ├── checkpoints/                      # Model weights (gitignored)
│   │   ├── actor_phase1_best.pth         # Best Phase 1 actor (v3, epoch 486)
│   │   ├── critic_phase1_best.pth
│   │   ├── discriminator_phase1_v2.pt    # Phase 1 discriminator (95.12% val acc)
│   │   ├── actor_phase2_best.pth
│   │   ├── critic_phase2_best.pth
│   │   └── discriminator_phase2.pt
│   ├── plots/
│   │   ├── first_vs_best_comparison.png  # Epoch 0 vs best checkpoint grid
│   │   ├── ppo_training_plot.png         # Reward / actor loss / critic loss curves
│   │   ├── discriminator_training_plot.png
│   │   ├── baseline_comparison.png       # PPO vs random bar chart (Phase 1)
│   │   └── baseline_comparison_phase2.png
│   ├── evaluation_report.json            # Phase 1 PPO eval (20 episodes)
│   ├── evaluation_report_phase2.json     # Phase 2 PPO eval (20 episodes)
│   ├── random_baseline_report.json       # Phase 1 random baseline
│   └── random_baseline_report_phase2.json
│
├── tests/                                # 17 unit + integration tests
│   ├── conftest.py
│   ├── test_actor.py
│   ├── test_beat_env.py
│   ├── test_critic.py
│   ├── test_discriminator.py
│   ├── test_download_samples.py
│   ├── test_integration.py
│   ├── test_process_groove.py
│   ├── test_reward.py
│   └── test_reward_sanity.py
│
├── docs/
│   ├── beat_report.tex                   # LaTeX project report
│   ├── beat_report.pdf                   # Compiled PDF (gitignored build artifacts)
│   ├── phase2_diagnostic.md              # Root-cause analysis of Phase 2 density issue
│   └── PROGRESS.md                       # Development status tracker
│
├── configs/
├── environment.yml
├── requirements.txt
├── Makefile
└── setup.py
```

---

## HPC Setup

All HPC scripts read three environment variables so no credentials are hard-coded in the repo.

| Variable | Required | Description |
|---|---|---|
| `HPC_USER` | ✅ Yes | Your cluster username |
| `HPC_REMOTE` | No (default: `explorer`) | SSH alias / hostname for the cluster |
| `HPC_SCRATCH` | No (default: `/scratch/${HPC_USER}`) | Path to scratch directory on the cluster |

**One-time setup:**

```bash
# 1. Copy the example and fill in your values
cp hpc/env.sh.example hpc/env.sh
# Edit hpc/env.sh: set HPC_USER (and optionally HPC_REMOTE / HPC_SCRATCH)

# 2. Source before using any HPC make targets
source hpc/env.sh

# 3. Use the Makefile as normal
make hpc-sync        # push code to cluster
make hpc-submit      # submit Phase 1 training jobs
make hpc-submit-p2   # submit Phase 2 training jobs
make hpc-status      # check job queue
make hpc-pull        # pull outputs back
```

> `hpc/env.sh` is listed in `.gitignore` and will never be committed.

---

## Setup

**Requirements:** Python 3.10, CUDA GPU (recommended), Freesound API key for downloading new samples.

```bash
git clone https://github.com/Atharv-Girish-Chaudhary/rl-beat-generation.git
cd rl-beat-generation

conda create -n rl-beats python=3.10
conda activate rl-beats

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -e .
```

Checkpoints and processed data are gitignored — run training scripts or pull from HPC to populate
`outputs/checkpoints/`.

---

## Usage

**Run the Streamlit app:**

```bash
conda activate rl-beats
streamlit run app.py
# Opens at localhost:8501 — use the Phase selector to switch between 4×16 and 8×16
```

**Retrain from scratch (Phase 1):**

```bash
conda activate rl-beats
python scripts/train_ppo.py --phase 1
# Saves actor_phase1_best.pth, critic_phase1_best.pth to outputs/checkpoints/
```

**Retrain from scratch (Phase 2):**

```bash
python scripts/train_ppo.py --phase 2
# Saves actor_phase2_best.pth, critic_phase2_best.pth to outputs/checkpoints/
```

**Generate a beat:**

```bash
python scripts/generate_audio.py --seed 42 --bpm 120 --n_beats 4
# Writes outputs/beat_sample.wav
```

**Evaluate checkpoints:**

```bash
# Phase 1
python evaluation/evaluate.py --n_episodes 20 --phase 1
python evaluation/evaluate_baseline.py --n_episodes 20 --phase 1

# Phase 2
python evaluation/evaluate.py --n_episodes 20 --phase 2
python evaluation/evaluate_baseline.py --n_episodes 20 --phase 2
```

**Retrain the discriminator** (requires Groove MIDI dataset):

```bash
wget https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip
unzip groove-v1.0.0-midionly.zip -d data/raw/groove
python scripts/process_groove.py              # → data/processed/groove_grids.npy
python scripts/train_discriminator.py --phase 1  # → discriminator_phase1_v2.pt
python scripts/train_discriminator.py --phase 2  # → discriminator_phase2.pt
```

**Compile the PDF report** (requires MacTeX / TeX Live):

```bash
# Output PDF and auxiliary files go into docs/
pdflatex -output-directory=docs docs/beat_report.tex
pdflatex -output-directory=docs docs/beat_report.tex  # run twice for cross-references
```

**Run tests:**

```bash
pytest tests/ -v
# 17 unit + integration tests should all pass
```

**Colab training** (recommended for GPU access):

Open `notebooks/train_ppo_colab.ipynb`. Set runtime to T4 GPU. All config is in Cell 3.

---

## Team

| Member | Contribution |
|--------|-------------|
| **Atharv Chaudhary** | PPO training loop, discriminator architecture and pre-training, Phase 2 expansion |
| **Taha Ucar** | Gymnasium environment, reward function, action masking |
| **Yixun Li** | Data pipeline (Groove MIDI, Freesound), audio rendering |

---

## References

- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (2017)
- Haarnoja et al., [Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL](https://arxiv.org/abs/1801.01290) (2018)
- Gillick et al., [Learning to Groove with Inverse Sequence Transformations](https://arxiv.org/abs/1905.06118) (2019) — Groove MIDI Dataset
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017)

---

## License

MIT
