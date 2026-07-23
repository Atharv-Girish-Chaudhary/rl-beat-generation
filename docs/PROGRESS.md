# PROGRESS.md

## 1. Project Overview

This project trains a reinforcement learning agent to compose drum beat grids using Proximal Policy Optimization (PPO) across two phases: a 4-instrument 4×16 grid (Phase 1) and an 8-instrument 8×16 grid (Phase 2). A transformer-based discriminator (2 encoder blocks, 4 attention heads) provides a learned reward signal by distinguishing agent-generated patterns from real drum data, blended with hand-crafted musical rules (α·rules + β·discriminator). The system produces playable audio output via a browser-based Streamlit interface, where users can inspect the generated beat grid, listen to rendered audio, and review per-step evaluation metrics.

---

## 2. Phase 1 Status — Complete ✅

### What Was Built

| Component | Description |
|---|---|
| `BeatGridEnv` | 4×16 beat grid environment with rule-based and discriminator rewards |
| PPO Agent | `CNNLayerStepSampleActor` trained with `train_ppo.py` |
| Discriminator | Transformer-encoder binary classifier (2 blocks, 4 heads) trained with `train_discriminator.py` |
| Audio Generation | `generate_audio.py` rendering 4 instruments via `pretty_midi` + `pedalboard` |
| Streamlit App | `app.py` — interactive beat grid viewer with audio playback |

### Final Metrics

| Metric | Value |
|---|---|
| Discriminator Val Accuracy | 94.75% (pinned: [`outputs/discriminator_phase1_eval.json`](../outputs/discriminator_phase1_eval.json), seed 7; training-run log printed 95.12%) |
| Best PPO Reward | 0.942 |
| EER-equivalent Discriminator Score | ~0.94 (near-perfect separation) |

---

## 3. Phase 2 Status — Complete ✅ (converged to a documented density-trap optimum)

### What's Done

- Environment expanded to 8×16 (8 instruments, 16 timesteps)
- Data pipeline updated to support 8-instrument MIDI examples
- Reward function v1 (rule-only) and v2 (rule + discriminator blended) implemented
- Discriminator retrained on expanded grid — **97.41% val accuracy**
- PPO training run 1 completed — **best reward: 0.520**
- `evaluate.py` and `evaluate_baseline.py` updated with `--phase` argument
- Streamlit `app.py` updated with Phase 1 / Phase 2 sidebar toggle

### Final Outcome

The retrained Phase 2 run converged: mean episode reward plateaued at ~0.595 (best 0.608)
within ~100 epochs and held flat for the remaining 400 — a stable density-spam local optimum,
not an under-trained run. Scored against the full Phase 2 objective (drums + melodic averaged,
seed 7), the agent reaches **0.5086 rule reward vs 0.3615 for a random baseline**. Root-cause
analysis (action-space density prior, clipped melodic penalties, Jaccard exploit) is in
[`phase2_diagnostic.md`](phase2_diagnostic.md).

### Known Issues (documented, analyzed)

| Issue | Detail |
|---|---|
| Groove consistency critically low | Agent: 0.269 — near the random-noise floor (0.25) |
| Discriminator score near zero | ~0.0008 — saturated; no learned-reward signal in the dense regime |
| Beat density too high | ~86% — agent over-fills the grid (density-trap equilibrium) |

---

## 4. Remaining Work

Phase 2 training, evaluation, and analysis are complete for the project's scope. Research
extensions beyond it — the Phase 2 reward redesign (density-trap fixes) and a possible
Soft Actor-Critic comparison study (no SAC code exists in this repo; `configs/sac.yaml` is
scaffolding only) — are tracked in [`FUTURE_WORK.md`](FUTURE_WORK.md).

---

## 5. Known Bugs / Tech Debt

| ID | File | Issue | Status |
|---|---|---|---|
| BUG-01 | `train_ppo.py` | Was saving checkpoints with generic names (e.g. `actor_best.pth`) instead of phase-tagged names | **Fixed** — training saves phase-tagged names; `evaluate.py` and `generate_audio.py` defaults updated to `actor_phase1_best.pth` |
| BUG-02 | `evaluate.py` / `evaluate_baseline.py` | Groove consistency metric is computed differently between the two scripts — results are not directly comparable | Open (documented) — footnoted in the README results tables |
| BUG-03 | `configs/ppo_phase2.yaml` | References `discriminator_phase2_best.pth` but actual saved file is `discriminator_phase2.pt` | **Fixed** — config paths corrected; configs now note that training is CLI-driven and does not read these YAMLs |

---

## 6. Metrics Tracking

| Metric | Phase 1 (v3) | Phase 2 — Run 1 | Phase 2 — Final |
|---|---|---|---|
| Best Training Reward | 0.942 | 0.520 | 0.608 (plateau ~0.595) |
| Discriminator Val Accuracy | 94.75%¹ | 97.41% | 97.41% |
| Rule Reward (eval, own objective) | 0.9585 | — | 0.5086 (random: 0.3615) |
| Discriminator Score (eval) | 0.0005 | ~0.00 | 0.0008 |
| Beat Density | 0.4508 | 0.87 | 0.8629 |
| Groove Consistency | 0.3971 | 0.267 | 0.2694 |

¹ Pinned by the committed, seeded eval artifact [`outputs/discriminator_phase1_eval.json`](../outputs/discriminator_phase1_eval.json)
(`python evaluation/evaluate_discriminator.py --seed 7`). The original training-run log printed 95.12%.

---

## 7. Detailed Evaluation Results

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

**Random baseline** (20 episodes, `evaluation/evaluate_baseline.py`) — values from the committed
`outputs/random_baseline_report.json`:

| Metric | Random Agent | PPO Agent (v3) |
|--------|-------------|----------------|
| Rule reward | 0.4170 ± 0.2129 | **0.9585 ± 0.1330** |
| Discriminator score | 0.00007 | **0.0005** |
| Beat density | 0.9438 | **0.4508** |
| Groove consistency | 0.8993* | **0.3971** |

The trained agent reaches **0.96 vs 0.42** rule reward — a **+130% relative improvement** — while
producing beats at roughly half the random agent's density.

*\* Baseline groove consistency uses a different formula (half-bar Jaccard) than the agent metric
(strong-beat fraction), so the two values are not directly comparable (BUG-02).*

**Training progression:**

| Run | Config | Best Reward |
|-----|--------|-------------|
| v1 | α=0.9, β=0.1, 250 ep, weak disc | 0.825 |
| v2 | α=0.6, β=0.4, 250 ep, weak disc | 0.779 |
| **v3 (final)** | **α=0.7, β=0.3, 500 ep, disc v2** | **0.942** |

### Phase 2 — 8×16 checkpoint

Evaluated over 20 episodes using `evaluation/evaluate.py --phase 2 --seed 7` with
`actor_phase2_best.pth` and `discriminator_phase2`. Rule reward is scored against the
**full Phase 2 objective** — the average of the drum and melodic rule sets that Phase 2
training actually optimized. (Earlier revisions reported **0.72**, but that was a drums-only
subscore: a metric bug scored Phase 2 grids against the Phase 1 rule set.)

| Metric | Phase 2 PPO | Random baseline | Notes |
|--------|-------------|-----------------|-------|
| Rule reward (Phase 2 objective) | **0.5086 ± 0.0212** | 0.3615 ± 0.0802 | +41% over random (Phase 1 was +130%) |
| Beat density | **0.8629 ± 0.0181** | 0.9379 ± 0.0163 | Density-spam local optimum |
| Groove consistency | **0.2694 ± 0.0104** | — | Near the random-noise floor (0.25)* |
| Discriminator score | **0.0008 ± 0.0003** | 0.0009 ± 0.0002 | Saturated at ~0 — no learned-reward signal |

*\* Baseline groove consistency is omitted: `evaluate_baseline.py` computes this metric with a
different formula (half-bar Jaccard), so the two columns are not comparable (BUG-02).*

**Phase 2 per-layer density:** Kick 0.991, Snare 0.847, HiHat **0.584** (learned throttle for the
hat-count rule), Clap 0.797, Bass 0.919, Melody 0.928, Pad 0.938, FX 0.900.

The melodic half of the Phase 2 objective sits on a zero-variance plateau (0.2501 for both the
agent and the random baseline — i.e. no learning signal), so training flatlined at the equilibrium
predicted in [`phase2_diagnostic.md`](phase2_diagnostic.md).

---

## 8. Architecture Details

**Observation.** Flat float32 vector of shape (L×T×(S+2),): channels 0–14 one-hot sample index
per cell, channel 15 silence flag, channel 16 `step_count / max_steps` (temporal progress).
Reshaped to (S+2, L, T) for CNN processing.

**Grid encoding.** `(L, 16)` int64 — −1 = empty, 0 = silence, 1–15 = sample index. An episode
fills all L×16 cells, then terminates. Action is a flat int decoded to (layer, step, sample).

**Actor — `CNNLayerStepSampleActor`.** Conv2d(S+2, 32) → ReLU → Conv2d(32, 64) → ReLU →
FC(64·L·T, 128) base, then three autoregressive heads: layer_head → L logits; step_head → T
logits (+ layer embedding); sample_head → S+1 logits (+ step embedding). Dynamic masking blocks
occupied cells and wrong-instrument samples per layer.

**Critic — `CNNBeatCritic`.** Conv2d(S+2, 32) → ReLU → Conv2d(32, 64) → ReLU → FC(64·L·T, 128)
→ ReLU → FC(128, 1) → V(s).

**Hybrid reward.** R = α·R_rules + β·R_disc with α = 0.7, β = 0.3 (v3 weights).
R_rules (terminal, [0,1]):

- +0.1 kick on step 0
- +0.1 kick on step 8
- +0.1 snare/clap on step 4
- +0.1 snare/clap on step 12
- +0.2 hi-hat count in [4, 12]
- +0.4 Jaccard similarity (first vs second half) in [0.6, 0.95)
- −0.01 per off-beat snare/clap hit

R_disc (terminal, [0,1]): sigmoid(BeatDiscriminator(binary_grid)).
R_intermediate: +0.05 for anchor/backbeat hits as they are placed.

**Discriminator — `BeatDiscriminator`.** Input (B, L, T) binary hit grid; token_embed
Linear(L, 64) + pos_embed Embedding(T, 64); 2× EncoderBlock (MultiHeadAttention with 4 heads +
LayerNorm + FFN); classifier Linear(64, 32) → ReLU → Linear(32, 1) → logit. Pre-trained on
Groove MIDI (real) vs synthetic negatives (fake).

---

## 9. Project Structure

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
│   ├── evaluate_baseline.py              # Random agent baseline (--phase 1|2)
│   └── evaluate_discriminator.py         # Discriminator val accuracy (evaluation-only, seeded)
│
├── configs/
│   ├── ppo_phase1.yaml                   # Phase 1 PPO hyperparameters (4×16)
│   ├── ppo_phase2.yaml                   # Phase 2 PPO hyperparameters (8×16)
│   ├── sac.yaml                          # SAC hyperparameters (reserved for future use)
│   └── discriminator.yaml                # Discriminator training config
│
├── hpc/                                  # HPC cluster scripts (SLURM / SSH)
│   ├── env.sh.example                    # Template — copy to env.sh and fill values
│   ├── env.sh                            # Local only — gitignored
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
│   ├── checkpoints/                      # Model weights (gitignored — download from the v1.0-checkpoints Release)
│   │   ├── actor_phase1_best.pth         # Best Phase 1 actor (v3, epoch 486)
│   │   ├── critic_phase1_best.pth
│   │   ├── discriminator_phase1_v2.pt    # Phase 1 discriminator (94.75% val acc — see outputs/discriminator_phase1_eval.json)
│   │   ├── actor_phase2_best.pth
│   │   ├── critic_phase2_best.pth
│   │   └── discriminator_phase2.pt
│   ├── plots/
│   │   ├── first_vs_best_comparison_phase2.png  # Epoch 0 vs best epoch grid (per-phase filenames)
│   │   ├── ppo_training_plot_phase2.png  # Reward / actor loss / critic loss curves (per-phase filenames)
│   │   ├── discriminator_training_plot.png
│   │   ├── baseline_comparison.png       # PPO vs random bar chart (Phase 1)
│   │   └── baseline_comparison_phase2.png
│   ├── beat_sample.wav                   # Committed demo render (Phase 1)
│   ├── discriminator_phase1_eval.json    # Pinned discriminator val accuracy (94.75%, seed 7)
│   ├── evaluation_report.json            # Phase 1 PPO eval (20 episodes)
│   ├── evaluation_report_phase2.json     # Phase 2 PPO eval (20 episodes, seed 7)
│   ├── random_baseline_report.json       # Phase 1 random baseline
│   └── random_baseline_report_phase2.json
│
├── tests/                                # 17 unit + integration tests
│
├── docs/
│   ├── beat_report.tex                   # LaTeX project report
│   ├── phase2_diagnostic.md              # Root-cause analysis of Phase 2 density issue
│   ├── FUTURE_WORK.md                    # Tracked improvements / research extensions
│   ├── img/                              # README screenshots
│   └── PROGRESS.md                       # This file
│
├── requirements.txt
├── Makefile
└── setup.py
```

---

## 10. HPC Workflow

All HPC scripts read three environment variables so no credentials are hard-coded in the repo.

| Variable | Required | Description |
|---|---|---|
| `HPC_USER` | Yes | Your cluster username |
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

`hpc/env.sh` is listed in `.gitignore` and will never be committed.

---

## 11. Retraining & Additional Usage

**Retrain from scratch:**

```bash
source .venv/bin/activate
python scripts/train_ppo.py --phase 1
# Saves actor_phase1_best.pth, critic_phase1_best.pth to outputs/checkpoints/

python scripts/train_ppo.py --phase 2
# Saves actor_phase2_best.pth, critic_phase2_best.pth to outputs/checkpoints/
```

**Generate a beat:**

```bash
python scripts/generate_audio.py --seed 42 --bpm 120 --n_beats 4
# Writes outputs/beat_sample.wav
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

**Colab training** (recommended for GPU access): open `notebooks/train_ppo_colab.ipynb`, set
runtime to T4 GPU. All config is in Cell 3.

**Streamlit app controls:**

| Control | Description |
|---------|-------------|
| Phase | Switch between Phase 1 (4×16) and Phase 2 (8×16) |
| BPM | Tempo (60–180) |
| Seed | Reproducible beat generation |
| N Bars | Loop length (1–8) |

---

## 12. References & Citation

- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (2017)
- Haarnoja et al., [Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL](https://arxiv.org/abs/1801.01290) (2018)
- Gillick et al., [Learning to Groove with Inverse Sequence Transformations](https://arxiv.org/abs/1905.06118) (2019) — Groove MIDI Dataset
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017)

If you reference this work:

```bibtex
@misc{chaudhary2026beatrl,
  author = {Chaudhary, Atharv and Ucar, Taha and Li, Yixun},
  title  = {RL Beat Generation: PPO with Hybrid Rule-Discriminator Reward},
  year   = {2026},
  note   = {CS 5180 Reinforcement Learning, Northeastern University},
  url    = {https://github.com/Atharv-Girish-Chaudhary/rl-beat-generation}
}
```
