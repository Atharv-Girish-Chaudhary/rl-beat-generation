# PROGRESS.md

## 1. Project Overview

This project trains a reinforcement learning agent to compose drum beat grids using Proximal Policy Optimization (PPO) and, in Phase 2, a Soft Actor-Critic (SAC) agent for continuous audio effects modulation. A CNN-based discriminator provides learned reward signal by distinguishing agent-generated patterns from real drum data. The system produces playable audio output via a browser-based Streamlit interface, where users can inspect the generated beat grid, listen to rendered audio, and review per-step evaluation metrics.

---

## 2. Phase 1 Status — Complete ✅

### What Was Built

| Component | Description |
|---|---|
| `BeatGridEnv` | 4×16 beat grid environment with rule-based and discriminator rewards |
| PPO Agent | `CNNLayerStepSampleActor` trained with `train_ppo.py` |
| Discriminator | CNN binary classifier trained with `train_discriminator.py` |
| Audio Generation | `generate_audio.py` rendering 4 instruments via `pretty_midi` + `pedalboard` |
| Streamlit App | `app.py` — interactive beat grid viewer with audio playback |

### Final Metrics

| Metric | Value |
|---|---|
| Discriminator Val Accuracy | 95.12% |
| Best PPO Reward | 0.942 |
| EER-equivalent Discriminator Score | ~0.94 (near-perfect separation) |

---

## 3. Phase 2 Status — In Progress 🚧

### What's Done

- Environment expanded to 8×16 (8 instruments, 16 timesteps)
- Data pipeline updated to support 8-instrument MIDI examples
- Reward function v1 (rule-only) and v2 (rule + discriminator blended) implemented
- Discriminator retrained on expanded grid — **97.41% val accuracy**
- PPO training run 1 completed — **best reward: 0.520**
- `evaluate.py` and `evaluate_baseline.py` updated with `--phase` argument
- Streamlit `app.py` updated with Phase 1 / Phase 2 sidebar toggle

### What's In Progress

- PPO retraining with improved reward weights — **job currently running on HPC**

### Known Issues

| Issue | Detail |
|---|---|
| Groove consistency critically low | Agent: 0.267 vs random baseline: 0.881 |
| Discriminator score near zero | Agent patterns not fooling the discriminator |
| Beat density too high | 87% — agent is over-filling the grid |

---

## 4. Phase 2 Remaining Work

- [ ] **Evaluate retrained PPO** — compare groove consistency, discriminator score, and beat density against run 1 results; target reward > 0.80
- [ ] **Tune reward weights further** if retrained run does not meet targets
- [ ] **Build `sac.py`** — SAC actor and critic network definitions for continuous FX modulation (`reverb_mix`, `delay_feedback`, `lpf_cutoff`)
- [ ] **Build `train_sac.py`** — SAC training script with environment integration and checkpoint saving
- [ ] **Update `generate_audio.py`** — add 4 new instruments (Phase 2 expansion) and integrate `pedalboard` DSP effects driven by SAC outputs
- [ ] **Update `evaluate.py` and `evaluate_baseline.py`** — add SAC-specific metrics (FX parameter distributions, audio feature changes)
- [ ] **Update `app.py`** — add SAC FX readout panel showing live parameter values alongside the beat grid

---

## 5. Known Bugs / Tech Debt

| ID | File | Issue | Status |
|---|---|---|---|
| BUG-01 | `train_ppo.py` | Was saving checkpoints with generic names (e.g. `actor_checkpoint.pth`) instead of phase-tagged names | Fixed in code; current HPC run still uses old naming — **manual rename required after job completes** |
| BUG-02 | `evaluate.py` / `evaluate_baseline.py` | Groove consistency metric is computed differently between the two scripts — results are not directly comparable | Open — needs reconciliation |
| BUG-03 | `configs/ppo_phase2.yaml` | References `discriminator_phase2_best.pth` but actual saved file is `discriminator_phase2.pt` | Open — path must be corrected before next training run |

---

## 6. Metrics Tracking

| Metric | Phase 1 | Phase 2 — Run 1 |
|---|---|---|
| Best Reward | 0.942 | 0.520 |
| Discriminator Val Accuracy | 95.12% | 97.41% |
| Rule Reward | — | — |
| Discriminator Score | ~0.94 | ~0.00 |
| Beat Density | — | 87% |
| Groove Consistency | — | 0.267 (random baseline: 0.881) |
