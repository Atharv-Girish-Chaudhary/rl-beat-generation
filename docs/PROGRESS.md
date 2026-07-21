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
| Discriminator Val Accuracy | 95.12% |
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
| Discriminator Val Accuracy | 95.12% | 97.41% | 97.41% |
| Rule Reward (eval, own objective) | 0.9585 | — | 0.5086 (random: 0.3615) |
| Discriminator Score (eval) | 0.0005 | ~0.00 | 0.0008 |
| Beat Density | 0.4508 | 0.87 | 0.8629 |
| Groove Consistency | 0.3971 | 0.267 | 0.2694 |
