# Phase 2 Diagnostic Audit

## Executive Summary
The Phase 2 PPO agent produces over-dense (87-94%) grids with terrible groove mostly because of an inherent 93.75% density exploration bias in the action space, combined with flatlined reward gradients under severe penalty clipping. At initialization, the agent is algorithmically forced to make a random choice across 16 possible options for every cell, where 15 options are "Hits" and only 1 is "Silence," cementing a near 94% hit rate. Because the density penalties in the Phase 2 melodic evaluation perfectly zero out and clip at this state, and the drum Jaccard similarity accidentally heavily rewards dense random noise, the agent rests in a stagnant local maximum where it exploits basic rule bounds and ignores the zeroed-out discriminator.

## Component-by-Component Findings

**beat_rl/env/beat_env.py**
- **[CRITICAL]** The environment forces exactly one action per cell (`len(self.empty_cells) == 0`). Silence is treated as just 1 equal valid action among 15 hit samples, mathematically guaranteeing ~93.75% density at untrained initialization.
- **[MINOR]** The spatial `obs` tensor expansion technically scales flawlessly from L=4 to L=8. No tensor shape mismatches are present.

**beat_rl/env/reward.py**
- **[CRITICAL]** Melodic density penalties heavily use `max(0.0, ...)` capping (`simul_score`, `pad_density`). In the 94% spam regime, these output perfectly flat `0.0` scores, offering zero gradient to guide the agent back to sparsity.
- **[MAJOR]** The Jaccard similarity index in `_evaluate_drums` awards `+0.4` when `0.6 <= jaccard < 0.95`. High-density random noise reliably has an intersection ratio that falls within this range, actively rewarding spamming behaviors.
- **[MAJOR]** The off-beat penalty is incredibly weak (`-0.01`). The agent happily absorbs a tiny `-0.28` penalty to guarantee the massive `+0.4` Jaccard variation bonus.
- **[MINOR]** `HIHAT = 2` hardcoded indexing is carried over safely due to array bounds checking, but technically violates the decoupled Phase 2 ideology.

**beat_rl/models/actor.py**
- **[CRITICAL]** The purely linear output `self.sample_head = nn.Linear(128, S+1)` ensures logits are un-biased around 0. Without deliberately pre-biasing the `index=0` (Silence) mask, Softmax naturally distributes ~94% probability cleanly across creating "Hits".
- **[MINOR]** Dynamic occupancy masking works correctly and cleanly prevents invalid hit assignments, but cannot save the agent from the uniform priority trap.

**beat_rl/models/critic.py**
- **[MINOR]** Operates exactly as intended. Vectorized reconstruction logic properly bounds and processes the incoming Phase 2 (8x16) `obs` grids.

**scripts/train_ppo.py**
- **[MAJOR]** Discriminator logit scaling isn't robust against the massive noise floor. A 94% dense grid yields a raw probability of `~0.0008`, which contributes virtually `0.0` useful gradient.
- **[MINOR]** Alpha and Beta values (0.7, 0.3) are entirely hardcoded as arguments in the `__main__` entrypoint, rendering configuration overrides useless.

**configs/ppo_phase2.yaml**
- **[MINOR]** Missing tuning hyperparameters for the Alpha / Beta reward distribution.

**outputs/evaluation_report_phase2.json**
- **[CRITICAL]** Demonstrates that the agent explicitly gamed the evaluation rules: while most layers are ~94% dense, `hihat` density sits precisely scaled back to `62.8%` (10 hits). The agent distinctly learned to strategically throttle just the hi-hat to slip between the `4 <= hat_count <= 12` rule, securing the sweet `+0.2` bonus while leaving everything else spammed.

## Root Cause Analysis
The agent produces 87-94% density and 0.267 groove because the environment forces it to visit every cell, and its action space logically represents "Silence" as just 1 of 16 options. A completely untrained neural network inherently defaults to 93.75% hits everywhere. 

Once initialized at this high density state, the agent evaluates a critically broken reward landscape:
1. All Phase 2 Melodic density penalties (like `simul_hits` > 4) are hard-capped at minimum `0.0`. Gradually shrinking wall clusters from 16 active lines down to 14 yields absolutely no observable reward signal, keeping the agent locked on a zero-gradient continuous plateau.
2. The Phase 1 Drum variation rules provide huge `+0.4` Jaccard bonuses for "slight variations". Dense random blocks coincidentally score massive points here just due to raw overlap probability.
3. The flattened discriminator outputs `0.0008` (effectively zero), meaning practically nothing is trying to steer the agent away from its ungodly spam behaviors toward real music.

## Reward Balance Calculation
- **Theoretical Maximum (Perfect Play):**
  - **Intermediate:** +0.25 (Pinging all 5 anchors manually).
  - **Terminal Drums:** 1.0 (Anchors, Hats logic, Positive Jaccard variation).
  - **Terminal Melodics:** 1.0 (Perfect bass lock, isolated sparse melodies, zero simul overlaps).
  - **Rule Multiplier Base:** (1.0 + 1.0) / 2 = 1.0. With `alpha=0.7` -> 0.70.
  - **Discriminator:** ~1.0. With `beta=0.3` -> 0.30.
  - **Max Expected Total:** 0.25 + 0.70 + 0.30 = **1.25**

- **Spamming (Current 90% Uniform Behavior):**
  - **Intermediate:** +0.25 (Accidentally strikes all 5 anchors within the noise wall).
  - **Terminal Drums:** (+0.4 anchors) + (+0.4 Jaccard overlap luck) + (+0.2 by throttling hats just below 13 hits) - (0.26 from tiny off-beat penalties) = **~0.74**
  - **Terminal Melodics:** (+0.25) from absolute perfect Bass Lock due to sheer probability. Everything else evaluates and hardclips out to 0.0 exactly. = **0.25**
  - **Rule Multiplier Base:** (0.74 + 0.25) / 2 = 0.495. With `alpha=0.7` -> 0.346.
  - **Discriminator:** 0.0008. With `beta=0.3` -> 0.000.
  - **Spammed Expected Total:** 0.25 + 0.346 + 0.000 = **~0.596**

Although "Perfect Play" is higher (1.25), "Spamming" represents a dominant and fiercely stuck local maximum (0.60). Moving toward sparse, optimal play actually requires radically dropping density in the immediate short-term, which actively strips the agent of its broken Bass-Lock (+0.25) and Jaccard (+0.4) bonuses long before the clipped density penalties ever un-flatline to offset the loss.

## Proposed Fixes in Priority Order

1. **Bias the Action Space Prior [CRITICAL]**
   - *What to Change:* Inject a substantial constant positive bias (e.g., `+2.5`) directly to the logit for the `index=0` (Silence) node inside `sample_head` in `actor.py`. 
   - *Expected Impact:* Forces an untrained network to lean strongly towards sparse configurations (e.g., <20% density) right out of the box, throwing it immediately into the proper discriminator domain.
   - *Effort Estimate:* Very Low (1 Line of Code).

2. **Remove Hard Clipping from Density Penalties [MAJOR]**
   - *What to Change:* Eliminate the `max(0.0, ...)` bounds on all `pad_density`, `fx_density`, and `simul_score` metrics inside `reward.py`. The RL agent must be allowed to incur severe negative bounds (e.g., `-0.8`) to feel the gradient descent back to sparse composition.
   - *Expected Impact:* Cures the "plateau of zero gradient".
   - *Effort Estimate:* Low.

3. **Gate the Jaccard Noise Exploit [MAJOR]**
   - *What to Change:* Introduce a density prerequisite in `_evaluate_drums` to receive the `+0.4` composition variation bonus. It absolutely cannot be awarded to configurations that exhibit >60% filled cells.
   - *Expected Impact:* Immediately removes the primary incentive allowing spam runs to easily hit high validation scores.
   - *Effort Estimate:* Low.

4. **Expose Alpha/Beta Configuration Parameters [MINOR]**
   - *What to Change:* Add the Alpha and Beta hyperparameters directly to `ppo_phase2.yaml` and read them dynamically via `train_ppo.py`. 
   - *Expected Impact:* Allows for crucial re-scaling experiments involving `beta=50` to force the RL node to bend to the discriminator.
   - *Effort Estimate:* Very Low.

## What NOT to change
- **The Masking Backbones:** The core dynamic execution environments inside `beat_env.py` and structural dependencies inside `actor.py`/`critic.py`. They operate safely without any major tensor dimension issues.
- **Phase 1 Legacy Rulesets:** Legacy indices representing purely Phase 1 layers (`grid[:4, :8]`) operate correctly due to strict, decoupled slice rules.
