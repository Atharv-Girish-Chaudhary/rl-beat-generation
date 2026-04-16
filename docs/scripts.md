# RL Beat Generation: Presentation Script

**Total estimated delivery:** ~12–13 minutes at a natural academic pace.

---

## Slide 1 — Cover
⏱ **Timing:** ~30 seconds

> Good [morning/afternoon], everyone. My project is called **RL Beat Generation** — a PPO reinforcement learning agent that autonomously composes structured drum and melodic beats on a discrete 8-by-16 grid. 
>
> The visual you see on the title slide is that grid: eight instrument layers across sixteen sixteenth-note time steps. 
> 
> The question I set out to answer is: **can an RL agent learn musical structure** — not just fill cells randomly — given the right reward signal? The short answer is yes, and I'm going to walk you through exactly how.

*(Click to Slide 2)*

---

## Slide 2 — Introduction
⏱ **Timing:** ~60–75 seconds

> Let me orient you to the overall system before we go deep on any one component.
>
> **The Agent:** The agent is a **PPO policy** whose job is to fill that 8-by-16 grid one cell at a time. At each decision step, it chooses three things: which instrument layer to place on, which time step, and which audio sample ID to use. That's a factored, autoregressive action — I'll come back to that when we talk about the actor network.
>
> **The Reward Signal:** The reward signal is what makes this interesting. The terminal reward is a weighted sum: **alpha times a rule-based score, plus beta times a discriminator score**. 
> - The **rule-based component** encodes hand-crafted music theory — backbeat placement, hi-hat density, Jaccard similarity between bar halves. 
> - The **discriminator component** is a pre-trained transformer classifier that distinguishes real Groove MIDI beats from synthetic ones, giving the agent a learned, data-driven signal on top of the hard rules.
>
> **Training Curriculum:** Training is split into two curriculum phases. 
> - **Phase 1** is a 4-drum, 4-by-16 grid — just kick, snare, hi-hat, clap. 
> - **Phase 2** opens up to the full 8-layer, 8-by-16 grid and activates a melodic reward branch. Separate discriminators are trained per phase to prevent checkpoint collision.
>
> The execution pipeline has eight scripts — from sample download and MIDI processing, through discriminator training, to the PPO training loop itself. Let me walk through each major piece.

*(Click to Slide 3)*

---

## Slide 3 — Data Pipeline
⏱ **Timing:** ~60 seconds

> The data pipeline has two distinct jobs:
>
> 1. `download_samples.py` handles audio asset acquisition. It queries the Freesound API for eight categories — kick, snare, hi-hat, clap, bass, melody, pad, and FX — matching exactly the eight grid rows. A manifest system makes it fully idempotent: re-runs skip anything already downloaded, with a half-second sleep between requests to respect rate limits. Importantly, this script is decoupled from training — it's only invoked by the audio synthesis output stage, not during the RL loop.
>
> 2. `process_groove.py` is where the real dataset construction happens. It maps 35 MIDI pitch values onto our 8 instrument channels, segments every four beats into one 8-by-16 binary grid, and outputs a NumPy array of shape **N-by-8-by-16**. 
> 
> One important housekeeping detail: about 5.86% of grids are all-zero (silent segments). Those get removed from the positives set before discriminator training, but they'll be re-added explicitly as synthetic negatives later. Everything downstream — the discriminator and the reward function — consumes these binary grids directly.

*(Click to Slide 4)*

---

## Slide 4 — BeatGridEnv
⏱ **Timing:** ~75 seconds

> The environment is a custom Gymnasium discrete environment called **BeatGridEnv**. Let me unpack the three key design decisions:
>
> - **Observation space:** The state is a flat float32 vector of shape `L-by-T-by-(S+2)`. Each cell carries `S+1` channels for a one-hot sample ID (zeros if unplayed), plus one extra channel broadcasting temporal progress as `step_count / max_steps`. During the actor and critic forward passes, this gets reshaped into a 2D spatial tensor of shape `(S+2, L, T)` for CNN processing.
>
> - **Action space:** It's a single flat integer. The decode logic peels it apart: sample varies fastest, then time step, then layer. With `S=15` and `T=16`, that gives 16 sample choices per cell including silence, across `L-by-T` positions. A pre-built boolean buffer gates which sample IDs are valid per layer — so the agent can never place a bass sample on the hi-hat row.
>
> - **Episode mechanics:** `max_steps` equals `L times T` exactly. Every episode terminates in precisely `L-by-T` actions, one per cell, with no truncation. The environment raises a `ValueError` if the actor tries to place on an already-occupied cell — though in practice, **dynamic occupancy masking** in the actor prevents that from ever happening at inference time.

*(Click to Slide 5)*

---

## Slide 5 — Reward Function
⏱ **Timing:** ~90 seconds

> The reward function is arguably the most important design artifact in this project, so I want to be precise.
>
> The terminal reward formula is: **`terminal_reward = α × r_rules + β × r_disc`**, with alpha and beta set to `0.7` and `0.3` in Phase 2. `r_rules` itself is an average of drum and melodic sub-rewards in Phase 2.
>
> There's also an **intermediate reward** computed at every step — but it's deliberately cheap. It only evaluates the single action just placed: `+0.05` for anchoring the kick on step zero, `+0.05` for a snare or clap on steps 4 or 12, zero otherwise. No full grid scan mid-episode.
>
> The **terminal drum evaluation** is richer. It rewards kick anchoring on beats 1 and 3, snare and clap on beats 2 and 4, penalizes off-beat snare hits, rewards hi-hat density in the 4-to-12 range, and critically — it uses **Jaccard similarity** between the first and second half of the bar to reward repetition without being robotically identical. A Jaccard above `0.95` triggers a penalty for being too repetitive.
>
> The **melodic evaluation** in Phase 2 rewards bass-to-kick lock (at least 50% overlap), sparse pad usage, and limited FX accents. It also penalizes polyphony violations where more than four instruments hit simultaneously.
>
> Finally, the **discriminator score** converts the binary grid to a tensor, runs it through `disc.eval()` under `no_grad`, and takes the sigmoid output as a continuous score in `[0, 1]`. Both unplayed and silence cells map to `0.0` before feeding the discriminator.

*(Click to Slide 6)*

---

## Slide 6 — Actor Network
⏱ **Timing:** ~90 seconds

> The actor is **CNNLayerStepSampleActor**, and its architecture is where the novelty really concentrates.
>
> The CNN backbone is shared across all heads: two `Conv2d` layers (`S+2 → 32 → 64` channels, `3×3` kernels with padding), then flatten and a 128-dimensional linear projection with ReLU. This gives us a 128-dimensional `base_features` vector that captures the current spatial state of the grid.
>
> What makes this architecture interesting is the **autoregressive three-head chain**. Rather than decoding a flat joint action, the actor factorizes it:
> 1. The **layer head** takes `base_features` and produces a distribution over `L` layers. 
> 2. Once a layer is sampled, its embedding is added to `base_features` before the **step head** produces a distribution over `T` time steps. 
> 3. The chosen step's embedding then conditions the **sample head**, which selects among `S+1` sample IDs. 
> 
> Log-probability for PPO is the sum across all three heads; entropy is summed similarly.
>
> The key technical contribution here is **dynamic occupancy masking**. The mask is reconstructed entirely from the observation tensor — not from stored environment state. This means it works correctly during the offline PPO gradient update phase, not just during rollout. Occupied cells are detected by checking whether the one-hot slice sums to greater than zero. Layers with no available cells are masked to negative infinity before sampling. Steps on the chosen layer that are occupied are masked. And sample IDs are gated per layer via a pre-built registered buffer. This guarantees the agent never produces illegal actions.

*(Click to Slide 7)*

---

## Slide 7 — Critic Network
⏱ **Timing:** ~45 seconds

> The critic, **CNNBeatCritic**, mirrors the actor's CNN frontend but is entirely independent — separate weights, separate Adam optimizer.
>
> The architecture is the same two-conv tower feeding into a 128-unit linear layer, but the output is a scalar `V(s)` with no activation. There are no autoregressive heads and no masking. It's called once per step under `no_grad` during rollout to produce the value estimates needed for GAE.
>
> The **GAE computation** is standard: temporal difference residuals `delta = r + γ·V(s') − V(s)`, accumulated in reverse with lambda decay. The critic is then updated with MSE loss between predicted values and computed returns, four gradient steps per epoch, with gradient clipping at max-norm `0.5`. The learning rate is `1e-3`, faster than the actor's `3e-4`, which reflects that the value loss surface is smoother and benefits from a slightly more aggressive update.

*(Click to Slide 8)*

---

## Slide 8 — Discriminator Model
⏱ **Timing:** ~75 seconds

> The **BeatDiscriminator** is a transformer encoder trained as a binary classifier: real Groove MIDI beats versus synthetically generated fakes.
>
> The input is a binary hit grid of shape `B-by-L-by-T`. It gets transposed to `B-by-T-by-L` — treating the 16 time steps as the sequence length — and projected into a 64-dimensional token embedding space. Learned positional embeddings are added, then dropout. 
> 
> Two stacked encoder blocks follow the standard pre-norm residual pattern: multi-head attention (`4 heads, d_k=16`) plus FFN, each with layer normalization and dropout. The sequence is mean-pooled across the time dimension, then passed through a two-layer classification head down to a single raw logit. Sigmoid is applied externally, in the reward function.
>
> The dataset construction is where I put real care into making the discriminator hard to fool. The `NegativeGenerator` produces five types of synthetic negatives with tuned probabilities: 
> - **Density outliers (30%)**: clearly out-of-range, providing the strongest signal.
> - **Random Bernoulli patterns (20%)**.
> - **All-zero silent grids (20%)**: re-added explicitly after being removed from positives.
> - **Row-shuffled real grids (15%)**: correct density but destroyed inter-row structure.
> - **PPO-generated agent beats (15%)**: sampled from the live training agent's output. This last category enables the adversarial refinement loop I'll mention in future work.

*(Click to Slide 9)*

---

## Slide 9 — Discriminator Training
⏱ **Timing:** ~60 seconds

> The training pipeline in `train_discriminator.py` is phase-gated. Phase 1 slices the grid array to the first four rows (drums only) and saves to `discriminator_phase1_v2.pt`. Phase 2 uses all eight layers and saves to `discriminator_phase2.pt`. The separate checkpoint names prevent any overwrite.
>
> The **five pipeline steps** are: 
> 1. Load and filter the `.npy` grids.
> 2. Construct the dataset with `2x` negatives per real sample and five-way negative sampling.
> 3. Do an 80/20 train/val split with `drop_last=True`.
> 4. Train with `BCEWithLogitsLoss` and Adam at `3e-4`.
> 5. Save only on improved validation accuracy.
>
> One implementation note worth flagging: `labels.view(-1, 1).float()` is required to prevent a broadcast dimension mismatch with `BCEWithLogitsLoss` — an easy bug to introduce if you're not careful.
>
> Looking at the training curves — the discriminator reaches validation accuracy above `0.97` within 15 epochs, and BCE loss converges in under five epochs. This gives us a highly confident learned reward signal for the PPO agent to optimize against.

*(Click to Slide 10)*

---

## Slide 10 — PPO Training
⏱ **Timing:** ~75 seconds

> `train_ppo.py` is the orchestrating script. Let me walk through the hyperparameters and the loop.
>
> **Key hyperparameters:** 500 epochs, 32 episodes per epoch, gamma of `0.99`, GAE lambda of `0.95`, clip ratio of `0.2` — standard PPO defaults. Entropy coefficient is `0.10`, which I kept relatively high to encourage exploration throughout training. Both actor and critic use gradient clipping at max-norm `0.5`. Alpha and beta for the reward blend are `0.7` and `0.3`.
>
> The **training loop** has four stages per epoch:
> 1. **Rollout collection:** 32 episodes of `L-by-T` steps each. The actor acts under masked Categorical with `no_grad`; the critic produces value estimates also under `no_grad`. Everything is stored in replay buffers.
> 2. **GAE computation:** temporal difference residuals are computed, then accumulated in reverse with lambda decay. Advantages are normalized to zero mean and unit variance before the update.
> 3. **PPO actor update:** four gradient steps with the clipped surrogate objective, minus the entropy bonus to maintain exploration pressure.
> 4. **Critic update:** four gradient steps with MSE loss on returns. Checkpoints are saved only when mean episode reward exceeds the running best.

*(Click to Slide 11)*

---

## Slide 11 — PPO Results
⏱ **Timing:** ~75 seconds

> Let me interpret what actually happened during training.
>
> The **mean episode reward curve** shows clear, monotonic improvement from around `0.35` at epoch zero up to the best of `0.523` at epoch 492. The improvement is particularly sharp in the first 150 epochs, then gradually plateaus — which is what you'd expect once the agent has locked in the high-value rule-based behaviors like backbeat and Jaccard and is squeezing marginal gains from the discriminator signal.
>
> The **actor loss** stays in a narrow range around `-0.62` to `-0.64` after initial stabilization — the PPO clip is doing its job keeping policy updates bounded. The **critic loss** drops sharply from a high of about `0.0025` in the first few epochs to near zero, indicating the value function is accurately predicting returns well before policy optimization plateaus.
>
> The **beat grid comparison** between Epoch 0 and Epoch 492 is qualitatively striking. The first grid is dense and somewhat scattered — the agent hasn't learned placement preferences yet. The best grid shows structured sparsity, especially visible in the snare, hi-hat, and clap rows, which now have clear gaps that mimic the breathing room in real musical patterns. The bass row also shows better kick alignment, reflecting the bass-lock reward.

*(Click to Slide 12)*

---

## Slide 12 — Baseline Comparison
⏱ **Timing:** ~30 seconds

> To quantify how much the agent actually learned, I ran a 
> random agent baseline — same environment, same reward 
> function, just sampling actions uniformly across 20 episodes.
> 
> The random agent averages 0.46 on rule reward, with four 
> complete collapses to zero — it has no concept of backbeat 
> or structure. The PPO agent reaches 0.96 consistently across 
> all runs. That 2x gap is the policy.

*(Click to Slide 13)*

---

## Slide 13 — Future Work
⏱ **Timing:** ~60 seconds

> I identified three directions worth pursuing from here.
>
> 1. **Adversarial discriminator refinement:** feeding the agent's best-epoch grids back into the training dataset and iteratively retraining the discriminator. This closes the adversarial loop and prevents reward hacking as the agent gets better at fooling a static discriminator.
> 2. **Multi-agent co-composition:** assigning separate PPO agents to instrument groups (drums, bass, melodic) with a shared coordination signal. The bass agent, for example, would receive the kick agent's grid as part of its observation, directly incentivizing bass-lock behavior structurally rather than through a reward term.
> 3. **VLM-based reward:** replacing the hand-coded music rules with a vision-language model like Qwen-VL that scores the beat grid against a natural-language music quality prompt. This enables richer, more flexible reward shaping.

*(Click to Slide 14)*

---

## Slide 14 — Q&A
⏱ **Timing:** ~15 seconds

> That covers the full system — from the data pipeline through discriminator training, actor architecture, and PPO results. I'm happy to go deeper on any component: the reward formulation, the autoregressive masking scheme, the discriminator's negative sampling strategy, or the phase curriculum design. 
> 
> Thank you.
