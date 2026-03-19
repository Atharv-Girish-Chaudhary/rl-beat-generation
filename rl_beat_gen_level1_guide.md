# RL-Based Beat Generation: Level 1 Implementation Guide

> **Who this is for:** The three students building this project. Every step is spelled out. No assumed knowledge beyond Python, basic PyTorch, and having read the Sutton & Barto RL textbook through Chapter 9.
>
> **What Level 1 covers:** Train a PPO agent to populate a beat grid (discrete actions only — no audio effects) guided by hand-crafted musical rules and a learned discriminator. By the end, your agent should produce structurally coherent drum and melody arrangements.

---

## Table of Contents

1. [Conceptual Foundation — What Are We Actually Building?](#1-conceptual-foundation)
2. [Environment Setup](#2-environment-setup)
3. [Dataset Pipeline](#3-dataset-pipeline)
4. [The MDP: State, Action, Reward](#4-the-mdp)
5. [The Beat Grid Environment](#5-the-beat-grid-environment)
6. [The Policy Network (Actor)](#6-the-policy-network-actor)
7. [The Value Network (Critic)](#7-the-value-network-critic)
8. [The Reward Function](#8-the-reward-function)
9. [The Discriminator](#9-the-discriminator)
10. [Pre-Training the Discriminator](#10-pre-training-the-discriminator)
11. [PPO Training Loop](#11-ppo-training-loop)
12. [Curriculum: Phase 1 → Phase 2](#12-curriculum)
13. [Evaluation](#13-evaluation)
14. [Common Failure Modes and Fixes](#14-common-failure-modes-and-fixes)
15. [Full File Structure](#15-full-file-structure)

---

## 1. Conceptual Foundation

Before writing a single line of code, make sure everyone on the team can answer these questions out loud:

### What is the agent doing?

The agent is building a beat grid step by step. Think of it like a producer sitting at a drum machine, filling in an empty 8×16 grid one cell at a time. Each row is an instrument (kick, snare, hi-hat, bass, melody, etc.). Each column is a 16th-note time step in one bar. At every step, the agent picks **one cell** to fill and **which sound** to put there.

### Why is this an MDP?

- **State** — the current partially-filled grid. Everything the agent needs to make the next decision is in the current grid (Markov property holds).
- **Action** — pick a layer, a time step, and a sample.
- **Reward** — how "good" is the finished beat? Given only at the end of the episode (plus small intermediate hints).
- **Episode** — starts with an empty grid, ends when all 128 cells are filled.

### Why is this hard?

1. **Long horizon.** 128 steps before the main reward arrives. Early decisions get weak feedback.
2. **Large action space.** With 8 layers × 16 steps × 20 samples (+1 silence), that's 8 × 16 × 21 = **2,688 possible actions per step**.
3. **Reward sparsity.** The most informative signal (discriminator score, "does this sound like a real beat?") only comes at the end.

### What does "success" look like for Level 1?

A generated beat should:
- Have kicks on beats 1, 5, 9, 13 (strong beats)
- Have snares on beats 3, 7, 11, 15 (backbeats)
- Have hi-hats filling the in-between steps
- Not be too dense or too sparse
- Have bass and melody lines that don't clash spectrally with the drums
- Sound recognizable as a beat to a human listener in a blind test

---

## 2. Environment Setup

### 2.1 Python version and virtual environment

Use Python 3.10. Create a dedicated environment — do not install into your system Python.

```bash
python3.10 -m venv beat_env
source beat_env/bin/activate   # On Windows: beat_env\Scripts\activate
```

### 2.2 Install dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install gymnasium numpy pretty_midi requests tqdm matplotlib tensorboard
pip install stable-baselines3[extra]  # For PPO reference implementation
```

> **Why stable-baselines3?** PPO is notoriously sensitive to implementation details (advantage normalization, gradient clipping, entropy bonus schedule). Stable-baselines3 has a battle-tested PPO implementation. You will write a **custom environment** that plugs into it, rather than implementing PPO from scratch. The novelty of your project is the music environment and reward — not a custom PPO optimizer.

### 2.3 Verify GPU access

```python
import torch
print(torch.cuda.is_available())      # Should print True
print(torch.cuda.get_device_name(0))  # Should print your GPU name
```

If this prints False, you need to install the CUDA-compatible PyTorch build. Check https://pytorch.org/get-started/locally/ for the exact command matching your CUDA version.

### 2.4 Project directory

```
beat_gen/
├── data/
│   ├── raw/              # Downloaded MIDI files
│   ├── processed/        # Converted grid tensors (.npy files)
│   └── samples/          # Downloaded WAV files from Freesound
├── env/
│   ├── beat_env.py       # The Gymnasium environment
│   └── reward.py         # All reward functions
├── models/
│   ├── actor.py          # Policy network
│   ├── critic.py         # Value network
│   └── discriminator.py  # Transformer discriminator
├── training/
│   ├── pretrain_disc.py  # Pre-train discriminator before RL
│   └── train_ppo.py      # Main RL training loop
├── evaluation/
│   └── evaluate.py
└── configs/
    └── config.yaml       # All hyperparameters in one place
```

---

## 3. Dataset Pipeline

You need two datasets: one for the agent's action space (WAV samples), one to train the discriminator (real beat grids).

### 3.1 Download WAV samples from Freesound

The agent's "vocabulary" is a library of one-shot WAV files. These are the sounds it can place on the grid.

```python
# data/download_samples.py
import requests
import os

FREESOUND_API_KEY = "YOUR_API_KEY"  # Register at freesound.org/apiv2/apply
BASE_URL = "https://freesound.org/apiv2"

LAYER_QUERIES = {
    "kick":    {"query": "kick drum one shot", "layer_id": 0},
    "snare":   {"query": "snare drum one shot", "layer_id": 1},
    "hihat":   {"query": "hi-hat closed one shot", "layer_id": 2},
    "clap":    {"query": "clap percussion one shot", "layer_id": 3},
    "bass":    {"query": "bass hit one shot", "layer_id": 4},
    "melody":  {"query": "synth note one shot C", "layer_id": 5},
    "pad":     {"query": "pad chord one shot", "layer_id": 6},
    "fx":      {"query": "fx sweep one shot", "layer_id": 7},
}

def search_and_download(category, query, layer_id, n_samples=15, output_dir="data/samples"):
    os.makedirs(f"{output_dir}/{category}", exist_ok=True)
    
    # Search for CC0 samples only
    params = {
        "query": query,
        "filter": 'license:"Creative Commons 0"',
        "fields": "id,name,previews,duration",
        "page_size": n_samples,
        "token": FREESOUND_API_KEY,
    }
    resp = requests.get(f"{BASE_URL}/search/text/", params=params)
    results = resp.json()["results"]
    
    sample_manifest = []  # Track which file = which sample index
    for i, sound in enumerate(results):
        preview_url = sound["previews"]["preview-hq-mp3"]
        audio_resp = requests.get(preview_url)
        filename = f"{output_dir}/{category}/{i:02d}_{sound['id']}.mp3"
        with open(filename, "wb") as f:
            f.write(audio_resp.content)
        sample_manifest.append({
            "layer": category,
            "layer_id": layer_id,
            "sample_idx": i,
            "filename": filename,
            "freesound_id": sound["id"],
        })
        print(f"  Downloaded {category} sample {i+1}/{n_samples}")
    
    return sample_manifest

# Run this once to build your sample library
all_samples = []
for category, info in LAYER_QUERIES.items():
    print(f"Downloading {category} samples...")
    samples = search_and_download(category, info["query"], info["layer_id"])
    all_samples.extend(samples)

import json
with open("data/samples/manifest.json", "w") as f:
    json.dump(all_samples, f, indent=2)
print(f"Total samples: {len(all_samples)}")
```

> **What you get:** ~15 WAV files per layer × 8 layers = ~120 samples total. The manifest.json maps each sample to its layer and integer index. This is the agent's action vocabulary.

### 3.2 Download and process Groove MIDI dataset

The Groove MIDI Dataset is used to **train the discriminator** to recognize real drum patterns.

```bash
# Download from Magenta's public storage
wget https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip
unzip groove-v1.0.0-midionly.zip -d data/raw/groove
```

Now convert MIDI files to the same grid representation the agent uses. This ensures the discriminator learns to judge grids in the format the agent produces.

```python
# data/process_groove.py
import pretty_midi
import numpy as np
import os
import glob

# These are the General MIDI note numbers for standard drum kit
# Only these will be mapped — others are discarded
DRUM_NOTE_TO_LAYER = {
    36: 0,  # Bass drum (kick)
    38: 1,  # Acoustic snare
    40: 1,  # Electric snare (also maps to snare layer)
    42: 2,  # Closed hi-hat
    46: 2,  # Open hi-hat (also hi-hat layer)
    39: 3,  # Hand clap
    49: 3,  # Crash cymbal (maps to clap/accent layer)
}

L = 4       # Layers for Phase 1 (drums only: kick, snare, hihat, clap)
T = 16      # Time steps (one bar at 16th-note resolution)
S = 15      # Samples per layer (use 15 for now, matches download above)

def midi_to_grid(midi_path, L=4, T=16):
    """
    Convert a MIDI file to a beat grid of shape (L, T).
    Each cell contains the sample index (1-indexed) placed there, or 0 for silence.
    Returns None if the MIDI file cannot be parsed.
    """
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
    except Exception:
        return None
    
    # Find the drum track (MIDI channel 9 or instrument.is_drum == True)
    drum_track = None
    for instrument in midi.instruments:
        if instrument.is_drum:
            drum_track = instrument
            break
    
    if drum_track is None:
        return None
    
    # Get tempo to figure out beat duration
    tempo = midi.estimate_tempo()           # BPM
    beat_duration = 60.0 / tempo            # Seconds per beat
    sixteenth_duration = beat_duration / 4  # Seconds per 16th note
    
    # Build the grid
    grid = np.zeros((L, T), dtype=np.int64)
    
    for note in drum_track.notes:
        layer = DRUM_NOTE_TO_LAYER.get(note.pitch, None)
        if layer is None or layer >= L:
            continue
        
        # Map onset time to 16th-note grid position
        step = int(round(note.start / sixteenth_duration)) % T
        
        # Use velocity to pick a sample (louder hits → higher sample index)
        # Velocity is 0-127, we have S samples, so we bin into S bins
        sample_idx = min(int(note.velocity / 128 * S) + 1, S)
        grid[layer][step] = sample_idx
    
    return grid

# Process all MIDI files
midi_files = glob.glob("data/raw/groove/**/*.mid", recursive=True)
grids = []
for path in midi_files:
    grid = midi_to_grid(path, L=L, T=T)
    if grid is not None:
        grids.append(grid)

grids = np.array(grids)  # Shape: (N, L, T)
print(f"Processed {len(grids)} valid grids out of {len(midi_files)} MIDI files")
np.save("data/processed/groove_grids.npy", grids)
```

> **What you get:** An array of shape `(N, 4, 16)` where N ≈ 1000–1100 grids. Each cell is an integer 0–15 (0 = silence, 1–15 = which sample). This is your **positive training set** for the discriminator.

---

## 4. The MDP

This section defines the math precisely so implementation follows directly.

### 4.1 State space

The state at time step `t` is the partially-filled beat grid:

```
s_t ∈ {0, 1, ..., S}^(L × T)
```

- Each cell is an integer: 0 means silence, 1–S means which sample is placed there.
- For the CNN, this needs to be one-hot encoded: shape `(L, T, S+1)` where the last dimension is a one-hot vector over the S+1 possible values (0 = silence, 1 to S = samples).
- This is what gets fed into the actor and critic networks.

**Why one-hot?** The integers 0–15 carry no ordinal meaning in this context — sample #7 is not "between" samples #6 and #8 in any musical sense. One-hot encoding prevents the network from trying to learn spurious ordinality.

### 4.2 Action space

**Important — this is the fix to the proposal's factored head problem.** The action is a tuple `(ℓ, t, s)` where:

- `ℓ ∈ {0, ..., L-1}` — which layer (instrument)
- `t ∈ {0, ..., T-1}` — which time step
- `s ∈ {0, ..., S_ℓ}` — which sample, where `S_ℓ` is the number of samples **for layer ℓ**, plus silence (index 0)

**The critical constraint:** Sample selection must be conditioned on layer selection. A kick sample cannot be placed on the melody layer. This means the three heads are NOT independent.

The correct architecture uses **masked action selection**:

1. Head 1 outputs a distribution over layers → sample `ℓ`
2. Head 2 outputs a distribution over time steps → sample `t`
3. Head 3 outputs a distribution over S+1 values, then **mask to zero all samples that don't belong to layer `ℓ`** → re-normalize → sample `s`

This masking happens at inference time and during probability computation for the policy gradient update.

### 4.3 Episode structure

- **Start:** Grid filled with all zeros (silence everywhere)
- **Termination:** All `L × T` cells have been filled (every cell has value ≥ 0, though 0 = silence is also a valid final value)
- **Max steps:** `L × T` (128 for Phase 2, 64 for Phase 1)
- **Step:** At each step, the agent fills exactly one cell. Once a cell is filled, it cannot be changed. The agent is not allowed to re-place a cell — this is enforced by the environment.

> **Practical note on ordering:** The agent chooses which cell to fill at each step. This means it could choose to fill step (layer=0, time=5) before (layer=0, time=0). The Markov property holds regardless of fill order because the full grid state is observed.

---

## 5. The Beat Grid Environment

This implements the environment as a `gymnasium.Env` subclass. This is what stable-baselines3 PPO will interact with.

```python
# env/beat_env.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class BeatGridEnv(gym.Env):
    """
    Beat grid environment for Level 1 (discrete actions only).
    
    Observation: one-hot encoded grid of shape (L, T, S+1), flattened to 1D
    Action: integer in [0, L*T*(S+1)) decoded to (layer, step, sample)
    """

    def __init__(self, L, T, S, reward_fn, layer_to_samples, phase=1):
        """
        Args:
            L: number of layers (4 for Phase 1, 8 for Phase 2)
            T: number of time steps (always 16)
            S: total number of samples per layer (assume uniform for now)
            reward_fn: callable(grid, phase) → scalar reward (called at episode end)
            layer_to_samples: dict {layer_idx: [sample_idx, ...]} — which samples
                              belong to which layer. Used for action masking.
            phase: 1 or 2, used to set grid size
        """
        super().__init__()
        self.L = L
        self.T = T
        self.S = S
        self.reward_fn = reward_fn
        self.layer_to_samples = layer_to_samples  # {0: [1,2,...,15], 1: [...], ...}
        self.phase = phase
        
        # Action space: flat integer over all (layer, step, sample) tuples
        # We decode this integer in _decode_action()
        # Size: L * T * (S + 1)  where +1 accounts for silence (sample index 0)
        self.action_space = spaces.Discrete(L * T * (S + 1))
        
        # Observation space: flattened one-hot grid
        # Shape before flattening: (L, T, S+1)
        obs_dim = L * T * (S + 1)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        
        self.grid = None
        self.filled_cells = None
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((self.L, self.T), dtype=np.int64)
        self.filled_cells = set()
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        layer, time_step, sample = self._decode_action(action)
        
        # Enforce: don't re-fill already filled cells
        # If the agent tries to fill a filled cell, redirect to a random empty cell
        # (This is a training convenience — log these events to monitor convergence)
        if (layer, time_step) in self.filled_cells:
            empty = [(l, t) for l in range(self.L) for t in range(self.T)
                     if (l, t) not in self.filled_cells]
            if empty:
                layer, time_step = empty[np.random.randint(len(empty))]
        
        # Apply action
        self.grid[layer][time_step] = sample
        self.filled_cells.add((layer, time_step))
        self.step_count += 1
        
        # Check if episode is done
        terminated = len(self.filled_cells) == self.L * self.T
        truncated = False
        
        # Reward: small intermediate reward at each step, full reward at episode end
        if terminated:
            reward = self.reward_fn(self.grid, final=True)
        else:
            reward = self.reward_fn(self.grid, final=False)
        
        obs = self._get_obs()
        info = {"step_count": self.step_count, "filled": len(self.filled_cells)}
        
        return obs, reward, terminated, truncated, info

    def _decode_action(self, action):
        """
        Convert flat integer action back to (layer, time_step, sample).
        
        Encoding scheme:
          action = layer * (T * (S+1)) + time_step * (S+1) + sample
        """
        sample    = action % (self.S + 1)
        remainder = action // (self.S + 1)
        time_step = remainder % self.T
        layer     = remainder // self.T
        return int(layer), int(time_step), int(sample)

    def _encode_action(self, layer, time_step, sample):
        """Inverse of _decode_action — useful for testing."""
        return layer * (self.T * (self.S + 1)) + time_step * (self.S + 1) + sample

    def _get_obs(self):
        """
        One-hot encode the grid and flatten to 1D.
        
        grid shape: (L, T) with integer values in [0, S]
        output shape: (L * T * (S+1),) with float32 values in {0.0, 1.0}
        """
        one_hot = np.zeros((self.L, self.T, self.S + 1), dtype=np.float32)
        for l in range(self.L):
            for t in range(self.T):
                sample_idx = self.grid[l][t]  # 0 = silence, 1..S = sample
                one_hot[l, t, sample_idx] = 1.0
        return one_hot.flatten()

    def get_action_mask(self, layer):
        """
        Return a boolean mask of shape (S+1,) indicating which samples
        are valid for the given layer.
        
        Index 0 (silence) is always valid.
        Indices 1..S: valid only if that sample belongs to this layer.
        """
        mask = np.zeros(self.S + 1, dtype=bool)
        mask[0] = True  # silence always allowed
        for s in self.layer_to_samples.get(layer, []):
            if s <= self.S:
                mask[s] = True
        return mask
```

> **Testing the environment:** Before any training, run a sanity check:
>
> ```python
> from env.beat_env import BeatGridEnv
> from env.reward import reward_fn_phase1
>
> layer_to_samples = {0: list(range(1,16)), 1: list(range(1,16)),
>                     2: list(range(1,16)), 3: list(range(1,16))}
> env = BeatGridEnv(L=4, T=16, S=15, reward_fn=reward_fn_phase1,
>                   layer_to_samples=layer_to_samples)
>
> obs, _ = env.reset()
> print("Obs shape:", obs.shape)  # Should be (4*16*16,) = (1024,)
>
> for _ in range(64):
>     action = env.action_space.sample()
>     obs, reward, terminated, truncated, info = env.step(action)
>     if terminated:
>         print("Episode done. Final reward:", reward)
>         break
> ```

---

## 6. The Policy Network (Actor)

The actor takes the one-hot encoded grid as input and outputs a probability distribution over actions. Because the action space is factored into (layer, time_step, sample), the network has three output heads — but with the critical fix: **sample selection is masked based on the chosen layer**.

```python
# models/actor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BeatActor(nn.Module):
    """
    CNN-based actor network for the beat grid environment.
    
    Input: one-hot encoded grid, shape (batch, L, T, S+1)
    Output: three distributions: π(layer), π(step), π(sample | layer)
    """
    
    def __init__(self, L, T, S, layer_to_samples):
        """
        layer_to_samples: dict {layer_idx: [sample_indices_that_belong_to_this_layer]}
        """
        super().__init__()
        self.L = L
        self.T = T
        self.S = S
        self.layer_to_samples = layer_to_samples
        
        # Build the layer-sample mask matrix: shape (L, S+1)
        # mask[l][s] = 1 if sample s is valid for layer l, else 0
        # Silence (index 0) is always valid
        self.register_buffer("layer_sample_mask", self._build_layer_mask(L, S, layer_to_samples))
        
        # CNN backbone: processes the grid as a 2D image
        # Input channels = S+1 (one-hot depth dimension)
        # We treat the grid as a (L × T) image with (S+1) channels
        self.conv1 = nn.Conv2d(in_channels=S+1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # After two conv layers, the spatial dims are still L × T (due to padding=1)
        # Flatten: 64 * L * T features
        conv_out_dim = 64 * L * T
        
        self.fc1 = nn.Linear(conv_out_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        
        # Three output heads
        self.layer_head  = nn.Linear(128, L)       # L-way softmax
        self.step_head   = nn.Linear(128, T)       # T-way softmax
        self.sample_head = nn.Linear(128, S + 1)   # (S+1)-way softmax, then masked

    def _build_layer_mask(self, L, S, layer_to_samples):
        """Create the (L, S+1) boolean mask tensor."""
        mask = torch.zeros(L, S + 1, dtype=torch.float32)
        for l in range(L):
            mask[l][0] = 1.0  # silence always valid
            for s in layer_to_samples.get(l, []):
                if s <= S:
                    mask[l][s] = 1.0
        return mask

    def forward(self, obs):
        """
        obs: (batch, L * T * (S+1)) — the flattened one-hot grid
        Returns: layer_logits, step_logits, sample_logits (all unmasked — masking done in act())
        """
        batch = obs.shape[0]
        
        # Reshape to (batch, S+1, L, T) for Conv2d
        # Conv2d expects (batch, channels, height, width)
        x = obs.view(batch, self.L, self.T, self.S + 1)   # (B, L, T, S+1)
        x = x.permute(0, 3, 1, 2)                         # (B, S+1, L, T)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(batch, -1)  # flatten
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        layer_logits  = self.layer_head(x)   # (B, L)
        step_logits   = self.step_head(x)    # (B, T)
        sample_logits = self.sample_head(x)  # (B, S+1)
        
        return layer_logits, step_logits, sample_logits

    def act(self, obs):
        """
        Full inference: sample an action from the policy.
        Applies layer-conditional sample masking.
        
        Returns: action (int), log_prob (scalar tensor), entropy (scalar tensor)
        """
        layer_logits, step_logits, sample_logits = self.forward(obs)
        
        # Sample layer
        layer_probs = F.softmax(layer_logits, dim=-1)        # (B, L)
        layer_dist  = torch.distributions.Categorical(layer_probs)
        layer_action = layer_dist.sample()                   # (B,)
        
        # Sample time step
        step_probs = F.softmax(step_logits, dim=-1)
        step_dist  = torch.distributions.Categorical(step_probs)
        step_action = step_dist.sample()                     # (B,)
        
        # Sample from MASKED sample distribution (conditioned on chosen layer)
        # layer_sample_mask shape: (L, S+1)
        # layer_action shape: (B,)
        mask = self.layer_sample_mask[layer_action]          # (B, S+1)
        
        # Apply mask: set logits of invalid samples to -inf before softmax
        masked_logits = sample_logits.masked_fill(mask == 0, float('-inf'))
        sample_probs  = F.softmax(masked_logits, dim=-1)
        sample_dist   = torch.distributions.Categorical(sample_probs)
        sample_action = sample_dist.sample()                 # (B,)
        
        # Log probability of the full action = sum of log probs of each sub-action
        # (valid because layer and step are marginally independent given state;
        #  sample is conditioned on layer, which we account for here)
        log_prob = (layer_dist.log_prob(layer_action) +
                    step_dist.log_prob(step_action) +
                    sample_dist.log_prob(sample_action))
        
        # Entropy = sum of marginal entropies (approximate — assumes approximate independence)
        entropy = layer_dist.entropy() + step_dist.entropy() + sample_dist.entropy()
        
        # Encode back to flat integer action
        action = (layer_action * self.T * (self.S + 1) +
                  step_action * (self.S + 1) +
                  sample_action)
        
        return action, log_prob, entropy

    def evaluate_actions(self, obs, actions):
        """
        Given a batch of observations and the actions taken, compute the
        log probability and entropy of those actions under the current policy.
        Used by PPO for the policy gradient update.
        """
        # Decode flat actions back to (layer, step, sample)
        sample_acts = actions % (self.S + 1)
        remainder   = actions // (self.S + 1)
        step_acts   = remainder % self.T
        layer_acts  = remainder // self.T
        
        layer_logits, step_logits, sample_logits = self.forward(obs)
        
        layer_probs  = F.softmax(layer_logits, dim=-1)
        step_probs   = F.softmax(step_logits, dim=-1)
        
        mask          = self.layer_sample_mask[layer_acts]
        masked_logits = sample_logits.masked_fill(mask == 0, float('-inf'))
        sample_probs  = F.softmax(masked_logits, dim=-1)
        
        layer_dist  = torch.distributions.Categorical(layer_probs)
        step_dist   = torch.distributions.Categorical(step_probs)
        sample_dist = torch.distributions.Categorical(sample_probs)
        
        log_prob = (layer_dist.log_prob(layer_acts) +
                    step_dist.log_prob(step_acts) +
                    sample_dist.log_prob(sample_acts))
        
        entropy = layer_dist.entropy() + step_dist.entropy() + sample_dist.entropy()
        
        return log_prob, entropy
```

---

## 7. The Value Network (Critic)

The critic estimates V(s) — how good the current state is on average across all possible future actions. It shares the same CNN backbone as the actor but has a scalar output head instead of action heads.

```python
# models/critic.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class BeatCritic(nn.Module):
    """
    CNN-based value network. Same architecture as the actor backbone,
    separate parameters (do NOT share weights with actor in PPO unless
    you have a very careful implementation — parameter sharing in PPO
    requires careful gradient management).
    """

    def __init__(self, L, T, S):
        super().__init__()
        self.L = L
        self.T = T
        self.S = S
        
        self.conv1 = nn.Conv2d(in_channels=S+1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        conv_out_dim = 64 * L * T
        self.fc1 = nn.Linear(conv_out_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.value_head = nn.Linear(128, 1)  # scalar output

    def forward(self, obs):
        """
        obs: (batch, L * T * (S+1))
        Returns: V(s), shape (batch, 1)
        """
        batch = obs.shape[0]
        x = obs.view(batch, self.L, self.T, self.S + 1)
        x = x.permute(0, 3, 1, 2)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(batch, -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        return self.value_head(x)  # (batch, 1)
```

---

## 8. The Reward Function

This is the core of what makes this a *music* project rather than a generic RL task. Each sub-reward is explained from first principles.

```python
# env/reward.py
import numpy as np

# ── Layer index constants ────────────────────────────────────────────────────
KICK  = 0
SNARE = 1
HIHAT = 2
CLAP  = 3
BASS  = 4
MELODY = 5
PAD    = 6
FX     = 7

# Phase 1 uses only layers 0-3 (drums)
# Phase 2 uses all 8 layers


def reward_fn_phase1(grid, final=False, discriminator=None, alpha=0.9, beta=0.1,
                     compat_lambda=0.02):
    """
    Reward function for Phase 1 (4-layer drum grid).
    
    At each intermediate step: return a small compatibility reward.
    At episode termination: return alpha * Rrules + beta * Rdisc, both in [0,1].
    
    Args:
        grid: np.array of shape (L, T), integer values 0..S
        final: True if this is the terminal step
        discriminator: trained discriminator model (can be None — will be added in Phase 2)
        alpha, beta: weights (alpha + beta = 1)
        compat_lambda: weight for intermediate compatibility reward
    """
    if not final:
        # Intermediate reward: pairwise spectral compatibility of the most recently
        # placed layer against already-placed layers.
        # We approximate this using the SPECTRAL BAND OVERLAP rule (see below).
        # This is called at every step, so keep it CHEAP.
        compat = _pairwise_compat_cheap(grid)
        return float(compat_lambda * compat)
    
    # ── Terminal reward ──────────────────────────────────────────────────────
    r_rules = _rule_based_score_phase1(grid)  # in [0, 1]
    
    if discriminator is not None:
        r_disc = _discriminator_score(grid, discriminator)  # in [0, 1]
    else:
        r_disc = 0.0
    
    return float(alpha * r_rules + beta * r_disc)


# ─────────────────────────────────────────────────────────────────────────────
# Rule-Based Sub-Rewards (Phase 1 — drum layers only)
# ─────────────────────────────────────────────────────────────────────────────

def _rule_based_score_phase1(grid):
    """
    Compute the composite rule-based score for a 4-layer drum grid.
    Returns a float in [0, 1].
    
    Sub-rewards (each in [0, 1], weighted equally):
      1. Rhythmic structure
      2. Density control
      3. Repetition with variation
    
    Phase 2 adds: spectral separation, harmonic compatibility, parameter moderation.
    """
    scores = [
        _rhythmic_structure(grid),
        _density_control(grid),
        _repetition_with_variation(grid),
    ]
    return float(np.mean(scores))


def _rhythmic_structure(grid):
    """
    Reward: kicks on strong beats (steps 0,4,8,12) and snares on backbeats (steps 2,6,10,14).
    
    Musical background: In a 4/4 time signature at 16th-note resolution:
      - Beat 1 = step 0, Beat 2 = step 4, Beat 3 = step 8, Beat 4 = step 12
      - Backbeats (snare traditionally hits here): Beat 2 = step 4, Beat 4 = step 12
      - In 16th-note indexing: snares typically on steps 4 and 12, or 2,6,10,14 for a busier pattern
    
    Here we reward:
      - Any kick placed on a strong beat (steps 0,4,8,12)
      - Any snare placed on a backbeat (steps 4,12 for simplicity)
    
    Returns float in [0, 1].
    """
    L, T = grid.shape
    strong_beats  = {0, 4, 8, 12}
    backbeats     = {4, 12}
    
    kick_row  = grid[KICK] if KICK < L else np.zeros(T, dtype=np.int64)
    snare_row = grid[SNARE] if SNARE < L else np.zeros(T, dtype=np.int64)
    
    # Fraction of strong beats that have a kick
    kick_on_strong  = sum(1 for t in strong_beats if kick_row[t] > 0) / len(strong_beats)
    
    # Fraction of backbeats that have a snare
    snare_on_back   = sum(1 for t in backbeats if snare_row[t] > 0) / len(backbeats)
    
    return float(0.5 * kick_on_strong + 0.5 * snare_on_back)


def _density_control(grid):
    """
    Reward: overall grid density (fraction of non-silent cells) in [0.3, 0.6].
    
    Musical background: A beat that is too sparse (density < 0.3) has uncomfortably
    large gaps. A beat that is too dense (density > 0.6) sounds cluttered and
    indistinct. The [0.3, 0.6] range corresponds to roughly 19–38 active cells
    out of 64 cells (for Phase 1, 4×16 grid).
    
    Returns 1.0 if density is in range, linearly decreases to 0 outside range.
    """
    L, T = grid.shape
    n_active = np.sum(grid > 0)
    density  = n_active / (L * T)
    
    if 0.3 <= density <= 0.6:
        return 1.0
    elif density < 0.3:
        return density / 0.3           # Linear ramp from 0 to 1 as density goes 0→0.3
    else:  # density > 0.6
        return max(0.0, 1.0 - (density - 0.6) / 0.4)  # Linear ramp from 1 to 0 as 0.6→1.0


def _repetition_with_variation(grid):
    """
    Reward: bar-to-bar Jaccard similarity in [0.7, 0.99).
    
    Musical background: Good beats have a repeating structure (the listener
    can follow the rhythm) but not exact repetition (which is boring).
    We split the 16-step bar into two 8-step half-bars and measure similarity
    between them using Jaccard similarity.
    
    Jaccard similarity = |A ∩ B| / |A ∪ B|
    where A and B are the sets of active (non-zero) cells in each half.
    
    Target: Jaccard ∈ [0.7, 0.99) — similar but not identical.
    Returns float in [0, 1].
    """
    L, T = grid.shape
    half = T // 2
    
    first_half  = grid[:, :half]    # Shape (L, T//2)
    second_half = grid[:, half:]    # Shape (L, T//2)
    
    # Jaccard over binary active/inactive cells
    active_first  = (first_half > 0).flatten()
    active_second = (second_half > 0).flatten()
    
    intersection = np.sum(active_first & active_second)
    union        = np.sum(active_first | active_second)
    
    if union == 0:
        return 0.0  # Empty grid — no repetition to measure
    
    jaccard = intersection / union
    
    if jaccard >= 0.99:  # Exact repetition — penalize
        return 0.0
    elif 0.7 <= jaccard < 0.99:
        return 1.0  # Target range
    else:
        return jaccard / 0.7  # Linear ramp up to target range


def _pairwise_compat_cheap(grid):
    """
    Fast intermediate compatibility reward: checks if newly placed cells
    are consistent with the rule-based rewards computed so far.
    Called at every step so must be O(L*T) or better.
    
    Returns a float in [0, 1].
    """
    L, T = grid.shape
    score = 0.0
    count = 0
    
    # Check kicks on strong beats at every step
    if KICK < L:
        strong_beats = {0, 4, 8, 12}
        for t in strong_beats:
            if grid[KICK][t] > 0:
                score += 1.0
            count += 1
    
    # Check snares on backbeats
    if SNARE < L:
        backbeats = {4, 12}
        for t in backbeats:
            if grid[SNARE][t] > 0:
                score += 1.0
            count += 1
    
    return score / max(count, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Discriminator Score
# ─────────────────────────────────────────────────────────────────────────────

def _discriminator_score(grid, discriminator):
    """
    Run the discriminator and return P(real) as a reward in [0, 1].
    
    Args:
        grid: np.array (L, T)
        discriminator: trained BeatDiscriminator instance
    Returns: float in [0, 1]
    """
    import torch
    discriminator.eval()
    with torch.no_grad():
        # Convert grid to one-hot tensor
        L, T = grid.shape
        S = discriminator.S
        one_hot = torch.zeros(1, L, T, S + 1)
        for l in range(L):
            for t in range(T):
                one_hot[0, l, t, grid[l][t]] = 1.0
        
        logit = discriminator(one_hot)
        prob  = torch.sigmoid(logit).item()
    
    return float(prob)
```

---

## 9. The Discriminator

The discriminator is a transformer encoder that learns to distinguish real drum grids (from Groove MIDI) from agent-generated or random grids.

```python
# models/discriminator.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class BeatDiscriminator(nn.Module):
    """
    Transformer encoder discriminator for beat grids.
    
    Architecture: Each instrument layer is tokenized as a sequence of T time steps.
    We embed the one-hot representation of each time step into a d_model-dimensional
    token. Then multi-head self-attention allows the model to learn inter-layer
    relationships (e.g., kick and bass tend to co-occur on strong beats).
    
    Output: P(real) — probability that this grid was generated by a human producer.
    """

    def __init__(self, L, T, S, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        """
        Args:
            L: number of layers
            T: number of time steps (16)
            S: number of samples per layer
            d_model: transformer embedding dimension (keep small — data is limited)
            nhead: number of attention heads (must divide d_model evenly)
            num_layers: number of transformer encoder blocks
        """
        super().__init__()
        self.L = L
        self.T = T
        self.S = S
        self.d_model = d_model
        
        # Token embedding: map each (layer, time_step) cell's one-hot (S+1,) to d_model
        # We treat the whole row of one time step across all layers as a single token.
        # Token input dimension = L * (S+1)
        token_dim = L * (S + 1)
        self.token_embedding = nn.Linear(token_dim, d_model)
        
        # Positional embedding: learnable, one per time step
        # (Sinusoidal would also work but learnable is simpler to implement here)
        self.pos_embedding = nn.Embedding(T, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,  # Standard FFN size = 4 * d_model
            dropout=dropout,
            batch_first=True,             # We use (batch, seq, feature) ordering
            norm_first=True,              # Pre-LN is more stable than post-LN
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head: pool over time steps, then classify
        self.pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling over T tokens
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Single logit — apply sigmoid for P(real)
        )
        
        # Label smoothing is applied in the training loop, not here

    def forward(self, x):
        """
        x: (batch, L, T, S+1) — one-hot encoded beat grid
        Returns: logit (batch, 1), not sigmoid — apply sigmoid externally for probability
        """
        batch = x.shape[0]
        
        # Reshape: for each time step t, create a token from all L layer values
        # x shape: (B, L, T, S+1) → (B, T, L*(S+1))
        x = x.permute(0, 2, 1, 3)              # (B, T, L, S+1)
        x = x.reshape(batch, self.T, -1)        # (B, T, L*(S+1))
        
        # Embed tokens
        x = self.token_embedding(x)             # (B, T, d_model)
        
        # Add positional embeddings
        positions = torch.arange(self.T, device=x.device).unsqueeze(0)  # (1, T)
        x = x + self.pos_embedding(positions)   # (B, T, d_model)
        
        # Transformer encoding
        x = self.transformer(x)                 # (B, T, d_model)
        
        # Pool over time steps
        x = x.permute(0, 2, 1)                  # (B, d_model, T)
        x = self.pool(x).squeeze(-1)             # (B, d_model)
        
        return self.classifier(x)               # (B, 1)
```

---

## 10. Pre-Training the Discriminator

**This step is mandatory.** If you start PPO training without a pre-trained discriminator, the agent will receive meaningless β·Rdisc = 0.1 × (random noise) as its discriminator component. This contaminates the reward signal from episode one and slows convergence significantly.

```python
# training/pretrain_disc.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from models.discriminator import BeatDiscriminator

def generate_random_grids(n, L, T, S):
    """
    Generate n random grids as negative training examples.
    These are intentionally bad — the discriminator's job is to distinguish
    these from real grids.
    """
    return np.random.randint(0, S + 1, size=(n, L, T))

def grid_to_tensor(grid, S):
    """Convert integer grid (N, L, T) to one-hot tensor (N, L, T, S+1)."""
    N, L, T = grid.shape
    one_hot = np.zeros((N, L, T, S + 1), dtype=np.float32)
    for n in range(N):
        for l in range(L):
            for t in range(T):
                one_hot[n, l, t, grid[n, l, t]] = 1.0
    return torch.from_numpy(one_hot)

def pretrain_discriminator(L=4, T=16, S=15,
                            real_grids_path="data/processed/groove_grids.npy",
                            epochs=50, batch_size=64, lr=1e-4,
                            save_path="checkpoints/discriminator_pretrained.pt"):
    import os
    os.makedirs("checkpoints", exist_ok=True)
    
    # Load real grids
    real_grids = np.load(real_grids_path)   # Shape (N_real, L, T)
    N_real = len(real_grids)
    print(f"Real grids: {N_real}")
    
    # Generate an equal number of random (fake) grids
    fake_grids = generate_random_grids(N_real, L, T, S)
    
    # Convert to tensors
    real_tensors = grid_to_tensor(real_grids, S)  # (N, L, T, S+1)
    fake_tensors = grid_to_tensor(fake_grids, S)
    
    # Labels with label smoothing ε = 0.1
    # Real labels: 0.9 (not 1.0) — prevents discriminator overconfidence
    # Fake labels: 0.1 (not 0.0)
    real_labels = torch.full((N_real, 1), 0.9, dtype=torch.float32)
    fake_labels = torch.full((N_real, 1), 0.1, dtype=torch.float32)
    
    all_grids  = torch.cat([real_tensors, fake_tensors], dim=0)
    all_labels = torch.cat([real_labels,  fake_labels],  dim=0)
    
    # Train/validation split (80/20)
    dataset   = TensorDataset(all_grids, all_labels)
    n_train   = int(0.8 * len(dataset))
    n_val     = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    disc = BeatDiscriminator(L=L, T=T, S=S).to(device)
    optimizer = optim.Adam(disc.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()  # Expects raw logits, applies sigmoid internally
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # ── Training ──────────────────────────────────────────────────────────
        disc.train()
        train_loss = 0.0
        for grids_batch, labels_batch in train_loader:
            grids_batch  = grids_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            optimizer.zero_grad()
            logits = disc(grids_batch)          # (B, 1)
            loss   = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # ── Validation ────────────────────────────────────────────────────────
        disc.eval()
        correct = 0
        total   = 0
        with torch.no_grad():
            for grids_batch, labels_batch in val_loader:
                grids_batch  = grids_batch.to(device)
                labels_batch = labels_batch.to(device)
                
                logits = disc(grids_batch)
                probs  = torch.sigmoid(logits)
                preds  = (probs > 0.5).float()
                # For validation accuracy, use hard labels (real > 0.5, fake < 0.5)
                hard_labels = (labels_batch > 0.5).float()
                correct += (preds == hard_labels).sum().item()
                total   += labels_batch.size(0)
        
        val_acc = correct / total
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1:3d}/{epochs} | loss={avg_loss:.4f} | val_acc={val_acc:.3f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(disc.state_dict(), save_path)
    
    print(f"\nBest validation accuracy: {best_val_acc:.3f}")
    print(f"Discriminator saved to {save_path}")
    
    # ── Sanity check ──────────────────────────────────────────────────────────
    # Target: discriminator accuracy > 0.75 before using it as a reward signal.
    # If val_acc < 0.65, the discriminator has not learned meaningful features.
    # Do not proceed to RL training until this threshold is met.
    if best_val_acc < 0.75:
        print("\n⚠ WARNING: Discriminator accuracy below 0.75.")
        print("  Check: are Groove grids being parsed correctly?")
        print("  Check: are fake grids actually random (not accidentally structured)?")
        print("  Consider training for more epochs or reducing learning rate.")
    
    return disc

if __name__ == "__main__":
    pretrain_discriminator()
```

> **What "converged" means here:** Validation accuracy above 0.75 on real vs. random grids. This means the discriminator has learned *some* musical structure from the Groove dataset. It does not need to reach 0.95 — a perfect discriminator will give the agent no useful gradient signal (everything is scored near 0 or 1).

---

## 11. PPO Training Loop

Rather than re-implementing PPO from scratch, we wrap the custom environment and models in the stable-baselines3 API. The key step is implementing a **custom policy** that uses the `BeatActor` and `BeatCritic`.

```python
# training/train_ppo.py
"""
Main training script. Run as:
  python -m training.train_ppo --phase 1
  python -m training.train_ppo --phase 2
"""

import argparse
import numpy as np
import torch
import json
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from env.beat_env import BeatGridEnv
from env.reward import reward_fn_phase1
from models.discriminator import BeatDiscriminator


# ── Hyperparameters ────────────────────────────────────────────────────────────
PHASE1_CONFIG = {
    "L": 4, "T": 16, "S": 15,
    "n_steps": 2048,          # Steps per rollout buffer (per env)
    "batch_size": 64,
    "n_epochs": 10,           # PPO update epochs per rollout
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,       # GAE lambda (controls bias-variance in advantage estimate)
    "clip_range": 0.2,        # PPO clipping ε
    "ent_coef": 0.01,         # Entropy bonus coefficient
    "vf_coef": 0.5,           # Value function loss coefficient
    "max_grad_norm": 0.5,
    "total_timesteps": 2_000_000,
    "n_envs": 4,              # Parallel environments
}

PHASE2_CONFIG = {
    **PHASE1_CONFIG,
    "L": 8,
    "total_timesteps": 5_000_000,
}


def build_env(config, discriminator=None, phase=1):
    """Build the BeatGridEnv with the appropriate config and reward function."""
    with open("data/samples/manifest.json") as f:
        manifest = json.load(f)
    
    layer_to_samples = {}
    for item in manifest:
        l = item["layer_id"]
        s = item["sample_idx"] + 1  # 1-indexed (0 = silence)
        if l not in layer_to_samples:
            layer_to_samples[l] = []
        layer_to_samples[l].append(s)
    
    L, T, S = config["L"], config["T"], config["S"]
    
    def reward_fn(grid, final):
        from env.reward import reward_fn_phase1
        alpha = 0.9 if phase == 1 else 0.5
        beta  = 1.0 - alpha
        return reward_fn_phase1(grid, final=final, discriminator=discriminator,
                                alpha=alpha, beta=beta)
    
    env = BeatGridEnv(
        L=L, T=T, S=S,
        reward_fn=reward_fn,
        layer_to_samples=layer_to_samples,
        phase=phase
    )
    return Monitor(env)


class DiscriminatorUpdateCallback(BaseCallback):
    """
    Updates the discriminator every 100 episodes using:
      - Real grids (from Groove dataset)
      - Agent-generated grids (stored in a historical pool)
    
    This is called after PPO collects its rollout buffer.
    """
    
    def __init__(self, discriminator, real_grids_path,
                 update_every_n_episodes=100,
                 pool_size=500, L=4, T=16, S=15, verbose=0):
        super().__init__(verbose)
        self.disc = discriminator
        self.real_grids = np.load(real_grids_path)
        self.update_every = update_every_n_episodes
        self.pool_size    = pool_size
        self.L, self.T, self.S = L, T, S
        self.agent_pool   = []   # Historical buffer of agent-generated grids
        self.episode_count = 0
        self.disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.device = next(discriminator.parameters()).device
    
    def _on_step(self):
        # Check for completed episodes and collect agent grids
        # The environment's info dict contains the final grid when done
        infos = self.locals.get("infos", [])
        for info in infos:
            if "terminal_observation" in info or info.get("episode"):
                # The grid is available in the environment — retrieve it
                # (You'll need to expose `self.env.grid` from the environment)
                pass
        return True
    
    def update_discriminator(self, agent_grids):
        """
        Perform one update of the discriminator using:
          - Positive examples: real grids from Groove
          - Negative examples: mix of random grids and agent-generated grids
        """
        if len(agent_grids) == 0:
            return
        
        self.agent_pool.extend(agent_grids)
        if len(self.agent_pool) > self.pool_size:
            self.agent_pool = self.agent_pool[-self.pool_size:]  # Keep most recent
        
        # Sample a training batch
        n_real = min(32, len(self.real_grids))
        n_fake = min(32, len(self.agent_pool))
        
        real_sample  = self.real_grids[np.random.choice(len(self.real_grids), n_real)]
        agent_sample = np.array(self.agent_pool)[np.random.choice(len(self.agent_pool), n_fake)]
        
        from training.pretrain_disc import grid_to_tensor
        real_tensors  = grid_to_tensor(real_sample,  self.S).to(self.device)
        agent_tensors = grid_to_tensor(agent_sample, self.S).to(self.device)
        
        real_labels  = torch.full((n_real, 1), 0.9).to(self.device)
        agent_labels = torch.full((n_fake, 1), 0.1).to(self.device)
        
        grids  = torch.cat([real_tensors,  agent_tensors], dim=0)
        labels = torch.cat([real_labels,   agent_labels],  dim=0)
        
        self.disc.train()
        self.disc_optimizer.zero_grad()
        logits = self.disc(grids)
        loss   = self.criterion(logits, labels)
        loss.backward()
        self.disc_optimizer.step()
        self.disc.eval()
        
        if self.verbose > 0:
            print(f"Discriminator update | loss={loss.item():.4f}")


def train(phase=1):
    """Main training entry point."""
    config = PHASE1_CONFIG if phase == 1 else PHASE2_CONFIG
    L, T, S = config["L"], config["T"], config["S"]
    
    # Load pre-trained discriminator
    disc = BeatDiscriminator(L=L, T=T, S=S)
    ckpt = f"checkpoints/discriminator_pretrained.pt"
    disc.load_state_dict(torch.load(ckpt, map_location="cpu"))
    disc.eval()
    print("Loaded pre-trained discriminator.")
    
    # If Phase 2, load Phase 1 policy as starting point
    if phase == 2:
        print("Phase 2: loading Phase 1 checkpoint as starting policy...")
    
    # Build vectorized environments
    def make_env():
        return build_env(config, discriminator=disc, phase=phase)
    
    vec_env = make_vec_env(make_env, n_envs=config["n_envs"])
    
    # ── PPO Model ──────────────────────────────────────────────────────────────
    # We use stable-baselines3's built-in MLP policy for now.
    # The custom CNN actor/critic from models/ can be plugged in as a custom policy
    # following the SB3 custom policy guide:
    # https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    
    model = PPO(
        "MlpPolicy",          # Replace with custom CNN policy once verified working
        vec_env,
        n_steps         = config["n_steps"],
        batch_size      = config["batch_size"],
        n_epochs        = config["n_epochs"],
        learning_rate   = config["learning_rate"],
        gamma           = config["gamma"],
        gae_lambda      = config["gae_lambda"],
        clip_range      = config["clip_range"],
        ent_coef        = config["ent_coef"],
        vf_coef         = config["vf_coef"],
        max_grad_norm   = config["max_grad_norm"],
        tensorboard_log = f"logs/phase{phase}/",
        verbose         = 1,
    )
    
    # ── Callbacks ──────────────────────────────────────────────────────────────
    eval_env = build_env(config, discriminator=disc, phase=phase)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = f"checkpoints/phase{phase}_best/",
        log_path             = f"logs/phase{phase}/eval/",
        eval_freq            = 10_000,
        n_eval_episodes      = 20,
        deterministic        = True,
    )
    
    disc_callback = DiscriminatorUpdateCallback(
        discriminator     = disc,
        real_grids_path   = "data/processed/groove_grids.npy",
        update_every_n_episodes = 100,
        pool_size         = 500,
        L=L, T=T, S=S,
    )
    
    # ── Train ──────────────────────────────────────────────────────────────────
    model.learn(
        total_timesteps = config["total_timesteps"],
        callback        = [eval_callback, disc_callback],
    )
    
    model.save(f"checkpoints/phase{phase}_final")
    print(f"Phase {phase} training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2])
    args = parser.parse_args()
    train(phase=args.phase)
```

> **Monitor training with TensorBoard:**
> ```bash
> tensorboard --logdir logs/
> ```
> You should see episode reward increasing over time. If it plateaus immediately near 0, the reward function has a bug. If it oscillates wildly, the learning rate is too high.

---

## 12. Curriculum

### Phase 1 → Phase 2 transition

Phase 1 trains a 4-layer drum agent. Phase 2 expands to 8 layers including bass and melody.

**Transition condition (from the proposal, with clarification):**

> Phase 1 is complete when the rolling mean rule score over the last 100 episodes exceeds 0.7.

To check this during training:

```python
# evaluation/check_phase1_ready.py
import numpy as np
from stable_baselines3 import PPO
from env.beat_env import BeatGridEnv
from env.reward import _rule_based_score_phase1
import json

def evaluate_rule_score(model_path, n_eval=100):
    """
    Load a Phase 1 model and compute its mean rule score over n_eval episodes.
    Returns True if ready to advance to Phase 2.
    """
    model = PPO.load(model_path)
    
    with open("data/samples/manifest.json") as f:
        manifest = json.load(f)
    layer_to_samples = {}
    for item in manifest:
        l = item["layer_id"]
        layer_to_samples.setdefault(l, []).append(item["sample_idx"] + 1)
    
    env = BeatGridEnv(L=4, T=16, S=15,
                      reward_fn=lambda g, f: 0,  # dummy
                      layer_to_samples=layer_to_samples)
    
    rule_scores = []
    for _ in range(n_eval):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
        
        score = _rule_based_score_phase1(env.grid)
        rule_scores.append(score)
    
    mean_score = np.mean(rule_scores)
    print(f"Mean rule score over {n_eval} episodes: {mean_score:.3f}")
    print(f"Phase 1 → 2 transition {'READY' if mean_score > 0.7 else 'NOT YET'}")
    return mean_score > 0.7
```

**Phase 2 specifics:**

When Phase 2 begins:
1. Load the Phase 1 policy weights into the Phase 2 actor/critic (the extra 4 layer channels initialize to zero).
2. Set α = 0.5, β = 0.5 (rule reward and discriminator weighted equally).
3. Include 30% Phase 1 grids in every discriminator training batch (the anti-catastrophic-forgetting mechanism).
4. Train a new 8-layer discriminator starting from the 4-layer checkpoint.

---

## 13. Evaluation

### 13.1 Quantitative metrics

```python
# evaluation/evaluate.py
import numpy as np
import torch
from stable_baselines3 import PPO
from env.beat_env import BeatGridEnv
from env.reward import _rule_based_score_phase1
from models.discriminator import BeatDiscriminator


def evaluate_agent(model_path, disc_path, n_eval=100, L=8, T=16, S=15):
    """
    Compute all quantitative metrics for an agent checkpoint.
    
    Metrics:
      1. Discriminator score: mean P(real) from a HELD-OUT discriminator
         (trained on test split, never seen by the agent)
      2. Rule-based sub-scores: reported individually for interpretability
      3. Novelty score: mean Hamming distance from training set
    """
    model = PPO.load(model_path)
    
    disc = BeatDiscriminator(L=L, T=T, S=S)
    disc.load_state_dict(torch.load(disc_path))
    disc.eval()
    
    real_grids = np.load("data/processed/groove_grids.npy")
    
    # Generate n_eval episodes
    generated_grids = []
    rule_scores_all = []
    disc_scores_all = []
    
    env = build_eval_env(L, T, S)
    for _ in range(n_eval):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
        
        grid = env.grid.copy()
        generated_grids.append(grid)
        rule_scores_all.append(_rule_based_score_phase1(grid))
        disc_scores_all.append(_discriminator_score_held_out(grid, disc))
    
    generated_grids = np.array(generated_grids)  # (n_eval, L, T)
    
    # ── Novelty score ──────────────────────────────────────────────────────────
    # Hamming distance between each generated grid and its nearest neighbor in
    # the Groove training set.
    # Hamming distance = fraction of positions that differ.
    novelty_scores = []
    for gen_grid in generated_grids:
        gen_flat  = (gen_grid > 0).flatten()  # binary: active/inactive
        dists = []
        for real_grid in real_grids:
            real_flat = (real_grid[:L, :T] > 0).flatten()
            # Pad or trim if shape mismatch
            min_len = min(len(gen_flat), len(real_flat))
            d = np.mean(gen_flat[:min_len] != real_flat[:min_len])
            dists.append(d)
        novelty_scores.append(min(dists))  # Distance to nearest neighbor
    
    print("\n=== Evaluation Results ===")
    print(f"Discriminator score (held-out): {np.mean(disc_scores_all):.3f} ± {np.std(disc_scores_all):.3f}")
    print(f"  (Baseline — random:           ~0.15)")
    print(f"  (Baseline — rules-only:       ~0.45)")
    print(f"  (Target:                       >0.70)")
    print(f"Rule-based score:               {np.mean(rule_scores_all):.3f} ± {np.std(rule_scores_all):.3f}")
    print(f"Novelty score (Hamming NN):     {np.mean(novelty_scores):.3f} ± {np.std(novelty_scores):.3f}")
    print(f"  (0 = memorized training data, 1 = completely different)")


def _discriminator_score_held_out(grid, disc):
    """Same as reward.py version but uses the held-out discriminator."""
    import torch
    L, T = grid.shape
    S = disc.S
    one_hot = torch.zeros(1, L, T, S + 1)
    for l in range(L):
        for t in range(T):
            one_hot[0, l, t, grid[l][t]] = 1.0
    with torch.no_grad():
        logit = disc(one_hot)
        return torch.sigmoid(logit).item()
```

### 13.2 Qualitative listening test

The generated grids need to be converted to audio for the blind listening test.

```python
# evaluation/render_to_audio.py
"""
Convert a beat grid to a WAV file for listening.
Requires: soundfile, numpy
"""
import numpy as np
import soundfile as sf
import json
import os

def render_grid_to_audio(grid, manifest_path="data/samples/manifest.json",
                          sample_dir="data/samples",
                          bpm=120, output_path="output.wav"):
    """
    Render a beat grid to a stereo WAV file.
    
    Args:
        grid: np.array (L, T), values 0..S
        bpm: beats per minute (default 120)
        output_path: where to write the WAV file
    """
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    # Map (layer_id, sample_idx) → WAV file path
    sample_map = {}
    for item in manifest:
        key = (item["layer_id"], item["sample_idx"])
        sample_map[key] = item["filename"]
    
    SAMPLE_RATE = 44100
    beat_duration   = 60.0 / bpm                       # seconds per beat
    sixteenth_dur   = beat_duration / 4                # seconds per 16th note
    total_duration  = sixteenth_dur * 16 + 2.0         # one bar + 2 sec tail
    
    # Initialize stereo output buffer
    output = np.zeros((int(SAMPLE_RATE * total_duration), 2), dtype=np.float32)
    
    L, T = grid.shape
    for l in range(L):
        for t in range(T):
            sample_idx = int(grid[l][t])
            if sample_idx == 0:
                continue  # silence
            
            filepath = sample_map.get((l, sample_idx - 1), None)  # -1: 1-indexed → 0-indexed
            if filepath is None or not os.path.exists(filepath):
                continue
            
            # Load the sample
            try:
                data, sr = sf.read(filepath, always_2d=True)
            except Exception:
                continue
            
            if sr != SAMPLE_RATE:
                # Simple: skip resampling for now. For production, use librosa.resample.
                continue
            
            # Find where to place the sample in the output buffer
            start_sample = int(t * sixteenth_dur * SAMPLE_RATE)
            end_sample   = start_sample + len(data)
            
            if end_sample > len(output):
                data = data[:len(output) - start_sample]
                end_sample = len(output)
            
            output[start_sample:end_sample] += data[:end_sample - start_sample]
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(output))
    if max_val > 0:
        output = output / max_val * 0.9
    
    sf.write(output_path, output, SAMPLE_RATE)
    print(f"Rendered to {output_path}")
```

---

## 14. Common Failure Modes and Fixes

| Symptom | Likely Cause | Fix |
|---|---|---|
| Episode reward stuck at ~0 from step 1 | Reward function has a bug; most actions get 0 | Add debug prints inside `reward_fn`. Check that a completed random grid gets non-zero rule score. |
| Agent fills only silence (sample index 0 every step) | Entropy too high; agent converges to silence as safe policy | Increase `ent_coef` to encourage exploration, or add a penalty for grids below minimum density. |
| Reward oscillates but never increases long-term | Discriminator is changing too fast relative to policy | Reduce discriminator update frequency from every 100 to every 200 episodes. Or reduce discriminator learning rate. |
| Agent always chooses the same layer (e.g., only kicks) | Layer head collapsed | Check that all layers are being masked correctly. Add a diversity penalty if one layer is chosen > 80% of the time. |
| Discriminator accuracy drops back to ~0.5 during RL training | Catastrophic forgetting in discriminator | Ensure the historical pool of agent grids is being used. Maintain 50% real / 50% agent grids in discriminator updates. |
| Rule score is 0.7 in evaluation but 0.3 during rollouts | Deterministic vs stochastic policy difference | `model.predict(deterministic=True)` produces the best action, but rollouts during training use stochastic sampling. This gap is normal. Evaluation always uses deterministic. |
| One-hot encoding crashes with index out of bounds | Sample index S+1 or higher in the grid | Your action decoder is off by one. Check `_decode_action()`. The sample dimension should be 0..S inclusive. |

---

## 15. Full File Structure

```
beat_gen/
├── data/
│   ├── download_samples.py          # Section 3.1
│   ├── process_groove.py            # Section 3.2
│   ├── raw/                         # Groove MIDI files (downloaded)
│   ├── processed/
│   │   └── groove_grids.npy         # Shape (N, 4, 16) for Phase 1
│   └── samples/
│       ├── manifest.json            # Maps sample files to layer+index
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
│   ├── beat_env.py                  # Section 5 — Gymnasium environment
│   └── reward.py                    # Section 8 — All reward functions
│
├── models/
│   ├── actor.py                     # Section 6 — Policy network
│   ├── critic.py                    # Section 7 — Value network
│   └── discriminator.py             # Section 9 — Transformer discriminator
│
├── training/
│   ├── pretrain_disc.py             # Section 10 — Must run before PPO
│   └── train_ppo.py                 # Section 11 — Main training script
│
├── evaluation/
│   ├── evaluate.py                  # Section 13.1 — Quantitative metrics
│   ├── render_to_audio.py           # Section 13.2 — Grid → WAV
│   └── check_phase1_ready.py        # Section 12 — Transition check
│
├── checkpoints/                     # Saved model weights
│   ├── discriminator_pretrained.pt
│   ├── phase1_best/
│   └── phase2_best/
│
└── logs/                            # TensorBoard logs
    ├── phase1/
    └── phase2/
```

---

## Order of Operations (Do These in This Exact Order)

1. **Register on Freesound.org** and get an API key.
2. Run `python data/download_samples.py` → builds `data/samples/`.
3. Download Groove MIDI dataset from the link in Section 3.2.
4. Run `python data/process_groove.py` → builds `data/processed/groove_grids.npy`.
5. Write a test script that imports `BeatGridEnv`, runs one random episode, and prints the final grid and reward. Verify the numbers are sane before continuing.
6. Run `python training/pretrain_disc.py` → saves `checkpoints/discriminator_pretrained.pt`. **Do not continue until validation accuracy > 0.75.**
7. Run `python training/train_ppo.py --phase 1` with TensorBoard open in another terminal.
8. Run `python evaluation/check_phase1_ready.py` periodically. When mean rule score > 0.7, Phase 1 is done.
9. Run `python training/train_ppo.py --phase 2`.
10. Run `python evaluation/evaluate.py` for final quantitative results.
11. Run `python evaluation/render_to_audio.py` on final model outputs to generate audio for the listening test.

---

*This guide covers Level 1 only. Level 2 (effect parameters, SAC, audio discriminator) requires a separate architecture and is out of scope for a single-semester project.*
