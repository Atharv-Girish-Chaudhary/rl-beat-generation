import numpy as np
from typing import Optional

# Layer Index Constants
KICK = 0
SNARE = 1
HIHAT = 2
CLAP = 3
BASS = 4
MELODY = 5
PAD = 6
FX = 7

def compute_reward(
    grid: np.ndarray, 
    final: bool = False, 
    phase: int = 1, 
    discriminator=None, 
    alpha: float = 0.9, 
    beta: float = 0.1
) -> float:
    """
    Master reward function. 
    Intermediate steps receive a lightweight heuristic score.
    Terminal steps receive the full theoretical and discriminator evaluation.
    """
    if not final:
        # Dense intermediate reward to guide the agent during the rollout
        return _fast_intermediate_reward(grid, phase)
        
    # Terminal Episode Evaluation
    r_drums = _evaluate_drums(grid)
    
    if phase == 1:
        r_rules = r_drums
    else:
        r_melodic = _evaluate_melodic_elements(grid)
        # Phase 2 rule score is the mean of drum competence and melodic competence
        r_rules = (r_drums + r_melodic) / 2.0

    # Discriminator Evaluation
    r_disc = 0.0
    if discriminator is not None:
        r_disc = _get_discriminator_score(grid, discriminator)

    # Final weighted sum in [0, 1]
    return float((alpha * r_rules) + (beta * r_disc))


def _fast_intermediate_reward(grid: np.ndarray, phase: int) -> float:
    """
    O(1) complexity checks to provide a breadcrumb trail for the PPO agent.
    If the agent places a Kick on 0 or a Snare on 4/12, reward it immediately.
    """
    reward = 0.0
    L = grid.shape[0]
    
    # 1. The Anchor (Step 0)
    if grid[KICK, 0] > 0: 
        reward += 0.05
        
    # 2. The Backbeat (Steps 4, 12)
    if L > SNARE and (grid[SNARE, 4] > 0 or grid[CLAP, 4] > 0):
        reward += 0.05
    if L > SNARE and (grid[SNARE, 12] > 0 or grid[CLAP, 12] > 0):
        reward += 0.05
        
    return reward


def _evaluate_drums(grid: np.ndarray) -> float:
    """
    Core 4/4 Drum Theory (Phase 1 & Phase 2 Foundation).
    Evaluates KICK, SNARE, HIHAT, CLAP.
    """
    T = grid.shape[1]
    kick_active = grid[KICK] > 0
    snare_active = grid[SNARE] > 0
    clap_active = grid[CLAP] > 0
    hihat_active = grid[HIHAT] > 0
    
    score = 0.0
    
    # --- Rhythmic Structure (0.0 to 0.4) ---
    # The "One" must exist
    if kick_active[0]: score += 0.1
    # Kick on step 8 establishes the half-bar groove
    if kick_active[8]: score += 0.1
    
    # Backbeat strictly on 4 and 12
    if snare_active[4] or clap_active[4]: score += 0.1
    if snare_active[12] or clap_active[12]: score += 0.1
    
    # Punish chaotic snare/clap placements on off-beats
    off_beats = [t for t in range(T) if t not in [4, 12]]
    off_beat_hits = np.sum(snare_active[off_beats]) + np.sum(clap_active[off_beats])
    score -= (off_beat_hits * 0.05)
    
    # --- Hi-Hat Momentum (0.0 to 0.2) ---
    hat_count = np.sum(hihat_active)
    if 4 <= hat_count <= 12:
        score += 0.2  # Rewards a steady 8th or 16th note pulse with some gaps
    
    # --- Repetition with Variation (0.0 to 0.4) ---
    # Good beats repeat, but not exactly [cite: 692-693].
    first_half = grid[:4, :8] > 0
    second_half = grid[:4, 8:] > 0
    
    intersection = np.sum(first_half & second_half)
    union = np.sum(first_half | second_half)
    
    if union > 0:
        jaccard = intersection / union
        if 0.6 <= jaccard < 0.95:
            score += 0.4
        elif jaccard >= 0.95:
            score -= 0.2  # Punish robotic, exact copying
            
    return float(np.clip(score, 0.0, 1.0))


def _evaluate_melodic_elements(grid: np.ndarray) -> float:
    """
    Harmonic and Spectral Rules (Phase 2 Only).
    Evaluates BASS, MELODY, PAD, FX interactions.
    """
    kick_active = grid[KICK] > 0
    bass_active = grid[BASS] > 0
    pad_active = grid[PAD] > 0
    fx_active = grid[FX] > 0
    melody_active = grid[MELODY] > 0
    
    score = 0.0
    
    # --- Kick & Bass Interlock (0.0 to 0.4) ---
    total_bass = np.sum(bass_active)
    if total_bass > 0:
        simultaneous = np.sum(kick_active & bass_active)
        lock_ratio = simultaneous / total_bass
        if lock_ratio >= 0.5:
            score += 0.4
        else:
            score += (lock_ratio * 0.4)
            
    # --- Pad & FX Sparsity (0.0 to 0.3) ---
    pad_count = np.sum(pad_active)
    fx_count = np.sum(fx_active)
    
    if pad_count <= 2: score += 0.15
    else: score -= (pad_count - 2) * 0.1  # Muddy mix penalty
    
    if fx_count <= 1: score += 0.15
    else: score -= (fx_count - 1) * 0.1   # Distracting transitions penalty
    
    # --- Global Clutter/Density Constraint (0.0 to 0.3) ---
    # Humans only have two hands. Punish > 4 instruments at once.
    active_per_step = np.sum(grid > 0, axis=0)
    violations = np.sum(active_per_step > 4)
    
    if violations == 0:
        score += 0.3
    else:
        score -= (violations * 0.1)
        
    return float(np.clip(score, 0.0, 1.0))

def _get_discriminator_score(grid: np.ndarray, discriminator) -> float:
    """
    Runs the Transformer discriminator to return P(real).
    Requires wrapping the grid to a PyTorch tensor.
    """
    import torch
    discriminator.eval()
    with torch.no_grad():
        L, T = grid.shape
        S = discriminator.S
        one_hot = torch.zeros(1, L, T, S + 1, device=next(discriminator.parameters()).device)
        for l in range(L):
            for t in range(T):
                one_hot[0, l, t, grid[l, t]] = 1.0
        
        logit = discriminator(one_hot)
        prob = torch.sigmoid(logit).item()
    return float(prob)