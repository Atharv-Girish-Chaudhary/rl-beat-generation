import numpy as np
import torch
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
    action_coord: Optional[tuple] = None,
    phase: int = 1, 
    discriminator=None, 
    alpha: float = 0.9, 
    beta: float = 0.1
) -> float:
    """
    Master reward function. 
    Intermediate steps receive a lightweight heuristic score based purely on the Delta (current action).
    Terminal steps receive the full theoretical and discriminator evaluation.
    """
    if not final:
        return _fast_intermediate_reward(grid, action_coord, phase)
        
    # Terminal Episode Evaluation
    r_drums = _evaluate_drums(grid)
    
    if phase == 1:
        r_rules = r_drums
    else:
        r_melodic = _evaluate_melodic_elements(grid)
        r_rules = (r_drums + r_melodic) / 2.0

    # Discriminator Evaluation
    r_disc = 0.0
    if discriminator is not None:
        r_disc = _get_discriminator_score(grid, discriminator)

    # Final weighted sum in [0, 1]
    return float((alpha * r_rules) + (beta * r_disc))


def _fast_intermediate_reward(grid: np.ndarray, action_coord: tuple, phase: int) -> float:
    """
    Evaluates the immediate Delta (action) to prevent infinite reward farming.
    """
    if action_coord is None:
        return 0.0
        
    reward = 0.0
    layer, time_step = action_coord
    
    # 1. The Anchor (Step 0 Kick)
    if layer == KICK and time_step == 0 and grid[layer, time_step] > 0: 
        reward += 0.05
        
    # 2. The Backbeat (Steps 4, 12 Snare/Clap)
    if layer in [SNARE, CLAP] and time_step in [4, 12] and grid[layer, time_step] > 0:
        reward += 0.05
        
    return float(reward)


def _evaluate_drums(grid: np.ndarray) -> float:
    T = grid.shape[1]
    kick_active = grid[KICK] > 0
    snare_active = grid[SNARE] > 0
    clap_active = grid[CLAP] > 0
    
    # Handle environment arrays of size L=4 or L=8 safely
    hihat_active = grid[HIHAT] > 0 if grid.shape[0] > HIHAT else np.zeros(T, dtype=bool)
    
    score = 0.0
    
    if kick_active[0]: score += 0.1
    if kick_active[8]: score += 0.1
    
    if snare_active[4] or clap_active[4]: score += 0.1
    if snare_active[12] or clap_active[12]: score += 0.1
    
    off_beats = [t for t in range(T) if t not in [4, 12]]
    off_beat_hits = np.sum(snare_active[off_beats]) + np.sum(clap_active[off_beats])
    score -= (off_beat_hits * 0.01)
    
    hat_count = np.sum(hihat_active)
    if 4 <= hat_count <= 12:
        score += 0.2 
    
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
    if grid.shape[0] < 8:
        return 0.0 # Phase 1 safety explicitly
        
    kick_active = grid[KICK] > 0
    bass_active = grid[BASS] > 0
    pad_active = grid[PAD] > 0
    fx_active = grid[FX] > 0
    
    score = 0.0
    
    total_bass = np.sum(bass_active)
    if total_bass > 0:
        simultaneous = np.sum(kick_active & bass_active)
        lock_ratio = simultaneous / total_bass
        if lock_ratio >= 0.5:
            score += 0.4
        else:
            score += (lock_ratio * 0.4)
            
    pad_count = np.sum(pad_active)
    fx_count = np.sum(fx_active)
    
    if pad_count <= 2: score += 0.15
    else: score -= (pad_count - 2) * 0.1 
    
    if fx_count <= 1: score += 0.15
    else: score -= (fx_count - 1) * 0.1   
    
    active_per_step = np.sum(grid > 0, axis=0)
    violations = np.sum(active_per_step > 4)
    
    if violations == 0:
        score += 0.3
    else:
        score -= (violations * 0.1)
        
    return float(np.clip(score, 0.0, 1.0))


def _get_discriminator_score(grid: np.ndarray, discriminator) -> float:
    """
    Mathematically compresses the S-sample grid down to a simple binary Hit/No-Hit tensor 
    perfectly matching the Discriminator's trained parameters (Batch, L, T).
    """
    discriminator.eval()
    with torch.no_grad():
        # Compress any Silence (0) or Unplayed (-1) into 0.0 (No Hit)
        # Any actual Sample (1-15) becomes 1.0 (Hit)
        binary_grid = (torch.tensor(grid) > 0).float()
        
        # Add the Batch dimension (B, L, T) -> (1, L, T)
        binary_grid = binary_grid.unsqueeze(0)
        
        # Pass safely through the token embeddings
        device = next(discriminator.parameters()).device
        logit, _ = discriminator(binary_grid.to(device))
        
        # Since we violently amputated nn.Sigmoid() from the Discriminator architecture, 
        # we perfectly apply it here ONCE to squash the raw logit into a valid [0, 1] reward prob.
        prob = torch.sigmoid(logit).item()
        
    return float(prob)