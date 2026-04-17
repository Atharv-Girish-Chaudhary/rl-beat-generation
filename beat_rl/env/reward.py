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
    """
    Evaluates the four melodic/harmonic layers in Phase 2 (8×16) grids.

    Sub-scores and weights:
      bass_lock    (0.25) — bass locks rhythmically to kick
      melody_groove (0.25) — melody hits align to strong beats, correct density
      pad_density  (0.15) — pad stays sparse (target 2–4 hits)
      fx_density   (0.10) — fx stays very sparse (target 1–2 hits)
      simul_hits   (0.25) — penalise vertical overcrowding (>4 layers active per step)
    """
    if grid.shape[0] < 8:
        return 0.0  # Phase 1 safety guard

    STRONG_BEATS = [0, 4, 8, 12]
    T = grid.shape[1]

    kick_active   = grid[KICK]   > 0
    bass_active   = grid[BASS]   > 0
    melody_active = grid[MELODY] > 0
    pad_active    = grid[PAD]    > 0
    fx_active     = grid[FX]     > 0

    # ── Sub-score 1: Bass lock to kick  (weight 0.25) ─────────────────────────
    # Rewards bass hits that coincide with kick hits.
    # lock_ratio = (kick∩bass) / total_bass.  Full credit at ≥50% alignment.
    total_bass = np.sum(bass_active)
    if total_bass > 0:
        simultaneous_bass = np.sum(kick_active & bass_active)
        lock_ratio = simultaneous_bass / total_bass
        bass_lock_score = min(lock_ratio / 0.5, 1.0)  # linear up to 50%, capped at 1
    else:
        bass_lock_score = 0.0

    # ── Sub-score 2: Melody groove  (weight 0.25) ─────────────────────────────
    # Rewards melody hits on strong beats (0,4,8,12).
    # Penalises off-beat melody hits; targets 2–6 total melody hits.
    total_melody = np.sum(melody_active)
    strong_melody_hits = np.sum(melody_active[STRONG_BEATS])
    off_beat_melody_hits = total_melody - strong_melody_hits

    # Density bonus: full credit for 2–6 hits, zero for 0 or >10
    if 2 <= total_melody <= 6:
        density_bonus = 1.0
    elif total_melody < 2:
        density_bonus = total_melody / 2.0          # partial if too few
    else:
        density_bonus = max(0.0, 1.0 - (total_melody - 6) * 0.15)  # decay above 6

    # Strong-beat alignment ratio (1.0 if every melody hit is on a strong beat)
    if total_melody > 0:
        alignment = strong_melody_hits / total_melody
    else:
        alignment = 0.0

    # Off-beat penalty: -0.05 per off-beat hit
    off_beat_penalty = min(off_beat_melody_hits * 0.05, 0.5)

    melody_groove_score = np.clip(
        0.5 * density_bonus + 0.5 * alignment - off_beat_penalty,
        0.0, 1.0
    )

    # ── Sub-score 3: Pad density  (weight 0.15) ───────────────────────────────
    # Target: 2–4 pad hits per bar.  Exponential penalty above threshold.
    pad_count = int(np.sum(pad_active))
    if pad_count <= 4:
        pad_density_score = 1.0 if pad_count >= 2 else pad_count / 2.0
    else:
        excess = pad_count - 4
        pad_density_score = max(0.0, 1.0 - (1 - 0.5 ** excess))  # halves per extra hit

    # ── Sub-score 4: FX density  (weight 0.10) ────────────────────────────────
    # Target: 1–2 fx hits.  Exponential penalty above threshold.
    fx_count = int(np.sum(fx_active))
    if fx_count <= 2:
        fx_density_score = 1.0 if fx_count >= 1 else 0.0
    else:
        excess = fx_count - 2
        fx_density_score = max(0.0, 1.0 - (1 - 0.5 ** excess))  # halves per extra hit

    # ── Sub-score 5: Simultaneous-hit penalty  (weight 0.25) ──────────────────
    # Penalise steps where >4 of the 8 layers are active at once (mud).
    active_per_step = np.sum(grid > 0, axis=0)
    violations = int(np.sum(active_per_step > 4))
    if violations == 0:
        simul_score = 1.0
    else:
        simul_score = max(0.0, 1.0 - violations * 0.1)

    # ── Weighted total ─────────────────────────────────────────────────────────
    score = (
        0.25 * bass_lock_score
        + 0.25 * melody_groove_score
        + 0.15 * pad_density_score
        + 0.10 * fx_density_score
        + 0.25 * simul_score
    )

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