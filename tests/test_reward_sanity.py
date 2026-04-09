import numpy as np
from beat_rl.env.reward import compute_reward, _evaluate_drums, KICK, SNARE, HIHAT, CLAP

def test_reward_sanity():
    # 4 instruments (L=4), 16 steps (T=16)
    L = 4
    T = 16
    
    # 1. grid_all_ones: all 4 instruments firing on all 16 steps
    grid_all_ones = np.ones((L, T))
    
    # 2. grid_musical: 
    # kick on steps 0,4,8,12 — snare on 4,12 — hihat on even steps — clap on 8
    grid_musical = np.zeros((L, T))
    grid_musical[KICK, [0, 4, 8, 12]] = 1
    grid_musical[SNARE, [4, 12]] = 1
    grid_musical[HIHAT, np.arange(0, 16, 2)] = 1
    grid_musical[CLAP, [8]] = 1
    
    # 3. grid_empty: all zeros
    grid_empty = np.zeros((L, T))
    
    grids = {
        "all_ones": grid_all_ones,
        "musical": grid_musical,
        "empty": grid_empty
    }
    
    rewards = {}
    
    for name, grid in grids.items():
        raw_score = _evaluate_drums(grid)
        # compute_reward with final=True, no discriminator (default)
        reward = compute_reward(grid, final=True)
        rewards[name] = reward
        print(f"Grid: {name:10} | Raw Drum Score: {raw_score:6.3f} | Final Reward: {reward:6.3f}")
        
    assert rewards["all_ones"] < 0, f"Expected reward for 'all_ones' to be < 0, got {rewards['all_ones']}"
    assert rewards["musical"] > rewards["all_ones"], "Expected 'musical' to have higher reward than 'all_ones'"
    assert rewards["musical"] > 0.5, f"Expected 'musical' reward > 0.5, got {rewards['musical']}"

if __name__ == "__main__":
    test_reward_sanity()
