import sys
import os
import numpy as np

# Force the Python path to recognize the root directory so imports work perfectly
sys.path.append(os.getcwd())

from grid_env.reward import compute_reward

def test_reward_sanity_check():
    print("--- Testing Reward Function (Phase 1) ---")
    
    # 1. Construct a theoretically perfect Phase 1 drum beat WITH variation
    perfect_grid = np.zeros((4, 16), dtype=np.int64)
    # Kicks on strong beats, PLUS a syncopated variation kick on step 10
    perfect_grid[0, [0, 4, 8, 10, 12]] = 1 
    # Snares locked perfectly on the backbeats (4, 12)
    perfect_grid[1, [4, 12]] = 2
    # Hi-hats driving a consistent 8th-note pulse
    perfect_grid[2, [0, 2, 4, 6, 8, 10, 12, 14]] = 3 
    
    perfect_score = compute_reward(perfect_grid, final=True, phase=1)
    print(f"Perfect drum beat score: {perfect_score:.4f} (Expected ~0.9000)")
    assert perfect_score > 0.8, "FAIL: Reward function did not highly score a perfect beat."
    
    # 2. Construct absolute random noise (density violation, backbeat missed, etc.)
    garbage_grid = np.random.randint(1, 16, size=(4, 16))
    garbage_score = compute_reward(garbage_grid, final=True, phase=1)
    print(f"Garbage beat score: {garbage_score:.4f}")
    assert garbage_score < 0.3, "FAIL: Reward function failed to severely penalize noise."
    
    # 3. Construct an empty grid (silence)
    empty_grid = np.zeros((4, 16), dtype=np.int64)
    empty_score = compute_reward(empty_grid, final=True, phase=1)
    print(f"Empty grid score: {empty_score:.4f}")
    assert empty_score < 0.5, "FAIL: Reward function is rewarding silence too highly."
    
    print("\nSUCCESS: Reward landscape is shaping correctly. Math is verified.")

if __name__ == "__main__":
    run_reward_sanity_check()