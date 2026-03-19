import numpy as np
from grid_env.beat_env import BeatGridEnv

# Dummy reward function to bypass the real one during testing
def dummy_reward_fn_phase1(grid, final=False):
    return 1.0 if final else 0.1

def run_sanity_check():
    # 4 layers (Phase 1), 15 samples each (1-indexed, 0 is silence)
    layer_to_samples = {
        0: list(range(1, 16)), 
        1: list(range(1, 16)),
        2: list(range(1, 16)), 
        3: list(range(1, 16))
    }
    
    env = BeatGridEnv(
        L=4, 
        T=16, 
        S=15, 
        reward_fn=dummy_reward_fn_phase1,
        layer_to_samples=layer_to_samples,
        phase=1
    )
    
    obs, info = env.reset()
    
    # Expected shape: 4 layers * 16 steps * 16 (15 samples + 1 silence) = 1024
    print(f"Observation shape: {obs.shape}") 
    assert obs.shape == (1024,), f"Expected shape (1024,), got {obs.shape}"
    print("Observation shape check passed.")

    # Run a full episode with random actions
    print("\nRunning random actions for 64 steps...")
    for step in range(64):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"Episode terminated precisely at step {info['step_count']}.")
            print(f"Final reward: {reward}")
            break

    assert info['step_count'] == 64, f"Episode terminated at wrong step count: {info['step_count']}"
    print("Termination logic check passed. The environment is rock solid.")

if __name__ == "__main__":
    run_sanity_check()