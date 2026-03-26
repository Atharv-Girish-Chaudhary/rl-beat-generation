import numpy as np
import sys
import os

# Ensure the root project is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from grid_env.beat_env import BeatGridEnv

def test_env_initialization():
    env = BeatGridEnv(L=4, T=16, S=15)
    obs, info = env.reset()
    
    # Check observation shape
    expected_dim = 4 * 16 * (15 + 2)
    assert obs.shape == (expected_dim,), f"Expected shape {(expected_dim,)}, got {obs.shape}"
    
    # Check time channel initialization (should be strictly 0.0)
    obs_3d = obs.reshape(4, 16, 17)
    assert np.all(obs_3d[:, :, -1] == 0.0), "Time channel not initialized to 0.0"
    
    # Check one-hot channels (should be strictly 0.0 since nothing is played)
    assert np.all(obs_3d[:, :, :-1] == 0.0), "One-hot channels not initialized to 0.0"

def test_temporal_progress():
    env = BeatGridEnv(L=4, T=16, S=15)
    env.reset()
    
    # Take 10 steps
    for _ in range(10):
        idx = np.random.randint(len(env.empty_cells))
        layer, time_step = env.empty_cells[idx]
        action = layer * (env.T * (env.S + 1)) + time_step * (env.S + 1) + 0
        env.step(action)
        
    idx = np.random.randint(len(env.empty_cells))
    layer, time_step = env.empty_cells[idx]
    action = layer * (env.T * (env.S + 1)) + time_step * (env.S + 1) + 0
    obs, _, _, _, _ = env.step(action) # Step 11
    
    obs_3d = obs.reshape(4, 16, 17)
    expected_time = 11 / (4 * 16)
    
    # Extract the temporal channel and check all values across the spatial grid
    assert np.allclose(obs_3d[:, :, -1], expected_time), f"Expected temporal value {expected_time}"

def test_collision_handling():
    env = BeatGridEnv(L=4, T=16, S=15)
    env.reset()
    
    # Pick a valid cell and fill it
    layer, time_step = env.empty_cells[0]
    action = layer * (env.T * (env.S + 1)) + time_step * (env.S + 1) + 0
    env.step(action)
    
    # Attempting to play it again should crash the matrix mathematically
    try:
        env.step(action)
        assert False, "Environment failed to block an illegal collision action!"
    except ValueError as e:
        assert "prevented by dynamic action masking" in str(e)

def test_terminal_state():
    env = BeatGridEnv(L=4, T=16, S=15)
    env.reset()
    
    done = False
    steps = 0
    while not done:
        # Simulate Actor Dynamic Masking (pick only from empty)
        idx = np.random.randint(len(env.empty_cells))
        layer, time_step = env.empty_cells[idx]
        action = layer * (env.T * (env.S + 1)) + time_step * (env.S + 1) + 0
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
        
    assert steps == 64
    assert len(env.empty_cells) == 0
    assert np.all(env.grid >= 0)

def test_visual_rollout():
    print("\n--- Running Visual Rollout Test ---")
    env = BeatGridEnv(L=4, T=16, S=15)
    env.reset()
    done = False
    
    while not done:
        # Simulate Actor Dynamic Masking
        idx = np.random.randint(len(env.empty_cells))
        layer, time_step = env.empty_cells[idx]
        action = layer * (env.T * (env.S + 1)) + time_step * (env.S + 1) + 0
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
    print("\nFinal Grid Representation (Rows=Layers, Cols=Time Steps):")
    print("('-' = mathematically unplayed. None should exist at termination!)")
    print("('0' = Silence Penalty inserted dynamically due to a collision.)")
    
    grid_str = ""
    for r in range(env.L):
        row_str = " ".join([f"{val:2d}" if val >= 0 else " -" for val in env.grid[r]])
        grid_str += f"Layer {r}: {row_str}\n"
    print(grid_str)
    
    print(f"Total Steps Taken: {info['step_count']} / 64")
    print("\nAttempting to trigger Matplotlib GUI for physical visual confirmation... (Check your screen!)")
    
    try:
        from grid_env.visualize_env import plot_beat_grid
        plot_beat_grid(env.grid, phase=1)
    except Exception as e:
        print(f"Matplotlib GUI failed (normal for headless instances): {e}")

if __name__ == "__main__":
    print("--- BeatGridEnv Unit Tests ---")
    test_env_initialization()
    test_temporal_progress()
    test_collision_handling()
    test_terminal_state()
    print("AUTOMATED TESTS SUCCESSFUL: Environment Mathematics are bulletproof!")
    
    # Run the manual visualization sequence
    test_visual_rollout()
