import sys
import os
import torch
sys.path.append(os.getcwd())

from models.critic import BeatCritic

def test_critic_sanity_check():
    print("\n--- Testing Critic Network ---")
    
    L, T, S = 8, 16, 15  # Let's test with Phase 2 dimensions this time
    critic = BeatCritic(L, T, S)
    
    # Create a dummy batch of 4 empty observations
    batch_size = 4
    obs_dim = L * T * (S + 1)
    dummy_obs = torch.zeros((batch_size, obs_dim), dtype=torch.float32)
    
    # Forward Pass
    values = critic.forward(dummy_obs)
    
    # The output MUST be exactly (batch_size, 1)
    assert values.shape == (batch_size, 1), f"FAIL: Value output shape mismatch. Expected {(batch_size, 1)}, got {values.shape}"
    print(f"SUCCESS: Critic forward pass tensor shape is perfectly (Batch, 1): {values.shape}")

if __name__ == "__main__":
    run_critic_sanity_check()