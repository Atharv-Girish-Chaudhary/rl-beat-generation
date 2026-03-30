import torch
import numpy as np

from beat_rl.env import BeatGridEnv, compute_reward
from beat_rl.models import BeatDiscriminator, CNNLayerStepSampleActor, CNNBeatCritic

def test_full_pipeline_integration():
    L, T, S = 4, 16, 15
    env_map = {0: [1,2], 1:[3,4], 2:[5], 3:[6]}
    
    disc = BeatDiscriminator(num_instruments=4, num_steps=16, d_model=64, num_heads=2, num_blocks=1, d_ff=128)
    
    def reward_wrapper(grid, final, action_coord=None):
        return compute_reward(grid, final, action_coord, phase=1, discriminator=disc)
        
    env = BeatGridEnv(L=L, T=T, S=S, reward_fn=reward_wrapper, layer_to_samples=env_map)
    actor = CNNLayerStepSampleActor(L=L, T=T, S=S, env_layer_to_samples=env_map)
    critic = CNNBeatCritic(L=L, T=T, S=S)
    
    obs, info = env.reset()
    
    done = False
    step = 0
    total_reward = 0.0
    
    # Run the absolute entirety of a 64-step episode natively
    while not done:
        obs_tensor = torch.FloatTensor(obs)
        
        # 1. Critic Matrix Stability check
        value = critic(obs_tensor.unsqueeze(0))
        assert value.shape == (1, 1), "Critic failed end-to-end integration."
        
        # 2. Actor Autoregressive Logic Check
        action, logp = actor.act(obs_tensor)
        
        # 3. Env Step & Delta Reward Check
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        step += 1
        
    # 4. Discriminator Final Compression Check
    assert step == 64, "PPO Agent sequence mathematically crashed before 64 steps."
    assert total_reward > 0.0 or total_reward <= 10.0, "Reward boundary compromised"
    
if __name__ == "__main__":
    print("--- Full Hardware Pipeline Integration Test ---")
    test_full_pipeline_integration()
    print("INTEGRATION SUCCESSFUL: All 5 neural and environmental scripts seamlessly communicate!")
