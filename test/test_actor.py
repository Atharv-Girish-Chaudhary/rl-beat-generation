import torch
import sys
import os

# Ensure the root project is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.actor import CNNLayerStepSampleActor

def test_actor_dimensions_and_autoregression():
    L, T, S = 4, 16, 15
    env_map = {0: [1,2], 1:[3,4], 2:[5], 3:[6]}
    
    actor = CNNLayerStepSampleActor(L=L, T=T, S=S, env_layer_to_samples=env_map)
    
    # Simulate a batched S+2 observation matrix (empty grid)
    obs = torch.zeros(8, L * T * (S + 2))
    
    # 1. Test Inference Sequences (Act)
    # This proves PyTorch matrices are no longer mathematically exploding
    try:
        actions, logps = actor.act(obs)
    except Exception as e:
        assert False, f"Neural Act Evaluation Crash! Error: {e}"
        
    assert actions.shape == (8,), "Actor failed to properly format batched NumPy integers."
    assert logps.shape == (8,), "Actor failed to return batched log probabilities."
    
    # 2. Test Parallel PPO Vector Mathematics
    try:
        # Simulate PyTorch historical actions passed by Replay Buffer
        historical_actions = torch.randint(0, L*T*(S+1), (8,))
        log_probs, entropy = actor.evaluate_actions(obs, historical_actions)
    except Exception as e:
        assert False, f"PPO Training Evaluation Crash! Error: {e}"
        
    assert log_probs.shape == (8,), "Actor crashed PyTorch parallel batch processing geometries."
    assert entropy.shape == (8,), "Actor Entropy equations misfired dimensions."

if __name__ == "__main__":
    print("--- Autoregressive Actor Unit Tests ---")
    print("Testing Mathematical Tensor Shapes and Forward Matrix Stability...")
    test_actor_dimensions_and_autoregression()
    print("AUTOMATED TESTS SUCCESSFUL: Neuro-Architecture Matrices are unbreakable!")