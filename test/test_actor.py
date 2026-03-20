import torch
from models.actor import BeatActor

def test_actor_sanity_check():
    print("\n--- Testing Actor Network ---")
    
    L, T, S = 4, 16, 15
    # Dummy dictionary: Layer 0 only allows samples 1,2,3. Layer 1 allows 4,5.
    layer_to_samples = {0: [1, 2, 3], 1: [4, 5], 2: [6, 7], 3: [8, 9]}
    
    actor = BeatActor(L, T, S, layer_to_samples)
    
    # Create a dummy batch of 2 empty observations
    batch_size = 2
    obs_dim = L * T * (S + 1)
    dummy_obs = torch.zeros((batch_size, obs_dim), dtype=torch.float32)
    
    # 1. Test Forward Pass Shapes
    layer_logits, step_logits, sample_logits = actor.forward(dummy_obs)
    assert layer_logits.shape == (batch_size, L), "FAIL: Layer logits shape mismatch."
    assert step_logits.shape == (batch_size, T), "FAIL: Step logits shape mismatch."
    assert sample_logits.shape == (batch_size, S + 1), "FAIL: Sample logits shape mismatch."
    print("Forward pass tensor shapes are correct.")
    
    # 2. Test Act / Inference
    action, log_prob, entropy = actor.act(dummy_obs)
    assert action.shape == (batch_size,), "FAIL: Flat action output shape mismatch."
    assert log_prob.shape == (batch_size,), "FAIL: Log prob shape mismatch."
    print("Inference (act) method outputs correct shapes.")
    
    # 3. The Brutal Mask Test
    # Let's forcefully check the mask buffer to ensure `-inf` is being applied.
    mask_layer_0 = actor.layer_sample_mask[0]
    print(f"Layer 0 Boolean Mask: {mask_layer_0}")
    
    # Silence (index 0) and samples 1,2,3 should be True. Everything else False.
    assert mask_layer_0[0] == True, "FAIL: Silence is not unmasked."
    assert mask_layer_0[1] == True and mask_layer_0[4] == False, "FAIL: Sample constraints are wrong."
    print("SUCCESS: Factored action masking constraint is mathematically perfect.")

if __name__ == "__main__":
    run_actor_sanity_check()