import numpy as np
import torch
import sys
import os

# Ensure the root project is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.discriminator import BeatDiscriminator, NegativeGenerator, BeatDataset

def test_discriminator_shape():
    # Phase 1 mathematical parameters
    L = 4
    T = 16
    B = 8 # Simulated Batch size
    
    disc = BeatDiscriminator(
        num_instruments=L, 
        num_steps=T, 
        d_model=64, 
        num_heads=4, 
        num_blocks=2, 
        d_ff=128
    )
    
    # Create fake batch of binary grids (representing 'B' episodes simultaneously)
    fake_grids = torch.rand((B, L, T))
    fake_grids = (fake_grids > 0.5).float()
    
    logits, attn_weights = disc(fake_grids)
    
    # Assert the matrix output is flawlessly squeezed to (Batch, 1) scalar scores
    assert logits.shape == (B, 1), f"Expected logit shape {(B, 1)}, got {logits.shape}"
    
    # Assert attention matrices are structurally intact
    assert len(attn_weights) == 2, "Failed to return attention matrix per Transformer block."
    
def test_double_sigmoid_removed():
    disc = BeatDiscriminator(
        num_instruments=4, num_steps=16, d_model=64, num_heads=4, num_blocks=2, d_ff=128
    )
    
    # To prove nn.Sigmoid() is truly gone, we force a massive, chaotic input tensor.
    # A Sigmoid would mathematically squash these infinite bounds down perfectly between 0.0 and 1.0.
    extreme_input = torch.randn(100, 4, 16) * 1000.0
    logits, _ = disc(extreme_input)
    
    # Since network weights are randomized, feeding 100 extreme batches guarantees 
    # some Logits will absolutely blow past 1.0 or fall below 0.0.
    has_raw_logits = torch.any((logits < 0.0) | (logits > 1.0)).item()
    
    assert has_raw_logits, "FATAL: All outputs are still bound to [0, 1]. The Double-Sigmoid crush failed to delete!"

def test_negative_generator():
    neg_gen = NegativeGenerator()
    
    fake_grid = neg_gen.random_grid(n_inst=4, n_steps=16)
    assert fake_grid.shape == (4, 16), "NegativeGenerator mathematically failed to match the specific 4x16 grid shape."
    
    # Ensure it only produces mathematical absolute 0s and 1s, preventing Sample ID corruption
    unique_vals = np.unique(fake_grid)
    for val in unique_vals:
        assert val in [0.0, 1.0], f"Dataset Generator produced illegal float value {val}"

def test_beat_dataset():
    # Mock a real dataset loaded from a theoretical .npy file
    mock_real = [np.ones((4, 16), dtype=np.float32) for _ in range(5)]
    
    # Initialize the PyTorch Dataset worker wrapper
    dataset = BeatDataset(real_grids=mock_real, num_samples=100)
    
    # Extract 1 item perfectly
    grid_tensor, label_tensor = dataset[42]
    
    assert grid_tensor.shape == (4, 16), "PyTorch Dataset Dataloader returned a corrupted mathematical dimension."
    assert label_tensor.shape == (1,), "Dataset missing the absolute binary label array."
    assert label_tensor.item() in [0.0, 1.0], "Dataset failed binary label assertion checking."
    
if __name__ == "__main__":
    print("--- BeatDiscriminator Unit Tests ---")
    print("Testing Mathematical Tensor Shapes...")
    test_discriminator_shape()
    print("Testing Sigmoid Eradication (Pure Logits vs Sigmoid Crush)...")
    test_double_sigmoid_removed()
    print("Testing Synthetic Dataset Generators...")
    test_negative_generator()
    print("Testing PyTorch DataLoader Formats...")
    test_beat_dataset()
    print("AUTOMATED TESTS SUCCESSFUL: Discriminator Mathematics are bulletproof!")
