import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, Tuple

class BeatActor(nn.Module):
    """
    CNN-based Policy Network (Actor) for the Beat Grid Environment.
    Outputs a factored, conditionally-masked action distribution.
    Optimized for PyTorch MPS (Metal Performance Shaders) on Apple Silicon.
    """
    def __init__(self, L: int, T: int, S: int, layer_to_samples: Dict[int, list]):
        super().__init__()
        self.L = L
        self.T = T
        self.S = S
        
        # Register the action mask as a PyTorch Buffer.
        # This guarantees it moves to your M4 GPU (MPS) automatically alongside the model weights.
        self.register_buffer("layer_sample_mask", self._build_layer_mask(L, S, layer_to_samples))
        
        # --- CNN Backbone ---
        # Input shape: (Batch, Channels, Height, Width) -> (B, S+1, L, T)
        self.conv1 = nn.Conv2d(in_channels=S+1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        conv_out_dim = 64 * L * T
        
        self.fc1 = nn.Linear(conv_out_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        
        # --- Factored Output Heads ---
        self.layer_head = nn.Linear(128, L)
        self.step_head = nn.Linear(128, T)
        self.sample_head = nn.Linear(128, S + 1)

    def _build_layer_mask(self, L: int, S: int, layer_to_samples: Dict[int, list]) -> torch.Tensor:
        """
        Builds a boolean tensor of shape (L, S+1) for ultra-fast action masking.
        True = Valid sample, False = Invalid sample.
        """
        mask = torch.zeros((L, S + 1), dtype=torch.bool)
        mask[:, 0] = True  # Silence (index 0) is universally valid across all layers
        
        for layer_idx, valid_samples in layer_to_samples.items():
            if layer_idx < L:
                valid_idx = [s for s in valid_samples if s <= S]
                if valid_idx:
                    mask[layer_idx, valid_idx] = True
                    
        return mask

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Passes the flattened one-hot grid through the CNN.
        Returns unmasked logits for Layer, Step, and Sample.
        """
        batch_size = obs.shape[0]
        
        # Reshape flat vector back to 3D grid: (B, L, T, S+1)
        x = obs.view(batch_size, self.L, self.T, self.S + 1)
        
        # Permute to meet PyTorch Conv2d expectations: (B, Channels, Height, Width) -> (B, S+1, L, T)
        x = x.permute(0, 3, 1, 2)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        x = x.reshape(batch_size, -1)  # Flatten
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        layer_logits = self.layer_head(x)
        step_logits = self.step_head(x)
        sample_logits = self.sample_head(x)
        
        return layer_logits, step_logits, sample_logits

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inference mode for environment rollouts. 
        Samples actions and applies the strict Layer -> Sample conditioning mask.
        """
        layer_logits, step_logits, sample_logits = self.forward(obs)
        
        # 1. Sample Layer
        layer_dist = Categorical(logits=layer_logits)
        layer_action = layer_dist.sample()
        
        # 2. Sample Time Step
        step_dist = Categorical(logits=step_logits)
        step_action = step_dist.sample()
        
        # 3. Apply Constraint Mask & Sample
        # Fetch the pre-computed boolean mask for the specific layers chosen by the batch
        mask = self.layer_sample_mask[layer_action]
        
        # Brutally enforce music theory: overwrite invalid sample logits to negative infinity
        # This mathematically forces their softmax probabilities to exactly 0.0
        masked_sample_logits = sample_logits.masked_fill(~mask, float('-inf'))
        
        sample_dist = Categorical(logits=masked_sample_logits)
        sample_action = sample_dist.sample()
        
        # PPO requires the joint log probability and joint entropy
        joint_log_prob = layer_dist.log_prob(layer_action) + step_dist.log_prob(step_action) + sample_dist.log_prob(sample_action)
        joint_entropy = layer_dist.entropy() + step_dist.entropy() + sample_dist.entropy()
        
        # Encode back to flat integer for the Gym environment
        flat_action = (layer_action * self.T * (self.S + 1)) + (step_action * (self.S + 1)) + sample_action
        
        return flat_action, joint_log_prob, joint_entropy

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Called during PPO gradient updates to evaluate old actions against the new policy.
        """
        # Decode the flat integer actions back into coordinates
        sample_acts = actions % (self.S + 1)
        remainder = actions // (self.S + 1)
        step_acts = remainder % self.T
        layer_acts = remainder // self.T
        
        layer_logits, step_logits, sample_logits = self.forward(obs)
        
        layer_dist = Categorical(logits=layer_logits)
        step_dist = Categorical(logits=step_logits)
        
        # Apply the exact same masking logic used during `act`
        mask = self.layer_sample_mask[layer_acts]
        masked_sample_logits = sample_logits.masked_fill(~mask, float('-inf'))
        sample_dist = Categorical(logits=masked_sample_logits)
        
        joint_log_prob = layer_dist.log_prob(layer_acts) + step_dist.log_prob(step_acts) + sample_dist.log_prob(sample_acts)
        joint_entropy = layer_dist.entropy() + step_dist.entropy() + sample_dist.entropy()
        
        return joint_log_prob, joint_entropy