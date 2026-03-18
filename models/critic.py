import torch
import torch.nn as nn
import torch.nn.functional as F

class BeatCritic(nn.Module):
    """
    CNN-based Value Network (Critic) for the Beat Grid Environment.
    Estimates the value V(s) of the current grid state.
    Maintains separate parameters from the Actor.
    """
    def __init__(self, L: int, T: int, S: int):
        super().__init__()
        self.L = L
        self.T = T
        self.S = S
        
        # --- CNN Backbone (Identical architecture to Actor, separate weights) ---
        # Input shape expected by Conv2d: (Batch, Channels, Height, Width) -> (B, S+1, L, T)
        self.conv1 = nn.Conv2d(in_channels=S + 1, out_channels=32, kernel_size=3, padding=1) # [cite: 565]
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1) # [cite: 565]
        
        # Calculate flattened dimension after 2 conv layers with padding=1
        conv_out_dim = 64 * L * T # [cite: 566]
        
        # --- Fully Connected Layers ---
        self.fc1 = nn.Linear(conv_out_dim, 256) # [cite: 567]
        self.fc2 = nn.Linear(256, 128) # [cite: 568]
        
        # --- Scalar Value Head ---
        self.value_head = nn.Linear(128, 1) # [cite: 570]

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Passes the flattened one-hot grid through the CNN.
        Returns the scalar state value V(s).
        """
        batch_size = obs.shape[0] # [cite: 579]
        
        # Reshape flat vector back to 3D grid: (B, L, T, S+1)
        x = obs.view(batch_size, self.L, self.T, self.S + 1) # [cite: 580]
        
        # Permute to meet PyTorch Conv2d expectations: (B, S+1, L, T)
        x = x.permute(0, 3, 1, 2) # [cite: 581]
        
        x = F.relu(self.conv1(x)) # [cite: 582]
        x = F.relu(self.conv2(x)) # [cite: 583]
        
        x = x.reshape(batch_size, -1)  # Flatten [cite: 584]
        
        x = F.relu(self.fc1(x)) # [cite: 586]
        x = F.relu(self.fc2(x)) # [cite: 587]
        
        # Returns V(s) with shape (batch, 1)
        return self.value_head(x) # [cite: 588]