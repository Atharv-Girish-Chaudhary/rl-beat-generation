import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBeatCritic(nn.Module):
    def __init__(self, L, T, S):
        super().__init__()
        self.L = L
        self.T = T
        self.S = S
        
        # Flaw 1 Fix: in_channels safely intercepts S + 2 arrays
        self.conv1 = nn.Conv2d(in_channels=S + 2, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Base CNN feature extractor structural dimension
        self.conv_out_dim = 64 * L * T
        
        self.fc1 = nn.Linear(self.conv_out_dim, 128)
        self.value_head = nn.Linear(128, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        # FLaw 2 Fix: Safely reconstruct spatial blocks geometrically before PyTorch permute
        x = obs.view(B, self.L, self.T, self.S + 2)
        x = x.permute(0, 3, 1, 2)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        x = x.contiguous().view(B, -1)
        x = F.relu(self.fc1(x))
        
        value = self.value_head(x)
        return value