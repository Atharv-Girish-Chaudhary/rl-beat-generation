import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, List, Optional

class CNNLayerStepSampleActor(nn.Module):
    def __init__(self, L: int, T: int, S: int, env_layer_to_samples: Dict[int, List[int]]):
        super().__init__()
        self.L = L
        self.T = T
        self.S = S
        self.env_layer_to_samples = env_layer_to_samples

        # in_channels cleanly intercepts our S + 2 temporal observation arrays
        self.conv1 = nn.Conv2d(in_channels=S + 2, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        self.conv_out_dim = 64 * L * T
        self.base_fc = nn.Linear(self.conv_out_dim, 128)

        # Sequential Autoregressive Heads
        self.layer_head = nn.Linear(128, L)
        self.layer_emb = nn.Embedding(L, 128)
        
        self.step_head = nn.Linear(128, T)
        self.step_emb = nn.Embedding(T, 128)
        
        self.sample_head = nn.Linear(128, S + 1)

        self._build_sample_mask()

    def _build_sample_mask(self):
        """Builds a registered boolean mask preventing illegal Audio/Instrument maps."""
        mask = torch.zeros(self.L, self.S + 1, dtype=torch.bool)
        for layer in range(self.L):
            mask[layer, 0] = True  # Silence (0) is universally valid
            valid_samples = self.env_layer_to_samples.get(layer, [])
            for s in valid_samples:
                if s <= self.S:
                    mask[layer, s] = True
        self.register_buffer("layer_sample_mask", mask)

    def _get_occupancy_mask(self, obs: torch.Tensor):
        """
        Reconstructs a (B, L, T) boolean occupancy mask from the flat observation.
        Returns: occupied (True = cell is filled), empty (True = cell is available)
        """
        B = obs.shape[0]
        obs_grid = obs.view(B, self.L, self.T, self.S + 2)
        # A cell is occupied if ANY of its S+1 one-hot sample channels is active
        occupied = obs_grid[:, :, :, :self.S + 1].sum(dim=-1) > 0  # (B, L, T)
        empty = ~occupied
        return empty

    def extract_base_features(self, obs: torch.Tensor):
        B = obs.shape[0]
        x = obs.view(B, self.L, self.T, self.S + 2)
        x = x.permute(0, 3, 1, 2)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.contiguous().view(B, -1)
        return F.relu(self.base_fc(x))

    def forward(self, obs: torch.Tensor, layer_act: Optional[torch.Tensor] = None, step_act: Optional[torch.Tensor] = None):
        """
        Calculates Neural logic autoregressively.
        When PPO is training, passes historical action vectors. When Playing casually, stops early.
        """
        base_features = self.extract_base_features(obs)
        
        layer_logits = self.layer_head(base_features)
        
        if layer_act is not None:
            layer_context = base_features + self.layer_emb(layer_act)
            step_logits = self.step_head(layer_context)
            
            if step_act is not None:
                step_context = layer_context + self.step_emb(step_act)
                sample_logits = self.sample_head(step_context)
                return layer_logits, step_logits, sample_logits
                
        return layer_logits, base_features

    def act(self, obs: torch.Tensor):
        """
        Samples a GUARANTEED VALID action using dynamic occupancy masking.
        Returns (flat_action, log_prob). Runs under no_grad.
        """
        is_unbatched = obs.dim() == 1
        if is_unbatched:
            obs = obs.unsqueeze(0)
        
        B = obs.shape[0]
        
        with torch.no_grad():
            empty_mask = self._get_occupancy_mask(obs)  # (B, L, T)
            
            # 1. Layer Sampling — mask out layers with zero empty cells
            layer_has_empty = empty_mask.any(dim=2)  # (B, L)
            layer_logits, base_features = self.forward(obs)
            masked_layer_logits = layer_logits.masked_fill(~layer_has_empty, float('-inf'))
            layer_dist = Categorical(logits=masked_layer_logits)
            layer_action = layer_dist.sample()
            
            # 2. Step Sampling — mask out occupied steps on the chosen layer
            step_mask = empty_mask[torch.arange(B, device=obs.device), layer_action]  # (B, T)
            layer_context = base_features + self.layer_emb(layer_action)
            step_logits = self.step_head(layer_context)
            masked_step_logits = step_logits.masked_fill(~step_mask, float('-inf'))
            step_dist = Categorical(logits=masked_step_logits)
            step_action = step_dist.sample()
            
            # 3. Sample Sampling — instrument validity mask (unchanged)
            step_context = layer_context + self.step_emb(step_action)
            sample_logits = self.sample_head(step_context)
            sample_mask = self.layer_sample_mask[layer_action]
            masked_sample_logits = sample_logits.masked_fill(~sample_mask, float('-inf'))
            sample_dist = Categorical(logits=masked_sample_logits)
            sample_action = sample_dist.sample()
            
            flat_action = layer_action * (self.T * (self.S + 1)) + step_action * (self.S + 1) + sample_action
            log_prob = layer_dist.log_prob(layer_action) + step_dist.log_prob(step_action) + sample_dist.log_prob(sample_action)
        
        action_out = flat_action.item() if is_unbatched else flat_action.detach().cpu().numpy()
        logp_out = log_prob.item() if is_unbatched else log_prob.detach().cpu().numpy()
        return action_out, logp_out

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        Called exclusively during PPO Training. Evaluates log_prob with dynamic masking.
        Reconstructs the occupancy mask from each obs to apply identical masks.
        """
        B = obs.shape[0]
        samples = actions % (self.S + 1)
        rem = actions // (self.S + 1)
        steps = rem % self.T
        layers = rem // self.T
        
        empty_mask = self._get_occupancy_mask(obs)  # (B, L, T)
        
        # Full autoregressive forward pass
        layer_logits, step_logits, sample_logits = self.forward(obs, layer_act=layers, step_act=steps)
        
        # Layer masking — identical to act()
        layer_has_empty = empty_mask.any(dim=2)  # (B, L)
        masked_layer_logits = layer_logits.masked_fill(~layer_has_empty, float('-inf'))
        dist_layer = Categorical(logits=masked_layer_logits)
        
        # Step masking — per chosen layer
        step_mask = empty_mask[torch.arange(B, device=obs.device), layers]  # (B, T)
        masked_step_logits = step_logits.masked_fill(~step_mask, float('-inf'))
        dist_step = Categorical(logits=masked_step_logits)
        
        # Sample masking — instrument validity
        sample_mask = self.layer_sample_mask[layers]
        masked_sample_logits = sample_logits.masked_fill(~sample_mask, float('-inf'))
        dist_sample = Categorical(logits=masked_sample_logits)
        
        log_prob_layer = dist_layer.log_prob(layers)
        log_prob_step = dist_step.log_prob(steps)
        log_prob_sample = dist_sample.log_prob(samples)
        
        log_probs = log_prob_layer + log_prob_step + log_prob_sample
        entropy = dist_layer.entropy() + dist_step.entropy() + dist_sample.entropy()
        
        return log_probs, entropy