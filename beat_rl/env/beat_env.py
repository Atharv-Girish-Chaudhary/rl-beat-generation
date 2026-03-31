import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any, Callable

class BeatGridEnv(gym.Env):
    """
    Beat Grid Environment (Supports Phase 1 and Phase 2).
    Optimized for high-throughput vectorized training.
    """
    def __init__(
        self, 
        L: int = 4, 
        T: int = 16, 
        S: int = 15, 
        reward_fn: Optional[Callable] = None, 
        layer_to_samples: Optional[Dict[int, list]] = None, 
        phase: int = 1
    ):
        super().__init__()
        self.L = L
        self.T = T
        self.S = S
        self.reward_fn = reward_fn or (lambda grid, final, action_coord=None: 0.0)
        self.layer_to_samples = layer_to_samples or {}
        self.phase = phase
        self.max_steps = self.L * self.T
        
        # Action space: Flat integer representing (layer, step, sample)
        self.action_space = spaces.Discrete(self.L * self.T * (self.S + 1))
        
        # Observation space: 
        # Channels: (S+1) for One-Hot samples, +1 for Normalized Temporal Progress.
        self.num_channels = self.S + 2
        obs_dim = self.L * self.T * self.num_channels
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        
        # Internal state tracking
        self.grid = np.full((self.L, self.T), -1, dtype=np.int64)
        self.empty_cells = []
        self.step_count = 0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.grid.fill(-1) # -1 strictly defines an unplayed/empty cell
        self.empty_cells = [(l, t) for l in range(self.L) for t in range(self.T)]
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        layer, time_step, sample = self._decode_action(action)
        
        coord = (layer, time_step)
        if coord not in self.empty_cells:
            raise ValueError(f"Action tried to overwrite occupied cell at {coord}. "
                             f"This should be prevented by dynamic action masking in the Actor.")
            
        # Register the cell as officially played
        self.empty_cells.remove((layer, time_step))
        self.grid[layer, time_step] = sample
        self.step_count += 1
        
        terminated = len(self.empty_cells) == 0
        truncated = False
        
        reward = float(self.reward_fn(self.grid, final=terminated, action_coord=(layer, time_step)))
            
        info = {
            "step_count": self.step_count, 
            "filled": self.max_steps - len(self.empty_cells),
            "executed_action": action
        }
        
        return self._get_obs(), reward, terminated, truncated, info

    def _decode_action(self, action: int) -> Tuple[int, int, int]:
        sample = action % (self.S + 1)
        remainder = action // (self.S + 1)
        time_step = remainder % self.T
        layer = remainder // self.T
        return int(layer), int(time_step), int(sample)

    def _get_obs(self) -> np.ndarray:
        """
        Flaw 1 & 2 Fix: Vectorized multi-channel encoding.
        0 to S: One-hot encoded samples. (Unplayed cells correctly register as all zeros).
        S+1: Temporal channel encoding elapsed time.
        """
        obs = np.zeros((self.L, self.T, self.num_channels), dtype=np.float32)
        
        filled_mask = self.grid >= 0
        if np.any(filled_mask):
            l_idx, t_idx = np.nonzero(filled_mask)
            samples = self.grid[filled_mask]
            obs[l_idx, t_idx, samples] = 1.0
            
        # The Temporal Channel fills uniformly with % of song completed
        time_fraction = self.step_count / self.max_steps
        obs[:, :, -1] = time_fraction
        
        return obs.flatten()

    def get_action_mask(self, layer: int) -> np.ndarray:
        mask = np.zeros(self.S + 1, dtype=bool)
        mask[0] = True
        
        valid_samples = self.layer_to_samples.get(layer, [])
        valid_samples = [s for s in valid_samples if s <= self.S]
        
        if valid_samples:
            mask[valid_samples] = True
            
        return mask