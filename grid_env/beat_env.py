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
        self.reward_fn = reward_fn or (lambda grid, final: 0.0)
        self.layer_to_samples = layer_to_samples or {}
        self.phase = phase
        
        # Action space: Flat integer representing (layer, step, sample)
        self.action_space = spaces.Discrete(self.L * self.T * (self.S + 1))
        
        # Observation space: Flattened one-hot encoded grid
        obs_dim = self.L * self.T * (self.S + 1)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        
        # Pre-allocate memory for the grid to avoid reallocation overhead
        self.grid = np.zeros((self.L, self.T), dtype=np.int64)
        self.filled_cells = set()
        self.step_count = 0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.grid.fill(0)
        self.filled_cells.clear()
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        layer, time_step, sample = self._decode_action(action)
        
        # Collision handling: Redirect to a random empty cell if already filled
        if (layer, time_step) in self.filled_cells:
            empty = [(l, t) for l in range(self.L) for t in range(self.T) 
                     if (l, t) not in self.filled_cells]
            if empty:
                # Pick a random available cell to maintain training momentum
                idx = np.random.randint(len(empty))
                layer, time_step = empty[idx]
                
        # Apply action
        self.grid[layer, time_step] = sample
        self.filled_cells.add((layer, time_step))
        self.step_count += 1
        
        # Terminal condition: all cells visited
        terminated = len(self.filled_cells) == (self.L * self.T)
        truncated = False
        
        reward = float(self.reward_fn(self.grid, final=terminated))
            
        info = {"step_count": self.step_count, "filled": len(self.filled_cells)}
        
        return self._get_obs(), reward, terminated, truncated, info

    def _decode_action(self, action: int) -> Tuple[int, int, int]:
        """Decodes flat integer into (layer, time_step, sample) using modulo arithmetic."""
        sample = action % (self.S + 1)
        remainder = action // (self.S + 1)
        time_step = remainder % self.T
        layer = remainder // self.T
        return int(layer), int(time_step), int(sample)

    def _get_obs(self) -> np.ndarray:
        """
        Vectorized one-hot encoding of the grid.
        """
        one_hot = np.zeros((self.L, self.T, self.S + 1), dtype=np.float32)
        # Advanced NumPy indexing to set the 1.0s instantly across the entire grid
        one_hot[np.arange(self.L)[:, None], np.arange(self.T), self.grid] = 1.0
        return one_hot.flatten()

    def get_action_mask(self, layer: int) -> np.ndarray:
        """Boolean mask for valid samples per layer."""
        mask = np.zeros(self.S + 1, dtype=bool)
        mask[0] = True  # Silence is universally valid
        
        valid_samples = self.layer_to_samples.get(layer, [])
        valid_samples = [s for s in valid_samples if s <= self.S]
        
        if valid_samples:
            mask[valid_samples] = True
            
        return mask