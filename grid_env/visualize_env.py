import matplotlib.pyplot as plt
import numpy as np
from grid_env.beat_env import BeatGridEnv

def plot_beat_grid(grid: np.ndarray, phase: int = 2):
    """
    Renders the L x T beat grid using Matplotlib.
    Creates a sequencer-style heat map of sample placements.
    """
    L, T = grid.shape
    
    # Map Y-axis labels based on the active Phase
    if L == 4:
        y_labels = ["Kick", "Snare", "Hihat", "Clap"]
    elif L == 8:
        y_labels = ["Kick", "Snare", "Hihat", "Clap", "Bass", "Melody", "Pad", "FX"]
    else:
        y_labels = [f"Layer {i}" for i in range(L)]

    # Size scales dynamically based on layer count
    fig, ax = plt.subplots(figsize=(10, L * 0.8))
    
    # Setup colormap: 0 (silence) is white, 1-15 get distinct colors
    cmap = plt.cm.get_cmap('viridis').copy()
    cmap.set_under('white')
    
    # Plot grid. vmin=0.1 forces the 0s into the 'under' color (white)
    cax = ax.imshow(grid, cmap=cmap, aspect='auto', vmin=0.1, vmax=15)
    
    # Primary axis formatting
    ax.set_xticks(np.arange(T))
    ax.set_yticks(np.arange(L))
    ax.set_xticklabels(np.arange(1, T + 1))
    ax.set_yticklabels(y_labels)
    
    # Draw standard grid lines around every cell
    ax.set_xticks(np.arange(-.5, T, 1), minor=True)
    ax.set_yticks(np.arange(-.5, L, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    # Draw thicker red structural lines on the downbeats (beats 2, 3, 4)
    # This visually divides the 16 steps into 4 distinct quarter-note beats
    for t in [3.5, 7.5, 11.5]:
        ax.axvline(x=t, color='red', linewidth=2, linestyle='--')

    plt.title(f"Beat Grid Sequence (Phase {phase} - {L}x{T})")
    plt.xlabel("16th Note Steps")
    
    # Add a colorbar to see which sample ID was chosen
    cbar = fig.colorbar(cax, ticks=np.arange(1, 16))
    cbar.set_label('Sample ID')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Sanity check: Run a fast random rollout and plot the final state
    print("Generating random Phase 2 rollout...")
    env = BeatGridEnv(L=8, phase=2)
    env.reset()
    
    terminated = False
    while not terminated:
        action = env.action_space.sample()
        _, _, terminated, _, _ = env.step(action)
        
    print("Rollout complete. Rendering plot...")
    plot_beat_grid(env.grid, phase=2)