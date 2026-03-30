import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import List

from beat_rl.env import BeatGridEnv, compute_reward
from beat_rl.models import CNNLayerStepSampleActor, CNNBeatCritic, BeatDiscriminator

def compute_gae(rewards, values, next_value, dones, gamma, lam):
    """Generalized Advantage Estimation"""
    advantages = []
    gae = 0
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_value * (1 - dones[step]) - values[step]
        gae = delta + gamma * lam * (1 - dones[step]) * gae
        advantages.insert(0, gae)
        next_value = values[step]
    return advantages

def render_grid(grid, epoch, save_dir="outputs/plots", ax=None):
    """Visualizes the final grid structure for inspection. Can render into a subplot axis."""
    standalone = ax is None
    if standalone:
        plt.figure(figsize=(10, 4))
        ax = plt.gca()
    
    ax.imshow(grid > 0, aspect='auto', cmap='Blues', interpolation='nearest')
    
    # Annotate the specific sample ID choices (1-15) directly onto the grid
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            val = grid[i, j]
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center", color="white" if val > 0 else "black", fontsize=10, fontweight='bold')
    
    ax.set_title(f'Generated Beat Grid - Epoch {epoch}')
    ax.set_ylabel('Layer (Instrument)')
    ax.set_xlabel('Time Step')
    ax.set_yticks(np.arange(grid.shape[0]))
    ax.set_yticklabels(['Kick', 'Snare', 'HiHat', 'Clap'][:grid.shape[0]])
    ax.set_xticks(np.arange(0, grid.shape[1], 4))
    ax.grid(color='black', linestyle='-', linewidth=0.5, axis='x')
    
    if standalone:
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"beat_grid_epoch_{epoch}.png"))
        plt.close()

def train_ppo(
    epochs: int = 250,
    episodes_per_epoch: int = 32,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_ratio: float = 0.2,
    pi_lr: float = 3e-4,
    v_lr: float = 1e-3,
    train_pi_iters: int = 4,
    train_v_iters: int = 4,
    device: str = "cpu"
):
    print("--- 🧠 Beat Generation PPO Pipeline ---")
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    print(f"Accelerating on Device: {device}")

    # Phase 1 parameters
    L, T, S = 4, 16, 15
    layer_to_samples = {i: list(range(1, 16)) for i in range(L)}

    # Optionally load discriminator
    disc = None
    disc_path = "checkpoints/discriminator_v1.pt"
    if os.path.exists(disc_path):
        print("Loading Pre-trained Discriminator...")
        disc = BeatDiscriminator(num_instruments=L, num_steps=T, d_model=64, num_heads=4, num_blocks=2, d_ff=128).to(device)
        disc.load_state_dict(torch.load(disc_path, map_location=device))
        disc.eval()

    # Create Environment
    def r_fn(grid, final, action_coord):
        return compute_reward(grid, final, action_coord, phase=1, discriminator=disc)

    env = BeatGridEnv(L=L, T=T, S=S, reward_fn=r_fn, layer_to_samples=layer_to_samples, phase=1)

    # Initialize Actor & Critic
    actor = CNNLayerStepSampleActor(L=L, T=T, S=S, env_layer_to_samples=layer_to_samples).to(device)
    critic = CNNBeatCritic(L=L, T=T, S=S).to(device)

    pi_optimizer = optim.Adam(actor.parameters(), lr=pi_lr)
    v_optimizer = optim.Adam(critic.parameters(), lr=v_lr)

    # Telemetry & Best Model Tracking
    history = {'mean_rewards': [], 'actor_loss': [], 'critic_loss': []}
    best_reward = -float('inf')
    best_epoch = 0
    first_grid = None
    first_reward = 0.0
    best_grid = None
    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)

    print("\nCommencing PPO Training Loop...")
    
    for epoch in range(epochs):
        obs_buf, act_buf, adv_buf, ret_buf, logp_buf = [], [], [], [], []
        epoch_rewards = []
        epoch_grids = []
        
        # Rollouts
        for ep_idx in range(episodes_per_epoch):
            obs, _ = env.reset()
            ep_reward = 0
            ep_obs, ep_act, ep_rew, ep_val, ep_logp = [], [], [], [], []
            
            while True:
                obs_t = torch.tensor(obs, dtype=torch.float32).to(device)
                action, logp = actor.act(obs_t)
                
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                with torch.no_grad():
                    val = critic(obs_t.unsqueeze(0)).item()
                    
                done = terminated or truncated
                ep_reward += reward

                ep_obs.append(obs)
                ep_act.append(action)
                ep_rew.append(reward)
                ep_val.append(val)
                ep_logp.append(logp)

                if done:
                    # Final Step
                    next_val = 0.0
                    dones = [0.0] * (len(ep_rew) - 1) + [1.0]
                    adv = compute_gae(ep_rew, ep_val, next_val, dones, gamma, lam)
                    returns = [a + v for a, v in zip(adv, ep_val)]
                    
                    obs_buf.extend(ep_obs)
                    act_buf.extend(ep_act)
                    adv_buf.extend(adv)
                    ret_buf.extend(returns)
                    logp_buf.extend(ep_logp)
                    epoch_rewards.append(ep_reward)
                    epoch_grids.append(env.grid.copy())
                        
                    break
                
                obs = next_obs
        
        mean_ep_reward = np.mean(epoch_rewards)
        history['mean_rewards'].append(mean_ep_reward)
        
        # Capture first epoch grid from the best episode of epoch 0
        if epoch == 0:
            best_ep_idx = int(np.argmax(epoch_rewards))
            first_grid = epoch_grids[best_ep_idx]
            first_reward = epoch_rewards[best_ep_idx]
            render_grid(first_grid, epoch=0)

        # Save best model checkpoint (uses best episode grid, not last episode)
        if mean_ep_reward > best_reward:
            best_reward = mean_ep_reward
            best_epoch = epoch
            best_ep_idx = int(np.argmax(epoch_rewards))
            best_grid = epoch_grids[best_ep_idx]
            torch.save(actor.state_dict(), "outputs/checkpoints/actor_best.pth")
            torch.save(critic.state_dict(), "outputs/checkpoints/critic_best.pth")
            render_grid(best_grid, epoch=epoch)
            print(f"  ✅ New best model saved! (Reward: {best_reward:.3f})")

        # Optimization Data Prep
        obs_t = torch.tensor(np.array(obs_buf), dtype=torch.float32).to(device)
        act_t = torch.tensor(act_buf, dtype=torch.long).to(device)
        adv_t = torch.tensor(adv_buf, dtype=torch.float32).to(device)
        ret_t = torch.tensor(ret_buf, dtype=torch.float32).to(device).unsqueeze(1)
        old_logp_t = torch.tensor(logp_buf, dtype=torch.float32).to(device)

        # Advantage Normalization
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # Optimize Actor
        for _ in range(train_pi_iters):
            pi_optimizer.zero_grad()
            logp_t, entropy = actor.evaluate_actions(obs_t, act_t)
            ratio = torch.exp(logp_t - old_logp_t)
            clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv_t
            loss_pi = -(torch.min(ratio * adv_t, clip_adv)).mean() - 0.05 * entropy.mean()
            loss_pi.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
            pi_optimizer.step()

        # Optimize Critic
        for _ in range(train_v_iters):
            v_optimizer.zero_grad()
            val_t = critic(obs_t)
            loss_v = ((val_t - ret_t) ** 2).mean()
            loss_v.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
            v_optimizer.step()

        history['actor_loss'].append(loss_pi.item())
        history['critic_loss'].append(loss_v.item())

        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"Mean Reward: {mean_ep_reward:.3f} | "
              f"Actor Loss: {loss_pi.item():.3f} | Critic Loss: {loss_v.item():.3f}")

    # Generate Final Visualization: Training Curves
    print("\nGenerating comprehensive PPO training charts...")
    plt.figure(figsize=(15, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['mean_rewards'], color='green', linewidth=2)
    plt.axhline(y=best_reward, color='green', linestyle='--', alpha=0.5, label=f'Best: {best_reward:.3f}')
    plt.title('Mean Episode Reward')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(history['actor_loss'], color='blue', linewidth=2)
    plt.title('Actor Loss (PPO Clip)')
    plt.xlabel('Epochs')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(history['critic_loss'], color='red', linewidth=2)
    plt.title('Critic Value Loss')
    plt.xlabel('Epochs')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("outputs/plots/ppo_training_plot.png")
    plt.close()
    
    # Generate Side-by-Side Comparison: First Epoch vs Best Epoch
    if first_grid is not None and best_grid is not None:
        print(f"Generating First vs Best comparison (Epoch 0 vs Epoch {best_epoch})...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))
        
        render_grid(first_grid, epoch=0, ax=ax1)
        ax1.set_title(f'Epoch 0 — First (Reward: {first_reward:.3f})', fontsize=14, fontweight='bold')
        
        render_grid(best_grid, epoch=best_epoch, ax=ax2)
        ax2.set_title(f'Epoch {best_epoch} — Best (Reward: {best_reward:.3f})', fontsize=14, fontweight='bold')
        
        fig.suptitle('Beat Grid Evolution: First vs Best', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig("outputs/plots/first_vs_best_comparison.png", bbox_inches='tight')
        plt.close()
        
        # Save standalone grid PNGs for epoch 0 and best
        render_grid(first_grid, epoch=0)
        render_grid(best_grid, epoch=best_epoch)
    
    print(f"\nPPO training finished. Best model at Epoch {best_epoch} (Reward: {best_reward:.3f}).")
    print(f"Saved: outputs/checkpoints/actor_best.pth, outputs/checkpoints/critic_best.pth")
    print(f"Saved: outputs/plots/first_vs_best_comparison.png")
    print(f"Saved: outputs/plots/beat_grid_epoch_0.png, outputs/plots/beat_grid_epoch_{best_epoch}.png")

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    train_ppo()
