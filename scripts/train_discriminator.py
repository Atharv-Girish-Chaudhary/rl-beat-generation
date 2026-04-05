import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt

from beat_rl.models import BeatDiscriminator, BeatDataset


class _Phase1Dataset(BeatDataset):
    """
    BeatDataset subclass for Phase 1 (4-instrument) training.

    The parent class's NegativeGenerator.random_grid() and density_wrong_grid()
    default to n_inst=8, which causes a shape mismatch when real_grids have been
    sliced to 4 instruments.  This subclass overrides __getitem__ to derive the
    instrument count directly from the sliced real_grids array.
    """
    def __getitem__(self, idx):
        n_inst, n_steps = self.real_grids.shape[1], self.real_grids.shape[2]

        if np.random.rand() < 0.5:
            i = np.random.randint(len(self.real_grids))
            return torch.tensor(self.real_grids[i].astype(np.float32)), torch.tensor([1.0])

        neg_type = np.random.choice(
            ["random", "shuffled", "density", "agent"],
            p=[0.3, 0.3, 0.2, 0.2]
        )

        if neg_type == "agent" and len(self.agent_pool) > 0:
            grid = self.agent_pool[np.random.randint(len(self.agent_pool))]
        elif neg_type == "shuffled":
            grid = self.neg_gen.shuffled_grid(
                self.real_grids[np.random.randint(len(self.real_grids))]
            )
        elif neg_type == "density":
            density = (np.random.uniform(0.0, 0.1)
                       if np.random.rand() > 0.5
                       else np.random.uniform(0.8, 1.0))
            grid = (np.random.rand(n_inst, n_steps) < density).astype(np.float32)
        else:  # random
            grid = (np.random.rand(n_inst, n_steps)
                    < np.random.uniform(0.1, 0.7)).astype(np.float32)

        return torch.tensor(grid.astype(np.float32)), torch.tensor([0.0])

def train_discriminator(
    data_path: str = "data/processed/groove_grids.npy",
    epochs: int = 15,       # Sufficient for simple binary rhythmic classification convergence
    batch_size: int = 128,
    lr: float = 3e-4,
    device: str = "cpu"
):
    print("--- 🧠 Discriminator Pytorch Gradient Pipeline ---")
    # Dynamically select optimized hardware acceleration
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps" 
        
    print(f"Hardware Accelerated on Device: {device}")
    
    # 1. Load Natively Processed Human Data
    if not os.path.exists(data_path):
        print(f"Error: Could not geometrically map pure dataset at {data_path}. Run sequence data_processing/process_groove.py first.")
        sys.exit(1)
        
    print(f"Harvesting Real Human Groove Data from {data_path}...")
    real_grids = np.load(data_path)
    print(f"Raw tensor shape: {real_grids.shape}")

    # Groove MIDI grids have 8 instrument rows; Phase 1 uses only the first 4
    # (kick, snare, hi-hat, clap).  Slice here so the discriminator is trained
    # on the same 4×16 representation the PPO agent generates.
    real_grids = real_grids[:, :4, :]
    print(f"Phase 1 slice shape: {real_grids.shape}  (kept rows 0–3 of 8)")
    
    # 2. Build PyTorch Dataset Integrator
    num_samples = len(real_grids) * 2  # Exactly 50% real human, 50% synthetic fake garbage
    dataset = _Phase1Dataset(real_grids=real_grids, num_samples=num_samples)
    
    # 3. Train/Validation Stratified Splitting (80 / 20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 4. Initialize Network Architecture
    L, T = real_grids.shape[1], real_grids.shape[2]
    print(f"Initializing Transformer network targeting spatial boundaries L={L}, T={T}...")
    
    model = BeatDiscriminator(
        num_instruments=L, 
        num_steps=T, 
        d_model=64, 
        num_heads=4, 
        num_blocks=2, 
        d_ff=128
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Mathematical Telemetry Arrays
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    print(f"\nCommencing Parallel Tensor Gradient Descent for {epochs} Epochs...")
    
    best_val_acc = 0.0
    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (grids, labels) in enumerate(train_loader):
            grids = grids.to(device)
            # CRITICAL SHAPE GUARD: Explicitly enforce (Batch, 1) to prevent catastrophic 'BCEWithLogitsLoss' broadcast matrices
            labels = labels.to(device).view(-1, 1).float()
            
            optimizer.zero_grad()
            logits, _ = model(grids)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Logit to probability translation manually evaluating Accuracy tracking:
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = correct / total
        
        # Validation Eval Sub-Branch
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for grids, labels in val_loader:
                grids = grids.to(device)
                labels = labels.to(device).view(-1, 1).float()
                
                logits, _ = model(grids)
                
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = val_correct / val_total
        
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)
        
        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")
              
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), "outputs/checkpoints/discriminator_phase1_v2.pt")
            print(f"  -> Checkpoint saved! New best val acc: {best_val_acc:.4f}")

    print(f"\nDiscriminator Hardware Training Terminated! Final Optimized Val Accuracy: {best_val_acc:.4f}")
    
    # 5. Plotting Telemetry Data
    print("\nGenerating comprehensive telemetry charts...")
    plt.figure(figsize=(14, 5))
    
    # Loss Line Plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history['train_loss'], label='Train BCE Loss', color='blue', linewidth=2)
    plt.plot(range(1, epochs + 1), history['val_loss'], label='Val BCE Loss', color='red', linewidth=2, linestyle='--')
    plt.title('Discriminator Gradient Loss Trajectory')
    plt.xlabel('Epochs')
    plt.ylabel('BCEWithLogitsLoss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Accuracy Line Plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), history['train_acc'], label='Train Accuracy', color='green', linewidth=2)
    plt.plot(range(1, epochs + 1), history['val_acc'], label='Val Accuracy', color='orange', linewidth=2, linestyle='--')
    plt.title('Binary Classification Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy Ratio [0-1]')
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    plot_path = "outputs/plots/discriminator_training_plot.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Telemetry visual charts successfully rendered and saved to: {plot_path}")
    
if __name__ == "__main__":
    import matplotlib
    # Force headless matplotlib backend to aggressively prevent terminal pop-up freezing during training runs
    matplotlib.use('Agg')
    train_discriminator()
