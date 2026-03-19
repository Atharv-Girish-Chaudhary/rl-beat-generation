"""
Beat Discriminator — Level 1
RL-Based Beat Generation | CS 5180 Final Project
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, seq_len, _ = x.shape
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        Q = Q.view(B, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        out = attn_weights @ V
        out = out.transpose(1, 2).contiguous().view(B, seq_len, self.d_model)
        return self.W_out(out), attn_weights


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, attn_weights = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x, attn_weights


class BeatDiscriminator(nn.Module):
    def __init__(self, num_instruments, num_steps, d_model, num_heads, num_blocks, d_ff, dropout=0.1):
        super().__init__()
        self.num_steps = num_steps
        self.d_model = d_model
        self.token_embed = nn.Linear(num_instruments, d_model)
        self.pos_embed = nn.Embedding(num_steps, d_model)
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_blocks)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1), nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, grid):
        B = grid.shape[0]
        x = self.token_embed(grid.transpose(1, 2))
        x = self.dropout(x + self.pos_embed(torch.arange(self.num_steps, device=grid.device)))
        all_attn = []
        for block in self.encoder_blocks:
            x, attn = block(x)
            all_attn.append(attn)
        return self.classifier(x.mean(dim=1)), all_attn


class NegativeGenerator:
    @staticmethod
    def random_grid(n_inst=4, n_steps=16):
        density = np.random.uniform(0.1, 0.7)
        return (np.random.rand(n_inst, n_steps) < density).astype(np.float32)

    @staticmethod
    def shuffled_grid(real_grid):
        grid = real_grid.copy()
        np.random.shuffle(grid)
        return grid

    @staticmethod
    def density_wrong_grid(n_inst=4, n_steps=16):
        density = np.random.uniform(0.0, 0.1) if np.random.rand() > 0.5 else np.random.uniform(0.8, 1.0)
        return (np.random.rand(n_inst, n_steps) < density).astype(np.float32)


class BeatDataset(Dataset):
    def __init__(self, real_grids, agent_pool=None, num_samples=5000):
        self.real_grids = real_grids
        self.neg_gen = NegativeGenerator()
        self.agent_pool = agent_pool or []
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if np.random.rand() < 0.5:
            i = np.random.randint(len(self.real_grids))
            return torch.tensor(self.real_grids[i].astype(np.float32)), torch.tensor([1.0])
        neg_type = np.random.choice(["random", "shuffled", "density", "agent"], p=[0.3, 0.3, 0.2, 0.2])
        if neg_type == "agent" and len(self.agent_pool) > 0:
            grid = self.agent_pool[np.random.randint(len(self.agent_pool))]
        elif neg_type == "shuffled":
            grid = self.neg_gen.shuffled_grid(self.real_grids[np.random.randint(len(self.real_grids))])
        elif neg_type == "density":
            grid = self.neg_gen.density_wrong_grid()
        else:
            grid = self.neg_gen.random_grid()
        return torch.tensor(grid.astype(np.float32)), torch.tensor([0.0])
