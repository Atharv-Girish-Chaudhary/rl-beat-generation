#!/bin/bash
#SBATCH --job-name=beats-phase1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=07:30:00
#SBATCH --output=logs/phase1_%j.out
#SBATCH --error=logs/phase1_%j.err

# ─── Setup ───
module load cuda/12.1
export PYTHONUNBUFFERED=1

PYTHON="/home/<YOUR_NEU_USERNAME>/.conda/envs/rl-beats/bin/python"

echo "=== Phase 1: PPO — Drums Only (4×16 grid) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo ""

# ─── Train ───
$PYTHON src/train.py \
    --phase 1 \
    --algorithm ppo \
    --grid-layers 4 \
    --grid-steps 16 \
    --total-timesteps 500000 \
    --batch-size 64 \
    --lr 3e-4 \
    --gamma 0.99 \
    --clip-epsilon 0.2 \
    --gae-lambda 0.95 \
    --reward-alpha 0.9 \
    --reward-beta 0.1 \
    --checkpoint-dir checkpoints/phase1 \
    --log-dir logs/tensorboard/phase1 \
    --seed 42

echo ""
echo "End: $(date)"
