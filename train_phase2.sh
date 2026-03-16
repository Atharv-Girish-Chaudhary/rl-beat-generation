#!/bin/bash
#SBATCH --job-name=beats-phase2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=07:30:00
#SBATCH --output=logs/phase2_%j.out
#SBATCH --error=logs/phase2_%j.err

# ─── Setup ───
module load cuda/12.1
export PYTHONUNBUFFERED=1

PYTHON="/home/<YOUR_NEU_USERNAME>/.conda/envs/rl-beats/bin/python"

echo "=== Phase 2: PPO — Full Grid (8×16) + Discriminator ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo ""

# ─── Train ───
$PYTHON src/train.py \
    --phase 2 \
    --algorithm ppo \
    --grid-layers 8 \
    --grid-steps 16 \
    --total-timesteps 1000000 \
    --batch-size 128 \
    --lr 1e-4 \
    --gamma 0.99 \
    --clip-epsilon 0.2 \
    --gae-lambda 0.95 \
    --reward-alpha 0.5 \
    --reward-beta 0.5 \
    --use-discriminator \
    --disc-update-freq 100 \
    --phase1-replay-ratio 0.3 \
    --checkpoint-dir checkpoints/phase2 \
    --resume-from checkpoints/phase1/best.pt \
    --log-dir logs/tensorboard/phase2 \
    --seed 42

echo ""
echo "End: $(date)"
