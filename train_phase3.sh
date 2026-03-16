#!/bin/bash
#SBATCH --job-name=beats-phase3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=07:30:00
#SBATCH --output=logs/phase3_%j.out
#SBATCH --error=logs/phase3_%j.err

# ─── Setup ───
module load cuda/12.1
export PYTHONUNBUFFERED=1

PYTHON="/home/<YOUR_NEU_USERNAME>/.conda/envs/rl-beats/bin/python"

echo "=== Phase 3: SAC — Hybrid Audio Effects ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo ""

# ─── Train ───
$PYTHON src/train.py \
    --phase 3 \
    --algorithm sac \
    --grid-layers 8 \
    --grid-steps 16 \
    --total-timesteps 2000000 \
    --batch-size 256 \
    --lr 3e-4 \
    --gamma 0.99 \
    --tau 0.005 \
    --alpha auto \
    --replay-buffer-size 1000000 \
    --reward-alpha 0.3 \
    --reward-beta 0.7 \
    --use-discriminator \
    --freeze-discrete-heads \
    --checkpoint-dir checkpoints/phase3 \
    --resume-from checkpoints/phase2/best.pt \
    --log-dir logs/tensorboard/phase3 \
    --seed 42

echo ""
echo "End: $(date)"
