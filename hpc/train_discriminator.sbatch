#!/bin/bash
#SBATCH --job-name=beat-disc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/rl-beat-generation/logs/disc_%j.out
#SBATCH --error=/scratch/%u/rl-beat-generation/logs/disc_%j.err

module load anaconda3/2024.06

cd ${HPC_SCRATCH:-/scratch/$USER}/rl-beat-generation

echo "Starting discriminator training on $(hostname) at $(date)"
export PYTHONUNBUFFERED=1
/home/$USER/.conda/envs/beat_env/bin/python scripts/train_discriminator.py
echo "Discriminator training finished at $(date)"
