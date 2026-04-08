#!/bin/bash
#SBATCH --job-name=beat-ppo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=07:30:00
#SBATCH --output=/scratch/chaudhary.at/rl-beat-generation/logs/ppo_%j.out
#SBATCH --error=/scratch/chaudhary.at/rl-beat-generation/logs/ppo_%j.err

module load anaconda3/2024.06

cd /scratch/chaudhary.at/rl-beat-generation

# echo "Pulling latest changes from git at $(date)"
# git pull origin main
echo "Starting PPO training on $(hostname) at $(date)"
export PYTHONUNBUFFERED=1
/home/chaudhary.at/.conda/envs/beat_env/bin/python scripts/train_ppo.py
echo "PPO training finished at $(date)"
