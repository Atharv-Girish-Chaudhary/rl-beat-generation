#!/bin/bash
#SBATCH --job-name=beat-disc-p2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/chaudhary.at/rl-beat-generation/logs/disc_p2_%j.out
#SBATCH --error=/scratch/chaudhary.at/rl-beat-generation/logs/disc_p2_%j.err

module load anaconda3/2024.06

cd /scratch/chaudhary.at/rl-beat-generation

echo "Starting discriminator training on $(hostname) at $(date)"
export PYTHONUNBUFFERED=1
/home/chaudhary.at/.conda/envs/beat_env/bin/python scripts/train_discriminator.py --phase 2
echo "Discriminator training finished at $(date)"
