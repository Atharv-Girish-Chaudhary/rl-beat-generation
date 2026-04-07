#!/bin/bash
# Run once on Explorer to create the conda environment.
# Usage: ssh explorer "bash /scratch/chaudhary.at/rl-beat-generation/hpc/setup_env.sh"

set -e

module load anaconda3/2024.06

conda create -n beat_env python=3.10 -y
source activate beat_env

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -e /scratch/chaudhary.at/rl-beat-generation

echo "Environment setup complete."
