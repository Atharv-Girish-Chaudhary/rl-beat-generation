#!/bin/bash
# Run once on Explorer to create the conda environment.
# Usage: ssh explorer "bash ${HPC_SCRATCH}/rl-beat-generation/hpc/setup_env.sh"

set -e

: "${HPC_USER:?HPC_USER is not set. Source hpc/env.sh first.}"
: "${HPC_SCRATCH:=/scratch/${HPC_USER}}"

module load anaconda3/2024.06

conda create -n beat_env python=3.10 -y
source activate beat_env

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -e ${HPC_SCRATCH}/rl-beat-generation
pip install matplotlib gymnasium

echo "Environment setup complete."
