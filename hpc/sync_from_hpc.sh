#!/bin/bash
# Pull outputs (checkpoints, plots, WAV) from Explorer back to local.
# Run from local machine: bash hpc/sync_from_hpc.sh

set -e

: "${HPC_USER:?HPC_USER is not set. Source hpc/env.sh first.}"
: "${HPC_REMOTE:=explorer}"
: "${HPC_SCRATCH:=/scratch/${HPC_USER}}"

REMOTE=${HPC_REMOTE}:${HPC_SCRATCH}/rl-beat-generation/outputs/
LOCAL=$(git rev-parse --show-toplevel)/outputs/

rsync -avz "$REMOTE" "$LOCAL"

echo "Sync from HPC complete. Outputs are in outputs/"
