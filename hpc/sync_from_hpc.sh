#!/bin/bash
# Pull outputs (checkpoints, plots, WAV) from Explorer back to local.
# Run from local machine: bash hpc/sync_from_hpc.sh

set -e

REMOTE=explorer:/scratch/chaudhary.at/rl-beat-generation/outputs/
LOCAL=/Users/atharvchaudhary/PERSONAL/GitHub/rl-beat-generation/outputs/

rsync -avz "$REMOTE" "$LOCAL"

echo "Sync from HPC complete. Outputs are in outputs/"
