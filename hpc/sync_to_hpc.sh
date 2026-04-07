#!/bin/bash
# Sync local code and data to Explorer scratch.
# Run from local machine: bash hpc/sync_to_hpc.sh

set -e

LOCAL=/Users/atharvchaudhary/PERSONAL/GitHub/rl-beat-generation/
REMOTE=explorer:/scratch/chaudhary.at/rl-beat-generation/

rsync -avz \
    --exclude='outputs/' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git/' \
    --exclude='*.egg-info/' \
    "$LOCAL" "$REMOTE"

echo "Sync to HPC complete."
