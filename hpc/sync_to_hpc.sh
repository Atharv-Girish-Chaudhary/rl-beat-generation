#!/bin/bash
# Sync local code and data to Explorer scratch.
# Run from local machine: bash hpc/sync_to_hpc.sh

set -e

: "${HPC_USER:?HPC_USER is not set. Source hpc/env.sh first.}"
: "${HPC_REMOTE:=explorer}"
: "${HPC_SCRATCH:=/scratch/${HPC_USER}}"

LOCAL=$(git rev-parse --show-toplevel)/
REMOTE=${HPC_REMOTE}:${HPC_SCRATCH}/rl-beat-generation/

rsync -avz \
    --exclude='outputs/' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git/' \
    --exclude='.venv/' \
    --exclude='*.egg-info/' \
    "$LOCAL" "$REMOTE"


echo "Sync to HPC complete."
