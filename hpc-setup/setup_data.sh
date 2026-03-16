#!/bin/bash
# ─── Setup dataset symlinks ───
# Run this once to link datasets from /scratch to the repo.
# Assumes datasets have been uploaded/downloaded to /scratch/<username>/datasets/

SCRATCH="/scratch/<YOUR_NEU_USERNAME>"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "Setting up dataset symlinks..."
echo "Scratch: $SCRATCH"
echo "Repo:    $REPO_DIR"
echo ""

# Create datasets dir in repo if it doesn't exist
mkdir -p "$REPO_DIR/datasets"

# Symlink each dataset component
ln -sfn "$SCRATCH/datasets/samples_processed" "$REPO_DIR/datasets/samples_processed"
ln -sfn "$SCRATCH/datasets/groove_grids.npy"  "$REPO_DIR/datasets/groove_grids.npy"
ln -sfn "$SCRATCH/datasets/slakh_grids.npy"   "$REPO_DIR/datasets/slakh_grids.npy"
ln -sfn "$SCRATCH/datasets/musdb_spectrograms.npy" "$REPO_DIR/datasets/musdb_spectrograms.npy"

echo "Done. Linked datasets:"
ls -la "$REPO_DIR/datasets/"
