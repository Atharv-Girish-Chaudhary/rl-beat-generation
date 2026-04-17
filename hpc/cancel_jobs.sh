#!/bin/bash
# Cancel all running/pending jobs for this user.
# Run ON the cluster: bash hpc/cancel_jobs.sh
# Or via Makefile: make hpc-cancel

: "${HPC_USER:?HPC_USER is not set. Source hpc/env.sh first.}"

scancel -u ${HPC_USER}
echo "All jobs for ${HPC_USER} have been cancelled."
echo "Check queue with: squeue -u ${HPC_USER}"
