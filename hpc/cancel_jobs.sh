#!/bin/bash
# Cancel all running/pending jobs for this user.
# Run ON the cluster: bash hpc/cancel_jobs.sh
# Or via Makefile: make hpc-cancel

scancel -u chaudhary.at
echo "All jobs for chaudhary.at have been cancelled."
echo "Check queue with: squeue -u chaudhary.at"
