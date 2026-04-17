#!/bin/bash
# Submit discriminator job, then chain PPO to start only after disc succeeds.
# Run ON the cluster: bash hpc/submit_jobs.sh
# Or via Makefile: make hpc-submit

set -e

: "${HPC_USER:?HPC_USER is not set. Source hpc/env.sh first.}"
: "${HPC_SCRATCH:=/scratch/${HPC_USER}}"

cd ${HPC_SCRATCH}/rl-beat-generation

mkdir -p logs

DISC_JOB=$(sbatch --parsable hpc/train_discriminator.sbatch)
echo "Submitted discriminator job: $DISC_JOB"

PPO_JOB=$(sbatch --parsable --dependency=afterok:$DISC_JOB hpc/train_ppo.sbatch)
echo "Submitted PPO job: $PPO_JOB (waits for job $DISC_JOB)"

echo ""
echo "make hpc-status"
echo "make hpc-cancel"
