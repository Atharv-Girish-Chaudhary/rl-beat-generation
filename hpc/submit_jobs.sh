#!/bin/bash
# Submit discriminator job, then chain PPO to start only after disc succeeds.
# Run ON the cluster: bash hpc/submit_jobs.sh
# Or via Makefile: make hpc-submit

set -e

cd /scratch/chaudhary.at/rl-beat-generation

mkdir -p logs

DISC_JOB=$(sbatch --parsable hpc/train_discriminator.sbatch)
echo "Submitted discriminator job: $DISC_JOB"

PPO_JOB=$(sbatch --parsable --dependency=afterok:$DISC_JOB hpc/train_ppo.sbatch)
echo "Submitted PPO job: $PPO_JOB (waits for job $DISC_JOB)"

echo ""
echo "make hpc-status"
echo "make hpc-cancel"
