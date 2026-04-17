#!/bin/bash
# Phase 2: Submit discriminator job, then chain PPO to start only after disc succeeds.
# Run ON the cluster: bash hpc/submit_jobs_phase2.sh

set -e

SCRATCH="${SCRATCH:-/scratch/${USER}}"

cd ${SCRATCH}/rl-beat-generation

mkdir -p logs

DISC_JOB=$(sbatch --parsable hpc/train_discriminator_phase2.sbatch)
echo "Submitted Phase 2 discriminator job: $DISC_JOB"

PPO_JOB=$(sbatch --parsable --dependency=afterok:$DISC_JOB hpc/train_ppo_phase2.sbatch)
echo "Submitted Phase 2 PPO job: $PPO_JOB (waits for job $DISC_JOB)"

echo ""
echo "make hpc-status"
echo "make hpc-cancel"
