HPC_REMOTE = explorer
HPC_PATH   = /scratch/chaudhary.at/rl-beat-generation

# ── HPC workflow ────────────────────────────────────────────────────────────

# Push local code + data to Explorer scratch (excludes outputs/)
hpc-sync:
	bash hpc/sync_to_hpc.sh

# First-time only: create conda env on Explorer
hpc-setup:
	ssh $(HPC_REMOTE) "bash $(HPC_PATH)/hpc/setup_env.sh"

# Submit discriminator job, then PPO job (chained)
hpc-submit:
	ssh $(HPC_REMOTE) "cd $(HPC_PATH) && bash hpc/submit_jobs.sh"

# Check job queue
hpc-status:
	ssh $(HPC_REMOTE) "squeue -u chaudhary.at"

# List recent log files on cluster
hpc-logs:
	ssh $(HPC_REMOTE) "ls -lt $(HPC_PATH)/logs/ | head -20"

# Tail the most recent log file
hpc-tail:
	ssh $(HPC_REMOTE) "tail -f \$$(ls -t $(HPC_PATH)/logs/*.out 2>/dev/null | head -1)"

# Pull outputs/ (checkpoints, plots, WAV) back to local
hpc-pull:
	bash hpc/sync_from_hpc.sh

# Cancel all your running/pending jobs
hpc-cancel:
	ssh $(HPC_REMOTE) "cd $(HPC_PATH) && bash hpc/cancel_jobs.sh"

# Cancel all jobs, then resubmit fresh
hpc-restart:
	ssh $(HPC_REMOTE) "scancel -u chaudhary.at && cd $(HPC_PATH) && bash hpc/submit_jobs.sh"

.PHONY: hpc-sync hpc-setup hpc-submit hpc-status hpc-logs hpc-tail hpc-pull hpc-cancel hpc-restart
