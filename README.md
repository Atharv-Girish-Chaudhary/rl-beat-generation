# HPC Setup — Northeastern Explorer Cluster

## Quick Start

### 1. Connect to Explorer
```bash
ssh <YOUR_NEU_USERNAME>@explorer.northeastern.edu
```

Start a **tmux** session so your work survives disconnects:
```bash
tmux new -s beats
```

### 2. Clone the repo (first time only)
```bash
cd /scratch/<YOUR_NEU_USERNAME>
git clone git@github.com:<YOUR_GITHUB_USERNAME>/rl-beat-generation.git
cd rl-beat-generation
```

> **Storage strategy:** Code lives in `~/` (permanent, backed up). Large datasets and checkpoints go in `/scratch/<YOUR_NEU_USERNAME>/` (fast, large quota, auto-purged after 30–90 days).

### 3. Set up the conda environment (first time only)
```bash
module load anaconda3/2024.06
conda create -n rl-beats python=3.11 -y
conda activate rl-beats

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy matplotlib tqdm librosa soundfile tensorboard stable-baselines3 gymnasium

# Verify CUDA
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

> **Important:** `conda activate` does NOT work inside SLURM batch scripts.
> Use the full path instead: `/home/<YOUR_NEU_USERNAME>/.conda/envs/rl-beats/bin/python`

### 4. Submit a training job
```bash
mkdir -p logs
sbatch hpc/train_phase1.sh
```

### 5. Monitor your jobs
```bash
squeue -u <YOUR_NEU_USERNAME>                    # Check status
tail -f logs/job_<JOB_ID>.out                    # Watch live output
sacct -j <JOB_ID> --format=JobID,State,Elapsed   # Post-run stats
scancel <JOB_ID>                                  # Cancel a job
```

## Using Open OnDemand (Browser-Based)

1. Go to https://ood.discovery.neu.edu
2. **Interactive Apps → Jupyter Notebook** for debugging
3. **Files** to browse your directories
4. **Jobs → Active Jobs** to monitor training

## GPU Partitions

| Partition | Max Time | GPU | Notes |
|-----------|----------|-----|-------|
| `gpu` | 8 hours | A100 / V100 | Main training — **1 GPU/job max, 4 concurrent jobs** |
| `gpu-short` | 2 hours | A100 / V100 | Quick tests, faster queue |
| `gpu-interactive` | 2 hours | A100 / V100 | Interactive `srun` sessions for debugging |
| `courses-gpu` | 24 hours | 34 GPUs | Needs `rc/courses` group membership |

### GPU Hardware

| GPU | VRAM | Best For |
|-----|------|----------|
| A100-SXM4 | 40 / 80 GB | Phase 2–3 training (larger models) |
| V100-SXM2 | 32 GB | Phase 1 training |
| V100-PCIE | 16 GB | Testing / quick runs |

## Interactive GPU Session (for debugging)

```bash
# A100
srun --partition=gpu-interactive --gres=gpu:a100:1 --mem=32G --cpus-per-task=4 --pty /bin/bash

# V100 (usually faster to get)
srun --partition=gpu-interactive --gres=gpu:v100:1 --mem=16G --cpus-per-task=4 --pty /bin/bash

# Once inside:
module load anaconda3/2024.06
conda activate rl-beats
python train.py
```

## File Structure
```
hpc/
├── README.md              ← You are here
├── train_phase1.sh        ← SLURM job: Phase 1 (PPO, drums-only)
├── train_phase2.sh        ← SLURM job: Phase 2 (PPO, full grid + discriminator)
├── train_phase3.sh        ← SLURM job: Phase 3 (SAC, audio effects)
├── setup_data.sh          ← Symlink datasets from scratch to repo
└── test_gpu.sh            ← Quick GPU sanity check
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `conda activate` fails in batch job | Use full path: `/home/<user>/.conda/envs/rl-beats/bin/python` |
| Python prints nothing in `.out` log | Add `export PYTHONUNBUFFERED=1` to SLURM script |
| `git pull` auth error | Switch to SSH: `git remote set-url origin git@github.com:user/repo.git` |
| Job rejected: `PartitionTimeLimit` | Lower `--time` to fit partition max (8h for `gpu`) |
| Job pending: `QOSMaxGRESPerJob` | Use `--gres=gpu:a100:1` (max 1 GPU per job) |
| Job pending forever | Try `gpu-short` partition for quicker access |
