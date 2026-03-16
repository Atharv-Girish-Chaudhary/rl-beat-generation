# HPC Setup Guide — Northeastern Explorer Cluster

**RL Beat Generation — CS 5180 Final Project**  
Team: Atharv Chaudhary, Taha Ucar, Yixun Li

---

## 1. Getting Access

You need an active Northeastern HPC account. If you haven't used Explorer before, request access at [rc.northeastern.edu](https://rc.northeastern.edu). Your credentials are your regular Northeastern username and password.

---

## 2. Connecting to Explorer

```bash
ssh YOUR_NEU_USERNAME@explorer.northeastern.edu
```

Replace `YOUR_NEU_USERNAME` with your actual username (e.g., `ucar.t` or `li.yix` — check your Northeastern email prefix).

> **💡 Pro Tip: Use tmux** so your session survives disconnects and laptop sleep.
>
> ```bash
> tmux new -s beats          # start a session
> # Ctrl+B then D            # detach (leaves it running)
> tmux attach -t beats       # reconnect later
> ```

---

## 3. First-Time Setup (One Time Only)

### Step 1: Clone the Repo

```bash
cd /scratch/$USER
git clone https://github.com/Atharv-Girish-Chaudhary/rl-beat-generation.git
cd rl-beat-generation
```

> **📁 Storage Locations**
>
> - `~/` (home) — Small quota (~50–100 GB), permanent, backed up. Use for code.
> - `/scratch/$USER/` — Large quota (~1–10 TB), fast, auto-purged after 30–90 days. Use for datasets & checkpoints.

### Step 2: Get a Compute Node

⚠️ **Never run heavy commands on the login node — it will kill your process.**

```bash
srun --partition=short --mem=8G --cpus-per-task=4 --time=01:00:00 --pty /bin/bash
```

Wait for it to allocate. You'll see a new prompt with a different hostname (e.g., `c0604`) when you're on a compute node.

### Step 3: Create the Conda Environment

```bash
module load anaconda3/2024.06
conda create -n rl-beats python=3.11 -y
```

Then install packages using the **conda env's pip directly**:

```bash
/home/$USER/.conda/envs/rl-beats/bin/pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu121

/home/$USER/.conda/envs/rl-beats/bin/pip install numpy scipy matplotlib \
    tqdm librosa soundfile tensorboard stable-baselines3 gymnasium
```

> **⚠️ Important: Always use the full pip path**
>
> - If you just type `pip install`, it installs to the system Python, not your env.
> - Similarly, `conda activate` does NOT work in SLURM batch scripts.
> - In scripts, always use: `/home/$USER/.conda/envs/rl-beats/bin/python`

### Step 4: Verify Installation

```bash
module load cuda/12.1
/home/$USER/.conda/envs/rl-beats/bin/python -c \
    "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

On a CPU node, CUDA will show `False` — that's expected. It detects the GPU when you submit a GPU job.

### Step 5: Exit the Compute Node

```bash
exit
```

You're back on the login node. The conda env is permanent — you never need to create it again.

---

## 4. Running GPU Jobs

Our SLURM scripts are in the repo root. Before first use, replace the placeholder username:

```bash
sed -i 's|<YOUR_NEU_USERNAME>|YOUR_ACTUAL_USERNAME|g' test_gpu.sh
```

### Submit a Job

```bash
mkdir -p logs
sbatch test_gpu.sh          # Quick GPU sanity check
```

Training scripts will be added as we build the actual training code.

### Monitor Your Jobs

```bash
squeue -u $USER                             # Check job status
watch -n 10 squeue -u $USER                 # Auto-refresh every 10s
tail -f logs/gpu_test_<JOB_ID>.out          # Watch live output
cat logs/gpu_test_<JOB_ID>.err              # Check for errors
scancel <JOB_ID>                            # Cancel a job
sacct -j <JOB_ID> --format=JobID,State,Elapsed   # Post-run stats
```

### Job States

| State | Meaning | What To Do |
|-------|---------|------------|
| `PD` | Pending — waiting in queue | Wait, or try a different partition |
| `R` | Running | Check logs with `tail -f` |
| *(gone)* | Finished | Check `.out` and `.err` log files |

---

## 5. GPU Partitions & Hardware

| Partition | Max Time | GPU | Notes |
|-----------|----------|-----|-------|
| `gpu` | 8 hours | A100 / V100 | Main training — 1 GPU/job max, 4 concurrent jobs |
| `gpu-short` | 2 hours | A100 / V100 | Quick tests, faster queue |
| `gpu-interactive` | 2 hours | A100 / V100 | Interactive debugging with `srun` |

| GPU | VRAM | Best For |
|-----|------|----------|
| A100-SXM4 | 40 / 80 GB | Phase 2–3 training (larger models) |
| V100-SXM2 | 32 GB | Phase 1 training |
| V100-PCIE | 16 GB | Testing / quick runs |

---

## 6. Interactive GPU Sessions (for Debugging)

```bash
# Request an A100
srun --partition=gpu-interactive --gres=gpu:a100:1 \
    --mem=32G --cpus-per-task=4 --time=02:00:00 --pty /bin/bash

# Or a V100 (usually faster to get)
srun --partition=gpu-interactive --gres=gpu:v100:1 \
    --mem=16G --cpus-per-task=4 --time=02:00:00 --pty /bin/bash

# Once on the node:
module load cuda/12.1
module load anaconda3/2024.06
conda activate rl-beats
python your_script.py
```

> **⚠️ If "Requested node configuration is not available"**
>
> - Try a different GPU type: change `v100` to `a100` or vice versa
> - Try a different partition: `gpu-interactive` → `gpu-short` → `gpu`
> - For non-GPU tasks: `--partition=short` (CPU only, faster to get)

---

## 7. Open OnDemand (Browser Alternative)

If you prefer a browser-based interface instead of SSH:

1. Go to **<https://ood.discovery.neu.edu>**
2. Log in with your Northeastern credentials
3. **Interactive Apps → Jupyter Notebook** for debugging
4. **Files** to browse your `/scratch/` and home directories
5. **Jobs → Active Jobs** to monitor running training jobs

---

## 8. Git Setup on Explorer

First time pushing from Explorer, set up your identity:

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@gmail.com"
```

GitHub requires a **Personal Access Token (PAT)** for HTTPS pushes — passwords don't work:

1. Go to **github.com → Settings → Developer Settings → Personal Access Tokens → Tokens (classic)**
2. Generate new token (classic) with **repo** scope
3. Copy the token — you won't see it again
4. When pushing, use the token as your password

```bash
# Save credentials so you only enter the token once:
git config --global credential.helper store
git push origin main
# Username: your-github-username
# Password: paste your token (characters won't show — that's normal)
```

---

## 9. Quick Reference — Daily Workflow

```bash
# 1. SSH in
ssh YOUR_USERNAME@explorer.northeastern.edu

# 2. Reconnect tmux (or start new)
tmux attach -t beats   # or: tmux new -s beats

# 3. Go to repo
cd /scratch/$USER/rl-beat-generation

# 4. Pull latest code
git pull origin main

# 5. Submit your job
sbatch test_gpu.sh

# 6. Monitor
squeue -u $USER
tail -f logs/gpu_test_<JOB_ID>.out
```

---

## 10. Troubleshooting

| Problem | Fix |
|---------|-----|
| `conda create` gets **Killed** | Don't run on login node. Get a compute node with `srun` first. |
| `No module named 'torch'` | Installed to wrong Python. Reinstall using full path: `/home/$USER/.conda/envs/rl-beats/bin/pip install ...` |
| CUDA not available | Either on CPU node (expected) or forgot `module load cuda/12.1` |
| Node configuration not available | GPU nodes busy/down. Try different GPU type or partition. |
| `git push` auth failure | Use Personal Access Token as password, not your GitHub password. |
| Job pending forever | Check reason with `squeue`. Try `gpu-short` or a different GPU type. |
| Python output not showing in logs | Add `export PYTHONUNBUFFERED=1` to your SLURM script. |
| "channel open failed" messages | Harmless SSH noise. Ignore them. |
