#!/bin/bash
#SBATCH --job-name=gpu-test
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/gpu_test_%j.out
#SBATCH --error=logs/gpu_test_%j.err

module load cuda/12.1
export PYTHONUNBUFFERED=1

PYTHON="/home/chaudhary.at/.conda/envs/rl-beats/bin/python"

echo "=== GPU Sanity Check ==="
echo "Node: $SLURM_NODELIST"
echo ""

nvidia-smi

echo ""
$PYTHON -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:             {torch.cuda.get_device_name(0)}')
    print(f'VRAM:            {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    # Quick matrix multiply test
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')
    torch.cuda.synchronize()
    import time
    start = time.time()
    for _ in range(100):
        z = x @ y
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f'MatMul test:     100x (1000×1000) in {elapsed:.3f}s')
    print()
    print('GPU is working.')
else:
    print('ERROR: No GPU detected!')
"
