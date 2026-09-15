#!/bin/bash
#SBATCH --partition=gpu-medium
#SBATCH --job-name=colormf-h100-2gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=32
#SBATCH --time=6-00:00:00
#SBATCH --output=/mnt/WORKSPACE/aza_workspace/ColorMF/slurm/colormf_h100_2gpu_%j.out
#SBATCH --error=/mnt/WORKSPACE/aza_workspace/ColorMF/slurm/colormf_h100_2gpu_%j.err

set -euo pipefail

PROJECT_ROOT=/mnt/WORKSPACE/aza_workspace/ColorMF
SIF="$PROJECT_ROOT/container/ColorMF.sif"

cd "$PROJECT_ROOT"
module load singularity/3.8.4
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export NCCL_ASYNC_ERROR_HANDLING=1

nvidia-smi --query-gpu=name,memory.total --format=csv

srun --kill-on-bad-exit=1 --wait=60 singularity exec --nv \
    --home /mnt/WORKSPACE/aza_workspace/AZA \
    -B /tmp:/tmp -B /mnt:/mnt \
    "$SIF" python train.py \
    --config configs/pmf_s_64_colorization_h100_2gpu.yaml