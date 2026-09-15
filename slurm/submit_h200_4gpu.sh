#!/bin/bash
#SBATCH --partition=gpu-medium
#SBATCH --job-name=colormf-h200-4gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --constraint=h200
#SBATCH --cpus-per-task=64
#SBATCH --time=6-00:00:00
#SBATCH --output=colormf_h200_4gpu_%j.out
#SBATCH --error=colormf_h200_4gpu_%j.err

set -euo pipefail

PROJECT_ROOT=/mnt/WORKSPACE/aza_workspace/ColorMF
SIF="$PROJECT_ROOT/container/ColorMF.sif"

cd "$PROJECT_ROOT"
module load singularity/3.8.4

nvidia-smi --query-gpu=name,memory.total --format=csv

srun singularity exec --nv \
    --home /mnt/WORKSPACE/aza_workspace/AZA \
    -B /tmp:/tmp -B /mnt:/mnt \
    "$SIF" python train.py \
    --config configs/pmf_s_64_colorization_h200_4gpu.yaml