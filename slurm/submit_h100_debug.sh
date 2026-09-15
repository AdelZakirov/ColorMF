#!/bin/bash
#SBATCH --partition=gpu-short
#SBATCH --job-name=aza-cmf-debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=colormf_debug_%j.out
#SBATCH --error=colormf_debug_%j.err

set -euo pipefail

PROJECT_ROOT=/mnt/WORKSPACE/aza_workspace/ColorMF
SIF="$PROJECT_ROOT/container/ColorMF.sif"
BASE_CONFIG="$PROJECT_ROOT/configs/pmf_s_64_colorization_h100_2gpu.yaml"
DEBUG_CONFIG="$PROJECT_ROOT/configs/.pmf_debug_${SLURM_JOB_ID}.yaml"

cd "$PROJECT_ROOT"
module load singularity/3.8.4

sed \
    -e 's/^experiment_name: .*/experiment_name: pMF-S-4-64-debug/' \
    -e 's/^  batch_size: 10$/  batch_size: 2/' \
    -e 's/^  num_workers: 8$/  num_workers: 2/' \
    -e 's/^  devices: 2$/  devices: 1/' \
    -e 's/^  max_epochs: 320$/  max_epochs: 1/' \
    -e 's/^  max_steps: null$/  max_steps: 2/' \
    -e 's/^  accumulate_grad_batches: 64$/  accumulate_grad_batches: 1/' \
    -e 's#checkpoints/pmf_s_4_64_h100_2gpu#checkpoints/debug_h100#' \
    -e 's#qualitative/pmf_s_4_64_h100_2gpu#qualitative/debug_h100#' \
    -e 's/pMF-S-4-64-colorization-h100-2gpu/pMF-S-4-64-debug/g' \
    "$BASE_CONFIG" > "$DEBUG_CONFIG"
trap 'rm -f "$DEBUG_CONFIG"' EXIT

nvidia-smi --query-gpu=name,memory.total --format=csv

srun singularity exec --nv \
    --home /mnt/WORKSPACE/aza_workspace/AZA \
    -B /tmp:/tmp -B /mnt:/mnt \
    "$SIF" python train.py --config "$DEBUG_CONFIG"