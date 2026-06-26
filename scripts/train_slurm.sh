#!/usr/bin/env bash
#SBATCH --job-name=pet_cdm_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00

set -euo pipefail

GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
CONFIG="${1:-configs/train.yaml}"

mkdir -p logs
torchrun --standalone --nnodes=1 --nproc_per_node="${GPUS_PER_NODE}" \
  train.py --config "${CONFIG}" --pretrained none
