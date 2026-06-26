#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/train.yaml}"

python train.py --config "${CONFIG}" --pretrained none
