#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${1:?Usage: scripts/infer_all.sh CHECKPOINT [OUTPUT_DIR] [GPU]}"
OUTPUT_DIR="${2:-outputs}"
GPU="${3:-0}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train.yaml}"

run_case() {
  local name="$1" fwhm="$2" angular="$3" radial="$4" dose="$5" background="$6"

  python infer.py \
    --config "${TRAIN_CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --model_type crossattn \
    --cond ct+lrpet \
    --sampler ddim \
    --ddim_steps 50 \
    --phase test \
    --gpu "${GPU}" \
    --out_dir "${OUTPUT_DIR}/${name}" \
    --dc_enable \
    --dc_fwhm_mm "${fwhm}" \
    --dc_rebin_factor "${angular}" \
    --dc_rebin_radial_factor "${radial}" \
    --dc_dose_alpha "${dose}" \
    --dc_background_beta "${background}" \
    --unit case
}

run_case "6mm" 4.5 2 1 0.10 0.05
run_case "8mm" 6.0 2 2 0.10 0.05
run_case "12mm" 8.0 3 2 0.05 0.10
