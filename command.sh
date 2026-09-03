#!/usr/bin/env bash
# Edit MOTIONDECODE_ROOT if the dataset is stored outside this repository.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOTIONDECODE_ROOT="${MOTIONDECODE_ROOT:-${PROJECT_ROOT}/dataset/MotionDecode/samples}"
JOBS="${JOBS:-4}"
LIMIT="${LIMIT:-0}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${PROJECT_ROOT}/.conda/bin/python" ]]; then
    PYTHON_BIN="${PROJECT_ROOT}/.conda/bin/python"
  elif [[ -x "/home/krixdina/miniconda3/envs/robot_retargeter/bin/python" ]]; then
    PYTHON_BIN="/home/krixdina/miniconda3/envs/robot_retargeter/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi
export PYTHON_BIN
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "${PROJECT_ROOT}/scripts/dr02_data_converter/run_dr02_batch.sh" \
  --input-dir "${MOTIONDECODE_ROOT}" --dataset-root "${MOTIONDECODE_ROOT}" \
  --jobs "${JOBS}" --limit "${LIMIT}" --fps 120
