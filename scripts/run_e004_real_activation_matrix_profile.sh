#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
LAB_ROOT="/home/meyowu/projects/nvfp4-doctor"
REVISION="ccd10a893cbca613259517c3efe08e151ddf2b8e"
PROFILE_STEM="$ROOT/.local/profiles/e004-real-activation-unfused-matrix"
RUN_EVIDENCE="$ROOT/artifacts/E004-qwen3-layer-capture/real-activation-matrix/unfused-matrix.json"
ARTIFACT_ROOT="$ROOT/artifacts/E004-qwen3-layer-capture/real-activation-matrix"
MODEL_DIR="$ROOT/models/nvidia--Qwen3-8B-NVFP4/$REVISION"

source "$LAB_ROOT/activate-nvfp4-lab.sh"
mkdir -p "$(dirname -- "$PROFILE_STEM")" "$ARTIFACT_ROOT"
rm -f -- "${PROFILE_STEM}.nsys-rep" "${PROFILE_STEM}.sqlite"

cd "$ROOT"
PYTHONPATH="$ROOT/src" nsys profile \
  --trace=cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --flush-on-cudaprofilerstop=true \
  --sample=none \
  --cpuctxsw=none \
  --wait=primary \
  --output="$PROFILE_STEM" \
  --force-overwrite=true \
  "$LAB_ROOT/.venv/bin/python" \
  -m scripts.run_e004_real_activation_matrix \
  --model-dir "$MODEL_DIR" \
  --artifact-root "$ARTIFACT_ROOT" \
  --output "$RUN_EVIDENCE" \
  --profile-capture

PYTHONPATH="$ROOT/src" "$LAB_ROOT/.venv/bin/python" \
  -m scripts.finalize_e004_real_activation_matrix \
  --run-evidence "$RUN_EVIDENCE" \
  --report "${PROFILE_STEM}.nsys-rep"
