#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE_STEM="$ROOT/.local/profiles/e004-layer-00-o-proj"
RUN_EVIDENCE="$ROOT/artifacts/E004-qwen3-layer-capture/replay/layer-00-o-proj.json"

source "$ROOT/activate-nvfp4-lab.sh"
mkdir -p "$(dirname -- "$PROFILE_STEM")" "$(dirname -- "$RUN_EVIDENCE")"
rm -f -- "${PROFILE_STEM}.sqlite"

cd "$ROOT"
PYTHONPATH="$ROOT/src" nsys profile \
  --trace=cuda,nvtx,osrt \
  --output="$PROFILE_STEM" \
  --force-overwrite=true \
  "$ROOT/.venv/bin/python" "$ROOT/scripts/run_e004_projection_replay.py" \
  --layer 0 \
  --projection o_proj \
  --rows 16 \
  --seed 0 \
  --repetitions 3 \
  --output "$RUN_EVIDENCE"

PYTHONPATH="$ROOT/src" "$ROOT/.venv/bin/python" \
  "$ROOT/scripts/finalize_e004_projection_replay.py" \
  --run-evidence "$RUN_EVIDENCE" \
  --report "${PROFILE_STEM}.nsys-rep"
