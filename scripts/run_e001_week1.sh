#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS="${1:-3}"

if ! [[ "$RUNS" =~ ^[0-9]+$ ]] || (( RUNS < 3 )); then
  echo "E001 requires at least three repetitions" >&2
  exit 2
fi

source "$ROOT/activate-nvfp4-lab.sh"
ARTIFACT_DIR="$ROOT/.local/artifacts/E001-kernel-identity/week1"
PROFILE_DIR="$ROOT/.local/profiles"
mkdir -p "$ARTIFACT_DIR" "$PROFILE_DIR"

manifests=()
for ((run = 1; run <= RUNS; run++)); do
  run_id="$(printf 'run-%02d' "$run")"
  report="$PROFILE_DIR/e001-$run_id.nsys-rep"
  manifest="$ARTIFACT_DIR/$run_id.json"

  echo "[$run_id] profiling"
  rm -f -- "${report%.nsys-rep}.sqlite"
  PYTHONPATH="$ROOT/src" nsys profile \
    --trace=cuda,nvtx,osrt \
    --output="${report%.nsys-rep}" \
    --force-overwrite=true \
    "$ROOT/.venv/bin/python" "$ROOT/smoke_nvfp4.py"

  echo "[$run_id] collecting manifest"
  PYTHONPATH="$ROOT/src" "$ROOT/.venv/bin/python" \
    "$ROOT/scripts/collect_e001_manifest.py" --output "$manifest"
  PYTHONPATH="$ROOT/src" "$ROOT/.venv/bin/python" \
    "$ROOT/scripts/attach_e001_nsys.py" \
    --manifest "$manifest" --report "$report"
  manifests+=("$manifest")
done

PYTHONPATH="$ROOT/src" "$ROOT/.venv/bin/python" \
  "$ROOT/scripts/summarize_e001_runs.py" \
  "${manifests[@]}" \
  --output "$ROOT/experiments/E001-kernel-identity/results.json"
cp -- "${manifests[0]}" "$ROOT/experiments/E001-kernel-identity/manifest.json"
