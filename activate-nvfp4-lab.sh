#!/usr/bin/env bash
# Source this file: source ~/projects/nvfp4-doctor/activate-nvfp4-lab.sh
NVFP4_DOCTOR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NVFP4_DOCTOR_ROOT
export XDG_CACHE_HOME="$NVFP4_DOCTOR_ROOT/.local/cache"
export HF_HOME="$NVFP4_DOCTOR_ROOT/.local/models/huggingface"
export UV_CACHE_DIR="$NVFP4_DOCTOR_ROOT/.local/cache/uv"
export TORCH_EXTENSIONS_DIR="$NVFP4_DOCTOR_ROOT/.local/cache/torch_extensions"
export CUDA_HOME="$NVFP4_DOCTOR_ROOT/.venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$NVFP4_DOCTOR_ROOT/.venv/bin:$CUDA_HOME/bin:$HOME/.local/bin:/usr/local/cuda-13.2/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MAX_JOBS=1
export FLASHINFER_NVCC_THREADS=1
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$UV_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" \
  "$NVFP4_DOCTOR_ROOT/.local/artifacts" "$NVFP4_DOCTOR_ROOT/.local/profiles"
