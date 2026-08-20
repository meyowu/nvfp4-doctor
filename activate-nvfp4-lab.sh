#!/usr/bin/env bash
# Source this file: source ~/src/nvfp4-doctor/activate-nvfp4-lab.sh
export CUDA_HOME="$HOME/src/nvfp4-doctor/.venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$HOME/src/nvfp4-doctor/.venv/bin:$HOME/.local/bin:/usr/local/cuda-13.2/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MAX_JOBS=1
export FLASHINFER_NVCC_THREADS=1
