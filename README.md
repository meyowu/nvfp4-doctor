# nvfp4-doctor

`nvfp4-doctor` is the implementation repository for QContract, an eight-week
research sprint on contract-driven correctness and fault injection for NVFP4
inference kernels.

The project treats low-precision correctness as a systems property. It records
the requested backend, actual kernel identity, tensor and scale metadata,
environment provenance, and oracle comparisons before making performance or
correctness claims.

## Current status

The first RTX 5080 / WSL2 environment is operational. A FlashInfer CUTLASS
NVFP4 quantize-and-GEMM smoke test completed on `sm_120`, and Nsight Systems
captured the run. This establishes environment viability only; it is not a
correctness proof or benchmark result.

See [docs/setup-windows-wsl2.md](docs/setup-windows-wsl2.md) for the reproducible
host setup and [docs/progress.md](docs/progress.md) for the verified baseline.
Research and engineering rules are defined in [AGENTS.md](AGENTS.md).

## Quick start

Inside Ubuntu on WSL2:

```bash
git clone https://github.com/meyowu/nvfp4-doctor.git
cd nvfp4-doctor
uv venv --python 3.12
source .venv/bin/activate
uv pip install vllm==0.27.1 --torch-backend=auto
uv pip install -r requirements-research.txt
source ./activate-nvfp4-lab.sh
python smoke_nvfp4.py
```

Package pins reflect the verified 2026-08-19 environment. Review the setup
guide before changing CUDA, PyTorch, vLLM, or FlashInfer versions together.

## Repository policy

- Do not commit virtual environments, model weights, profiler traces, caches,
  tokens, or raw activation data.
- Every experiment must have a manifest, immutable artifacts, an oracle, and a
  stated acceptance criterion.
- Separate observed results from hypotheses and planned work.

Licensed under Apache-2.0.
