# nvfp4-doctor

`nvfp4-doctor` is the implementation repository for QContract, an eight-week
research sprint on contract-driven correctness and fault injection for NVFP4
inference kernels.

The project treats low-precision correctness as a systems property. It records
the requested backend, actual kernel identity, tensor and scale metadata,
environment provenance, and oracle comparisons before making performance or
correctness claims.

## Current status

Week 1 / Gate 0 is complete with a `go` decision. On the pinned RTX 5080 / WSL2
environment, three FlashInfer CUTLASS NVFP4 quantize-and-GEMM smoke runs
completed on `sm_120`. Nsight Systems attributed the same target kernel set to
the `e001:nvfp4_gemm` range in all three runs, and the bounded classifier did
not detect a known fallback signature in that range.

This establishes repeatable environment and dispatch evidence for the tested
matrix only. It is not a format-correctness proof or benchmark result.

See [docs/setup-windows-wsl2.md](docs/setup-windows-wsl2.md) for the reproducible
host setup and [docs/progress.md](docs/progress.md) for the verified baseline.
The model-validation sequence is specified in
[docs/research-plan.md](docs/research-plan.md), and the research boundary is
frozen in [docs/scope.md](docs/scope.md). Research and engineering rules are
defined in [AGENTS.md](AGENTS.md).

To resume a research session, read [research-state.md](research-state.md) and run:

```bash
python scripts/research_status.py
```

## Quick start

Inside Ubuntu on WSL2:

```bash
git clone https://github.com/meyowu/nvfp4-doctor.git
cd nvfp4-doctor
uv venv --python 3.12
source .venv/bin/activate
uv pip install vllm==0.27.1 --torch-backend=auto
uv pip install -r requirements-research.txt
uv pip install -r requirements-dev.txt
source ./activate-nvfp4-lab.sh
python smoke_nvfp4.py
./scripts/run_e001_week1.sh 3
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
