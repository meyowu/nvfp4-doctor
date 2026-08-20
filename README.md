# nvfp4-doctor

`nvfp4-doctor` is the implementation repository for QContract, an eight-week
research sprint on contract-driven correctness and fault injection for NVFP4
inference kernels.

The project treats low-precision correctness as a systems property. It records
the requested backend, actual kernel identity, tensor and scale metadata,
environment provenance, and oracle comparisons before making performance or
correctness claims.

## Current status

Week 2 / Gate 1 and the bounded E003 synthetic fault experiment are complete
with a `continue` decision toward Gate 2. The repository contains a CPU-only oracle for E2M1,
UE4M3 scales, packed values, CUTLASS 128x4 scale layout, padding, scalar global
scales, and exact hierarchical reconstruction. Its semantics are pinned to
versioned public NVIDIA sources and hand-authored fixtures rather than the
candidate FlashInfer implementation.

Three constructed RTX 5080 differential cases matched FlashInfer's packed and
scale bytes exactly, including boundary padding and multi-atom layouts. This
supports the declared dense row-wise block-16 format contract only. It does not
establish arbitrary quantization rounding, GEMM correctness, model quality, or
performance.

Three E003 slices add exact structural/evidence contracts, six deterministic
format faults, five stride/backend-identity faults, and three packed-value
permutation families. The final held-out matrix applied block, row, and column
permutations to three new clean artifacts. All nine held-out faults were
detected and localized with zero clean false rejects or fault false accepts.
Real dispatch replay, GEMM, and model-level fault propagation remain pending.

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
PYTHONPATH=src python scripts/run_e002_gate1.py
PYTHONPATH=src python scripts/run_e003_format_faults.py
PYTHONPATH=src python scripts/run_e003_execution_faults.py
PYTHONPATH=src python scripts/run_e003_heldout_permutations.py
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
