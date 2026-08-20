# Verified Research Baseline

Verified on 2026-08-20. This document records observations, not portability or
performance claims.

## Host and software

| Component | Verified value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5080, 16 GB |
| Compute capability | 12.0 (`sm_120`) |
| Windows driver | 595.95 |
| WSL | 2.7.12, kernel 6.18.33.2 |
| Distribution | Ubuntu 24.04.4 LTS |
| Python | 3.12 |
| uv | 0.12.5 |
| PyTorch | 2.13.0+cu132 |
| vLLM | 0.27.1 |
| FlashInfer | 0.6.16.post3 |
| CUDA compiler-side packages | 13.2.86 |
| CUDA nvdisasm | 13.3.73 |
| Nsight Systems | 2026.1.3 |
| Nsight Compute | 2026.2.1 |

Compiler-side CUDA packages were pinned to 13.2.86 to avoid mixing CUDA 13.2
headers with a CUDA 13.3 compiler. The independently packaged `nvdisasm` binary
is pinned to 13.3.73 because CUTLASS DSL 4.6 requires `>=13.3,<14`; the final
`uv pip check` reported that all 205 installed packages were compatible.

## Gate 0 observation

The public package skeleton now contains the frozen framework-independent
boundaries for environment evidence, formats, checkpoints, capture, backends,
oracles, contracts, faults, minimization, and reporting. Import tests cover all
boundaries without importing PyTorch, vLLM, or FlashInfer.

Three controlled repetitions of the checked-in smoke test exercised the
FlashInfer CUTLASS NVFP4 quantizer and FP4 GEMM at shape `(16, 128)`. Every run
reported finite output with maximum absolute error
`0.45729243755340576` and mean absolute error `0.10718676447868347` against
the unquantized FP32 matrix-multiplication reference.

These values prove only that the selected path executed end to end. They do not
establish an error bound, cross-backend agreement, model-level quality, or a
performance advantage.

Nsight Systems produced three local reports. Each manifest recorded 23 unique
kernel names, and the four names attributed to `e001:nvfp4_gemm` had the same
set hash in all repetitions. The target range contained the expected SM120
block-scaled CUTLASS E2M1 kernel and no known fallback signature, so the bounded
assessment was `not_detected` three times. The repository-tracked
`experiments/E001-kernel-identity/results.json` records the report and manifest
hashes. Raw profiler reports remain local and excluded from Git.

## Operational note

The first JIT build exceeded available memory under parallel compilation. The
verified workaround limits build parallelism with `MAX_JOBS=1` and
`FLASHINFER_NVCC_THREADS=1`; `activate-nvfp4-lab.sh` exports both variables.

## Next gate

Gate 0 passed with a `go` decision. Gate 1 is the independent format oracle:
exhaustive E2M1 decoding, E4M3 scale handling, packing and layout golden
fixtures, and exact scale-index reconstruction. No Week 1 observation is a
claim that those format semantics are correct.
