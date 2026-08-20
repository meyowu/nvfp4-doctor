# Verified Research Baseline

Verified on 2026-08-19. This document records observations, not portability or
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
| Nsight Systems | 2026.1.3 |
| Nsight Compute | 2026.2.1 |

Compiler-side CUDA packages were pinned to 13.2.86 to avoid mixing CUDA 13.2
headers with a CUDA 13.3 compiler.

## Smoke-test observation

The checked-in smoke test exercised the FlashInfer CUTLASS NVFP4 quantizer and
FP4 GEMM at shape `(16, 128)` on the RTX 5080. The output was finite. Against
the unquantized FP32 matrix-multiplication reference, the observed maximum
absolute error was `0.45729243755340576` and the mean absolute error was
`0.10718676447868347`.

These values prove only that the selected path executed end to end. They do not
establish an error bound, cross-backend agreement, model-level quality, or a
performance advantage.

Nsight Systems also produced a valid report for the run. Profiler reports are
local artifacts and are intentionally excluded from Git.

## Operational note

The first JIT build exceeded available memory under parallel compilation. The
verified workaround limits build parallelism with `MAX_JOBS=1` and
`FLASHINFER_NVCC_THREADS=1`; `activate-nvfp4-lab.sh` exports both variables.

## Next gate

The next milestone is E001: capture backend request and actual kernel identity,
packed tensor shape/dtype/stride, scale metadata, version and hardware
provenance, deterministic seeds, and artifact hashes. Only after this record is
complete should contract and fault-injection experiments be treated as evidence.
