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

## Gate 1 observation

The independent Gate 1 oracle is CPU-only and does not import PyTorch,
FlashInfer, vLLM, or CUDA. It exhaustively checks all 16 E2M1 payloads and all
127 finite UE4M3 scale codes. Hand-authored fixtures cover packing, signed zero,
subnormal and maximum scales, CUTLASS 128x4 offsets, physical padding, and a
hierarchical reconstruction with an explicit scalar FP32 global scale.

Hypothesis properties generated reversible packed values and scale layouts over
boundary shapes. A pinned FlashInfer differential then exercised `(128, 64)`,
`(17, 80)`, and `(129, 80)` BF16 matrices at three global scales. Packed bytes,
linear scale bytes, and CUTLASS 128x4 scale bytes matched the independent oracle
exactly in all cases; reconstructed inputs had maximum absolute error `0.0`.
The largest and multi-atom cases each covered all 127 finite UE4M3 codes.

The final verification reported 58 CPU tests plus 210 subtests passing, 23
Mypy source files clean, 47 Python files formatted, and all 207 installed
packages compatible.

These observations support only the declared dense row-wise block-16 format
contract. They do not establish arbitrary quantizer rounding, GEMM numerical
correctness, model-level quality, or support for adapter-specific layouts not
listed in the contract.

## E003 format-fault slice observation

The first E003 slice defines six exact contracts with explicit domains,
preconditions, invariants, mismatch metrics, zero-mismatch thresholds, and
limitations. It injects deterministic, reversible, `synthetic` positive
controls for packed-nibble order, scale-index shift, block-scale reversal,
global scale, scale-layout metadata, and physical padding.

Three clean CPU artifacts produced 18 passing contract evaluations. All six
faults were detected, and every observed failed-contract set exactly matched
its declared expected set. Clean false rejects, fault false accepts,
localization failures, and reversibility failures were all zero.

The padding control changed one of 1,403 physical padding bytes in a `(129, 80)`
CUTLASS-layout artifact. The scale-padding contract detected it while logical
reconstruction remained unchanged. This is evidence for the bounded synthetic
matrix only, not a frequency estimate for real faults.

The second E003 slice separates six exact evidence contracts for recorded
stride, row-major contiguity, requested backend, reported backend, observed
kernel tuple, and bounded fallback status. One clean snapshot passed every
contract. Two stride faults and three backend-identity faults were all detected
with exact failed-contract sets and zero false accepts, clean false rejects,
localization failures, or reversibility failures.

Changing only the reported backend left the requested-backend and
observed-kernel contracts passing. The observed fallback kernel was an explicit
synthetic input, not a profiler observation from this run; real dispatch
identity still requires target-range profiler evidence.

The final CPU verification for these slices reported 77 tests plus 248 subtests
passing, 27 Mypy source files clean, 56 Python files formatted, and all 207
installed packages compatible. No GPU integration was required for this
CPU-only work.

## E003 held-out permutation observation

The final E003 slice added deterministic, reversible cyclic permutations of
complete packed-value blocks, rows, and logical columns. It evaluated all three
fault families over held-out `(3, 48)`, `(5, 64)`, and `(131, 80)` artifacts
with distinct data salts and both linear and CUTLASS 128x4 scale layouts.

The exact zero-mismatch thresholds came from Gate 1 and were frozen before this
matrix was constructed; none was tuned on the held-out cases. All 18 clean
contract evaluations passed. All nine injected faults were detected and
localized to the packed-value and reconstruction contracts, with zero false
accepts, clean false rejects, localization failures, or reversibility failures.

The final CPU verification reported 83 tests plus 261 subtests passing, 27 Mypy
source files clean, 58 Python files formatted, and all 207 installed packages
compatible. These results complete the declared synthetic E003 matrix only;
they do not estimate naturally occurring fault rates or establish runtime,
GEMM, or model-level detection.

## Next gate

Gate 1 and E003 passed their bounded criteria with a `continue` decision. The
first E004 slice pinned revision
`ccd10a893cbca613259517c3efe08e151ddf2b8e` of
`nvidia/Qwen3-8B-NVFP4` and inspected four public metadata files without
downloading weights. The declarations agree on NVFP4 group size 16, and each
planned projection family has four quantization tensor names in all 36 layers.

The second E004 slice issued four exact HTTP range requests totaling
134,032 bytes. Both shard URLs returned HTTP 206 with matching range and length
boundaries. The parsed headers described 1,227 tensors whose names exactly
matched the pinned index; all tensor intervals were contiguous and covered the
declared payloads. No tensor payload byte was downloaded.

The five capture targets account for 720 indexed tensors. Their packed weights
are stored as U8, block scales as F8_E4M3, and the two scalar scales as F32,
uniformly across 36 layers. These stored shapes and dtypes support acquisition
planning only; they do not establish logical layouts, runtime strides, backend
support, or numerical behavior.

The third E004 slice fixed layers 0, 18, and 35 as early, middle, and late
representatives. It mapped four required tensors for each of five projection
families to 60 unique, non-overlapping, in-bounds HTTP ranges. Each layer
accounts for 103,809,064 bytes and the full plan accounts for 311,427,192 bytes.
Construction repeated only the four header requests and downloaded zero payload
bytes.

The next bounded step requires separate authorization to acquire those planned
payload ranges into ignored local storage and hash each result. No current
observation is a claim about payload validity, real dispatch, GEMM accumulation,
or model outputs.
