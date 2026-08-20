# Research State

This is the canonical handoff record for resumable research sessions. Update it
only from observed repository, environment, test, and experiment evidence.

## Current handoff

- Current experiment: `E003-synthetic-faults`
- Current gate: `Gate 2 preparation — Synthetic Fault Injection`
- Status: `in_progress`
- Decision: `continue`
- Verified baseline commit: `18d9e14cad813abccce09821099517f6ff769be5`
- Verified Gate 1 implementation commit: `f0be51513892d8b10968090fb081a8dafbee0b89`
- Active local branch: `exp/e003-synthetic-faults` (uncommitted)
- Active Codex worktree: `/mnt/c/Users/meyow/Documents/Codex/2026-08-19/referenced-chatgpt-conversation-this-is-an/nvfp4-doctor`
- Canonical WSL checkout: `/home/meyowu/projects/nvfp4-doctor`
- Canonical GPU environment: `/home/meyowu/projects/nvfp4-doctor/.venv`
- Hardware boundary: one RTX 5080 (`sm_120`, 16 GB)

## Completed evidence

- The RTX 5080 WSL2 research environment is operational.
- A FlashInfer CUTLASS NVFP4 quantize-and-GEMM smoke run completed with finite
  output at shape `(16, 128)`.
- Nsight Systems 2026.1.3 produced a retained, ignored profiler report for the
  smoke run with SHA-256
  `62fecf2f03911cd092c32c539c80145bc74a97711fbbe4d7f64d0b9e7507df86`.
- The initial `src/nvfp4_doctor` package and CPU test boundaries exist.
- Project code, environment, caches, models, artifacts, and profiler outputs now
  have a canonical project-local directory layout.
- E001 manifest schema version 1 now has an immutable, CPU-only data model and a
  hand-authored golden JSON fixture.
- Strict tests reject unknown fields, unsupported schema versions, missing
  backend evidence, and invalid tensor stride ranks.
- CPU-only collectors now capture the single-GPU fingerprint and host/software
  versions through read-only commands and package metadata. Their logic is
  tested with injected observations and does not import GPU frameworks.
- A real WSL collector run observed driver `595.95`, runtime package `13.2.75`,
  CUDA toolkit build `13.2.86`, WSL `2.7.12.0`, and the pinned PyTorch, vLLM,
  and FlashInfer versions.
- Manifest assembly now combines host/software, Git, command, tensor, and
  requested-backend evidence while leaving profiler-derived fields unknown.
- A local pre-profiler manifest was generated under ignored artifact storage;
  profiler evidence was then attached without overwriting unknown conclusions.
- Independent parsing of the Nsight `cuda_gpu_kern_sum` CSV observed 18 unique
  CUDA kernel names, including an SM120 block-scaled CUTLASS kernel with E2M1
  operands and the TensorRT-LLM NVFP4 block-quantization kernel.
- The project activation script now exports `LIBRARY_PATH` for package-local
  CUDA runtime and WSL driver libraries; this repaired an observed JIT linker
  failure for `-lcudart` and `-lcuda`.
- The synthetic workload now uses separate NVTX ranges for quantization, NVFP4
  GEMM, and the unquantized reference, preventing reference cuBLAS kernels from
  contaminating the fallback assessment.
- The E001 classifier returns `not_detected` only when the expected SM120
  block-scaled CUTLASS E2M1 signature is present and no known fallback signature
  occurs inside `e001:nvfp4_gemm`; unfamiliar evidence remains `unknown`.
- Three controlled runs each completed with identical numerical observations,
  23 unique observed kernel names, fallback status `not_detected`, and the same
  four-kernel target-range set hash
  `98aa16a78e2cdb3d74f27a0568bc2ed9f35de0e8584936717180c5e9c76ef10a`.
- The generated repeatability summary records stable environment and target
  kernel evidence, complete profiler hashes, `gate0_repeatability=pass`, and a
  `go` decision.
- The repository-tracked representative manifest records inputs, packed FP4 storage,
  block-scale logical/physical layouts, output metadata, environment, command,
  observed kernels, and the retained report hash.
- CUTLASS DSL 4.6 requires `nvidia-cuda-nvdisasm>=13.3,<14`; pinning the
  independent inspection tool to 13.3.73 removed four package-metadata
  incompatibilities while leaving the CUDA 13.2 compiler/runtime pins intact.
- E002 records the public NVFP4 semantics used by the oracle, including the
  E2M1 value table, low-nibble-first packing, UE4M3 decoding, a scalar FP32
  global scale, 16-value block scales, and the CUTLASS 128x4 physical layout.
- The CPU-only oracle reconstructs values as
  `E2M1 value * UE4M3 block scale * scalar global scale` without importing
  PyTorch, CUDA, FlashInfer, or vLLM.
- Hand-authored fixtures cover all 16 E2M1 bit patterns, selected UE4M3
  boundary values, eight packing bytes, eleven CUTLASS layout offsets, and a
  directly checkable hierarchical reconstruction case.
- Property tests cover nibble packing round trips, scale-layout round trips,
  and layout offset bijection and bounds over generated shapes and payloads.
- The E002 GPU differential runner compared FlashInfer `nvfp4_quantize`
  outputs byte-for-byte against independently computed packed E2M1 values and
  UE4M3 scale bytes for three tensors spanning linear and CUTLASS 128x4 scale
  layouts, padding, row-atom and scale-atom boundaries, and global scales 0.5,
  1.0, and 2.0.
- All three controlled GPU cases matched exactly. Together they exercised all
  127 finite UE4M3 codes, and independent reconstruction of the candidate
  bytes had maximum absolute error 0.0 for the deliberately representable
  inputs.
- E002 retains source provenance, tensor metadata, requested backend evidence,
  artifact hashes, and a bounded `go` decision in its manifest and results.
- The final E002 manifest was regenerated from clean commit
  `f0be51513892d8b10968090fb081a8dafbee0b89`; it records `dirty=false` and the
  unchanged semantic source-bundle SHA-256
  `88f5b9afaa4f2d83b25b298d0d981026740b810a928b3f1f037e312e00c12154`.
- E003's first CPU-only slice defines six exact format contracts with explicit
  domains, preconditions, invariants, mismatch metrics, zero-mismatch
  thresholds, and limitations.
- Six deterministic, reversible faults cover nibble swapping, scale-index
  shifting, block-scale reversal, global-scale multiplication, layout
  mislabeling, and physical padding corruption. Every artifact is labeled
  `synthetic`, and the immutable clean tensor is retained separately.
- Three clean artifacts produced 18 passing contract evaluations. All six
  faults were detected, every failed-contract set matched its declared expected
  set, and clean false rejects, fault false accepts, localization failures, and
  reversibility failures were zero.
- The padding control changed one of 1,403 physical padding bytes while logical
  reconstruction remained exact, demonstrating a bounded structurally invalid
  but numerically silent positive control.
- E003's second CPU-only slice defines separate exact contracts for recorded
  stride, row-major contiguity, requested backend, reported backend, observed
  kernel tuple, and bounded fallback status.
- One clean execution-evidence snapshot passed all six contracts. Two stride
  faults and three backend-identity faults were detected with exact localization
  and zero clean false rejects, fault false accepts, localization failures, or
  reversibility failures.
- A reported-backend-only fault left requested-backend and observed-kernel
  contracts passing. This directly checks that the three identity fields remain
  separate rather than being inferred from one another.
- The fallback-kernel string in this slice is labeled synthetic and is not
  represented as a profiler observation from the run.

These observations complete Gate 1 and support two E003 CPU synthetic-fault
slices. E003 remains in progress. They do not establish real runtime stride or
dispatch detection, arbitrary-input rounding, NVFP4 GEMM correctness or
performance, production model quality, or model-level fault propagation.

## Last verification

```bash
source /home/meyowu/projects/nvfp4-doctor/activate-nvfp4-lab.sh
cd /mnt/c/Users/meyow/Documents/Codex/2026-08-19/referenced-chatgpt-conversation-this-is-an/nvfp4-doctor
bash -n activate-nvfp4-lab.sh scripts/run_e001_week1.sh
/home/meyowu/projects/nvfp4-doctor/.venv/bin/ruff format --check src tests scripts smoke_nvfp4.py
/home/meyowu/projects/nvfp4-doctor/.venv/bin/ruff check src tests scripts smoke_nvfp4.py
PYTHONPATH=src /home/meyowu/projects/nvfp4-doctor/.venv/bin/mypy src
PYTHONPATH=src /home/meyowu/projects/nvfp4-doctor/.venv/bin/python -m pytest -q
/home/meyowu/projects/nvfp4-doctor/.venv/bin/python -m compileall -q src tests scripts
uv pip check --python /home/meyowu/projects/nvfp4-doctor/.venv/bin/python
PYTHONPATH=src /home/meyowu/projects/nvfp4-doctor/.venv/bin/python scripts/run_e002_gate1.py
PYTHONPATH=src /home/meyowu/projects/nvfp4-doctor/.venv/bin/python scripts/run_e003_format_faults.py
PYTHONPATH=src /home/meyowu/projects/nvfp4-doctor/.venv/bin/python scripts/run_e003_execution_faults.py
```

Observed result: 77 tests plus 248 subtests passed and Python compilation
completed. Ruff formatting and lint checks passed, Mypy reported no issues in
27 source files, and `uv pip check` found all 207 installed packages compatible.
The two E003 CPU runners reported 24 clean contract passes, eleven of eleven
faults detected, exact localization, zero false accepts or rejects, zero
reversibility failures, slice status `pass`, and decision `continue`.

## Next action

Add deterministic packed-value block, row, and column permutation controls,
then evaluate all E003 detectors on a held-out clean/fault matrix that is not
used to tune later thresholds. Measure clean false rejects, fault false accepts,
localization, and reversibility before deciding whether E003 is complete.

## Blockers and limitations

- The fallback classifier recognizes only documented E001 signatures; unknown
  implementations intentionally remain `unknown`.
- E002 validates constructed, exactly representable quantization inputs; it does
  not yet characterize arbitrary-input rounding or saturation behavior in the
  candidate implementation.
- E002 validates linear and CUTLASS 128x4 scale layouts only. Other layouts,
  shuffles, per-token scaling, and two-dimensional training recipes remain out
  of scope.
- NVFP4 GEMM correctness, throughput, accuracy on real model weights and
  activations, and end-to-end model quality remain future-gate questions.
- The first E003 slice covers six deterministic CPU format controls only. It
  does not yet cover stride, non-contiguous storage, backend mismatch, fallback,
  arbitrary corruptions, or held-out fault distributions.
- The second E003 slice covers metadata and backend-identity fields using
  synthetic evidence. It does not establish actual runtime storage or dispatch;
  profiler-backed replay remains a later experiment boundary.
- Packed-value block, row, and column permutations plus a held-out evaluation
  matrix remain required before E003 can be considered complete.
- The E002 manifest pins the clean implementation commit rather than the later
  evidence-only handoff commit. Regenerate it after any semantic source edit;
  the source-bundle hash detects such changes.
- The migration backup and pre-merge WSL stash remain local recovery artifacts.

## Working-tree expectation

Preserve the uncommitted E003 format-fault slice on
`exp/e003-synthetic-faults`. Review and commit it only after explicit user
authorization. Do not infer permission to push, open a PR, merge, download
model weights, or publish external results from this state file.
