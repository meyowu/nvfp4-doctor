# Research State

This is the canonical handoff record for resumable research sessions. Update it
only from observed repository, environment, test, and experiment evidence.

## Current handoff

- Current experiment: `E003-synthetic-faults`
- Current gate: `Gate 2 entry — Qwen3 checkpoint pinning and inspection`
- Status: `complete`
- Decision: `continue`
- Verified baseline commit: `e9cbd9055be5e74a9f32569e1cadedf546dce84e`
- Verified Gate 1 implementation commit: `f0be51513892d8b10968090fb081a8dafbee0b89`
- Verified E003 implementation commit: `2116fbb5ed76d51119ea79b89da701c25b0ef0d5`
- Verified E003 completion commit: `326522ce74b70ee5934373a2a5fa72d512cd8390`
- Active local branch: `exp/e003-heldout-permutations`
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
- The final format-fault manifest was regenerated from clean implementation
  commit `2116fbb5ed76d51119ea79b89da701c25b0ef0d5`; it records `dirty=false` and
  source-bundle SHA-256
  `1029f17f569ee74f509673225ea00977a018adcd5590af08247344a5f2a8bad6`.
- The final execution-evidence manifest was regenerated from clean provenance
  commit `4c7cab930f74c7a7f5457b94d1ffc7bf642191cb`; it records `dirty=false` and
  source-bundle SHA-256
  `0a9e5587e74205092338448d151b3ce80ed6c03003b5d6ebddebc72c984b1f23`.
- E003's final slice adds reversible cyclic permutations of complete packed
  value blocks, rows, and logical columns without changing scale storage or
  metadata.
- Three held-out clean artifacts with shapes `(3, 48)`, `(5, 64)`, and
  `(131, 80)` produced 18 passing contract evaluations. Their data salts and
  linear/CUTLASS layouts differ from the development matrix.
- Gate 1 exact zero-mismatch thresholds were frozen before the held-out matrix
  was constructed and were not tuned on its cases.
- All nine held-out permutation faults were detected and localized exactly to
  packed values plus reconstruction. Clean false rejects, fault false accepts,
  localization failures, and reversibility failures were zero.
- The held-out manifest was regenerated from clean completion commit
  `326522ce74b70ee5934373a2a5fa72d512cd8390`; it records `dirty=false`, result
  SHA-256 `a228b0a75b50c99ffa661d37640c5d8bbdf211f7aae6c79942fd9f698dbf5e74`,
  and source-bundle SHA-256
  `fe3a273cbd28ae1b3354ff8ea7b02402d0df891b9640dffe185cd76b4cc88d94`.

These observations complete Gate 1 and the declared three-slice E003 CPU
synthetic-fault matrix. They do not establish real runtime stride or dispatch
detection, arbitrary-input rounding, NVFP4 GEMM correctness or performance,
production model quality, or model-level fault propagation.

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
PYTHONPATH=src /home/meyowu/projects/nvfp4-doctor/.venv/bin/python scripts/run_e003_heldout_permutations.py
```

Observed result: 83 tests plus 261 subtests passed and Python compilation
completed. Ruff formatting and lint checks passed, Mypy reported no issues in
27 source files, and `uv pip check` found all 207 installed packages compatible.
The two E003 CPU runners reported 24 clean contract passes, eleven of eleven
faults detected, exact localization, zero false accepts or rejects, zero
reversibility failures, slice status `pass`, and decision `continue`.
The held-out runner added 18 clean contract passes and detected all nine packed
permutation faults with the same zero-failure metrics; it reported E003 status
`complete` and decision `continue`.

## Next action

Start E004 by resolving and recording an immutable revision for
`nvidia/Qwen3-8B-NVFP4`, then inspect its public quantization configuration and
checkpoint metadata before downloading model weights or designing capture.

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
- The first E003 slice covers six deterministic CPU format controls only; the
  later execution-evidence and held-out permutation slices extend that matrix
  without turning it into a distribution of naturally occurring faults.
- The second E003 slice covers metadata and backend-identity fields using
  synthetic evidence. It does not establish actual runtime storage or dispatch;
  profiler-backed replay remains a later experiment boundary.
- The held-out E003 permutations are cyclic and axis-aligned; arbitrary partial
  permutations and compound faults remain outside the completion boundary.
- The E002 manifest pins the clean implementation commit rather than the later
  evidence-only handoff commit. Regenerate it after any semantic source edit;
  the source-bundle hash detects such changes.
- The migration backup and pre-merge WSL stash remain local recovery artifacts.

## Working-tree expectation

Keep verified E003 completion work committed and pushed on the focused
`exp/e003-heldout-permutations` branch. The configured research closeout workflow
authorizes commit and push for verified in-scope changes only; opening a PR,
merging, downloading model weights, or publishing external results still
requires separate explicit authorization.
