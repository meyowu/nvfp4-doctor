# Research State

This is the canonical handoff record for resumable research sessions. Update it
only from observed repository, environment, test, and experiment evidence.

## Current handoff

- Current experiment: `E002-format-oracle`
- Current gate: `Gate 1 — Format Oracle`
- Status: `complete`
- Decision: `go`
- Verified baseline commit: `fdc0beab037c1f35115de10a0b1539deeeadf38e`
- Verified Gate 1 implementation commit: `f0be51513892d8b10968090fb081a8dafbee0b89`
- Active local branch: `exp/e002-format-oracle`
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

These observations complete Gate 1 for the specified public semantics and
constructed differential cases. They do not establish arbitrary-input rounding,
NVFP4 GEMM correctness or performance, production model quality, or untested
layout and scaling recipes.

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
```

Observed result: 58 tests plus 210 subtests passed and Python compilation
completed. Ruff formatting and lint checks passed, Mypy reported no issues in
23 source files, and `uv pip check` found all 207 installed packages compatible.
The E002 runner completed three synchronized CUDA differential cases with exact
packed-value and scale-byte matches, maximum reconstruction error 0.0, and a
`go` decision.

## Next action

Create E003 and define deterministic, reversible positive-control faults for
scale shifts or permutations, nibble-order corruption, and scale-layout or
padding corruption. Measure their detection against the clean Gate 1 oracle
before expanding to model workloads.

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
- The E002 manifest pins the clean implementation commit rather than the later
  evidence-only handoff commit. Regenerate it after any semantic source edit;
  the source-bundle hash detects such changes.
- The migration backup and pre-merge WSL stash remain local recovery artifacts.

## Working-tree expectation

Gate 1 changes are committed on `exp/e002-format-oracle`. Begin E003 from the
updated `main` after this branch is merged. Do not infer permission to download
model weights or publish external results from this state file.
