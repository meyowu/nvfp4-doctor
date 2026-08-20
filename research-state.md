# Research State

This is the canonical handoff record for resumable research sessions. Update it
only from observed repository, environment, test, and experiment evidence.

## Current handoff

- Current experiment: `E002-format-oracle`
- Current gate: `Gate 1 — Format Oracle`
- Status: `not_started`
- Decision: `go`
- Verified baseline commit: `77ca69b3eac94f8dc9f92e1d69653c90f26ec473`
- Canonical WSL checkout: `/home/meyowu/projects/nvfp4-doctor`
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

These observations complete Gate 0 for the pinned synthetic matrix. They do not
establish format correctness or exclude fallback modes unknown to the bounded
classifier.

## Last verification

```bash
bash -n activate-nvfp4-lab.sh
.venv/bin/ruff format --check src tests scripts smoke_nvfp4.py
.venv/bin/ruff check src tests scripts smoke_nvfp4.py
PYTHONPATH=src .venv/bin/mypy src
PYTHONPATH=src .venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q src tests scripts
uv pip check --python .venv/bin/python
./scripts/run_e001_week1.sh 3
```

Observed result: thirty-four CPU tests plus eleven subtests passed and Python
compilation completed. All three profiled runs printed `NVFP4_CUTLASS_OK`; the
generated summary reported three repetitions, stable environment and target
kernel sets, complete profiler evidence, repeatability `pass`, and decision
`go`. Ruff 0.16.3 formatting and lint checks passed, and Mypy 2.3.1 reported no
issues in seventeen source files. `uv pip check` found all 205 installed
packages compatible.

## Next action

Create E002 and begin the independent format oracle with a public-semantics
source record and exhaustive hand-authored E2M1 decode fixtures.

## Blockers and limitations

- The fallback classifier recognizes only documented E001 signatures; unknown
  implementations intentionally remain `unknown`.
- Schema v1 cannot represent zero-dimensional global-scale tensor metadata
  without inventing a logical shape; Gate 1 must address this explicitly.
- The migration backup and pre-merge WSL stash remain local recovery artifacts.

## Working-tree expectation

Start from a clean `main`. Create a focused branch before implementation. Do
not infer permission to commit, push, open a PR, merge, download model weights,
or publish results from this state file.
