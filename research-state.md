# Research State

This is the canonical handoff record for resumable research sessions. Update it
only from observed repository, environment, test, and experiment evidence.

## Current handoff

- Current experiment: `E001-kernel-identity`
- Current gate: `Gate 0 — Foundation`
- Status: `in_progress`
- Decision: `continue`
- Last verified commit: `5885d04d467af8d99aa76b1d24fa70e03ae66746`
- Canonical WSL checkout: `/home/meyowu/projects/nvfp4-doctor`
- Hardware boundary: one RTX 5080 (`sm_120`, 16 GB)

## Completed evidence

- The RTX 5080 WSL2 research environment is operational.
- A FlashInfer CUTLASS NVFP4 quantize-and-GEMM smoke run completed with finite
  output at shape `(16, 128)`.
- Nsight Systems 2026.1.3 produced a retained, ignored profiler report for the
  smoke run with SHA-256
  `d9d5086200b4bebcc51df806e7ccfa933b59baf40906fd385f3bd0e681aea1c5`.
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

These observations establish kernel identity for one trace. They do not yet
establish absence of fallback, repeatability, or Gate 0 completion.

## Last verification

```bash
bash -n activate-nvfp4-lab.sh
PYTHONPATH=src .venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q src tests
PYTHONPATH=src .venv/bin/python smoke_nvfp4.py
PYTHONPATH=src nsys profile --trace=cuda,nvtx,osrt \
  --output=.local/profiles/e001-smoke --force-overwrite=true \
  .venv/bin/python smoke_nvfp4.py
PYTHONPATH=src .venv/bin/python scripts/attach_e001_nsys.py
```

Observed result: twenty-one CPU tests plus four subtests passed, Python
compilation completed, both smoke runs printed `NVFP4_CUTLASS_OK`, and 18
unique kernel names were attached to the ignored manifest. Ruff and mypy were
not installed, so lint and static type checks were explicitly skipped rather
than reported as passing.

## Next action

Define and test a deterministic backend/fallback classifier whose positive and
negative rules use independent evidence and explicitly document what kernel
names cannot prove. Then repeat the profiled run before considering Gate 0.

## Blockers and limitations

- Fallback detection is not implemented.
- Only one successful profiler capture has been structured; repeatability is
  not established.
- The migration backup and pre-merge WSL stash remain local recovery artifacts.
- Profiler extraction, tests, experiment notes, and the activation-path repair
  are uncommitted on branch `research/e001-profiler-evidence`.

## Working-tree expectation

Start from a clean `main`. Create a focused branch before implementation. Do
not infer permission to commit, push, open a PR, merge, download model weights,
or publish results from this state file.
