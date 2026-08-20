# Research State

This is the canonical handoff record for resumable research sessions. Update it
only from observed repository, environment, test, and experiment evidence.

## Current handoff

- Current experiment: `E001-kernel-identity`
- Current gate: `Gate 0 — Foundation`
- Status: `in_progress`
- Decision: `continue`
- Last verified commit: `62d6727`
- Canonical WSL checkout: `/home/meyowu/projects/nvfp4-doctor`
- Hardware boundary: one RTX 5080 (`sm_120`, 16 GB)

## Completed evidence

- The RTX 5080 WSL2 research environment is operational.
- A FlashInfer CUTLASS NVFP4 quantize-and-GEMM smoke run completed with finite
  output at shape `(16, 128)`.
- Nsight Systems produced a local profiler report for the smoke run.
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
  it correctly recorded `git.dirty=true`, no observed kernels, and fallback
  status `unknown`.

These observations establish environment viability and repository structure
only. They do not yet establish observed kernel identity, absence of fallback,
or Gate 0 completion.

## Last verification

```bash
bash -n activate-nvfp4-lab.sh
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
.venv/bin/python -m compileall -q src tests
```

Observed result: sixteen CPU tests passed and Python compilation completed
successfully. Ruff and mypy were not installed, so lint and static type checks
were explicitly skipped rather than reported as passing.

## Next action

Implement profiler artifact hashing and independent kernel-name extraction from
the existing Nsight Systems report. Only then evaluate reported backend and
fallback status; do not infer either from the requested backend.

## Blockers and limitations

- Actual CUDA kernel identity has not been extracted into structured evidence.
- Fallback detection is not implemented.
- The existing profiler observation predates the E001 manifest schema.
- The migration backup and pre-merge WSL stash remain local recovery artifacts.
- The manifest schema, collectors, assembly, tests, and activation-path correction are uncommitted on branch
  `research/e001-manifest-schema`.

## Working-tree expectation

Start from a clean `main`. Create a focused branch before implementation. Do
not infer permission to commit, push, open a PR, merge, download model weights,
or publish results from this state file.
