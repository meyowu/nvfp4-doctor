# Research State

This is the canonical handoff record for resumable research sessions. Update it
only from observed repository, environment, test, and experiment evidence.

## Current handoff

- Current experiment: `E001-kernel-identity`
- Current gate: `Gate 0 — Foundation`
- Status: `in_progress`
- Decision: `continue`
- Last verified commit: `19a43b5`
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

These observations establish environment viability and repository structure
only. They do not yet establish observed kernel identity, absence of fallback,
or Gate 0 completion.

## Last verification

```bash
bash -n activate-nvfp4-lab.sh
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
.venv/bin/python -m compileall -q src tests
```

Observed result: one CPU package-structure test passed; shell syntax and Python
compilation completed successfully.

## Next action

Implement the versioned E001 environment-manifest schema and CPU-only data
model. Add golden serialization tests before adding host or GPU collectors.

## Blockers and limitations

- Actual CUDA kernel identity has not been extracted into structured evidence.
- Fallback detection is not implemented.
- The existing profiler observation predates the E001 manifest schema.
- The migration backup and pre-merge WSL stash remain local recovery artifacts.

## Working-tree expectation

Start from a clean `main`. Create a focused branch before implementation. Do
not infer permission to commit, push, open a PR, merge, download model weights,
or publish results from this state file.
