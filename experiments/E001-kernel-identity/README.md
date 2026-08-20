# E001: Environment and Kernel Identity

## Hypothesis

On the pinned RTX 5080 environment, an NVFP4 smoke run can be accompanied by a
machine-readable environment fingerprint and independent profiler evidence that
identifies the executed CUDA kernel and exposes a silent backend fallback.

## Completion criterion

E001 is complete only when a repeated run records:

- the requested format and backend separately from observed execution evidence;
- GPU, driver, WSL, CUDA, Python, package, Git, tensor, and command provenance;
- an observed kernel identity extracted from a retained local profiler report;
- a deterministic fallback assessment with documented limitations;
- artifact hashes without committing profiler reports or model data.

## Controlled variables

- GPU: RTX 5080 (`sm_120`)
- smoke-test seed and matrix shapes
- pinned Python package versions
- requested FlashInfer backend

## Independent variable

The execution run and its profiler capture. No fault is injected in E001.

## Expected outcome

The requested CUTLASS NVFP4 path executes repeatedly and profiler evidence is
sufficient to identify the actual kernel without relying on the requested
backend string alone.

## Actual observations

A local pre-profiler schema-v1 manifest was generated from the checked-in smoke
test definition and the current WSL environment. It recorded the requested
backend as CUTLASS, `git.dirty=true`, no observed kernels, and fallback status
`unknown`. This is an assembly/collection observation, not kernel-identity
evidence and not a completed E001 result.

## Threats to validity

- Profiler-visible symbol names may be abbreviated, mangled, or unstable across
  versions.
- A kernel name alone may not prove the full data-format contract.
- WSL profiler support may differ from native Linux.
- JIT cache state may change the visible execution sequence.

## Decision

Pending.
