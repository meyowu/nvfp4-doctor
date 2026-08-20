# E001: Environment and Kernel Identity

## Hypothesis

On the pinned RTX 5080 environment, an NVFP4 smoke run can be accompanied by a
machine-readable environment fingerprint and independent profiler evidence that
identifies the executed CUDA kernel and exposes a silent backend fallback.

## Rationale

Requested backend strings are configuration, not execution evidence. E001 uses
NVTX-scoped profiler observations so unrelated reference kernels cannot be
mistaken for the NVFP4 GEMM implementation.

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

On 2026-08-20, three controlled executions completed under Nsight Systems
2026.1.3 on the RTX 5080. All produced the same finite output metrics. Each
manifest contained 23 unique CUDA kernel names, and the four kernels launched
inside `e001:nvfp4_gemm` had the same set hash
`98aa16a78e2cdb3d74f27a0568bc2ed9f35de0e8584936717180c5e9c76ef10a`.

The target range contained an SM120 block-scaled CUTLASS device kernel whose
signature names E2M1 values and UE4M3 scales. No known fallback signature was
observed inside that range, so the deterministic assessment was
`not_detected` for all three runs. `reported_backend` remains unset because
FlashInfer did not independently report a backend; profiler observations are
kept in `observed_kernels` instead.

The repository-tracked [manifest](manifest.json) is the first repetition's complete
environment record. [results.json](results.json) records all three manifest and
profiler hashes. Raw reports remain under ignored `.local/profiles` storage.

The first rebuild also exposed a missing linker search path for package-local
CUDA libraries. Exporting `LIBRARY_PATH` alongside `LD_LIBRARY_PATH` restored
both unprofiled and profiled smoke execution. The repeat runner also removes
only its stale derived SQLite exports before replacing a report.

## Interpretation

The hypothesis is supported for the tested environment and matrix: the
requested CUTLASS path is accompanied by stable, independently scoped kernel
evidence, and the current known-fallback rule is negative in three repetitions.
This does not establish the NVFP4 format contract or exclude unknown fallback
modes.

## Threats to validity

- Profiler-visible symbol names may be abbreviated, mangled, or unstable across
  versions.
- A kernel name alone may not prove the full data-format contract.
- WSL profiler support may differ from native Linux.
- JIT cache state may change the visible execution sequence.
- The schema-v1 tensor list cannot represent zero-dimensional global-scale
  tensors without inventing a shape, so global-scale structure is deferred to
  the Gate 1 format schema.
- `not_detected` is limited to known signatures within the named target range;
  it is not proof that every silent fallback mode is impossible.

## Decision

Go. E001 and Gate 0 are complete for the pinned RTX 5080 matrix. Proceed to
E002 and Gate 1 without promoting this dispatch result into a format-correctness
claim.
