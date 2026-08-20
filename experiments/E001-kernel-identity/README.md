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

On 2026-08-20, the checked-in smoke test completed both without and under
Nsight Systems 2026.1.3 on the RTX 5080. The retained ignored report is
`.local/profiles/e001-smoke.nsys-rep` with SHA-256
`d9d5086200b4bebcc51df806e7ccfa933b59baf40906fd385f3bd0e681aea1c5`.

Independent parsing of the `cuda_gpu_kern_sum` CSV found 18 unique CUDA kernel
names. The observations include an SM120 block-scaled CUTLASS device kernel
whose operands name `cutlass::float_e2m1_t` and `cutlass::float_ue4m3_t`, plus
`tensorrt_llm::kernels::quantize_with_block_size`. These names support kernel
identity for this trace but do not, by themselves, establish the entire format
contract or prove that no fallback occurred. Accordingly, `reported_backend`
remains unset and `fallback_status` remains `unknown` in the manifest.

The first rebuild also exposed a missing linker search path for package-local
CUDA libraries. Exporting `LIBRARY_PATH` alongside `LD_LIBRARY_PATH` in the
project activation script restored both the unprofiled and profiled smoke run.

## Threats to validity

- Profiler-visible symbol names may be abbreviated, mangled, or unstable across
  versions.
- A kernel name alone may not prove the full data-format contract.
- WSL profiler support may differ from native Linux.
- JIT cache state may change the visible execution sequence.

## Decision

Continue. Kernel evidence is now structured, but the deterministic fallback
rule and repeated-run evidence required by the completion criterion are still
pending.
