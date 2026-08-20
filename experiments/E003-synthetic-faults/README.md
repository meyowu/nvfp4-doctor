# E003: Synthetic Fault Injection

## Current slices

Three CPU-only E003 slices cover NVFP4 format positive controls, tensor
stride/contiguity, requested/reported/observed backend evidence, and held-out
packed-value block/row/column permutations. The declared synthetic matrix is
complete. Real dispatch replay, GEMM, and model propagation remain later
experiment boundaries.

## Hypothesis

Exact contracts built on the independent Gate 1 oracle can detect and localize
deterministic corruption of packed values, logical scale mapping, global scale,
declared scale layout, and physical padding without rejecting the clean
artifacts.

The second-slice hypothesis is that exact evidence contracts can detect and
localize stride corruption and deliberate backend-identity disagreement without
conflating caller requests, adapter reports, or observed kernel strings.

The final-slice hypothesis is that the same frozen exact contracts detect and
localize packed-value block, row, and column permutations on a clean/fault
matrix not used to tune thresholds.

## Completion criterion

This slice passes only if:

- every fault is labeled `synthetic`, parameterized, non-mutating, and exactly
  reversible;
- three clean artifacts pass all six exact contracts;
- all six positive controls are detected;
- observed failed-contract sets exactly match their independently declared
  expected sets;
- clean false rejects, fault false accepts, localization failures, and
  reversibility failures are all zero.

The execution-evidence slice applies the same criteria to one clean snapshot,
six exact contracts, and five stride/backend positive controls.

The held-out slice passes only if three new clean artifacts produce zero false
rejects and all nine permutation cases are detected, exactly localized, and
reversible with zero false accepts. Gate 1 zero-mismatch thresholds are frozen
before the held-out matrix is constructed and are not tuned on its cases.

## Contract definitions

Each contract in `src/nvfp4_doctor/contracts/format.py` records its domain,
preconditions, invariant, mismatch metric, exact threshold, and limitations.
The contracts separately inspect:

- declared metadata;
- packed E2M1 bytes;
- layout-normalized logical UE4M3 scales;
- non-logical CUTLASS 128x4 padding bytes;
- the scalar global scale;
- explicitly reconstructed logical values.

All thresholds are zero mismatches. This slice does not introduce numerical
tolerances.

The execution-evidence contracts separately inspect recorded stride,
row-major-contiguity classification, requested backend, reported backend,
observed kernel tuple, and bounded fallback status. Kernel strings in this
slice are explicit synthetic inputs; they are not presented as profiler output.

## Controlled variables

- execution device: CPU;
- deterministic seed: 0;
- clean logical shapes: `(2, 32)`, `(128, 64)`, and `(129, 80)`;
- layouts: linear and CUTLASS 128x4;
- block size: 16;
- clean global scale: 0.5.

No CUDA kernel or model checkpoint is used.

The execution-evidence clean snapshot records physical shape `(4, 8)`, stride
`(8, 1)`, requested and reported backend `cutlass`, one synthetic CUTLASS kernel
string, and fallback status `not_detected`.

The held-out clean matrix uses shapes `(3, 48)`, `(5, 64)`, and `(131, 80)`,
data salts 11, 23, and 37, and linear plus CUTLASS 128x4 scale layouts. Its
deterministic seed is `20260820`.

## Independent variables

| Synthetic fault | Parameter | Expected failed contracts |
| --- | --- | --- |
| Packed-nibble swap | every packed byte | packed values, reconstruction |
| Scale-index shift | offset 1 | logical scales, reconstruction |
| Block-scale reversal | per row | logical scales, reconstruction |
| Global-scale multiplier | factor 2.0 | global scale, reconstruction |
| Scale-layout mislabel | CUTLASS 128x4 to linear | metadata, logical scales, reconstruction |
| Padding corruption | first padding offset | scale padding only |

The second slice varies one evidence category at a time:

| Synthetic fault | Parameter | Expected failed contracts |
| --- | --- | --- |
| Stride-axis permutation | `(8, 1)` to `(1, 8)` | stride, contiguity |
| Stride gap | outer stride `+8` | stride, contiguity |
| Requested-backend mismatch | `cutlass` to `cublas` | requested backend |
| Reported-backend mismatch | `cutlass` to `cublas` | reported backend |
| Observed fallback kernel | synthetic `cublasGemmEx` | observed kernels, fallback status |

The held-out slice applies each packed-value fault family to all three new
artifacts:

| Synthetic fault | Parameter family | Expected failed contracts |
| --- | --- | --- |
| Packed block permutation | cyclic offsets 1, 2, 3 | packed values, reconstruction |
| Packed row permutation | cyclic offsets 1, 2, 3 | packed values, reconstruction |
| Packed column permutation | cyclic offsets 3, 5, 7 | packed values, reconstruction |

## Actual observations

The runner evaluated 18 clean contract outcomes over three artifacts and
reported zero clean false rejects. All six injected faults were detected and
their failed-contract sets matched the expected sets exactly. It reported zero
false accepts, localization failures, and reversibility failures.

Padding corruption was structurally detected while reconstruction remained
unchanged. In the tested `(129, 80)` artifact, one corrupted byte was localized
among 1,403 physical padding bytes. This is a useful positive control for a
fault that is real at the storage-contract level but numerically silent under
logical dequantization.

The second runner evaluated six clean evidence contracts and reported zero
clean false rejects. All five stride/backend faults were detected, with exact
failed-contract sets, zero false accepts, zero localization failures, and zero
reversibility failures. Changing only `reported_backend` left the requested
backend and observed-kernel contracts passing, as required by the evidence
separation rule.

The held-out runner evaluated 18 clean contract outcomes and nine fault cases.
All clean outcomes passed, and every block, row, and column permutation was
detected and localized exactly to packed values plus reconstruction. False
accepts, clean false rejects, localization failures, and reversibility failures
were zero.

Run the slice with:

```bash
source ./activate-nvfp4-lab.sh
PYTHONPATH=src python scripts/run_e003_format_faults.py
PYTHONPATH=src python scripts/run_e003_execution_faults.py
PYTHONPATH=src python scripts/run_e003_heldout_permutations.py
```

Small structured evidence is retained in [results.json](results.json) and
[manifest.json](manifest.json) for format faults, and in
[results-execution.json](results-execution.json) and
[manifest-execution.json](manifest-execution.json) for execution evidence. The
held-out matrix is retained in [results-heldout.json](results-heldout.json) and
[manifest-heldout.json](manifest-heldout.json).

## Interpretation

The hypothesis is supported for this six-fault CPU matrix. The exact contracts
distinguished the clean artifacts from every injected control and localized
each fault to the expected contract layer.

The second-slice hypothesis is also supported for its five-fault matrix. The
results demonstrate field-level separation on synthetic evidence; they do not
establish what backend actually ran in any CUDA execution.

The held-out hypothesis is supported for the three declared permutation
families. Because exact thresholds were frozen before matrix construction, this
is an evaluation slice rather than another threshold-development slice.

This does not yet demonstrate detector power for runtime metadata, dispatch,
GEMM output, arbitrary corruptions, or model workloads.

## Threats to validity

- Expected failure sets were declared for six hand-selected deterministic
  controls; they are not a distribution of naturally occurring faults.
- The initial six format controls use their development artifact family. The
  permutation slice is held out, but no E003 matrix estimates naturally
  occurring faults or justifies later numerical thresholds.
- Reconstruction equality is exact because the clean inputs and mutations are
  synthetic and CPU interpreted.
- Layout mislabeling is tested only where linear and CUTLASS physical byte
  lengths coincide, allowing invalid metadata to remain shape-valid.
- Padding corruption is intentionally numerically silent and requires retained
  physical storage evidence.
- Backend strings and kernels in the second slice are synthetic controls. Real
  dispatch identity still requires profiler evidence scoped to the target NVTX
  range.
- Stride faults alter metadata only; a later capture/replay experiment must
  compare metadata with actual framework storage without silently making it
  contiguous.
- Held-out artifacts are deterministic synthetic tensors, not an estimate of
  naturally occurring permutation frequency or model-level impact.
- Permutations are cyclic and axis-aligned. Arbitrary partial permutations and
  compound faults remain outside this E003 completion boundary.

## Decision

Continue. All three slices passed their bounded criteria, so E003 is complete.
Proceed to E004 by pinning and inspecting the primary Qwen3-8B NVFP4 checkpoint
before designing metadata-preserving representative-layer capture.
