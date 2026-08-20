# E002: Independent NVFP4 Format Oracle

## Hypothesis

Public NVIDIA semantics are sufficient to implement a CPU-only NVFP4 format
oracle, independent of FlashInfer, that exactly predicts E2M1 values, finite
UE4M3 block scales, low-nibble-first packed storage, CUTLASS 128x4 scale
layout, padding, and hierarchical reconstruction for the selected dense
block-16 operand contract.

## Rationale

E001 observed a requested NVFP4 path and its kernels, but a successful dispatch
does not establish that bytes and scales have the expected meaning. E002 fixes
that dependency order: public semantics and hand-authored fixtures define the
oracle before a candidate implementation is used for differential evidence.

## Completion criterion

E002 is complete only if:

- all 16 E2M1 payloads match a hand-authored decode table;
- all 127 finite UE4M3 scale codes decode monotonically and round-trip;
- representative subnormal, normal, boundary, maximum, and invalid UE4M3
  encodings have independent golden checks;
- low-nibble-first packing and CUTLASS 128x4 offsets match hand-authored
  fixtures;
- layout conversion is reversible across generated boundary shapes;
- a scalar FP32 global scale and exact block-scale index reconstruction are
  explicit in the data model;
- constructed, exactly representable GPU cases agree byte-for-byte with the
  pinned FlashInfer quantizer for both linear and CUTLASS 128x4 scale layouts;
- the experiment records sources, environment, tensor metadata, commands,
  hashes, limitations, and a gate decision.

## Public-semantics boundary

The authoritative source record is [sources.json](sources.json). It pins
CUTLASS 4.6.0 to commit
`e6233cbac5d7c7a865c19c91cd684ceece19513c` and records versioned NVIDIA PTX
9.2 and Transformer Engine 2.16 documentation. The executable contract is
documented in [../../docs/nvfp4-contract.md](../../docs/nvfp4-contract.md).

No FlashInfer encoder, decoder, dequantization helper, or CUDA kernel is
imported by `src/nvfp4_doctor/formats` or `src/nvfp4_doctor/oracle`. FlashInfer
is used only as the candidate in the final differential script.

## Controlled variables

- GPU: RTX 5080 (`sm_120`)
- PyTorch: 2.13.0+cu132
- FlashInfer: 0.6.16.post3
- input dtype: BF16
- block size: 16
- requested quantizer backend: `cuda`
- layouts: logical linear and CUTLASS 128x4
- deterministic seed: 0

No model checkpoint is loaded in E002.

## Independent variables

- matrix shapes `(128, 64)`, `(17, 80)`, and `(129, 80)`;
- dequantization global scales `1.0`, `0.5`, and `2.0`;
- logical versus padded physical scale shapes;
- all finite UE4M3 scale codes embedded in exactly representable blocks.

## Expected outcome

The independent oracle predicts every constructed packed value and scale byte.
FlashInfer's linear scale output and CUTLASS 128x4 output match those predictions
exactly, and explicit reconstruction recovers the constructed BF16 inputs with
zero error.

## Actual observations

The fixture runner checked all 16 E2M1 payloads, all 127 finite UE4M3 codes,
eleven hand-authored layout offsets, eight golden packed bytes, and one
hand-computable hierarchical reconstruction. All checks passed.

Three RTX 5080 differential cases then completed:

| Logical shape | Global scale | Physical scale shape | Finite scale codes covered | Max reconstruction error |
| --- | ---: | --- | ---: | ---: |
| `(128, 64)` | 1.0 | `(128, 4)` | 127 | 0.0 |
| `(17, 80)` | 0.5 | `(128, 8)` | 85 | 0.0 |
| `(129, 80)` | 2.0 | `(256, 8)` | 127 | 0.0 |

For every case, the candidate's packed bytes matched the independent expected
bytes in both requested layouts. Linear scale bytes and CUTLASS 128x4 scale
bytes also matched exactly, including padded and multi-atom cases. The
independent oracle reconstructed every source value exactly.

[results.json](results.json) records the per-case checks and content hashes.
[manifest.json](manifest.json) records the environment, requested adapter,
tensor shapes/dtypes/strides, scalar global-scale metadata, source-bundle hash,
and artifact hashes. The candidate did not report a backend or kernel during
this experiment, so both fields remain null instead of being inferred. E001
remains the dispatch-evidence experiment.

The complete CPU suite reported 58 tests plus 210 subtests passing. Ruff,
Mypy, Python compilation, shell syntax, and dependency compatibility checks
also passed. The GPU experiment was rerun after the final source record was
fixed and returned `adapter_differential=pass` and `decision=go`.

## Interpretation

The hypothesis is supported for the declared dense row-wise block-16 contract
and tested layouts. Public semantics were sufficient to build an independent
oracle, and the pinned FlashInfer quantizer agreed exactly on constructed cases
that exercise every finite local scale code plus row/scale padding boundaries.

This observation supports advancing past Gate 1. It is not evidence that
arbitrary quantization rounding, GEMM accumulation, a checkpoint adapter, or a
model output is correct.

## Threats to validity

- Constructed values are exactly representable and do not characterize
  arbitrary input rounding or numerical error distributions.
- The GPU differential uses one FlashInfer version and one RTX 5080 software
  stack.
- CUTLASS 128x4 and logical linear layouts are covered; FlashInfer-specific
  8x4, post-quantization shuffle, and per-token modes are outside this contract.
- Transformer Engine's two-dimensional training-scale recipe is not treated as
  the dense row-wise GEMM layout.
- Exact reconstruction validates format interpretation, not the FP4 GEMM
  reduction or output dtype behavior.
- The manifest records an uncommitted branch plus a content hash of the oracle,
  experiment runner, contract, fixture, and source record. It must be regenerated
  after the work receives a permanent commit.

## Decision

Go. E002 and Gate 1 are complete for the stated contract. The next experiment
should use this oracle for deterministic synthetic format faults before any
model-level capture is treated as trustworthy.
