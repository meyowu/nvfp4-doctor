# Eight-Week Research Plan

## Research question

Can executable NVFP4 contracts distinguish normal quantization error from
silent format, scale, layout, and dispatch failures, localize those failures,
and predict their downstream effect in a real Qwen3-8B inference workload?

The model study validates the contract work; it does not turn this project into
a general LLM benchmark or model-quality leaderboard.

## Evaluation ladder

Evidence advances through five levels:

1. Tiny hand-computable fixtures for encoding, packing, scales, and layouts.
2. Synthetic matrix families covering normal and boundary shapes.
3. Representative linear layers captured from Qwen3-8B.
4. End-to-end inference with a pinned Qwen3-8B NVFP4 checkpoint.
5. A bounded cross-check against the high-precision checkpoint and, if feasible,
   one independently produced Qwen3-8B NVFP4 checkpoint.

A later level cannot repair an unvalidated earlier level. In particular,
end-to-end model quality cannot establish that a kernel obeyed its format
contract.

## Checkpoints and resource boundary

- Primary candidate: `nvidia/Qwen3-8B-NVFP4`, pinned to an immutable revision.
- High-precision reference: the corresponding upstream Qwen3-8B checkpoint,
  also pinned to an immutable revision.
- Optional cross-toolchain candidate: one independently produced Qwen3-8B
  NVFP4 checkpoint, selected only after its format metadata is inspected.
- Hardware boundary: one RTX 5080 with 16 GB VRAM under WSL2.

Run checkpoints sequentially. Do not keep both high-precision and NVFP4 models
resident on the GPU. Store only selected layer outputs required by a declared
experiment, move them to CPU or local artifact storage, and record content
hashes rather than committing model data.

## Experiment sequence

| ID | Experiment | Required evidence |
| --- | --- | --- |
| E001 | Environment and kernel identity | Pinned environment, requested backend, observed kernel, no silent fallback |
| E002 | Independent NVFP4 format oracle | Exhaustive E2M1 checks, E4M3 scale checks, packing and layout golden fixtures |
| E003 | Synthetic fault injection | Deterministic scale, packing, layout, stride, padding, and dispatch faults with labels |
| E004 | Qwen3-8B layer extraction | Reproducible captures for representative attention and MLP projections |
| E005 | Layer oracle versus production kernel | Quantization error separated from contract/kernel error |
| E006 | Layer fault propagation | Per-layer error metrics and fault signatures under positive controls |
| E007 | End-to-end clean baseline | Pinned inputs, logits, top-k agreement, KL divergence, perplexity, latency, and memory |
| E008 | End-to-end injected faults | Downstream signatures for faults that remain finite and shape-correct |
| E009 | Cross-checkpoint comparison | Metadata, mixed-precision scope, kernel behavior, and output differences across toolchains |
| E010 | Diagnostic evaluation | Detection, classification, localization, false-positive rate, and overhead |

## Metrics

Tensor and layer metrics:

- maximum, mean, and percentile absolute error;
- relative error with an explicitly defined near-zero policy;
- cosine similarity and signal-to-quantization-noise ratio;
- structured error maps aligned to quantization blocks;
- finite-value, shape, stride, layout, and scale-index contract results.

Model metrics:

- final-logit absolute error and KL divergence;
- top-1 and top-k token agreement;
- perplexity on a pinned, fixed evaluation slice;
- deterministic generation divergence under an explicitly fixed decoding policy;
- latency, peak memory, and diagnostic overhead with warm-up and uncertainty.

Do not use subjective inspection of a few generated responses as the primary
quality metric.

## Eight-week schedule and gates

### Week 1: Foundation — Gate 0

Freeze scope, capture the environment, and validate repeatable `sm_120` CUDA and
NVFP4 execution. Go only if backend execution can be reproduced in a pinned
environment.

### Week 2: Format oracle — Gate 1

Implement independent E2M1/E4M3, packing, layout, and scale reconstruction
tests. Pivot if the public format semantics cannot be established independently
of the candidate implementation.

### Week 3: Qwen3 layer capture — Gate 2

Pin the primary checkpoint, inspect its quantization configuration, and capture
representative `q_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`
cases. Go only if metadata-preserving replay and actual kernel identification
are reliable. Otherwise pivot to checkpoint-only plus synthetic analysis.

### Week 4: Numerical reference — Gate 3

Compare the production kernel against explicit dequantization and a tiled
higher-precision matmul over synthetic and captured shapes. Stop kernel
correctness claims if the oracle fails independent golden tests.

### Week 5: Fault propagation — Gate 4

Inject deterministic scale permutation, global-scale, nibble, layout, stride,
padding, and fallback faults into representative layers. Measure both local
errors and downstream logits/perplexity on a fixed slice. Go only if the suite
detects relevant positive controls with an acceptable false-positive rate and
at least one silent fault has a reproducible model-level signature.

### Week 6: Backend matrix and minimization — Gate 5

Compare genuinely distinct implementations when available, verify dispatch,
and minimize failures. If only one genuine NVFP4 backend is available on
`sm_120`, retain NVFP4 as a single-backend diagnostic study or pivot the
comparative component to FP8.

### Week 7: End-to-end checkpoint study — Gate 6

Run the clean NVIDIA NVFP4 and high-precision checkpoints sequentially on fixed
inputs. If resources permit, add one independently produced NVFP4 checkpoint.
Measure diagnostic accuracy, localization, false positives, latency, memory,
and overhead. Require a minimized reproducer or a documented negative result.

### Week 8: Reproduction and release — Gate 7

Rebuild from an empty pinned environment, rerun the evidence set, publish
manifests and small auditable artifacts, document limitations, and make an
explicit Go/Pivot/Stop decision for phase two.

## Global pivot and stop rules

Pivot or stop the affected claim when any of the following holds:

- backend identity cannot be verified;
- the oracle depends on the candidate encoder, decoder, or kernel;
- relevant positive controls are not detected better than simple baselines;
- a result fails three disciplined reproduction attempts;
- model-level effects disappear under fixed inputs and repeated runs;
- the 16 GB hardware boundary prevents a valid comparison without changing the
  research question;
- checkpoint or dataset licensing prevents reproducible publication.

A negative result is valid: the project may conclude that selected contract
faults are reliably caught by simpler checks, do not propagate materially in
the tested workload, or cannot be distinguished under the available evidence.

## Claims boundary

Results apply only to the pinned checkpoints, revisions, layers, inputs,
software stack, hardware, and observed kernels. A finite output is not evidence
of correctness. A perplexity or token-ranking change is not evidence of a
kernel defect until normal quantization error has been separated from
contract/kernel error through the independent oracle and positive controls.
