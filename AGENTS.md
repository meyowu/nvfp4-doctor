# AGENTS.md — QContract / nvfp4-doctor Research Guide

## Mission

Build `nvfp4-doctor`, an open, reproducible diagnostic toolkit for NVFP4 checkpoint layout, hierarchical scaling, backend identity, and numerical contracts on NVIDIA Blackwell GPUs. The first eight-week milestone is a credible research track record, not a production inference system or a claim that any backend is correct or broken.

Primary question: given a real NVFP4 checkpoint, real layer activations, and a precisely recorded environment, can structural errors in packed data, scale mapping, shape boundaries, or kernel dispatch be detected before full serving?

## Scope for the first eight weeks

In scope:

- One GPU: RTX 5080 (`sm_120`), WSL2/Ubuntu as the initial environment.
- Dense NVFP4 linear/GEMM, followed by bounded model-level validation on Qwen3-8B.
- Synthetic fixtures, representative Qwen3-8B layers, then an end-to-end checkpoint study.
- Format decoding, capture/replay, backend identity, a high-precision reference, contracts, fault injection, minimization, and JSON/Markdown reports.
- vLLM and FlashInfer adapters only after framework-independent core logic is tested.
- `nvidia/Qwen3-8B-NVFP4` as the primary model checkpoint, its upstream
  high-precision Qwen3-8B checkpoint as the clean reference, and one independently
  produced Qwen3-8B NVFP4 checkpoint as an optional cross-toolchain comparison.

Out of scope unless a gate explicitly changes it:

- MoE, attention-kernel correctness, KV-cache quantization, multi-GPU, training,
  CPU offload, a web dashboard, a new optimized CUDA kernel, or production-safe dispatch.
- Benchmark leaderboard work without a falsifiable research question.
- Uploading model weights, private data, full activations, or large traces to Git.
- Generalizing model-level findings beyond the tested Qwen3 checkpoints, layers,
  prompts, datasets, revisions, and execution backends.

## Expected repository shape

```text
src/nvfp4_doctor/
  env/             # immutable environment fingerprints
  formats/         # E2M1, E4M3, packing, layout, scales
  checkpoint/      # metadata and model adapters
  capture/         # versioned capture schema and storage
  backends/        # backend selection and identity evidence
  oracle/          # dequantization, tiled reference matmul, metrics
  contracts/       # structural, execution, numerical, metamorphic
  faults/          # controlled positive controls
  minimize/        # failure shrinking
  report/          # stable JSON schema and Markdown rendering
tests/{unit,property,integration,fixtures}/
experiments/E###-short-name/
docs/{research-thesis,scope,architecture,nvfp4-contract,limitations,failure-taxonomy}.md
```

Keep framework-independent code importable and testable without CUDA, vLLM, or FlashInfer. GPU/framework dependencies belong behind adapters. Do not let an adapter redefine core format semantics.

## NVFP4 rules

Treat these as separate objects and record all of them explicitly:

- Packed E2M1 values and nibble order.
- Logical tensor shape, physical shape, dtype, stride, padding, and permutation.
- E4M3 block scales, block size, scale layout/index mapping, and global scale.
- Backend requested, backend reported, loaded libraries, and observed kernel identity.

Never infer layout from tensor shape alone. Never silently reshape, make contiguous, transpose, pad, or cast captured data. If a conversion is required, record the source and destination metadata and test the conversion independently.

Use three distinct oracle layers:

1. **Format oracle:** exact assertions for packing, E2M1/E4M3 decoding, indices, shape, strides, padding, and scale mapping.
2. **Mathematical reference:** dequantize explicitly and perform tiled/chunked higher-precision matmul. It is a reference, not an automatic definition of the only legal backend result.
3. **Implementation envelope:** tolerances justified by output dtype, reduction length, accumulation behavior, backend documentation, and the empirical clean-run distribution.

Structural failures and floating-point deviations must have different result categories. Bitwise disagreement between backends is not, by itself, a bug.

## Contracts and fault injection

Every contract must state its domain, preconditions, invariant, metric, threshold, and known limitations. Prefer exact checks for structure and distribution-aware checks for numerics. Metamorphic relations must preserve the stated mathematical semantics; test the relation against the reference before using it to judge a backend.

Required positive controls include:

- Scale index shift and block-scale permutation.
- Packed-nibble swap or corruption.
- Block/row/column permutation.
- Padding, stride, and non-contiguous-layout mistakes.
- Incorrect global scale and deliberate backend/fallback mismatch.

Fault injection must be deterministic, parameterized, reversible, labeled as synthetic, and never overwrite the clean artifact. A detector has no demonstrated power until it detects relevant positive controls. Track false accepts and false rejects; do not tune thresholds on the same cases used as the final evaluation.

## Reproducibility and experiment discipline

Each experiment lives in `experiments/E###-short-name/` and contains:

- `README.md`: falsifiable hypothesis, rationale, controlled/independent variables, expected outcome, actual observations, interpretation, threats to validity, and decision.
- `manifest.json`: UTC time, OS/WSL/kernel, GPU and compute capability, driver, CUDA runtime/toolkit, Python, PyTorch, vLLM, FlashInfer, relevant commit hashes, model/revision, backend, shapes/dtypes/strides, seeds, and exact command.
- Small results and hashes. Large tensors/traces stay outside Git and are referenced by content hash plus acquisition instructions.

Distinguish observation, inference, and claim. Preserve failed experiments and negative results when they affect a decision. Never hand-edit raw output. Record warm-up, synchronization, repetitions, summary statistic, and uncertainty for benchmarks. Compare one independent variable at a time whenever possible.

Release evidence must rebuild from an empty pinned environment. Keep a stable environment and exploratory environments separate. Lock dependencies and record wheel sources; never replace a stable result with an unpinned nightly result.

For model-level experiments, pin both model and dataset revisions. Never load the
high-precision and NVFP4 Qwen3-8B checkpoints on the 16 GB test GPU at the same
time. Run them separately, retain only the minimal selected outputs needed for
comparison, move those outputs off the GPU promptly, and reference large local
artifacts by content hash. Treat prompts and datasets as experimental inputs,
not as evidence by themselves.

## Testing requirements

Before changing code:

1. Read the relevant contract, experiment, and adapter boundary.
2. State the intended behavior and smallest test that can falsify it.
3. Run the narrow baseline test; record pre-existing failures.

While changing code:

- Add or update tests with the implementation.
- For format code, use exhaustive tests where the domain is small (all E2M1 values) and property-based tests for packing/layout.
- Use tiny hand-computable golden fixtures independent of production encoder/decoder code.
- Never generate expected output with the function under test.

Before completion:

- Run formatter/linter, type checks, CPU unit/property tests, then the narrowest relevant GPU integration test.
- Report exact commands and outcomes. Do not claim tests ran when hardware or dependencies were unavailable.
- A GPU test must synchronize before timing or accepting asynchronous success.
- New bug fixes require a regression test; schema changes require migration/versioning tests.

## Claims discipline

Allowed language is proportional to evidence: “observed,” “consistent with,” “detected under these conditions,” or “not detected in the tested matrix.” Do not write “proved correct,” “production safe,” “all NVFP4,” “framework bug,” or causal/performance claims without evidence that supports that exact scope.

Always disclose hardware, software versions, shapes, model/revision, backend evidence, repetitions, uncertainty, and limitations. A clean run means only that no tested contract failed. FP64/reference output is not automatically the specification. A backend difference becomes an upstream bug candidate only after independent oracle validation, repeatability, backend identity evidence, and minimization.

At model level, report layer-output error separately from final-logit and task
metrics. Do not infer user-visible quality from one prompt, equate token
disagreement with semantic failure, or attribute a perplexity change to a kernel
bug until quantization error and contract/kernel error have been separated.

Do not open an upstream issue merely to meet a milestone. An issue/PR needs a minimal public reproducer, pinned environment, expected-versus-actual behavior, artifact hashes, impact, bisect evidence when feasible, and neutral wording. Otherwise publish a negative result or limitation.

## Eight-week roadmap and gates

- **Week 1 — Foundation / Gate 0:** public skeleton, frozen scope, environment manifest, synthetic CUDA sanity check. Go only if `sm_120` is visible and a pinned Linux environment can run repeatably.
- **Week 2 — Format oracle / Gate 1:** exhaustive E2M1, E4M3 scale handling, packing/layout fixtures, exact scale reconstruction. Pivot if public format semantics cannot be independently established.
- **Week 3 — Qwen3 layer capture and replay / Gate 2:** pin and inspect
  `nvidia/Qwen3-8B-NVFP4`; capture representative `q_proj`, `o_proj`, `gate_proj`,
  `up_proj`, and `down_proj` cases without losing metadata; replay them
  deterministically; identify the actual backend/kernel. Pivot to synthetic plus
  checkpoint-only analysis if capture is not reliable or the checkpoint cannot
  execute within the 16 GB GPU boundary.
- **Week 4 — Numerical reference / Gate 3:** tiled high-precision reference, metrics, clean-run envelope across boundary shapes. Stop correctness claims if the oracle cannot pass independent golden tests.
- **Week 5 — Contracts and model-level positive controls / Gate 4:** structural
  and metamorphic suite plus deterministic scale, packing, layout, and dispatch
  faults applied to representative Qwen3 layers. Measure layer outputs, final
  logits, top-k agreement, KL divergence, and perplexity on a fixed evaluation
  slice. Go only if relevant faults are detected with acceptable false alarms and
  at least one silent fault has a reproducible downstream signature.
- **Week 6 — Backend matrix and minimization / Gate 5:** compare at least two genuinely distinct implementations if available, prove no silent fallback, shrink failures. If only one real NVFP4 backend exists on `sm_120`, pivot the comparative study to FP8 or retain NVFP4 as a single-backend diagnostic tool.
- **Week 7 — End-to-end checkpoint study / Gate 6:** compare the clean NVIDIA
  Qwen3-8B NVFP4 checkpoint with its high-precision reference across fixed inputs;
  optionally compare one independently produced Qwen3-8B NVFP4 checkpoint.
  Measure diagnostic accuracy, layer localization, false-positive rate, runtime
  overhead, memory, and latency. Produce an upstream-quality reproducer or a
  documented negative result.
- **Week 8 — Release / Gate 7:** clean-room reproduction, tagged release, technical report, limitations, demo, and explicit Go/Pivot/Stop decision for phase two.

Global stop/pivot triggers: unverifiable backend identity; an oracle that depends on the candidate implementation; no detection advantage over simple baselines after positive controls; results that cannot be reproduced after three disciplined attempts; or a scope that requires unavailable hardware. Stopping or narrowing is a valid research result.

## Daily engineering workflow

1. Select one issue and write a falsifiable hypothesis plus completion criterion.
2. Pull current state; inspect working-tree changes and preserve unrelated user work.
3. Capture the baseline environment and failing/passing tests.
4. Make the smallest reviewable change on a focused branch (`feat/`, `fix/`, `exp/`, `docs/`).
5. Test from narrow to broad; save manifests, seeds, commands, and artifact hashes.
6. Review the diff for silent casts/layout changes, leaked data, generated files, and exaggerated claims.
7. Commit a meaningful unit, for example `exp(E014): measure clean error at M boundaries`.
8. End with one decision: supports hypothesis, does not support it, inconclusive, or invalid; then choose continue, repeat, pivot, or stop.

Never create empty “daily commits.” Research value comes from auditable decisions and reproducible evidence.
