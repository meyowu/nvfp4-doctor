# Frozen Research Scope

This scope was frozen for the eight-week QContract research track on
2026-08-20. A gate decision may narrow or pivot it, but exploratory results do
not silently expand it.

## Research question

Can executable NVFP4 contracts distinguish expected quantization error from
silent format, scale, layout, and dispatch failures, localize those failures,
and measure their downstream effect in a bounded Qwen3-8B workload?

## In scope

- One NVIDIA RTX 5080 (`sm_120`, 16 GB) under Windows 11 and Ubuntu 24.04 on
  WSL2.
- Dense NVFP4 quantization and GEMM, beginning with synthetic inputs and then
  representative Qwen3-8B linear layers.
- Environment provenance, packed-format and scale oracles, capture/replay,
  execution contracts, deterministic fault injection, minimization, and
  machine-readable reports.
- FlashInfer and vLLM adapters behind framework-independent contracts.
- Sequential, bounded comparison of the pinned NVIDIA NVFP4 checkpoint and its
  high-precision upstream reference.

## Out of scope

- Training, MoE, attention correctness, KV-cache quantization, multi-GPU,
  production serving, CPU offload, or development of a new optimized kernel.
- General performance ranking or claims about GPUs, models, formats, or
  backends outside the recorded test matrix.
- Committing model weights, full activations, profiler reports, caches, tokens,
  private inputs, or other large raw artifacts.

## Evidence and claims boundary

Each gate requires a falsifiable experiment, exact environment and command
provenance, independent observations, artifact hashes, limitations, and an
explicit Go/Pivot/Stop decision. A finite output, a kernel name, one prompt, or
agreement with the candidate implementation is not a correctness proof.

Week 1 establishes only that the pinned local environment repeatedly executes
the scoped synthetic NVFP4 path and that no known fallback signature was
observed inside the target NVTX range. Format correctness begins at Gate 1.
