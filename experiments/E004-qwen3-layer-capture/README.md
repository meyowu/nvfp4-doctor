# E004: Qwen3-8B Layer Capture

## Current slice

Nine completed E004 slices progress from public checkpoint metadata and bounded
payload acquisition through synthetic projection replay, complete pinned
snapshot acquisition, and one real Qwen activation capture. The latest slice
reuses one full-model load and the same fixed hashed token-ID request to capture
and replay the unfused `o_proj` and `down_proj` modules at layers 0, 18, and 35.
All six cases passed their preregistered transfer, replay, regression, and
range-scoped profiler criteria. Gate 2 remains open because fused production
module boundaries are not covered.

## Hypothesis

The immutable public metadata for `nvidia/Qwen3-8B-NVFP4` is sufficient to
establish a consistent NVFP4 quantization declaration, enumerate the required
quantization tensors for the five planned capture targets, and bound the weight
shards before any large download or runtime experiment.

The second-slice hypothesis is that exact prefix and JSON-header ranges are
sufficient to validate file boundaries, align all tensor names with the pinned
index, and establish stored dtypes and shapes for the five capture targets
without retrieving weight payloads.

The third-slice hypothesis is that validated headers alone are sufficient to
select early, middle, and late layers and construct exact shard-local byte
ranges for every required projection tensor before authorizing payload access.

The fourth-slice hypothesis is that all 60 authorized ranges can be acquired
independently with exact partial-content boundaries and retained as
content-addressed local artifacts without downloading either complete shard.

The fifth-slice hypothesis is that the acquired layer-0 `o_proj` tensors can be
loaded without an implicit cast or layout inference, transformed exactly into
the selected CUTLASS representation, and replayed deterministically through the
vLLM-selected FlashInfer kernel with profiler-backed dispatch evidence.

The sixth-slice hypothesis is that the same explicit loading and transformation
contract holds over the frozen 3-layer by 5-projection matrix, with stable finite
outputs and no case-specific weight mutation or padding.

The seventh-slice hypothesis is that all files in the immutable repository
snapshot can be acquired into ignored local storage and verified against the
pinned Hub checksum inventory before any full-model execution.

The eighth-slice hypothesis is that one real layer-0 `o_proj` prefill activation
can be captured with explicit tensor metadata, replayed deterministically
through the same loaded module, and attributed to the expected SM120 NVFP4
kernel without committing prompt contents or raw tensors.

The ninth-slice hypothesis is that one model load and one fixed hashed request
can produce exactly one metadata-preserving prefill capture for each unfused
`o_proj` and `down_proj` module at layers 0, 18, and 35, and that every captured
input can be replayed three times through its original module with stable,
logical-byte-exact output and independent range-scoped backend evidence.

The tenth-slice hypothesis is that the same request can capture the production
fused `qkv_proj` and `gate_up_proj` module boundaries at those three layers,
that the runtime fused tensors can be reconstructed independently from their
ordered checkpoint components, and that each captured input can be replayed
three times through its original fused module with stable, logical-byte-exact
output and independently scoped backend evidence. This is the preregistered
completion slice for Gate 2.

## Completion criterion

This metadata slice passes only if:

- the Hub resolves the declared revision to the same immutable SHA;
- `config.json` and `hf_quant_config.json` agree on algorithm, group size,
  exclusions, KV-cache scheme, and producer;
- `q_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj` each expose
  `weight`, `input_scale`, `weight_scale`, and `weight_scale_2` in every layer;
- the index shard inventory and remote LFS sizes/hashes are recorded;
- no safetensors weight file is downloaded; and
- normalized results and source files are content-hashed.

The header slice passes only if:

- every request returns HTTP 206 with the exact requested `Content-Range` and
  `Content-Length` before its body is read;
- the 8-byte prefix length, JSON header length, and remote file size produce
  valid payload boundaries for both shards;
- tensor intervals are dtype/shape-consistent, contiguous, and cover each
  declared payload exactly;
- combined header names exactly match all 1,227 pinned index entries;
- all five capture targets have uniform stored dtypes and shapes over 36
  layers; and
- payload bytes downloaded remain zero.

The acquisition-plan slice passes only if:

- layers 0, 18, and 35 represent early, middle, and late positions;
- all four quantization tensors for all five projection families occur exactly
  once for each selected layer;
- the resulting 60 ranges are non-empty, non-overlapping, and within their
  declared shard boundaries;
- the exact planned transfer size is recorded; and
- no payload request is executed while constructing the plan.

The payload-acquisition slice passes only if:

- all 60 planned requests return HTTP 206 with exact `Content-Range` and
  `Content-Length` values before their bodies are accepted;
- every local file length matches its planned tensor interval;
- a SHA-256 is recorded and reverified for every local tensor artifact;
- the combined acquired size is exactly 311,427,192 bytes; and
- neither complete safetensors shard is downloaded.

The single-projection replay slice passes only if:

- all four layer-0 `o_proj` artifacts match their recorded lengths and hashes;
- packed weight bytes remain unchanged and no weight padding is required;
- the independent 128x4 scale transform matches vLLM byte-for-byte;
- three synchronized replays produce the same finite BF16 `(16, 4096)` output;
- vLLM selects its FlashInfer CUTLASS NVFP4 kernel; and
- Nsight attributes an SM120 block-scaled E2M1 CUTLASS kernel, with no known
  fallback signature, to the exact E004 NVTX range.

The representative replay matrix passes only if:

- all 15 layer/projection cases are present exactly once;
- every case revalidates its four acquired source tensors;
- every packed weight remains byte-identical and requires zero padding;
- every independent scale swizzle matches vLLM byte-for-byte;
- three synchronized repeats per case are finite and hash-stable; and
- fused-family cases remain labeled as individual kernel preflights rather than
  production model-layer replays.

The full-model acquisition slice passes only if all 15 pinned snapshot files
and 6,413,063,143 bytes are present, every pinned checksum verifies, both weight
shards match their immutable identities, and model bytes remain ignored.

The first real-activation slice passes only if:

- the full checkpoint loads as `Qwen3ForCausalLM` under the frozen single-GPU,
  eager, no-CPU-offload configuration;
- one fixed hashed request produces exactly one layer-0 `o_proj` prefill input;
- the captured BF16 input has shape `(9, 4096)` and preserves shape, dtype,
  stride, storage offset, and canonical logical bytes across the recorded
  CUDA-to-CPU transfer;
- three synchronized replays are finite, hash-stable, and byte-exact to the
  captured module output;
- vLLM selects `FlashInferCutlassNvFp4LinearKernel`; and
- the exact NVTX range contains the expected SM120 block-scaled CUTLASS
  E2M1/UE4M3 signature and no known fallback.

The representative unfused real-activation matrix passes only if:

- the exact ordered Cartesian product of layers 0, 18, and 35 with `o_proj`
  and `down_proj` is captured from one model load and one request;
- all twelve hooks fire once in model execution order, and all eighteen tensor
  artifacts have unique ignored paths and preserve shape, dtype, stride,
  storage offset, byte length, and canonical logical bytes across transfer;
- every runtime packed-weight and swizzled-scale identity matches the tracked
  representative replay dependency and requires zero weight padding;
- one warm-up and three synchronized replays per case are finite, hash-stable,
  and logical-byte-exact to the corresponding captured module output;
- the overlapping layer-0 `o_proj` input and output match the prior real case;
  and
- all six unique NVTX ranges contain activation quantization and the expected
  SM120 block-scaled CUTLASS E2M1/UE4M3 signature with no known fallback.

The representative fused real-activation matrix passes only if:

- the exact ordered Cartesian product of layers 0, 18, and 35 with
  `qkv_proj` and `gate_up_proj` is captured from one model load and the same
  fixed hashed request;
- the runtime boundaries are exactly `QKVParallelLinear` with logical widths
  `[4096, 1024, 1024]` and `MergedColumnParallelLinear` with logical widths
  `[12288, 12288]`, both at tensor-parallel size one;
- the checkpoint components are independently loaded in production order,
  their packed rows concatenate byte-exactly to the runtime weight, and their
  linear scale rows reproduce the runtime scale only after the independent
  128x4 swizzle oracle is applied;
- every component within a fused group has an equal `input_scale` and equal
  `weight_scale_2`, the recorded maximum reduction is therefore lossless, and
  the reduced runtime scales, reciprocal, and float32 alpha match exactly;
- all twelve hooks fire once in model execution order, and all eighteen tensor
  artifacts preserve the declared logical metadata and bytes across transfer;
- one warm-up and three synchronized whole-module replays per case are finite,
  hash-stable, and logical-byte-exact to the corresponding captured fused
  module output; and
- all six unique NVTX ranges contain activation quantization and the expected
  SM120 block-scaled CUTLASS E2M1/UE4M3 signature with no known fallback.

If any fused construction, capture, replay, or range-scoped identity check is
inconclusive, Gate 2 remains open or pivots to checkpoint-only plus synthetic
analysis. A pass permits a Gate 2 `go` decision only for metadata-preserving
real-activation replay and kernel identification; it does not establish NVFP4
numerical correctness, independent component replay, post-activation
equivalence, prompt diversity, final logits, or model quality.

## Controlled variables

- repository: `nvidia/Qwen3-8B-NVFP4`;
- revision: `ccd10a893cbca613259517c3efe08e151ddf2b8e`;
- metadata files: `README.md`, `config.json`, `hf_quant_config.json`, and
  `model.safetensors.index.json`;
- execution device: one RTX 5080 (`sm_120`) for replay slices; and
- full snapshot location: ignored project-local `models/` storage.

The four small source files are retained under ignored local artifact storage.
The tracked normalized result records their immutable URLs and hashes.

Header requests use `Accept-Encoding: identity`, a 16 MiB per-shard header
limit, and fail before reading a body if the endpoint does not return exact
partial content. Raw headers are parsed in memory and are not committed.

The representative selection is fixed at layer 0, layer 18 (the first layer of
the model's second half), and layer 35. The plan covers `weight`,
`input_scale`, `weight_scale`, and `weight_scale_2` for each target projection.

Authorized payloads are stored under the ignored project-local
`artifacts/E004-qwen3-layer-capture/tensor-payloads/` directory. The downloader
uses atomic per-tensor writes and an ignored progress index so interrupted runs
can resume only from artifacts whose length, metadata, and SHA-256 still match.

The first replay uses a deterministic BF16 `(16, 4096)` activation whose values
are defined by a recorded integer sequence. It stays below the checkpoint's
declared input calibration bound. Activation quantization happens before the
target NVTX range; the range contains the selected projection GEMM only.

The matrix uses the same 16-row recipe and seed independently for every case.
Each case runs in a separate process so no preceding projection weight remains
resident on the 16 GB GPU.

The real-activation case fixes a nine-token identity by SHA-256 while omitting
the token array and prompt text from tracked evidence. vLLM runs in single-
process eager mode with WSL2 pinned memory enabled, FlashInfer sampling disabled
as unrelated to the target, tensor parallelism 1, CPU offload 0, and a 256 MiB
KV cache. Its target NVTX range includes the production module's activation
quantization and NVFP4 GEMM, while validation and device copies remain outside.

The real-activation matrix reuses that exact request and environment. It
installs hooks only after model initialization, profiles replay rather than the
live hook path, and uses one profiler session containing six sibling NVTX
ranges. The matrix is limited to production-aligned unfused module boundaries;
fused `qkv_proj` and `gate_up_proj` require a later design.

## Actual observations

The pinned repository is public and ungated. The two quantization declarations
agree on `NVFP4`, group size 16, static 4-bit floating-point weights and input
activations, exclusion of `lm_head`, FP8 KV cache, and ModelOpt 0.35.0.

The model config declares `Qwen3ForCausalLM`, 36 layers, hidden size 4096,
intermediate size 12288, 32 attention heads, 8 key/value heads, BF16 as the
unquantized model dtype field, and vocabulary size 151936.

The safetensors index contains 1,227 tensor names over two shards. Each of the
five planned capture projections has four quantization tensors in all 36
layers, or 144 indexed tensors per projection. The two remote weight files total
6,397,066,384 bytes; the index reports 6,396,932,352 tensor-payload bytes. Their
134,032-byte difference is recorded as safetensors header overhead rather than
model data.

Only 112,965 bytes of metadata were downloaded. No model weight file was
downloaded.

Both shard URLs honored two exact ranges: the 8-byte prefix followed by the
declared JSON header. The first header is 129,048 bytes and describes 1,181
tensors; the second is 4,968 bytes and describes 46 tensors. Prefixes plus
headers total 134,032 bytes, matching the previously observed aggregate header
overhead. All tensor intervals are contiguous, end at the declared payload
boundary, and align exactly with the pinned index. No payload byte was read.

Across all 36 layers, `input_scale` and `weight_scale_2` are scalar F32 and
`weight_scale` is F8_E4M3. Stored packed weights are U8 with shapes:

| Projection | Stored weight shape | Stored block-scale shape |
| --- | --- | --- |
| `q_proj` | `(4096, 2048)` | `(4096, 256)` |
| `o_proj` | `(4096, 2048)` | `(4096, 256)` |
| `gate_proj` | `(12288, 2048)` | `(12288, 256)` |
| `up_proj` | `(12288, 2048)` | `(12288, 256)` |
| `down_proj` | `(4096, 6144)` | `(4096, 768)` |

These are stored checkpoint shapes. In particular, a U8 weight element packs
two four-bit values; the table is not a runtime stride or layout declaration.

The acquisition planner found all 60 selected tensors exactly once and mapped
their payload-relative offsets to absolute HTTP ranges. Each layer accounts for
103,809,064 bytes, for a planned total of 311,427,192 bytes. Forty ranges occur
in the first shard and twenty in the second. Construction executed four header
requests totaling 134,032 bytes and zero payload requests.

The authorized acquisition then completed all 60 exact range requests and
wrote 311,427,192 bytes to ignored local files. Every response boundary and
length matched the plan, and an independent local pass recomputed all 60
SHA-256 values without a mismatch. No complete weight shard was downloaded.

Layer-0 `o_proj` loaded as U8 `(4096, 2048)`, representing a logical
`(4096, 4096)` weight, plus F8_E4M3 scales at `(4096, 256)` and two scalar F32
values. Packed weight SHA-256 remained
`1db669cf9be8e913653ff5aea1e30d4db2da2e86a2a635c952f8fa8346056f8a`.
The independently swizzled scale SHA-256 was
`4e6992cbfa93bd7136816762fbf212861a103944f1d6054cd6d13eac15347be2`,
matching vLLM byte-for-byte; weight padding was zero.

vLLM selected `FlashInferCutlassNvFp4LinearKernel`. One warm-up and three
synchronized measured replays produced finite BF16 `(16, 4096)` outputs with
the same SHA-256,
`ae14a5d1cf6c304e3c57cbeee2f024fbc7a26963e7b53d59d34963a77b4e3faf`.
Nsight Systems attributed the expected SM120 block-scaled CUTLASS E2M1 kernel
to `e004:layer_00:o_proj:nvfp4_gemm`; no known fallback signature occurred in
that range. The retained ignored profiler report has SHA-256
`faadecc958a5a0b2730a90e6325f650bf4e761b79ea377079699e9f2edf3702d`.

All 15 early/middle/late matrix cases passed. Every case preserved its packed
weight bytes, required zero padding, matched the independent scale transform,
and produced three identical finite output hashes. The 15 case outputs had 15
distinct hashes. Six `o_proj`/`down_proj` cases are labeled
`production_aligned_unfused`; nine `q_proj`/`gate_proj`/`up_proj` cases are
labeled `individual_fused_family_preflight`.

The complete immutable snapshot was then acquired under ignored `models/`
storage. Its 15 files total 6,413,063,143 bytes, including 6,397,066,384 bytes
across the two safetensors shards, and all pinned Hub checksums verified.

The full model loaded on the RTX 5080 and produced one real BF16 layer-0
`o_proj` prefill input with shape `(9, 4096)` and SHA-256
`c4c16cadca3b8981e8bdfdd7bd20b2b7b6c6e2be7b34ffa9e98cd6f23890893c`.
Three synchronized standalone replays were finite and byte-exact to the
captured module output, with common SHA-256
`da6b9fd682b8fd312ca95379f9993ca4fd1dec4a1f38ca3b1629c87f3b0abf2f`.
vLLM selected `FlashInferCutlassNvFp4LinearKernel`; Nsight found the expected
SM120 block-scaled CUTLASS E2M1/UE4M3 signature and no known fallback in
`e004:real_activation:layer_00:o_proj:nvfp4_gemm`.

The representative real-activation matrix then captured `o_proj` and
`down_proj` at layers 0, 18, and 35 from one model load and the same fixed
hashed request. All 12 hooks fired once in the expected model order. The three
`o_proj` inputs had shape `(9, 4096)`, the three `down_proj` inputs had shape
`(9, 12288)`, and all outputs had shape `(9, 4096)`. The 18 ignored tensor
artifacts had unique paths, and the recorded CUDA-to-CPU transfers preserved
shape, dtype, stride, storage offset, byte length, and canonical logical bytes.

Every runtime packed weight and swizzled scale matched its tracked replay-matrix
dependency, and every case required zero weight padding. One warm-up and three
synchronized measured replays per case were finite, within-case hash-stable,
and logical-byte-exact to the captured module output. All six input hashes and
all six output hashes were distinct. The layer-0 `o_proj` input and output
hashes exactly matched the earlier single real-activation result.

| Case | Captured input SHA-256 | Captured/replayed output SHA-256 |
| --- | --- | --- |
| layer 0 `o_proj` | `c4c16cadca3b8981e8bdfdd7bd20b2b7b6c6e2be7b34ffa9e98cd6f23890893c` | `da6b9fd682b8fd312ca95379f9993ca4fd1dec4a1f38ca3b1629c87f3b0abf2f` |
| layer 0 `down_proj` | `bb2448b43501160994fdb336417df8a2cde32efed0c609ec1cc24beed8f5f4bb` | `9b9daeb4bb6b9f0f654ef1790530107e3f41f6daa4271f17b336a8dd94854f37` |
| layer 18 `o_proj` | `44254d76073176c98be92708b4a414ed9c8a78383f733ec9edddf7a23b6e76cf` | `9a4130925666d7452b2de53859e4fe052e3ce405515f89c649d53f860e6b122e` |
| layer 18 `down_proj` | `b16c1fb4f76e3716a4c69606b5e60e1e5cd7e577fbdfd9470074b845e054b6d4` | `33eb9efc23365b8eb6b50f2491f008cb243a0e4ae9f90a12309b6d7a42c02fd9` |
| layer 35 `o_proj` | `3a54eb38bcdd52cf844144c716d8571e03faa84fc34b99cd82145a7a3b9bd57d` | `6eec4e6a5988f920f1573b14d13b69e84ec713d6bec614b480b4ffacf0924603` |
| layer 35 `down_proj` | `2ff622a107f58392bbb0f755828992927910a18f25609d5c67d24a03d18f1ceb` | `e4d0699df7af0f47539c48a77dc0b981b6c0b390cb6cc50327e572ccf42a4477` |

vLLM selected `FlashInferCutlassNvFp4LinearKernel`. All six exact NVTX ranges
contained both the vLLM BF16-to-FP4 activation conversion and the expected
SM120 block-scaled CUTLASS E2M1/UE4M3 kernel, with no known fallback signature.
The retained ignored profiler report has SHA-256
`96351a88df4c7b256e909a38b0ea79e51ef68e3da1e3a16da02dbeb40973c79c`.
The normalized result has SHA-256
`0ea500a2823aff6017d4027ce4ad1bcc84c03205400fdde43258a56af1c97651`,
the clean source bundle has SHA-256
`d669b40b1dd27da800daeb2a9d28089981df4238427b01f6dbaed537d2445181`,
and the run peaked at 7,189,731,328 allocated GPU bytes.

Run the slice with network access using:

```bash
source ./activate-nvfp4-lab.sh
PYTHONPATH=src python scripts/run_e004_checkpoint_metadata.py
PYTHONPATH=src python scripts/run_e004_safetensors_headers.py
PYTHONPATH=src python scripts/run_e004_acquisition_plan.py
PYTHONPATH=src python scripts/run_e004_tensor_acquisition.py
bash scripts/run_e004_projection_profile.sh
PYTHONPATH=src python scripts/run_e004_replay_matrix.py
PYTHONPATH=src python scripts/run_e004_full_model_acquisition.py
bash scripts/run_e004_real_activation_profile.sh
bash scripts/run_e004_real_activation_matrix_profile.sh
```

The normalized evidence is retained in [metadata.json](metadata.json) and
[manifest-metadata.json](manifest-metadata.json).
Header evidence is retained in [headers.json](headers.json) and
[manifest-headers.json](manifest-headers.json).
The exact representative-layer plan is retained in
[acquisition-plan.json](acquisition-plan.json) and
[manifest-acquisition-plan.json](manifest-acquisition-plan.json).
The tracked hash and range inventory is retained in [payloads.json](payloads.json)
and [manifest-payloads.json](manifest-payloads.json); the payload bytes remain
ignored and local.
The normalized first replay and its provenance are retained in
[replay-single-projection.json](replay-single-projection.json) and
[manifest-replay.json](manifest-replay.json); the raw run and Nsight report
remain ignored and local.
The complete bounded matrix and its clean provenance are retained in
[replay-matrix.json](replay-matrix.json) and
[manifest-replay-matrix.json](manifest-replay-matrix.json); per-case raw runs
remain ignored and local.
Complete-snapshot integrity is retained in
[full-model-acquisition.json](full-model-acquisition.json) and
[manifest-full-model-acquisition.json](manifest-full-model-acquisition.json).
The normalized real-activation observation and clean provenance are retained in
[real-activation-replay.json](real-activation-replay.json) and
[manifest-real-activation-replay.json](manifest-real-activation-replay.json);
raw tensors and the Nsight report remain ignored and local.
The normalized representative unfused observation and clean provenance are
retained in
[real-activation-unfused-matrix.json](real-activation-unfused-matrix.json) and
[manifest-real-activation-unfused-matrix.json](manifest-real-activation-unfused-matrix.json);
its 18 raw tensors, raw run, and Nsight report remain ignored and local.

## Interpretation

The hypothesis is supported for checkpoint planning. The pinned metadata is
internally consistent for the declared NVFP4 scheme, and all five target
projection families have complete layer-level quantization tensor names.

This is not evidence that the stored tensor shapes or layouts match the Gate 1
oracle, that a supported backend can load the checkpoint on RTX 5080, or that
any kernel executes correctly. Those require safetensors-header inspection and
later metadata-preserving runtime capture.

The header hypothesis is also supported. The stored dtype/shape inventory is
now sufficient to design a bounded representative-layer acquisition plan, but
it does not define logical NVFP4 layout semantics or authorize payload download.

The acquisition-plan hypothesis is supported. The next payload step is bounded
to 311,427,192 bytes rather than either complete shard, and every intended range
can be audited before transfer. This result plans a transfer; it does not
authorize or validate one.

The payload-acquisition hypothesis is also supported. The selected stored bytes
are now reproducible by immutable source revision, exact range, length, and
local SHA-256. This establishes acquisition integrity only; decoding, scale
mapping, runtime views, and backend execution remain separate questions.

The single-projection hypothesis is supported for the bounded layer-0 `o_proj`
case. Stored and runtime metadata, the exact scale permutation, deterministic
output, vLLM kernel selection, and range-scoped profiler identity are now
recorded separately. Output finiteness and repeatability are execution evidence,
not a numerical-correctness comparison.

The representative-matrix hypothesis is also supported. The checkpoint-to-
runtime transform and deterministic execution preflight held for all acquired
cases. This expands structural and execution coverage but still supplies no real
Qwen activations or numerical reference outputs.

The full-snapshot integrity hypothesis and the first real-activation hypothesis
are supported for their bounded cases. The latter establishes one captured
module input, stable same-module replay, and range-scoped kernel identity. It is
not a numerical reference comparison or a representative activation matrix.

The representative unfused real-activation hypothesis is supported for one
fixed request. The result expands real activation, same-module replay, and
independently scoped backend identity across early, middle, and late `o_proj`
and `down_proj` modules. It does not cover the fused `qkv_proj` or
`gate_up_proj` production boundaries and is not a numerical reference or model
quality result.

## Threats to validity

- Safetensors headers establish stored shapes, dtypes, and byte offsets but not
  runtime strides, framework views, or NVFP4 scale-layout semantics.
- The index `total_parameters` field is recorded as upstream metadata and is not
  interpreted as the architectural parameter count of the packed model.
- The model card describes TensorRT-LLM and B200 deployment; it does not
  establish vLLM/FlashInfer execution or RTX 5080 compatibility.
- An immutable repository revision prevents upstream drift but does not
  independently validate the producer's quantization claims.
- The tracked result stores only hashes for the fixed token IDs and generated
  token; it does not establish prompt diversity or tokenizer-level semantics.
- Individual tensor ranges do not have upstream per-tensor hashes. A future
  clean-room reproduction must compare newly computed local hashes while still
  pinning each source shard by its immutable LFS SHA-256.
- Exact transport and local hashes do not prove that the producer's logical
  NVFP4 layout matches the Gate 1 oracle.
- The 15-case checkpoint replay matrix uses synthetic activations. The latest
  six-case matrix adds real Qwen prefill activations only for unfused `o_proj`
  and `down_proj` modules.
- `o_proj` is unfused in the inspected vLLM path. Individual `q_proj`,
  `gate_proj`, and `up_proj` replays would not reproduce their fused model-layer
  loading behavior without the companion tensors.
- Six profiled unfused projections establish selected-kernel identity only for
  this pinned environment and these cases; they do not establish numerical
  correctness.
- One fixed request and six unfused cases do not characterize other prompts,
  fused module families, final logits, model quality, or a high-precision
  reference.

## Decision

Continue within Gate 2. The representative unfused real-activation matrix now
passes for `o_proj` and `down_proj` at early, middle, and late layers. The next
bounded step is to design real-activation capture and replay at the production
fused `qkv_proj` and `gate_up_proj` boundaries. Gate 2 is not complete, and no
numerical-correctness or model-quality claim is made.
