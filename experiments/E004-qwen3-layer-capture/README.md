# E004: Qwen3-8B Layer Capture

## Current slice

Three E004 slices now pin public checkpoint metadata, validate the two remote
safetensors headers, and produce a metadata-only acquisition plan for
representative layers. They do not download tensor payloads, load the model,
capture activations, or execute a backend.

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

## Controlled variables

- repository: `nvidia/Qwen3-8B-NVFP4`;
- revision: `ccd10a893cbca613259517c3efe08e151ddf2b8e`;
- metadata files: `README.md`, `config.json`, `hf_quant_config.json`, and
  `model.safetensors.index.json`;
- execution device: CPU, metadata only;
- model weights downloaded: false.

The four small source files are retained under ignored local artifact storage.
The tracked normalized result records their immutable URLs and hashes.

Header requests use `Accept-Encoding: identity`, a 16 MiB per-shard header
limit, and fail before reading a body if the endpoint does not return exact
partial content. Raw headers are parsed in memory and are not committed.

The representative selection is fixed at layer 0, layer 18 (the first layer of
the model's second half), and layer 35. The plan covers `weight`,
`input_scale`, `weight_scale`, and `weight_scale_2` for each target projection.

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

Run the slice with network access using:

```bash
source ./activate-nvfp4-lab.sh
PYTHONPATH=src python scripts/run_e004_checkpoint_metadata.py
PYTHONPATH=src python scripts/run_e004_safetensors_headers.py
PYTHONPATH=src python scripts/run_e004_acquisition_plan.py
```

The normalized evidence is retained in [metadata.json](metadata.json) and
[manifest-metadata.json](manifest-metadata.json).
Header evidence is retained in [headers.json](headers.json) and
[manifest-headers.json](manifest-headers.json).
The exact representative-layer plan is retained in
[acquisition-plan.json](acquisition-plan.json) and
[manifest-acquisition-plan.json](manifest-acquisition-plan.json).

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

## Threats to validity

- Safetensors headers establish stored shapes, dtypes, and byte offsets but not
  runtime strides, framework views, or NVFP4 scale-layout semantics.
- The index `total_parameters` field is recorded as upstream metadata and is not
  interpreted as the architectural parameter count of the packed model.
- The model card describes TensorRT-LLM and B200 deployment; it does not
  establish vLLM/FlashInfer execution or RTX 5080 compatibility.
- An immutable repository revision prevents upstream drift but does not
  independently validate the producer's quantization claims.
- No safetensors payload, tokenizer, prompt, activation, GPU execution, or
  observed kernel is part of this slice.
- Individual tensor ranges do not have upstream per-tensor hashes. A future
  authorized acquisition must record locally computed range hashes while still
  pinning each source shard by its immutable LFS SHA-256.

## Decision

Continue. With separate explicit authorization for payload access, the next
bounded step is to retrieve the 60 planned ranges into ignored local artifact
storage, verify every length and range boundary, and record a SHA-256 for each
acquired tensor. Without that authorization, stop at this metadata-only
boundary.
