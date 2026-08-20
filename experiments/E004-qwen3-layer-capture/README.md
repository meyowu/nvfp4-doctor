# E004: Qwen3-8B Layer Capture

## Current slice

The first E004 slice pins and validates public checkpoint metadata only. It does
not download model weights, read safetensors headers or payloads, load the
model, capture activations, or execute a backend.

## Hypothesis

The immutable public metadata for `nvidia/Qwen3-8B-NVFP4` is sufficient to
establish a consistent NVFP4 quantization declaration, enumerate the required
quantization tensors for the five planned capture targets, and bound the weight
shards before any large download or runtime experiment.

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

## Controlled variables

- repository: `nvidia/Qwen3-8B-NVFP4`;
- revision: `ccd10a893cbca613259517c3efe08e151ddf2b8e`;
- metadata files: `README.md`, `config.json`, `hf_quant_config.json`, and
  `model.safetensors.index.json`;
- execution device: CPU, metadata only;
- model weights downloaded: false.

The four small source files are retained under ignored local artifact storage.
The tracked normalized result records their immutable URLs and hashes.

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

Run the slice with network access using:

```bash
source ./activate-nvfp4-lab.sh
PYTHONPATH=src python scripts/run_e004_checkpoint_metadata.py
```

The normalized evidence is retained in [metadata.json](metadata.json) and
[manifest-metadata.json](manifest-metadata.json).

## Interpretation

The hypothesis is supported for checkpoint planning. The pinned metadata is
internally consistent for the declared NVFP4 scheme, and all five target
projection families have complete layer-level quantization tensor names.

This is not evidence that the stored tensor shapes or layouts match the Gate 1
oracle, that a supported backend can load the checkpoint on RTX 5080, or that
any kernel executes correctly. Those require safetensors-header inspection and
later metadata-preserving runtime capture.

## Threats to validity

- The safetensors index contains names and shard assignments but not tensor
  shapes, dtypes, strides, or layout semantics.
- The index `total_parameters` field is recorded as upstream metadata and is not
  interpreted as the architectural parameter count of the packed model.
- The model card describes TensorRT-LLM and B200 deployment; it does not
  establish vLLM/FlashInfer execution or RTX 5080 compatibility.
- An immutable repository revision prevents upstream drift but does not
  independently validate the producer's quantization claims.
- No safetensors payload, tokenizer, prompt, activation, GPU execution, or
  observed kernel is part of this slice.

## Decision

Continue. The next bounded step is to retrieve only the two remote safetensors
headers with HTTP range requests, validate their hashes/length boundaries, and
record exact dtypes and shapes for the five projection families before choosing
representative layers or downloading weight payloads.
