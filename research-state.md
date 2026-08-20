# Research State

This is the canonical handoff record for resumable research sessions. Update it
only from observed repository, environment, test, and experiment evidence.

## Current handoff

- Current experiment: `E004-qwen3-layer-capture`
- Current gate: `Gate 2 — representative real-activation coverage`
- Status: `single_real_activation_replay_pass`
- Decision: `continue`
- Verified baseline commit: `3dfd06fbd3ca5e01b8218967def5506135b4bba4`
- Verified Gate 1 implementation commit: `f0be51513892d8b10968090fb081a8dafbee0b89`
- Verified E003 implementation commit: `2116fbb5ed76d51119ea79b89da701c25b0ef0d5`
- Verified E003 completion commit: `326522ce74b70ee5934373a2a5fa72d512cd8390`
- Verified E004 metadata commit: `68cb4e0328598dd33d3626b0046e704c97ac39b5`
- Verified E004 header commit: `ee716253c14f0f6b1a8364d2f92253ec01cbe571`
- Verified E004 acquisition-plan implementation commit: `2f4f6301447dd6f6dc73164d4f30536679bcbb8b`
- Verified E004 acquisition-plan evidence commit: `76ef175ad885b741068dd21db08149b0902840bd`
- Verified E004 tensor-acquisition implementation commit: `62db873d3951bbcdcabe638983165a941f62eea2`
- Verified E004 tensor-acquisition evidence commit: `60c05df86936ca958088d4685b25001f77ad0da0`
- Verified E004 tensor-acquisition seal commit: `597d076e01bbbc7fb9df246220d67b049cb62f4e`
- Verified E004 payload-loader commit: `729eaadd32233e806756d0a58a5cf75b012533b4`
- Verified E004 single-replay implementation commit: `816fee61f0de6014029730c065a73a14a7b8a3be`
- Verified E004 single-replay evidence commit: `63d0d7a971dd5c30e9309b9451de6c399b5365c9`
- Verified E004 replay-matrix implementation commit: `603e914253548f8b4e7adeb83d07f13b170b0433`
- Verified E004 replay-matrix evidence commit: `7fc5cec640affb0af2143ce90a8887b0460a3293`
- Verified E004 full-snapshot verifier commit: `6dfa5abe03e72f8c852d3e686a474994304a86ef`
- Verified E004 full-snapshot evidence commit: `96cfaed51e0357a58219c597961bcf7e7ed3a669`
- Verified E004 real-activation capture commit: `95bc5d7dbf6256bee32814e347903facd1d77ace`
- Verified E004 real-activation finalizer commit: `c7884800ce20a9669386e46a53772ed4f4d18796`
- Verified E004 real-activation evidence commit: `3dfd06fbd3ca5e01b8218967def5506135b4bba4`
- Active local branch: `exp/e004-real-activation`
- Active Codex worktree: `/mnt/c/Users/meyow/Documents/Codex/2026-08-19/referenced-chatgpt-conversation-this-is-an/nvfp4-doctor`
- Canonical WSL checkout: `/home/meyowu/projects/nvfp4-doctor`
- Canonical GPU environment: `/home/meyowu/projects/nvfp4-doctor/.venv`
- Hardware boundary: one RTX 5080 (`sm_120`, 16 GB)

## Completed evidence

- The RTX 5080 WSL2 research environment is operational.
- A FlashInfer CUTLASS NVFP4 quantize-and-GEMM smoke run completed with finite
  output at shape `(16, 128)`.
- Nsight Systems 2026.1.3 produced a retained, ignored profiler report for the
  smoke run with SHA-256
  `62fecf2f03911cd092c32c539c80145bc74a97711fbbe4d7f64d0b9e7507df86`.
- The initial `src/nvfp4_doctor` package and CPU test boundaries exist.
- Project code, environment, caches, models, artifacts, and profiler outputs now
  have a canonical project-local directory layout.
- E001 manifest schema version 1 now has an immutable, CPU-only data model and a
  hand-authored golden JSON fixture.
- Strict tests reject unknown fields, unsupported schema versions, missing
  backend evidence, and invalid tensor stride ranks.
- CPU-only collectors now capture the single-GPU fingerprint and host/software
  versions through read-only commands and package metadata. Their logic is
  tested with injected observations and does not import GPU frameworks.
- A real WSL collector run observed driver `595.95`, runtime package `13.2.75`,
  CUDA toolkit build `13.2.86`, WSL `2.7.12.0`, and the pinned PyTorch, vLLM,
  and FlashInfer versions.
- Manifest assembly now combines host/software, Git, command, tensor, and
  requested-backend evidence while leaving profiler-derived fields unknown.
- A local pre-profiler manifest was generated under ignored artifact storage;
  profiler evidence was then attached without overwriting unknown conclusions.
- Independent parsing of the Nsight `cuda_gpu_kern_sum` CSV observed 18 unique
  CUDA kernel names, including an SM120 block-scaled CUTLASS kernel with E2M1
  operands and the TensorRT-LLM NVFP4 block-quantization kernel.
- The project activation script now exports `LIBRARY_PATH` for package-local
  CUDA runtime and WSL driver libraries; this repaired an observed JIT linker
  failure for `-lcudart` and `-lcuda`.
- The synthetic workload now uses separate NVTX ranges for quantization, NVFP4
  GEMM, and the unquantized reference, preventing reference cuBLAS kernels from
  contaminating the fallback assessment.
- The E001 classifier returns `not_detected` only when the expected SM120
  block-scaled CUTLASS E2M1 signature is present and no known fallback signature
  occurs inside `e001:nvfp4_gemm`; unfamiliar evidence remains `unknown`.
- Three controlled runs each completed with identical numerical observations,
  23 unique observed kernel names, fallback status `not_detected`, and the same
  four-kernel target-range set hash
  `98aa16a78e2cdb3d74f27a0568bc2ed9f35de0e8584936717180c5e9c76ef10a`.
- The generated repeatability summary records stable environment and target
  kernel evidence, complete profiler hashes, `gate0_repeatability=pass`, and a
  `go` decision.
- The repository-tracked representative manifest records inputs, packed FP4 storage,
  block-scale logical/physical layouts, output metadata, environment, command,
  observed kernels, and the retained report hash.
- CUTLASS DSL 4.6 requires `nvidia-cuda-nvdisasm>=13.3,<14`; pinning the
  independent inspection tool to 13.3.73 removed four package-metadata
  incompatibilities while leaving the CUDA 13.2 compiler/runtime pins intact.
- E002 records the public NVFP4 semantics used by the oracle, including the
  E2M1 value table, low-nibble-first packing, UE4M3 decoding, a scalar FP32
  global scale, 16-value block scales, and the CUTLASS 128x4 physical layout.
- The CPU-only oracle reconstructs values as
  `E2M1 value * UE4M3 block scale * scalar global scale` without importing
  PyTorch, CUDA, FlashInfer, or vLLM.
- Hand-authored fixtures cover all 16 E2M1 bit patterns, selected UE4M3
  boundary values, eight packing bytes, eleven CUTLASS layout offsets, and a
  directly checkable hierarchical reconstruction case.
- Property tests cover nibble packing round trips, scale-layout round trips,
  and layout offset bijection and bounds over generated shapes and payloads.
- The E002 GPU differential runner compared FlashInfer `nvfp4_quantize`
  outputs byte-for-byte against independently computed packed E2M1 values and
  UE4M3 scale bytes for three tensors spanning linear and CUTLASS 128x4 scale
  layouts, padding, row-atom and scale-atom boundaries, and global scales 0.5,
  1.0, and 2.0.
- All three controlled GPU cases matched exactly. Together they exercised all
  127 finite UE4M3 codes, and independent reconstruction of the candidate
  bytes had maximum absolute error 0.0 for the deliberately representable
  inputs.
- E002 retains source provenance, tensor metadata, requested backend evidence,
  artifact hashes, and a bounded `go` decision in its manifest and results.
- The final E002 manifest was regenerated from clean commit
  `f0be51513892d8b10968090fb081a8dafbee0b89`; it records `dirty=false` and the
  unchanged semantic source-bundle SHA-256
  `88f5b9afaa4f2d83b25b298d0d981026740b810a928b3f1f037e312e00c12154`.
- E003's first CPU-only slice defines six exact format contracts with explicit
  domains, preconditions, invariants, mismatch metrics, zero-mismatch
  thresholds, and limitations.
- Six deterministic, reversible faults cover nibble swapping, scale-index
  shifting, block-scale reversal, global-scale multiplication, layout
  mislabeling, and physical padding corruption. Every artifact is labeled
  `synthetic`, and the immutable clean tensor is retained separately.
- Three clean artifacts produced 18 passing contract evaluations. All six
  faults were detected, every failed-contract set matched its declared expected
  set, and clean false rejects, fault false accepts, localization failures, and
  reversibility failures were zero.
- The padding control changed one of 1,403 physical padding bytes while logical
  reconstruction remained exact, demonstrating a bounded structurally invalid
  but numerically silent positive control.
- E003's second CPU-only slice defines separate exact contracts for recorded
  stride, row-major contiguity, requested backend, reported backend, observed
  kernel tuple, and bounded fallback status.
- One clean execution-evidence snapshot passed all six contracts. Two stride
  faults and three backend-identity faults were detected with exact localization
  and zero clean false rejects, fault false accepts, localization failures, or
  reversibility failures.
- A reported-backend-only fault left requested-backend and observed-kernel
  contracts passing. This directly checks that the three identity fields remain
  separate rather than being inferred from one another.
- The fallback-kernel string in this slice is labeled synthetic and is not
  represented as a profiler observation from the run.
- The final format-fault manifest was regenerated from clean implementation
  commit `2116fbb5ed76d51119ea79b89da701c25b0ef0d5`; it records `dirty=false` and
  source-bundle SHA-256
  `1029f17f569ee74f509673225ea00977a018adcd5590af08247344a5f2a8bad6`.
- The final execution-evidence manifest was regenerated from clean provenance
  commit `4c7cab930f74c7a7f5457b94d1ffc7bf642191cb`; it records `dirty=false` and
  source-bundle SHA-256
  `0a9e5587e74205092338448d151b3ce80ed6c03003b5d6ebddebc72c984b1f23`.
- E003's final slice adds reversible cyclic permutations of complete packed
  value blocks, rows, and logical columns without changing scale storage or
  metadata.
- Three held-out clean artifacts with shapes `(3, 48)`, `(5, 64)`, and
  `(131, 80)` produced 18 passing contract evaluations. Their data salts and
  linear/CUTLASS layouts differ from the development matrix.
- Gate 1 exact zero-mismatch thresholds were frozen before the held-out matrix
  was constructed and were not tuned on its cases.
- All nine held-out permutation faults were detected and localized exactly to
  packed values plus reconstruction. Clean false rejects, fault false accepts,
  localization failures, and reversibility failures were zero.
- The held-out manifest was regenerated from clean completion commit
  `326522ce74b70ee5934373a2a5fa72d512cd8390`; it records `dirty=false`, result
  SHA-256 `a228b0a75b50c99ffa661d37640c5d8bbdf211f7aae6c79942fd9f698dbf5e74`,
  and source-bundle SHA-256
  `fe3a273cbd28ae1b3354ff8ea7b02402d0df891b9640dffe185cd76b4cc88d94`.
- E004 pins public, ungated repository `nvidia/Qwen3-8B-NVFP4` at immutable
  revision `ccd10a893cbca613259517c3efe08e151ddf2b8e`.
- `config.json` and `hf_quant_config.json` agree on NVFP4, static 4-bit float
  weights and input activations, group size 16, excluded `lm_head`, FP8 KV
  cache, and ModelOpt 0.35.0.
- The model config declares Qwen3ForCausalLM with 36 layers, hidden size 4096,
  intermediate size 12288, 32 attention heads, 8 key/value heads, and BF16 as
  the unquantized model dtype field.
- The safetensors index contains 1,227 names over two shards. Each of `q_proj`,
  `o_proj`, `gate_proj`, `up_proj`, and `down_proj` has `weight`, `input_scale`,
  `weight_scale`, and `weight_scale_2` entries in all 36 layers.
- The two remote weight shards total 6,397,066,384 bytes and expose immutable
  LFS SHA-256 values. Only 112,965 bytes of four metadata files were downloaded
  to ignored artifact storage; no weight shard was downloaded.
- The E004 metadata manifest was regenerated from clean commit
  `68cb4e0328598dd33d3626b0046e704c97ac39b5`; it records `dirty=false`, result
  SHA-256 `3d87bff0ff6147dd022398e465968ebc1ae06a8fb868efcd2059e8dc4f547e81`,
  and source-bundle SHA-256
  `a8dd847464f82efcb1f6213b09816f084573aaedb7b3f0032dfff574bdc5f0b6`.
- Four bounded HTTP range requests retrieved only the 8-byte prefix and declared
  JSON header for each pinned weight shard. Both URLs returned exact HTTP 206
  ranges and lengths; 134,032 bytes were downloaded and payload bytes remained
  zero.
- The headers describe 1,181 and 46 tensors respectively. Their combined 1,227
  names match the pinned index exactly, and all dtype/shape-sized intervals are
  contiguous and cover each declared payload boundary.
- The five capture-target families account for 720 tensors over 36 layers.
  Packed weights are stored as U8, block scales as F8_E4M3, and input/global
  scales as scalar F32. Stored weight shapes are `(4096, 2048)` for `q_proj`
  and `o_proj`, `(12288, 2048)` for `gate_proj` and `up_proj`, and
  `(4096, 6144)` for `down_proj`.
- The E004 header manifest was regenerated from clean commit
  `ee716253c14f0f6b1a8364d2f92253ec01cbe571`; it records `dirty=false`, result
  SHA-256 `8d6b32d2bc36b8f3477b88e11b5bc8542fcabc70778a12bdbe3f1b8fd5fd64d9`,
  and source-bundle SHA-256
  `c5789594fe59ae05bb22efdd10836cb37143fdde6a64d9ea875bbe82b11c8049`.
- E004 fixes layers 0, 18, and 35 as early, middle, and late representatives
  and covers `q_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`.
- The metadata-only planner resolved all four quantization tensors for each
  selected projection to 60 unique, non-overlapping, shard-local HTTP ranges.
- Each selected layer accounts for 103,809,064 planned bytes. The complete
  plan accounts for 311,427,192 bytes across 40 first-shard and 20 second-shard
  ranges.
- Plan construction executed four header range requests totaling 134,032 bytes
  and no payload requests; payload bytes downloaded remained zero.
- The acquisition-plan manifest was regenerated from clean evidence commit
  `76ef175ad885b741068dd21db08149b0902840bd`; it records `dirty=false`, result
  SHA-256 `c69847d3a8ef8b886bd4904b49e34bcc97cc8e80b6657a8d6d37859068d902c2`,
  and source-bundle SHA-256
  `edc9a1efce1c5fa8cec2cdf82b1aecd08272be98f9310485eb85b0abe7ce78d8`.
- All 60 authorized tensor ranges returned HTTP 206 with the exact planned
  `Content-Range` and `Content-Length` values before their bodies were accepted.
- The acquisition wrote 311,427,192 bytes into 60 ignored project-local files.
  Neither complete safetensors shard was downloaded.
- Every local tensor length matched its planned interval, and an independent
  verification pass recomputed all 60 recorded SHA-256 values without a
  mismatch.
- The payload manifest was regenerated from clean evidence commit
  `60c05df86936ca958088d4685b25001f77ad0da0`; it records `dirty=false`, result
  SHA-256 `6ae6088720f14d8a681896a524a638e94b9732c3b1d4d751769548a4d5feb062`,
  and source-bundle SHA-256
  `8093fc825d4d2eb9e98c8bc891234738481fa50db36cf0eda0c141a8fcece4dd`.
- The strict payload loader reads exactly four tensors for a selected projection,
  validates recorded length, dtype, shape, and SHA-256, parses scalar F32 values
  explicitly as little-endian, and rejects negative or non-finite E4M3 scale
  codes. It does not silently cast, transpose, pad, or infer layout from shape.
- ModelOpt checkpoint weights are row-major U8 with two low-nibble-first E2M1
  values per byte. Their block scales are stored in linear F8_E4M3 order and
  require an explicit CUTLASS 128x4 byte permutation. The stored scale shape can
  remain unchanged through that permutation, so shape alone cannot identify it.
- The bounded layer-0 `o_proj` replay preserved the packed weight SHA-256,
  required zero weight padding, and produced independent and vLLM scale-swizzle
  SHA-256
  `4e6992cbfa93bd7136816762fbf212861a103944f1d6054cd6d13eac15347be2`.
- vLLM selected `FlashInferCutlassNvFp4LinearKernel` on the RTX 5080. Three
  synchronized layer-0 `o_proj` replays produced the same finite BF16
  `(16, 4096)` output SHA-256
  `ae14a5d1cf6c304e3c57cbeee2f024fbc7a26963e7b53d59d34963a77b4e3faf`.
- Nsight Systems attributed the expected SM120 block-scaled CUTLASS E2M1/UE4M3
  kernel to the exact E004 target range. No known fallback signature occurred
  in that range; the retained report SHA-256 is
  `faadecc958a5a0b2730a90e6325f650bf4e761b79ea377079699e9f2edf3702d`.
- All 15 early/middle/late projection cases completed three synchronized
  replays with finite, hash-stable outputs. Every packed weight remained
  byte-identical, every weight padding count was zero, and every scale swizzle
  matched vLLM byte-for-byte.
- Six `o_proj` and `down_proj` cases are production-aligned unfused replays.
  Nine `q_proj`, `gate_proj`, and `up_proj` cases remain individual fused-family
  kernel preflights and are not represented as exact model-layer replay.
- The complete pinned repository contains 6,413,063,143 bytes, including
  6,397,066,384 weight-shard bytes and 15,996,759 bytes of tokenizer and small
  files. All 15 pinned files were acquired under ignored `models/` storage and
  passed the Hub checksum inventory.
- One fixed hashed token-ID request produced a real BF16 layer-0 `o_proj`
  prefill input at shape `(9, 4096)` with SHA-256
  `c4c16cadca3b8981e8bdfdd7bd20b2b7b6c6e2be7b34ffa9e98cd6f23890893c`.
- Three synchronized replays were finite, hash-stable, and byte-exact to the
  captured module output with SHA-256
  `da6b9fd682b8fd312ca95379f9993ca4fd1dec4a1f38ca3b1629c87f3b0abf2f`.
- The real-activation range contained the expected SM120 block-scaled CUTLASS
  E2M1/UE4M3 signature and no known fallback. The retained ignored Nsight report
  has SHA-256
  `394f15f9e944713a01cd04ff7e200df59e411b77bb9f0decd90e6af1f35dc38a`.

These observations complete Gate 1 and E003, and support E004 metadata, header,
acquisition, strict layout transformation, deterministic real-weight replay,
complete-snapshot integrity, and one profiler-backed real-activation replay.
They do not establish representative real-activation coverage, arbitrary-input
rounding, NVFP4 GEMM numerical correctness or performance, production model
quality, or model-level fault propagation.

## Last verification

```bash
source /home/meyowu/projects/nvfp4-doctor/activate-nvfp4-lab.sh
cd /mnt/c/Users/meyow/Documents/Codex/2026-08-19/referenced-chatgpt-conversation-this-is-an/nvfp4-doctor
bash -n activate-nvfp4-lab.sh scripts/run_e001_week1.sh scripts/run_e004_projection_profile.sh scripts/run_e004_real_activation_profile.sh
/home/meyowu/projects/nvfp4-doctor/.venv/bin/ruff format --check src tests scripts smoke_nvfp4.py
/home/meyowu/projects/nvfp4-doctor/.venv/bin/ruff check src tests scripts smoke_nvfp4.py
PYTHONPATH=src /home/meyowu/projects/nvfp4-doctor/.venv/bin/mypy src
PYTHONPATH=src /home/meyowu/projects/nvfp4-doctor/.venv/bin/python -m pytest -q
/home/meyowu/projects/nvfp4-doctor/.venv/bin/python -m compileall -q src tests scripts
uv pip check --python /home/meyowu/projects/nvfp4-doctor/.venv/bin/python
PYTHONPATH=src /home/meyowu/projects/nvfp4-doctor/.venv/bin/python scripts/run_e002_gate1.py
PYTHONPATH=src /home/meyowu/projects/nvfp4-doctor/.venv/bin/python scripts/run_e003_format_faults.py
PYTHONPATH=src /home/meyowu/projects/nvfp4-doctor/.venv/bin/python scripts/run_e003_execution_faults.py
PYTHONPATH=src /home/meyowu/projects/nvfp4-doctor/.venv/bin/python scripts/run_e003_heldout_permutations.py
PYTHONPATH=src /home/meyowu/projects/nvfp4-doctor/.venv/bin/python scripts/run_e004_checkpoint_metadata.py
PYTHONPATH=src /home/meyowu/projects/nvfp4-doctor/.venv/bin/python scripts/run_e004_safetensors_headers.py
PYTHONPATH=src /home/meyowu/projects/nvfp4-doctor/.venv/bin/python scripts/run_e004_acquisition_plan.py
PYTHONPATH=src /home/meyowu/projects/nvfp4-doctor/.venv/bin/python scripts/run_e004_tensor_acquisition.py
PYTHONPATH=src /home/meyowu/projects/nvfp4-doctor/.venv/bin/python scripts/run_e004_projection_replay.py
PYTHONPATH=src /home/meyowu/projects/nvfp4-doctor/.venv/bin/python scripts/run_e004_replay_matrix.py
PYTHONPATH=src /home/meyowu/projects/nvfp4-doctor/.venv/bin/python scripts/run_e004_full_model_acquisition.py
bash scripts/run_e004_real_activation_profile.sh
```

Observed result: 163 tests plus 419 subtests passed and Python compilation
completed. Ruff formatting and lint checks passed, Mypy reported no issues in
31 source files, Ruff reported 90 Python files formatted, and `uv pip check`
found all 207 installed packages compatible.
The two E003 CPU runners reported 24 clean contract passes, eleven of eleven
faults detected, exact localization, zero false accepts or rejects, zero
reversibility failures, slice status `pass`, and decision `continue`.
The held-out runner added 18 clean contract passes and detected all nine packed
permutation faults with the same zero-failure metrics; it reported E003 status
`complete` and decision `continue`.
The E004 metadata runner resolved the immutable revision, downloaded 112,965
metadata bytes, validated all five target projection inventories, downloaded no
weight files, and reported status `pass` and decision `continue`.
The header runner validated four exact partial-content responses, full index
alignment, exact payload boundaries, and uniform stored shapes/dtypes for 720
capture-target tensors while downloading zero payload bytes. It reported status
`pass` and decision `continue`.
The acquisition-plan runner selected layers 0, 18, and 35, produced 60 exact
in-bounds ranges totaling 311,427,192 planned bytes, and downloaded zero payload
bytes. It reported status `pass` and decision `continue`.
The tensor-acquisition runner completed all 60 planned requests, wrote
311,427,192 bytes, recorded a SHA-256 for each ignored local artifact, and
reported status `pass` and decision `continue`. A separate local verification
recomputed all lengths and hashes successfully.
The single replay selected FlashInfer CUTLASS, preserved all recorded source
metadata, matched the independent scale transform, produced three identical
finite outputs, and retained range-scoped Nsight kernel evidence. The matrix
runner then passed all 15 cases and 45 measured repetitions without a weight
mutation, padding requirement, scale-transform mismatch, non-finite output, or
within-case hash change.
The full-snapshot verifier passed all 15 pinned files and 6,413,063,143 bytes.
The real-activation workflow then loaded the complete model, captured one BF16
layer-0 `o_proj` prefill input, preserved its declared transfer metadata, and
produced three finite, hash-stable, byte-exact same-module replays. Nsight found
the expected SM120 block-scaled CUTLASS signature and no known fallback in the
exact target range; the normalized result reports status `pass` and decision
`continue`.

## Next action

Extend the real-activation capture into a bounded representative matrix. Start
with unfused `o_proj` and `down_proj` at early, middle, and late layers, then
design fused `qkv_proj` and `gate_up_proj` cases that preserve production module
boundaries. Keep prompt identities hashed, raw activations ignored, and each
case separately profiled or tied to an explicit backend-identity anchor.

## Blockers and limitations

- The fallback classifier recognizes only documented E001 signatures; unknown
  implementations intentionally remain `unknown`.
- E002 validates constructed, exactly representable quantization inputs; it does
  not yet characterize arbitrary-input rounding or saturation behavior in the
  candidate implementation.
- E002 validates linear and CUTLASS 128x4 scale layouts only. Other layouts,
  shuffles, per-token scaling, and two-dimensional training recipes remain out
  of scope.
- NVFP4 GEMM correctness, throughput, accuracy on real model weights and
  activations, and end-to-end model quality remain future-gate questions.
- The first E003 slice covers six deterministic CPU format controls only; the
  later execution-evidence and held-out permutation slices extend that matrix
  without turning it into a distribution of naturally occurring faults.
- The second E003 slice covers metadata and backend-identity fields using
  synthetic evidence. It does not establish actual runtime storage or dispatch;
  profiler-backed replay remains a later experiment boundary.
- The held-out E003 permutations are cyclic and axis-aligned; arbitrary partial
  permutations and compound faults remain outside the completion boundary.
- E004 now validates config, index, and stored header shapes/dtypes/offsets.
  Stored U8 shapes describe packed bytes; they do not establish logical NVFP4
  shapes, runtime strides, framework views, or scale-layout semantics.
- Upstream provides whole-shard LFS hashes rather than per-tensor hashes. The
  acquired ranges therefore use local SHA-256 values for repeatability while
  retaining the immutable shard hashes as source identity.
- The model card documents TensorRT-LLM on B200. The observed RTX 5080 result
  establishes only the pinned vLLM/FlashInfer environment and tested cases, not
  general support.
- The first real-activation result covers one fixed request and layer-0
  `o_proj`. It does not cover prompt diversity, other layers, fused module
  families, final logits, model quality, or a numerical reference comparison.
- The E002 manifest pins the clean implementation commit rather than the later
  evidence-only handoff commit. Regenerate it after any semantic source edit;
  the source-bundle hash detects such changes.
- The migration backup and pre-merge WSL stash remain local recovery artifacts.

## Working-tree expectation

Keep verified E004 real-activation work on the focused
`exp/e004-real-activation` branch.
Use the configured research closeout
workflow for commit, push, PR creation, and merge. Acquiring additional model
payloads outside the pinned snapshot, deleting branches, or publishing external
results still requires separate explicit authorization.
