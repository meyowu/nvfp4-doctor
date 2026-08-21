# nvfp4-doctor

`nvfp4-doctor` is the implementation repository for QContract, an eight-week
research sprint on contract-driven correctness and fault injection for NVFP4
inference kernels.

The project treats low-precision correctness as a systems property. It records
the requested backend, actual kernel identity, tensor and scale metadata,
environment provenance, and oracle comparisons before making performance or
correctness claims.

## Current status

Gate 1, the bounded E003 synthetic fault experiment, and E004 / Gate 2 are
complete. Gate 2 records a `go` decision toward the Gate 3 numerical reference.
The repository contains a CPU-only oracle for E2M1, UE4M3 scales, packed values,
CUTLASS 128x4 scale layout, padding, scalar global scales, and exact hierarchical
reconstruction. Its semantics are pinned to versioned public NVIDIA sources and
hand-authored fixtures rather than the candidate FlashInfer implementation.

Three constructed RTX 5080 differential cases matched FlashInfer's packed and
scale bytes exactly, including boundary padding and multi-atom layouts. This
supports the declared dense row-wise block-16 format contract only. It does not
establish arbitrary quantization rounding, GEMM correctness, model quality, or
performance.

Three E003 slices add exact structural/evidence contracts, six deterministic
format faults, five stride/backend-identity faults, and three packed-value
permutation families. The final held-out matrix applied block, row, and column
permutations to three new clean artifacts. All nine held-out faults were
detected and localized with zero clean false rejects or fault false accepts.
Profiler-backed real dispatch and NVFP4 GEMM execution now exist in E004;
numerical GEMM comparison and model-level fault propagation remain pending.

E004 has started with a metadata-only inspection of the immutable
`nvidia/Qwen3-8B-NVFP4` revision
`ccd10a893cbca613259517c3efe08e151ddf2b8e`. Its two public quantization
declarations agree on NVFP4 group size 16, and all five planned projection
families expose the expected quantization tensor names across 36 layers. Only
112,965 bytes of small metadata were downloaded; no model weight was fetched.

Bounded range requests subsequently read 134,032 bytes covering only the two
safetensors prefixes and JSON headers. All 1,227 tensor names aligned with the
pinned index, and the five capture-target families had uniform stored dtypes and
shapes across 36 layers. Payload bytes downloaded remained zero.

A metadata-only acquisition plan now fixes layers 0, 18, and 35 and maps all
four NVFP4 tensors for the five projection families to 60 exact shard-local
ranges. The planned payload is 311,427,192 bytes; plan construction downloaded
only the 134,032 header bytes and zero payload bytes.

Those 60 authorized ranges have now been acquired into ignored local artifact
storage. All exact partial-content boundaries and lengths matched the plan, and
all 311,427,192 local bytes were reverified against recorded per-tensor SHA-256
values. Neither complete weight shard was downloaded.

A strict loader then replayed acquired layer-0 `o_proj` weights with a
deterministic synthetic activation. Packed weight bytes were preserved, the
independent CUTLASS 128x4 scale transform matched vLLM byte-for-byte, and vLLM
selected its FlashInfer CUTLASS NVFP4 kernel. Three synchronized outputs were
finite and hash-identical; Nsight observed the expected SM120 block-scaled E2M1
kernel in the exact target range with no known fallback signature there.

The same bounded preflight passed all 15 early/middle/late projection cases and
45 measured repetitions. This is real-weight, synthetic-activation execution
evidence.

The complete pinned repository snapshot has now been acquired into ignored
local `models/` storage. All 15 files and 6,413,063,143 bytes passed the pinned
Hugging Face checksum inventory; the two weight shards account for
6,397,066,384 bytes. This establishes local acquisition integrity only.

A first real-activation slice then ran the full Qwen3 checkpoint with one fixed,
hashed token-ID request and captured the layer-0 `o_proj` prefill input. The BF16
activation had shape `(9, 4096)`. Three synchronized standalone replays were
finite, hash-stable, and byte-exact to the captured module output. vLLM selected
`FlashInferCutlassNvFp4LinearKernel`, and Nsight found the expected SM120
block-scaled CUTLASS signature with no known fallback in the exact target range.
This first slice covered one prompt and one unfused projection; by itself it did
not complete Gate 2, and no numerical-correctness or model-quality conclusion
followed from it.

A representative unfused matrix now extends that same hashed request to
`o_proj` and `down_proj` at layers 0, 18, and 35 in one model pass. All six BF16
captures preserved their declared transfer metadata, and all 18 synchronized
replays were finite, hash-stable, and logical-byte-exact to their captured
module outputs. Six separate Nsight ranges each contained activation
quantization and the expected SM120 CUTLASS NVFP4 signature with no known
fallback. This supports representative unfused execution coverage for one
request; it does not establish numerical correctness, final logits, or model
quality.

The Gate 2 completion slice captures the production-fused `qkv_proj` and
`gate_up_proj` boundaries at the same three layers and with the same hashed
request. Their ordered q/k/v and gate/up checkpoint components reconstruct the
runtime packed weights and independently swizzled scales byte-for-byte. All 18
measured whole-module replays were finite, stable, and logical-byte-exact, and
all six exact Nsight ranges contained activation quantization plus the expected
SM120 CUTLASS NVFP4 signature with no known fallback. Combined with the unfused
matrix, this satisfies the bounded Gate 2 replay and backend-identity criterion
with a `go` decision. Numerical correctness, prompt diversity, final logits,
model quality, performance, and high-precision equivalence remain open for later
gates.

See [docs/setup-windows-wsl2.md](docs/setup-windows-wsl2.md) for the reproducible
host setup and [docs/progress.md](docs/progress.md) for the verified baseline.
The model-validation sequence is specified in
[docs/research-plan.md](docs/research-plan.md), and the research boundary is
frozen in [docs/scope.md](docs/scope.md). Research and engineering rules are
defined in [AGENTS.md](AGENTS.md).

To resume a research session, read [research-state.md](research-state.md) and run:

```bash
python scripts/research_status.py
```

## Quick start

Inside Ubuntu on WSL2:

```bash
git clone https://github.com/meyowu/nvfp4-doctor.git
cd nvfp4-doctor
uv venv --python 3.12
source .venv/bin/activate
uv pip install vllm==0.27.1 --torch-backend=auto
uv pip install -r requirements-research.txt
uv pip install -r requirements-dev.txt
source ./activate-nvfp4-lab.sh
python smoke_nvfp4.py
./scripts/run_e001_week1.sh 3
PYTHONPATH=src python scripts/run_e002_gate1.py
PYTHONPATH=src python scripts/run_e003_format_faults.py
PYTHONPATH=src python scripts/run_e003_execution_faults.py
PYTHONPATH=src python scripts/run_e003_heldout_permutations.py
PYTHONPATH=src python scripts/run_e004_checkpoint_metadata.py
PYTHONPATH=src python scripts/run_e004_safetensors_headers.py
PYTHONPATH=src python scripts/run_e004_acquisition_plan.py
PYTHONPATH=src python scripts/run_e004_tensor_acquisition.py
bash scripts/run_e004_projection_profile.sh
PYTHONPATH=src python scripts/run_e004_replay_matrix.py
PYTHONPATH=src python scripts/run_e004_full_model_acquisition.py
bash scripts/run_e004_real_activation_profile.sh
bash scripts/run_e004_real_activation_matrix_profile.sh
bash scripts/run_e004_real_activation_fused_matrix_profile.sh
```

Package pins reflect the verified 2026-08-19 environment. Review the setup
guide before changing CUDA, PyTorch, vLLM, or FlashInfer versions together.

## Repository policy

- Do not commit virtual environments, model weights, profiler traces, caches,
  tokens, or raw activation data.
- Every experiment must have a manifest, immutable artifacts, an oracle, and a
  stated acceptance criterion.
- Separate observed results from hypotheses and planned work.

Licensed under Apache-2.0.
