# RTX 5080 Research Setup on Windows 11 and WSL2

Verified on 2026-08-19. Windows is the host; CUDA compilation, vLLM,
FlashInfer, tests, and experiments run inside Ubuntu on WSL2.

## 1. Install the Windows driver and WSL2

Install a current NVIDIA Windows driver for the RTX 5080, reboot, and verify in
PowerShell:

```powershell
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
wsl --update
wsl --install -d Ubuntu-24.04
wsl --status
wsl --list --verbose
```

Ubuntu must show WSL version 2. Do not install a Linux NVIDIA display driver
inside WSL; GPU access is projected from the Windows driver.

## 2. Prepare Ubuntu

Run inside Ubuntu:

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y build-essential git curl ca-certificates pkg-config ninja-build
nvidia-smi
```

Keep repositories under the Linux filesystem, for example `~/projects`, rather
than under `/mnt/c`. Use one subdirectory per project.

## 3. Install uv and Python 3.12

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/meyowu/nvfp4-doctor.git
cd nvfp4-doctor
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
```

## 4. Install the verified Python stack

Install vLLM first so its PyTorch selection is coherent, then apply the
compiler-side pins used by the verified environment:

```bash
uv pip install vllm==0.27.1 --torch-backend=auto
uv pip install -r requirements-research.txt
```

Verify the device:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
x = torch.randn(1024, 1024, device="cuda")
print("finite:", torch.isfinite(x @ x).all().item())
PY
```

## 5. CUDA Toolkit and profiling tools

Python wheels provide their own CUDA user-mode runtime. Install a toolkit only
when `nvcc`, binary inspection, sanitizers, or extension compilation is needed.
Use NVIDIA's WSL-Ubuntu repository instructions and install a toolkit-only
package that matches the chosen stack. Never install `cuda-drivers` in WSL.

Check the tools after installation:

```bash
nvcc --version
compute-sanitizer --version
cuobjdump --version
nvdisasm --version
nsys --version
ncu --version
```

The verified machine used CUDA compiler-side packages 13.2.86, Nsight Systems
2026.1.3, and Nsight Compute 2026.2.1. Driver, wheel runtime, and local toolkit
versions are distinct; record all three in experiment manifests.

## 6. Run the first NVFP4 smoke test

```bash
source ./activate-nvfp4-lab.sh
python smoke_nvfp4.py
nsys profile --trace=cuda,nvtx,osrt --output=/tmp/nvfp4-smoke \
  --force-overwrite=true python smoke_nvfp4.py
```

The test is an environment check, not a benchmark or proof of correctness.
Record the actual backend and kernel identity before using later runs as
research evidence.

## 7. Troubleshooting order

1. If `nvidia-smi` fails on Windows, repair or update the Windows driver.
2. If it works on Windows but fails in WSL, update WSL and confirm version 2.
3. If PyTorch reports no CUDA device, confirm that a CUDA wheel was installed.
4. If a kernel image is unavailable, use a build containing `sm_120` cubin or PTX.
5. For undefined symbols, rebuild a clean environment instead of overlaying packages.
6. For JIT out-of-memory failures, source `activate-nvfp4-lab.sh` to limit jobs.
7. If Nsight Compute cannot read counters, enable GPU performance-counter access
   in NVIDIA Control Panel and consult the current Nsight WSL release notes.

## Reproducibility boundary

Do not commit tokens, complete environment dumps, model weights, activations,
JIT caches, or profiler reports. Do record package versions, wheel sources,
hardware and driver identity, random seeds, commands, kernel identity, and
cryptographic hashes for immutable artifacts.
