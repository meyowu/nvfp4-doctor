import hashlib
import os
import unittest
from pathlib import Path

import torch

from scripts.run_e004_real_activation_capture import (
    FROZEN_VLLM_ENVIRONMENT,
    PROMPT_TOKEN_IDS,
    RealActivationCaptureError,
    _bitwise_tensor_match,
    _canonical_token_bytes,
    _copy_to_cpu_preserving_stride,
    _cuda_status_code,
    _linear_output,
    _project_relative,
    _tensor_metadata,
)


class RealActivationCaptureTests(unittest.TestCase):
    def test_wsl_compatible_vllm_runner_is_frozen(self) -> None:
        self.assertEqual(os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"], "0")
        self.assertEqual(os.environ["VLLM_WSL2_ENABLE_PIN_MEMORY"], "1")
        self.assertEqual(os.environ["VLLM_USE_FLASHINFER_SAMPLER"], "0")
        self.assertEqual(FROZEN_VLLM_ENVIRONMENT["VLLM_WSL2_ENABLE_PIN_MEMORY"], "1")

    def test_token_encoding_is_fixed_little_endian_int32(self) -> None:
        payload = _canonical_token_bytes(PROMPT_TOKEN_IDS)
        self.assertEqual(len(payload), 4 * len(PROMPT_TOKEN_IDS))
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            "154c66e5fa3bad6f385105bb54d93b6f2ab1e3bc9e3b1452bffcbfa6fd97413e",
        )

    def test_tensor_metadata_records_stride_and_storage(self) -> None:
        tensor = torch.arange(24, dtype=torch.bfloat16).reshape(4, 6).t()
        metadata = _tensor_metadata(tensor)
        self.assertEqual(metadata["shape"], [6, 4])
        self.assertEqual(metadata["stride"], [1, 6])
        self.assertFalse(metadata["contiguous"])
        self.assertEqual(metadata["byte_length"], 48)
        self.assertEqual(len(metadata["sha256"]), 64)

    def test_bitwise_match_distinguishes_signed_zero(self) -> None:
        positive_zero = torch.tensor([0.0], dtype=torch.bfloat16)
        negative_zero = torch.tensor([-0.0], dtype=torch.bfloat16)
        self.assertTrue(torch.equal(positive_zero, negative_zero))
        self.assertFalse(_bitwise_tensor_match(positive_zero, negative_zero))
        self.assertTrue(_bitwise_tensor_match(positive_zero, positive_zero.clone()))

    def test_cpu_copy_preserves_strided_metadata(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required by the synchronized capture helper")
        source = torch.arange(24, dtype=torch.bfloat16, device="cuda").reshape(4, 6).t()
        copied = _copy_to_cpu_preserving_stride(source)
        self.assertEqual(copied.dtype, source.dtype)
        self.assertEqual(copied.shape, source.shape)
        self.assertEqual(copied.stride(), source.stride())
        self.assertTrue(torch.equal(copied, source.cpu()))

    def test_cpu_copy_rejects_nonzero_storage_offset(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required by the synchronized capture helper")
        source = torch.arange(25, dtype=torch.bfloat16, device="cuda")[1:]
        self.assertEqual(source.storage_offset(), 1)
        with self.assertRaisesRegex(RealActivationCaptureError, "storage offsets"):
            _copy_to_cpu_preserving_stride(source)

    def test_linear_output_accepts_vllm_tuple(self) -> None:
        tensor = torch.zeros((2, 3), dtype=torch.bfloat16)
        self.assertIs(_linear_output((tensor, None)), tensor)
        with self.assertRaisesRegex(RealActivationCaptureError, "unexpected"):
            _linear_output((None, tensor))

    def test_project_paths_and_cuda_status_are_explicit(self) -> None:
        self.assertEqual(
            _project_relative(
                Path("artifacts/E004-qwen3-layer-capture/real-activation"),
                label="artifact",
            ).parts[0],
            "artifacts",
        )
        with self.assertRaisesRegex(RealActivationCaptureError, "repository root"):
            _project_relative(
                Path("/tmp/e004-output.json"),
                label="output",
            )
        self.assertEqual(_cuda_status_code(None), 0)
        self.assertEqual(_cuda_status_code((0,)), 0)


if __name__ == "__main__":
    unittest.main()
