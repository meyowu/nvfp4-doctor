import tempfile
import unittest
from pathlib import Path

import httpx

from scripts.run_e004_tensor_acquisition import (
    TensorAcquisitionError,
    _download_range,
    _validate_plan,
)


def _record() -> dict[str, object]:
    return {
        "tensor_name": "model.layers.0.self_attn.q_proj.input_scale",
        "layer": 0,
        "projection": "q_proj",
        "suffix": "input_scale",
        "shard": "model-00001-of-00002.safetensors",
        "source_url": (
            "https://huggingface.co/nvidia/Qwen3-8B-NVFP4/resolve/"
            "ccd10a893cbca613259517c3efe08e151ddf2b8e/"
            "model-00001-of-00002.safetensors"
        ),
        "shard_lfs_sha256": "a" * 64,
        "dtype": "F32",
        "shape": [],
        "file_start": 8,
        "file_end_exclusive": 12,
        "http_range": "bytes=8-11",
        "byte_length": 4,
        "shard_file_size": 100,
    }


class TensorAcquisitionTests(unittest.TestCase):
    def test_download_accepts_only_exact_partial_content(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.headers["range"], "bytes=8-11")
            self.assertEqual(request.headers["accept-encoding"], "identity")
            return httpx.Response(
                206,
                headers={
                    "Content-Range": "bytes 8-11/100",
                    "Content-Length": "4",
                },
                content=b"abcd",
            )

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "value.bin"
            with httpx.Client(transport=httpx.MockTransport(handler)) as client:
                evidence = _download_range(client, _record(), destination)

            self.assertEqual(destination.read_bytes(), b"abcd")
            self.assertEqual(evidence["content_length"], 4)
            self.assertEqual(
                evidence["sha256"],
                "88d4266fd4e6338d13b845fcf289579d209c897823b9217da3e161936f031589",
            )

    def test_download_rejects_full_response_without_writing_artifact(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"abcd")

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "value.bin"
            with httpx.Client(transport=httpx.MockTransport(handler)) as client:
                with self.assertRaisesRegex(TensorAcquisitionError, "refusing body"):
                    _download_range(client, _record(), destination)
            self.assertFalse(destination.exists())

    def test_download_rejects_inexact_content_range(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                206,
                headers={
                    "Content-Range": "bytes 9-12/100",
                    "Content-Length": "4",
                },
                content=b"abcd",
            )

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "value.bin"
            with httpx.Client(transport=httpx.MockTransport(handler)) as client:
                with self.assertRaisesRegex(TensorAcquisitionError, "boundaries"):
                    _download_range(client, _record(), destination)
            self.assertFalse(destination.exists())

    def test_plan_validator_rejects_incomplete_matrix(self) -> None:
        plan = {
            "repository": {
                "id": "nvidia/Qwen3-8B-NVFP4",
                "revision": "ccd10a893cbca613259517c3efe08e151ddf2b8e",
            },
            "request_policy": {"planned_payload_bytes": 311427192},
            "ranges": [_record()],
        }
        with self.assertRaisesRegex(TensorAcquisitionError, "exactly 60"):
            _validate_plan(plan)


if __name__ == "__main__":
    unittest.main()
