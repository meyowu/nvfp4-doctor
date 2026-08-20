import unittest

from scripts.finalize_e004_projection_replay import (
    ReplayFinalizationError,
    _validate_run,
)


def _run() -> dict[str, object]:
    return {
        "status": "pass",
        "decision": "pending_profiler",
        "case": {
            "layer": 0,
            "projection": "o_proj",
            "activation_provenance": "synthetic_deterministic",
        },
        "replay": {
            "repetitions": 3,
            "all_finite": True,
            "output_hash_stable": True,
            "output_shape": [16, 4096],
            "output_dtype": "bfloat16",
        },
        "backend": {
            "requested_backend": "cutlass",
            "selected_vllm_kernel": "FlashInferCutlassNvFp4LinearKernel",
            "reported_backend": None,
            "target_nvtx_range": "e004:layer_00:o_proj:nvfp4_gemm",
        },
    }


class E004ReplayFinalizationTests(unittest.TestCase):
    def test_accepts_the_frozen_runtime_observation(self) -> None:
        range_name, backend = _validate_run(_run())
        self.assertEqual(range_name, "e004:layer_00:o_proj:nvfp4_gemm")
        self.assertIsNone(backend["reported_backend"])

    def test_rejects_inferred_reported_backend(self) -> None:
        run = _run()
        run["backend"]["reported_backend"] = "cutlass"
        with self.assertRaisesRegex(ReplayFinalizationError, "identity"):
            _validate_run(run)

    def test_rejects_unstable_output(self) -> None:
        run = _run()
        run["replay"]["output_hash_stable"] = False
        with self.assertRaisesRegex(ReplayFinalizationError, "invariants"):
            _validate_run(run)


if __name__ == "__main__":
    unittest.main()
