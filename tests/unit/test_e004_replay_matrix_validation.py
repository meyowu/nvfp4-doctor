import unittest

from scripts.run_e004_replay_matrix import ReplayMatrixError, _validate_case


def _run() -> dict[str, object]:
    return {
        "status": "pass",
        "decision": "pending_profiler",
        "case": {"layer": 18, "projection": "down_proj", "seed": 0},
        "replay": {
            "repetitions": 3,
            "all_finite": True,
            "output_hash_stable": True,
            "output_sha256s": ["a" * 64] * 3,
        },
        "backend": {
            "requested_backend": "cutlass",
            "selected_vllm_kernel": "FlashInferCutlassNvFp4LinearKernel",
            "reported_backend": None,
        },
        "transforms": [
            {
                "name": "packed_weight_materialization",
                "source_sha256": "b" * 64,
                "destination_sha256": "b" * 64,
                "padding_bytes": 0,
            },
            {
                "name": "weight_scale_swizzle",
                "vllm_candidate_byte_exact": True,
            },
        ],
    }


class E004ReplayMatrixValidationTests(unittest.TestCase):
    def test_accepts_a_complete_case(self) -> None:
        _validate_case(_run(), layer=18, projection="down_proj")

    def test_rejects_weight_mutation(self) -> None:
        run = _run()
        run["transforms"][0]["destination_sha256"] = "c" * 64
        with self.assertRaisesRegex(ReplayMatrixError, "weight transform"):
            _validate_case(run, layer=18, projection="down_proj")

    def test_rejects_reported_backend_inference(self) -> None:
        run = _run()
        run["backend"]["reported_backend"] = "cutlass"
        with self.assertRaisesRegex(ReplayMatrixError, "backend"):
            _validate_case(run, layer=18, projection="down_proj")


if __name__ == "__main__":
    unittest.main()
