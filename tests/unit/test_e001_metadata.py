import unittest

from nvfp4_doctor.env import e001_smoke_tensors


class E001MetadataTests(unittest.TestCase):
    def test_packed_values_preserve_logical_and_physical_shapes(self) -> None:
        tensors = {tensor.name: tensor for tensor in e001_smoke_tensors()}
        self.assertEqual(tensors["a_fp4"].logical_shape, (16, 256))
        self.assertEqual(tensors["a_fp4"].physical_shape, (16, 128))
        self.assertEqual(tensors["b_fp4"].logical_shape, (128, 256))
        self.assertEqual(tensors["b_fp4"].physical_shape, (128, 128))

    def test_scale_padding_is_explicit(self) -> None:
        tensors = {tensor.name: tensor for tensor in e001_smoke_tensors()}
        self.assertEqual(tensors["a_scale"].logical_shape, (16, 16))
        self.assertEqual(tensors["a_scale"].physical_shape, (128, 16))
        self.assertEqual(tensors["a_scale"].stride, (16, 1))

    def test_output_metadata_matches_smoke_contract(self) -> None:
        tensors = {tensor.name: tensor for tensor in e001_smoke_tensors()}
        self.assertEqual(tensors["output"].logical_shape, (16, 128))
        self.assertEqual(tensors["output"].dtype, "bfloat16")


if __name__ == "__main__":
    unittest.main()
