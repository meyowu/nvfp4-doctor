import importlib
import unittest


class PackageStructureTests(unittest.TestCase):
    def test_e001_boundaries_import_without_gpu_dependencies(self) -> None:
        modules = (
            "nvfp4_doctor",
            "nvfp4_doctor.env",
            "nvfp4_doctor.backends",
            "nvfp4_doctor.report",
        )

        for module in modules:
            with self.subTest(module=module):
                importlib.import_module(module)


if __name__ == "__main__":
    unittest.main()
