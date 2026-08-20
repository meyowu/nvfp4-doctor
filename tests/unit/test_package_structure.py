import importlib
import unittest


class PackageStructureTests(unittest.TestCase):
    def test_e001_boundaries_import_without_gpu_dependencies(self) -> None:
        modules = (
            "nvfp4_doctor",
            "nvfp4_doctor.backends",
            "nvfp4_doctor.capture",
            "nvfp4_doctor.checkpoint",
            "nvfp4_doctor.contracts",
            "nvfp4_doctor.env",
            "nvfp4_doctor.faults",
            "nvfp4_doctor.formats",
            "nvfp4_doctor.minimize",
            "nvfp4_doctor.oracle",
            "nvfp4_doctor.report",
        )

        for module in modules:
            with self.subTest(module=module):
                importlib.import_module(module)


if __name__ == "__main__":
    unittest.main()
