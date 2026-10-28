import subprocess
import sys
import unittest
from unittest.mock import patch

from scripts.install_dropps import (
    NvidiaProbe,
    add_extra,
    build_pip_command,
    parse_cuda_version,
    probe_nvidia,
    select_install_extra,
)


class CudaSelectionTests(unittest.TestCase):
    def test_parse_cuda_version(self):
        self.assertEqual(parse_cuda_version("CUDA Version: 12.8"), (12, 8))
        self.assertEqual(parse_cuda_version("CUDA Version: 13"), (13, 0))
        self.assertIsNone(parse_cuda_version("CUDA Version: N/A"))

    def test_auto_selects_cuda_generation_from_driver(self):
        cuda12 = NvidiaProbe(("NVIDIA A100",), (12, 9))
        cuda13 = NvidiaProbe(("NVIDIA B200",), (13, 0))

        self.assertEqual(select_install_extra(cuda12, "auto"), "cuda12")
        self.assertEqual(select_install_extra(cuda13, "auto"), "cuda13")

    def test_auto_uses_base_install_without_visible_gpu(self):
        probe = NvidiaProbe((), None, "nvidia-smi was not found")

        self.assertEqual(select_install_extra(probe, "auto"), "latest")
        with self.assertRaisesRegex(RuntimeError, "No visible NVIDIA GPU"):
            select_install_extra(probe, "auto", require_cuda=True)

    def test_old_driver_is_rejected_when_gpu_is_visible(self):
        probe = NvidiaProbe(("NVIDIA GPU",), (11, 8))

        with self.assertRaisesRegex(RuntimeError, "require CUDA 12 or newer"):
            select_install_extra(probe, "auto")

    def test_explicit_cuda_generation_overrides_probe(self):
        probe = NvidiaProbe((), None, "nvidia-smi was not found")

        self.assertEqual(select_install_extra(probe, "12"), "cuda12")
        self.assertEqual(select_install_extra(probe, "13"), "cuda13")


class InstallerCommandTests(unittest.TestCase):
    def test_extra_is_added_to_source_or_wheel_target(self):
        self.assertEqual(add_extra(".", "cuda12"), ".[cuda12]")
        self.assertEqual(
            add_extra("dist/dropps.whl", "cuda13"),
            "dist/dropps.whl[cuda13]",
        )

    def test_existing_extra_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must not already contain extras"):
            add_extra("dropps[latest]", "cuda12")

    def test_editable_pip_command_uses_current_interpreter(self):
        self.assertEqual(
            build_pip_command(".", "cuda12", editable=True),
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--editable",
                ".[cuda12]",
            ],
        )

    @patch("scripts.install_dropps.shutil.which", return_value="/usr/bin/nvidia-smi")
    @patch("scripts.install_dropps.subprocess.run")
    def test_probe_reports_devices_and_driver_cuda_version(self, run, _which):
        run.side_effect = [
            subprocess.CompletedProcess([], 0, "NVIDIA A100\nNVIDIA A100\n", ""),
            subprocess.CompletedProcess([], 0, "CUDA Version: 12.8\n", ""),
        ]

        probe = probe_nvidia()

        self.assertEqual(probe.devices, ("NVIDIA A100", "NVIDIA A100"))
        self.assertEqual(probe.max_cuda_version, (12, 8))
        self.assertIsNone(probe.error)


if __name__ == "__main__":
    unittest.main()
