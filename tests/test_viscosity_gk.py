import tempfile
import unittest
from pathlib import Path

import numpy as np

from dropps.analysis.viscosity_gk import (
    analyze_green_kubo,
    getargs_viscosity_gk,
    viscosity_gk,
)
from dropps.analysis.viscosity_gk_core import (
    autocorrelation_fft,
    cumulative_trapezoid,
)


class ViscosityCoreTests(unittest.TestCase):
    def test_fft_autocorrelation_matches_direct_unbiased_result(self):
        values = np.asarray([1.0, 4.0, -2.0, 3.0, 0.5])
        centered = values - np.mean(values)
        expected = np.asarray(
            [
                np.dot(centered[: len(values) - lag], centered[lag:])
                / (len(values) - lag)
                for lag in range(len(values))
            ]
        )
        np.testing.assert_allclose(
            autocorrelation_fft(values), expected, rtol=1.0e-13, atol=1.0e-13
        )

    def test_cumulative_trapezoid_integrates_linear_series(self):
        values = np.asarray([0.0, 1.0, 2.0, 3.0])
        np.testing.assert_allclose(
            cumulative_trapezoid(values, 0.5), [0.0, 0.25, 1.0, 2.25]
        )

    def test_green_kubo_analysis_returns_block_uncertainty(self):
        rng = np.random.default_rng(1215)
        sample_count = 400
        times = np.arange(sample_count, dtype=float) * 0.2
        pressure = np.zeros((sample_count, 6), dtype=float)
        innovations = rng.normal(size=(sample_count, 3))
        for index in range(1, sample_count):
            pressure[index, 3:6] = 0.9 * pressure[index - 1, 3:6] + innovations[index]
        data = np.column_stack(
            (
                np.arange(sample_count),
                times,
                np.full(sample_count, 125.0),
                pressure,
            )
        )
        result = analyze_green_kubo(data, 300.0, blocks=5)
        self.assertEqual(len(result["block_etas_pa_s"]), 5)
        self.assertTrue(np.isfinite(result["eta_pa_s"]))
        self.assertTrue(np.isfinite(result["sem_pa_s"]))
        self.assertEqual(result["lag_times_ps"].shape, result["viscosity_pa_s"].shape)


class ViscosityCommandTests(unittest.TestCase):
    def test_command_reads_reporter_file_and_writes_curve(self):
        rng = np.random.default_rng(7)
        sample_count = 200
        pressure = rng.normal(size=(sample_count, 6))
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            input_path = directory / "run.stress.xvg"
            output_path = directory / "viscosity.xvg"
            with input_path.open("w", encoding="utf-8") as handle:
                handle.write(
                    "# pressure_unit=kJ_mol^-1_nm^-3 temperature_K=310 strain=0.0003\n"
                )
                handle.write(
                    "# columns: step time_ps volume_nm3 Pxx Pyy Pzz Pxy Pxz Pyz\n"
                )
                for index, row in enumerate(pressure):
                    values = " ".join(str(value) for value in row)
                    handle.write(f"{index} {index * 0.1} 100 {values}\n")

            args = getargs_viscosity_gk(
                ["-f", str(input_path), "-o", str(output_path), "--blocks", "4"]
            )
            viscosity_gk(args)
            output = output_path.read_text(encoding="utf-8")
            self.assertIn("eta0_Pa_s=", output)
            self.assertIn("block_sem_Pa_s=", output)
            data_lines = [
                line
                for line in output.splitlines()
                if line and not line.startswith(("#", "@"))
            ]
            self.assertGreater(len(data_lines), 2)
            self.assertTrue(all(len(line.split()) == 4 for line in data_lines))


if __name__ == "__main__":
    unittest.main()
