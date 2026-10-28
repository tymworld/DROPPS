import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from dropps.commands.eneconv import eneconv
from dropps.share.energy_reporter import EDRWriter, ENERGY_FIELDS, read_edr


def _write_edr(path, rows, *, metadata=None, fields=("temperature", "volume")):
    writer = EDRWriter(path, fields, ENERGY_FIELDS, metadata=metadata)
    for step, time_ps, temperature, volume in rows:
        values = {"temperature": temperature, "volume": volume}
        writer.write(step, time_ps, {field: values[field] for field in fields})
    writer.close()


def _args(inputs, output, **overrides):
    values = {
        "input": [str(path) for path in inputs],
        "output": str(output),
        "start_time": None,
        "end_time": None,
        "delta_time": None,
        "time_unit": "ps",
        "allow_incompatible": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class EneconvTests(unittest.TestCase):
    def test_merges_edr_and_later_file_replaces_duplicate_step(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            first = directory / "first.edr"
            second = directory / "second.edr"
            output = directory / "complete.edr"
            metadata = {"production_temperature_K": 300.0}
            _write_edr(
                first,
                [(1, 1.0, 299.0, 100.0), (2, 2.0, 298.0, 101.0)],
                metadata=metadata,
            )
            _write_edr(
                second,
                [(2, 2.0, 302.0, 102.0), (3, 3.0, 301.0, 103.0)],
                metadata=metadata,
            )

            eneconv(_args([first, second], output))

            data = read_edr(output)
            np.testing.assert_array_equal(data.steps, [1, 2, 3])
            np.testing.assert_allclose(data.times_ps, [1.0, 2.0, 3.0])
            np.testing.assert_allclose(
                data.values["temperature"], [299.0, 302.0, 301.0]
            )
            self.assertEqual(len(data.header.metadata["eneconv_sources"]), 2)

    def test_time_selection_is_applied_to_merged_samples(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            source = directory / "source.edr"
            output = directory / "selected.edr"
            _write_edr(
                source,
                [(step, float(step), 300.0 + step, 100.0) for step in range(1, 7)],
            )

            eneconv(
                _args(
                    [source],
                    output,
                    start_time=2.0,
                    end_time=6.0,
                    delta_time=2.0,
                )
            )

            data = read_edr(output)
            np.testing.assert_array_equal(data.steps, [2, 4, 6])

    def test_rejects_scientific_metadata_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            first = directory / "first.edr"
            second = directory / "second.edr"
            _write_edr(
                first,
                [(1, 1.0, 300.0, 100.0)],
                metadata={"production_temperature_K": 300.0},
            )
            _write_edr(
                second,
                [(2, 2.0, 310.0, 100.0)],
                metadata={"production_temperature_K": 310.0},
            )

            with self.assertRaisesRegex(ValueError, "scientific metadata"):
                eneconv(_args([first, second], directory / "output.edr"))


if __name__ == "__main__":
    unittest.main()
