import tempfile
import unittest
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from dropps.analysis.energy import energy, interactive_select, resolve_terms
from dropps.share.energy_reporter import (
    EDRWriter,
    ENERGY_FIELDS,
    read_edr,
)


class EDRIOTests(unittest.TestCase):
    def test_writer_reader_and_append_preserve_schema(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "run.edr"
            fields = ("temperature", "boxX")
            writer = EDRWriter(
                path,
                fields,
                ENERGY_FIELDS,
                metadata={"production_temperature_K": 300.0},
            )
            writer.write(10, 0.02, {"temperature": 301.5, "boxX": 12.0})
            writer.close()

            writer = EDRWriter(path, fields, ENERGY_FIELDS, append=True)
            writer.write(20, 0.04, {"temperature": 299.5, "boxX": 12.1})
            writer.close()

            data = read_edr(path)
            self.assertEqual(data.header.fields, fields)
            self.assertEqual(data.header.units["temperature"], "K")
            np.testing.assert_array_equal(data.steps, [10, 20])
            np.testing.assert_allclose(data.times_ps, [0.02, 0.04])
            np.testing.assert_allclose(data.values["temperature"], [301.5, 299.5])

    def test_append_rejects_schema_change(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "run.edr"
            writer = EDRWriter(path, ("temperature",), ENERGY_FIELDS)
            writer.close()
            with self.assertRaisesRegex(ValueError, "different field schema"):
                EDRWriter(path, ("volume",), ENERGY_FIELDS, append=True)

    def test_non_dropps_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "foreign.edr"
            path.write_text("step,time_ps,temperature\n1,1,300\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing format marker"):
                read_edr(path)


class EnergySelectionTests(unittest.TestCase):
    fields = ("potentialEnergy", "temperature", "totalEnergy", "volume")

    def test_resolve_numbers_names_prefixes_and_all(self):
        self.assertEqual(
            resolve_terms(["2", "pot", "volume", "2"], self.fields),
            ["temperature", "potentialEnergy", "volume"],
        )
        self.assertEqual(resolve_terms(["all"], self.fields), list(self.fields))

    def test_ambiguous_prefix_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            resolve_terms(["t"], self.fields)

    def test_interactive_selection_ends_with_zero(self):
        answers = iter(["1 temp", "4 0"])
        output = StringIO()
        header = SimpleNamespace(
            fields=self.fields,
            units={name: "unit" for name in self.fields},
        )
        selected = interactive_select(
            header, input_func=lambda _prompt: next(answers), stream=output
        )
        self.assertEqual(selected, ["potentialEnergy", "temperature", "volume"])
        self.assertIn("Available energy terms", output.getvalue())

    def test_energy_command_exports_selected_xvg(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            source = directory / "run.edr"
            output = directory / "temperature"
            writer = EDRWriter(source, ("temperature", "volume"), ENERGY_FIELDS)
            writer.write(1, 1.0, {"temperature": 299.0, "volume": 100.0})
            writer.write(2, 2.0, {"temperature": 301.0, "volume": 101.0})
            writer.close()

            energy(
                SimpleNamespace(
                    input=str(source),
                    output=str(output),
                    list=False,
                    terms=["temperature"],
                    start_time=None,
                    end_time=None,
                    delta_time=None,
                    x_axis="time",
                )
            )
            contents = Path(f"{output}.xvg").read_text(encoding="utf-8")
            self.assertIn("# columns: time_ps temperature", contents)
            self.assertIn('@ s0 legend "temperature (K)"', contents)
            self.assertIn("2\t301", contents)


if __name__ == "__main__":
    unittest.main()
