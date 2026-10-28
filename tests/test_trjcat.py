import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from MDAnalysis.lib.formats.libmdaxdr import XTCFile

from dropps.commands.trjcat import trjcat


def _write_xtc(path, atom_count, frames, precision=1000.0):
    box = np.diag([4.0, 4.0, 4.0]).astype(np.float32)
    with XTCFile(str(path), "w") as writer:
        for step, time_ps, coordinate in frames:
            positions = np.full((atom_count, 3), coordinate, dtype=np.float32)
            writer.write(positions, box, step, time_ps, precision)


def _args(inputs, output, **overrides):
    values = {
        "input": [str(path) for path in inputs],
        "output": str(output),
        "start_time": None,
        "end_time": None,
        "delta_time": None,
        "time_unit": "ps",
        "sort": True,
        "keep_duplicates": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TrjcatTests(unittest.TestCase):
    def test_mixed_compressed_precisions_preserve_the_finest_input(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            first, second, output = [directory / name for name in ("a.xtc", "b.xtc", "out.xtc")]
            _write_xtc(first, 12, [(0, 0.0, 0.123)], precision=1000.0)
            _write_xtc(second, 12, [(1, 1.0, 0.1234)], precision=10000.0)
            trjcat(_args([first, second], output))
            with XTCFile(str(output), "r") as reader:
                frames = list(reader)
            self.assertEqual([frame.prec for frame in frames], [10000.0, 10000.0])
            np.testing.assert_allclose(frames[1].x, 0.1234, atol=1.0e-5)

    def test_sorts_segments_and_later_file_replaces_duplicate_time(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            first = directory / "first.xtc"
            second = directory / "second.xtc"
            output = directory / "complete.xtc"
            _write_xtc(first, 2, [(0, 0.0, 0), (1, 1.0, 1), (2, 2.0, 2)])
            _write_xtc(second, 2, [(2, 2.0, 20), (3, 3.0, 3), (4, 4.0, 4)])

            # Deliberately reverse the command-line order; chronological sorting
            # still gives the later-time segment priority at its boundary.
            trjcat(_args([second, first], output))

            with XTCFile(str(output), "r") as reader:
                frames = list(reader)
            self.assertEqual([frame.step for frame in frames], [0, 1, 2, 3, 4])
            np.testing.assert_allclose(
                [frame.time for frame in frames], [0.0, 1.0, 2.0, 3.0, 4.0]
            )
            np.testing.assert_allclose(frames[2].x, 20.0, atol=1.0e-3)

    def test_time_window_and_interval_are_applied_after_merge(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            source = directory / "source.xtc"
            output = directory / "selected.xtc"
            _write_xtc(
                source,
                1,
                [(step, float(step), step) for step in range(6)],
            )

            trjcat(
                _args(
                    [source],
                    output,
                    start_time=1.0,
                    end_time=5.0,
                    delta_time=2.0,
                )
            )

            with XTCFile(str(output), "r") as reader:
                self.assertEqual([frame.step for frame in reader], [1, 3, 5])

    def test_rejects_different_atom_counts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            first = directory / "first.xtc"
            second = directory / "second.xtc"
            _write_xtc(first, 1, [(0, 0.0, 0)])
            _write_xtc(second, 2, [(1, 1.0, 1)])

            with self.assertRaisesRegex(ValueError, "different atom counts"):
                trjcat(_args([first, second], directory / "output.xtc"))


if __name__ == "__main__":
    unittest.main()
