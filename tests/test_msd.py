import importlib
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import MDAnalysis as mda
import numpy as np
from openmm import unit

from dropps.share.pbc import find_nearest

msd_module = importlib.import_module("dropps.analysis.msd")


class MSDRegressionTests(unittest.TestCase):
    def test_strided_msd_unwraps_intermediate_saved_frames(self):
        # A bead moves +4 nm per saved frame in a 10 nm box. Sampling every
        # second frame before unwrapping would mistake +8 nm for -2 nm.
        universe = mda.Universe.empty(1, trajectory=True)
        positions = np.zeros((7, 1, 3), dtype=np.float32)
        positions[:, 0, 0] = (np.arange(7) * 40) % 100
        universe.load_new(
            positions,
            dimensions=np.tile([100, 100, 100, 90, 90, 90], (7, 1)),
            dt=1000.0,
        )
        fake = SimpleNamespace(
            Universe=universe,
            time2frame=lambda *_: (2, 6, 2),
            getSelection=lambda *_: (universe.atoms, "System"),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "msd.xvg"
            args = msd_module.getargs_msd([
                "-s", "unused.tpr", "-f", "unused.xtc", "-o", str(output),
                "--selection", "0", "-b", "2", "-e", "6", "-dt", "2", "-t", "x",
            ])
            with patch.object(msd_module, "trajectory_class", return_value=fake):
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    msd_module.msd(args)
            data = np.loadtxt(output, comments=("#", "@"))
            np.testing.assert_allclose(data[:, 0], [0, 2, 4])
            np.testing.assert_allclose(data[:, 1], [0, 64, 256], atol=1e-5)

    def test_spatial_unwrapping_handles_more_than_one_box_crossing(self):
        result = find_nearest([24.0, 0, 0], [8.0, 0, 0], [10, 10, 10] * unit.nanometer)
        np.testing.assert_allclose(result, [28.0, 0, 0])


if __name__ == "__main__":
    unittest.main()
