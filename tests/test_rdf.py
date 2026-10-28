import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from dropps.analysis.rdf import _radius_prepass, getargs_rdf, rdf
from dropps.share.all_commands import all_commands


class _Frames:
    def __init__(self, frames):
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def __iter__(self):
        return iter(self.frames)

    def __getitem__(self, index):
        return self.frames[index]


class _Index:
    def __init__(self):
        self.index_groups = [
            SimpleNamespace(name="Reference", indices=[0, 2]),
            SimpleNamespace(name="Target", indices=[1, 3]),
        ]

    def print_all(self):
        return None


class _Selection:
    def __init__(self, indices):
        self.indices = np.asarray(indices, dtype=int)

    def __len__(self):
        return len(self.indices)


class _SyntheticTrajectory:
    def __init__(self, *_args, **_kwargs):
        frames = []
        for frame_id in range(2):
            frames.append(
                SimpleNamespace(
                    time=frame_id * 1000.0,
                    positions=np.asarray(
                        [
                            [0.0, 0.0, 0.0],
                            [2.0, 0.0, 0.0],
                            [10.0, 0.0, 0.0],
                            [12.0, 0.0, 0.0],
                        ],
                        dtype=float,
                    ),
                    dimensions=np.asarray(
                        [20.0, 20.0, 20.0, 90.0, 90.0, 90.0],
                        dtype=float,
                    ),
                )
            )
        self.Universe = SimpleNamespace(
            trajectory=_Frames(frames),
            bonds=SimpleNamespace(indices=np.asarray([[0, 1], [2, 3]])),
        )
        self.index = _Index()
        self.id2chainID = [0, 0, 1, 1]
        self.id2resID = [1, 1, 1, 1]
        self.id2masses = [1.0, 1.0, 1.0, 1.0]
        self.tpr = SimpleNamespace(itp_list=[])

    def getSelection(self, specification, _help):
        group_id = int(str(specification).replace("group", "").strip())
        group = self.index.index_groups[group_id]
        return _Selection(group.indices), group.name

    def num_atoms(self):
        return 4


class RdfArgumentTests(unittest.TestCase):
    def test_parser_exposes_gromacs_compatible_controls(self):
        args = getargs_rdf(
            [
                "-s",
                "run.tpr",
                "-f",
                "run.xtc",
                "-ref",
                "1",
                "-sel",
                "2",
                "-o",
                "rdf.xvg",
                "--norm",
                "number_density",
                "--geometry",
                "xy",
                "--no-pbc",
            ]
        )

        self.assertEqual(args.normalization, "number-density")
        self.assertEqual(args.geometry, "xy")
        self.assertFalse(args.pbc)

    def test_surface_implies_none_and_rejects_topology_exclusions(self):
        args = getargs_rdf(
            [
                "-s",
                "run.tpr",
                "-f",
                "run.xtc",
                "-ref",
                "1",
                "-sel",
                "2",
                "-o",
                "rdf.xvg",
                "--surface",
                "molecule",
            ]
        )
        self.assertEqual(args.surface, "chain")
        self.assertEqual(args.normalization, "none")

        with self.assertRaises(SystemExit):
            getargs_rdf(
                [
                    "-s",
                    "run.tpr",
                    "-f",
                    "run.xtc",
                    "-ref",
                    "1",
                    "-sel",
                    "2",
                    "-o",
                    "rdf.xvg",
                    "--surface",
                    "chain",
                    "--exclude",
                    "topology",
                ]
            )


class RdfCommandTests(unittest.TestCase):
    def test_rdf_is_a_public_command(self):
        self.assertIn("rdf", all_commands.command_names)

    def test_atom_rdf_and_cumulative_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rdf.xvg"
            cumulative = Path(directory) / "rdf-cn.xvg"
            args = getargs_rdf(
                [
                    "-s",
                    "run.tpr",
                    "-f",
                    "run.xtc",
                    "-ref",
                    "0",
                    "-sel",
                    "1",
                    "-o",
                    str(output),
                    "-cn",
                    str(cumulative),
                    "--norm",
                    "none",
                    "--bin-width",
                    "0.2",
                    "--rmax",
                    "0.9",
                ]
            )
            with patch("dropps.analysis.rdf.trajectory_class", _SyntheticTrajectory):
                rdf(args)

            rdf_data = np.loadtxt(output, comments=("#", "@"))
            cumulative_data = np.loadtxt(cumulative, comments=("#", "@"))
            np.testing.assert_allclose(rdf_data[:, 0], [0.0, 0.2, 0.4, 0.6, 0.8])
            np.testing.assert_allclose(rdf_data[:, 1], [0.0, 5.0, 0.0, 0.0, 5.0])
            np.testing.assert_allclose(cumulative_data[:, 1], [0.0, 1.0, 1.0, 1.0, 2.0])
            text = output.read_text(encoding="utf-8")
            self.assertIn("normalization=none", text)
            self.assertIn("actual_rmax_nm=0.9", text)
            self.assertIn('@ s0 legend "1"', text)

    def test_automatic_radius_uses_smallest_selected_frame(self):
        frames = _Frames(
            [
                SimpleNamespace(
                    time=0.0,
                    dimensions=np.asarray([20.0, 20.0, 20.0, 90.0, 90.0, 90.0]),
                ),
                SimpleNamespace(
                    time=1000.0,
                    dimensions=np.asarray([16.0, 18.0, 20.0, 90.0, 90.0, 90.0]),
                ),
            ]
        )
        args = SimpleNamespace(
            pbc=True,
            rmax=None,
            geometry="3d",
            bin_width=0.2,
            cut=0.0,
        )

        grid, information = _radius_prepass(frames, [0, 1], args)

        self.assertAlmostEqual(information["safe_radius_nm"], 0.8)
        self.assertEqual(information["limiting_frame"], 1)
        self.assertAlmostEqual(grid.actual_rmax, 0.7)


if __name__ == "__main__":
    unittest.main()
