import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from dropps.analysis.distance import _frame_positions_nm, distance, getargs_distance
from dropps.analysis.distance_core import build_endpoint_plan, project_endpoint
from dropps.analysis.image_interaction_core import build_unwrap_plan
from dropps.share.all_commands import all_commands


class _Frames:
    def __init__(self, frames):
        self.frames = frames
        self.n_frames = len(frames)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        return self.frames[index]


class _Index:
    def __init__(self):
        self.index_groups = [
            SimpleNamespace(name="System", indices=list(range(6))),
            SimpleNamespace(name="A", indices=[0, 2, 4]),
            SimpleNamespace(name="B", indices=[1, 3, 5]),
        ]

    def print_all(self):
        return None

    def phraseSelection(self, expression):
        raise AssertionError(f"Unexpected compound selection: {expression}")


class _SyntheticTrajectory:
    def __init__(self, *_args, **_kwargs):
        frames = []
        for frame_id, offset in enumerate((0.0, 0.1)):
            frames.append(
                SimpleNamespace(
                    time=frame_id * 1000.0,
                    positions=np.asarray(
                        [
                            [0.0, 0.0, 0.0],
                            [2.0 + offset, 0.0, 0.0],
                            [10.0, 0.0, 0.0],
                            [12.0 + offset, 0.0, 0.0],
                            [20.0, 0.0, 0.0],
                            [22.0 + offset, 0.0, 0.0],
                        ],
                        dtype=float,
                    ),
                    dimensions=np.asarray(
                        [100.0, 100.0, 100.0, 90.0, 90.0, 90.0]
                    ),
                )
            )
        self.Universe = SimpleNamespace(trajectory=_Frames(frames))
        self.index = _Index()
        self.id2chainID = [0, 0, 1, 1, 2, 2]
        self.id2resID = [10, 50, 10, 50, 10, 50]
        self.id2masses = [1.0] * 6

    def time2frame(self, *_args):
        return 0, 1, 1


class DistanceCommandTests(unittest.TestCase):
    def test_distance_is_public_and_legacy_commands_are_hidden_aliases(self):
        self.assertIn("distance", all_commands.command_names)
        self.assertNotIn("idist", all_commands.command_names)
        self.assertNotIn("odist", all_commands.command_names)
        self.assertTrue(all_commands.exist("idist"))
        self.assertTrue(all_commands.exist("odist"))

    def test_parser_exposes_many_chain_scope_and_statistics(self):
        args = getargs_distance(
            [
                "-s",
                "run.tpr",
                "-f",
                "run.xtc",
                "-ref",
                "1",
                "-sel",
                "2",
                "--scope",
                "inter-chain",
                "--position",
                "cog",
                "--mode",
                "pair",
                "-ostat",
                "inter.csv",
                "--statistics-scope",
                "both",
                "--no-pbc",
            ]
        )
        self.assertEqual(args.scope, "inter-chain")
        self.assertEqual(args.position, "cog")
        self.assertEqual(args.statistics_scope, "both")
        self.assertFalse(args.pbc)

    def test_intra_chain_command_writes_pooled_and_per_chain_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            stats = Path(directory) / "intra.csv"
            average = Path(directory) / "intra.xvg"
            histogram = Path(directory) / "intra-hist.xvg"
            args = getargs_distance(
                [
                    "-s",
                    "run.tpr",
                    "-f",
                    "run.xtc",
                    "-ref",
                    "1",
                    "-sel",
                    "2",
                    "--scope",
                    "intra-chain",
                    "--position",
                    "cog",
                    "--mode",
                    "pair",
                    "-ostat",
                    str(stats),
                    "--statistics-scope",
                    "both",
                    "-oav",
                    str(average),
                    "-oh",
                    str(histogram),
                    "--no-pbc",
                ]
            )
            with patch(
                "dropps.analysis.distance.trajectory_class",
                _SyntheticTrajectory,
            ):
                distance(args)

            with stats.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0]["scope"], "pooled")
            self.assertEqual(int(rows[0]["samples"]), 6)
            self.assertAlmostEqual(float(rows[0]["mean_nm"]), 0.205)
            self.assertEqual([row["scope"] for row in rows[1:]], ["channel"] * 3)
            self.assertTrue(all(int(row["samples"]) == 2 for row in rows[1:]))

            data_lines = [
                line
                for line in average.read_text(encoding="utf-8").splitlines()
                if line and not line.startswith(("#", "@"))
            ]
            self.assertEqual(len(data_lines), 2)
            first = np.fromstring(data_lines[0], sep="\t")
            second = np.fromstring(data_lines[1], sep="\t")
            np.testing.assert_allclose(first, [0.0, 0.2, 0.0], atol=1.0e-12)
            np.testing.assert_allclose(second, [1.0, 0.21, 0.0], atol=1.0e-12)

            histogram_text = histogram.read_text(encoding="utf-8")
            self.assertIn("# hist_scope=job normalization=probability", histogram_text)
            histogram_rows = np.loadtxt(histogram, comments=("#", "@"))
            self.assertAlmostEqual(float(histogram_rows[:, 1].sum()), 1.0)

    def test_pbc_center_reconstructs_complete_bonded_chain(self):
        unwrap_plan = build_unwrap_plan(
            [np.asarray([0, 1])],
            np.asarray([0, 1]),
            np.asarray([[0, 1]]),
        )
        timestep = SimpleNamespace(
            positions=np.asarray([[9.8, 0.0, 0.0], [0.2, 0.0, 0.0]]),
            dimensions=np.asarray([10.0, 10.0, 10.0, 90.0, 90.0, 90.0]),
        )
        positions = _frame_positions_nm(timestep, unwrap_plan, True)
        endpoint = build_endpoint_plan(
            [0, 1],
            "wrapped-chain",
            "chain",
            "cog",
            [0, 0],
            [1, 1],
        )
        center = project_endpoint(
            endpoint,
            positions,
            [1.0, 1.0],
            [1.0, 1.0, 1.0, 90.0, 90.0, 90.0],
            True,
        )[0]
        np.testing.assert_allclose(center, [[1.0, 0.0, 0.0]], atol=1.0e-12)


if __name__ == "__main__":
    unittest.main()
