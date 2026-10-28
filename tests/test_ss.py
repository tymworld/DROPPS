import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from dropps.analysis.ss import getargs_ss, ss


def _strand_coordinates(length=8):
    angle = np.deg2rad(28.0)
    directions = np.column_stack(
        (
            np.full(length - 1, np.cos(angle)),
            np.sin(angle) * np.where(np.arange(length - 1) % 2 == 0, 1.0, -1.0),
            np.zeros(length - 1),
        )
    )
    return np.vstack((np.zeros(3), np.cumsum(0.38 * directions, axis=0)))


class _Frames:
    def __init__(self, frames):
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        return self.frames[index]


class _Index:
    def __init__(self, atom_count, residue_ids):
        self.atom_molnames = ["PROT"] * atom_count
        self.atom_names = ["BB"] * atom_count
        self.atom_abbrs = ["BB"] * atom_count
        self.atom_resnames = ["RES"] * atom_count
        self.atom_resids = residue_ids

    def print_all(self):
        return None


class _SyntheticStrandTrajectory:
    def __init__(self, *_args, **_kwargs):
        length = 8
        chain = _strand_coordinates(length)
        positions = 10.0 * np.vstack((chain, chain + [0.0, 0.0, 0.5]))
        dimensions = np.asarray([100.0, 100.0, 100.0, 90.0, 90.0, 90.0])
        frames = [
            SimpleNamespace(time=0.0, positions=positions, dimensions=dimensions),
            SimpleNamespace(time=1000.0, positions=positions.copy(), dimensions=dimensions),
        ]
        bonds = np.vstack(
            (
                np.column_stack((np.arange(7), np.arange(1, 8))),
                np.column_stack((np.arange(8, 15), np.arange(9, 16))),
            )
        ).astype(np.int64)
        self.Universe = SimpleNamespace(
            trajectory=_Frames(frames),
            bonds=SimpleNamespace(indices=bonds),
        )
        self.id2chainID = [0] * length + [1] * length
        self.id2resID = list(range(1, length + 1)) * 2
        self.index = _Index(2 * length, self.id2resID)
        atom_records = [
            SimpleNamespace(
                name="BB",
                abbr="BB",
                residueid=index + 1,
                residuename="RES",
            )
            for index in range(length)
        ]
        topology = SimpleNamespace(molecule_name="PROT", atoms=atom_records)
        self.tpr = SimpleNamespace(itp_list=[topology, topology])

    def getSelection(self, expression, _purpose=None):
        return SimpleNamespace(indices=np.arange(16, dtype=np.int64)), expression

    def getSelection_interactive_multiple(self, _purpose=None):
        return [SimpleNamespace(indices=np.arange(16, dtype=np.int64))], ["System"]

    def time2frame(self, *_args):
        return 0, 1, 1


class SecondaryStructureArgumentTests(unittest.TestCase):
    def test_psea_defaults_and_pairing_options(self):
        args = getargs_ss(
            [
                "-s",
                "run.tpr",
                "-f",
                "run.xtc",
                "-o",
                "states.csv",
                "--select",
                "mol PROT",
                "--strand-pairing",
                "interchain",
                "--strand-pair-output",
                "pairs.csv",
            ]
        )

        self.assertEqual(args.method, "psea")
        self.assertTrue(args.normalize_bonds)
        self.assertEqual(args.min_helix_length, 4)
        self.assertEqual(args.min_strand_length, 3)
        self.assertEqual(args.helix_angle, (89.0, 12.0))
        self.assertEqual(args.strand_angle, (124.0, 14.0))
        self.assertEqual(args.selection_expressions, ["mol PROT"])


class SecondaryStructureCommandTests(unittest.TestCase):
    def test_command_writes_state_map_counts_occupancy_and_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            state = directory / "state.csv"
            state_map = directory / "map.xpm"
            counts = directory / "counts.xvg"
            occupancy = directory / "occupancy.xvg"
            strand_stat = directory / "strand-stat.xvg"
            pairs = directory / "pairs.csv"
            args = getargs_ss(
                [
                    "-s",
                    "run.tpr",
                    "-f",
                    "run.xtc",
                    "-o",
                    str(state),
                    "-map",
                    str(state_map),
                    "-num",
                    str(counts),
                    "-occ",
                    str(occupancy),
                    "--strand-stat-output",
                    str(strand_stat),
                    "--strand-pair-output",
                    str(pairs),
                    "--select",
                    "mol PROT",
                    "--method",
                    "angle",
                    "--strand-pairing",
                    "interchain",
                ]
            )

            with patch(
                "dropps.analysis.ss.trajectory_class",
                _SyntheticStrandTrajectory,
            ):
                ss(args)

            with state.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            with pairs.open(newline="") as handle:
                pair_rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 16)
            self.assertTrue(all(row["dominant"] == "E" for row in rows))
            self.assertTrue(all(float(row["e_final_fraction"]) == 1.0 for row in rows))
            self.assertGreaterEqual(len(pair_rows), 2)
            self.assertTrue(all(row["orientation"] == "parallel" for row in pair_rows))
            self.assertIn('/* type:    "Discrete" */', state_map.read_text())
            self.assertIn('legend "mol PROT:PROT E"', counts.read_text())
            self.assertIn('legend "E"', occupancy.read_text())
            self.assertIn('legend "E final"', strand_stat.read_text())


if __name__ == "__main__":
    unittest.main()
