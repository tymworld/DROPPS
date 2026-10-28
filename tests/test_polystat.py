import csv
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from dropps.analysis.polystat import (
    _build_cohorts,
    getargs_polystat,
    polystat,
)


class _Frames:
    def __init__(self, frames):
        self.frames = frames
        self.n_frames = len(frames)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        return self.frames[index]


class _Index:
    def __init__(self, atom_molnames, atom_resids):
        self.atom_molnames = atom_molnames
        self.atom_names = [f"B{resid}" for resid in atom_resids]
        self.atom_abbrs = ["BB"] * len(atom_resids)
        self.atom_resids = atom_resids
        self.atom_resnames = ["RES"] * len(atom_resids)

    def print_all(self):
        return None


class _SyntheticTrajectory:
    def __init__(self, *_args, **_kwargs):
        chain_ids = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
        residue_ids = [1, 2, 3, 1, 2, 3, 1, 2, 3, 4]
        molecule_names = ["A"] * 6 + ["B"] * 4
        bonds = np.asarray(
            [[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8], [8, 9]],
            dtype=np.int64,
        )
        positions = np.asarray(
            [
                [98.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [10.0, 10.0, 0.0],
                [12.0, 10.0, 0.0],
                [14.0, 10.0, 0.0],
                [20.0, 20.0, 0.0],
                [23.0, 20.0, 0.0],
                [26.0, 20.0, 0.0],
                [29.0, 20.0, 0.0],
            ],
            dtype=float,
        )
        dimensions = np.asarray([100.0, 100.0, 100.0, 90.0, 90.0, 90.0])
        frames = [
            SimpleNamespace(time=0.0, positions=positions, dimensions=dimensions),
            SimpleNamespace(time=1000.0, positions=positions.copy(), dimensions=dimensions),
        ]
        atoms = SimpleNamespace(masses=np.ones(len(chain_ids), dtype=float))
        self.Universe = SimpleNamespace(
            trajectory=_Frames(frames),
            atoms=atoms,
            bonds=SimpleNamespace(indices=bonds),
        )
        self.id2chainID = chain_ids
        self.id2resID = residue_ids
        self.index = _Index(molecule_names, residue_ids)
        itp_list = []
        for molecule_name, length in (("A", 3), ("A", 3), ("B", 4)):
            atom_records = [
                SimpleNamespace(
                    name=f"B{atom_id + 1}",
                    abbr="BB",
                    residueid=atom_id + 1,
                    residuename="RES",
                    mass=1.0,
                )
                for atom_id in range(length)
            ]
            itp_list.append(
                SimpleNamespace(molecule_name=molecule_name, atoms=atom_records)
            )
        self.tpr = SimpleNamespace(itp_list=itp_list)

    def getSelection(self, expression, _purpose=None):
        if expression == "mol A":
            indices = np.arange(6, dtype=np.int64)
            name = "A"
        else:
            indices = np.arange(10, dtype=np.int64)
            name = "System"
        return SimpleNamespace(indices=indices), name

    def getSelection_interactive_multiple(self, _purpose=None):
        return [SimpleNamespace(indices=np.arange(10))], ["System"]

    def time2frame(self, *_args):
        return 0, 1, 1


class _ManyCopyTrajectory:
    def __init__(self, copies=1000, length=3):
        atom_count = copies * length
        self.id2chainID = np.repeat(np.arange(copies), length).tolist()
        self.id2resID = np.tile(np.arange(1, length + 1), copies).tolist()
        molecule_names = ["PROT"] * atom_count
        self.index = _Index(molecule_names, self.id2resID)
        local_bonds = np.column_stack(
            (np.arange(length - 1), np.arange(1, length))
        )
        bonds = np.concatenate(
            [local_bonds + chain_id * length for chain_id in range(copies)],
            axis=0,
        )
        self.Universe = SimpleNamespace(
            atoms=SimpleNamespace(masses=np.ones(atom_count, dtype=float)),
            bonds=SimpleNamespace(indices=bonds),
        )
        atom_records = [
            SimpleNamespace(
                name=f"B{atom_id + 1}",
                abbr="BB",
                residueid=atom_id + 1,
                residuename="RES",
                mass=1.0,
            )
            for atom_id in range(length)
        ]
        topology = SimpleNamespace(molecule_name="PROT", atoms=atom_records)
        self.tpr = SimpleNamespace(itp_list=[topology] * copies)


class PolystatArgumentTests(unittest.TestCase):
    def _parse_quietly(self, argv):
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            return getargs_polystat(argv)

    def test_many_protein_defaults_and_optional_outputs(self):
        args = self._parse_quietly(
            [
                "-s",
                "run.tpr",
                "-f",
                "run.xtc",
                "-o",
                "polystat.xvg",
                "--select",
                "mol A",
                "-i",
                "internal.csv",
                "-p",
                "persistence.xvg",
                "--tangent-output",
                "tangent.csv",
            ]
        )

        self.assertEqual(args.metrics, ("ree", "rg", "shape"))
        self.assertEqual(args.weight, "mass")
        self.assertEqual(args.lag_step, 1)
        self.assertTrue(args.pbc)
        self.assertEqual(args.selection_expressions, ["mol A"])

    def test_group_and_expression_selection_sources_are_exclusive(self):
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as raised:
                getargs_polystat(
                    [
                        "-s",
                        "run.tpr",
                        "-f",
                        "run.xtc",
                        "-o",
                        "polystat.xvg",
                        "-sel",
                        "1",
                        "--select",
                        "mol A",
                    ]
                )
        self.assertEqual(raised.exception.code, 2)


class PolystatCohortTests(unittest.TestCase):
    def test_mixed_molecule_types_are_never_pooled(self):
        trajectory = _SyntheticTrajectory()
        args = getargs_polystat(
            ["-s", "run.tpr", "-f", "run.xtc", "-o", "out.xvg"]
        )

        cohorts, _, selected, _ = _build_cohorts(
            trajectory,
            [("System", np.arange(10, dtype=np.int64))],
            args,
        )

        self.assertEqual([cohort.molecule_name for cohort in cohorts], ["A", "B"])
        self.assertEqual([cohort.chain_ids.size for cohort in cohorts], [2, 1])
        self.assertEqual([cohort.selected_particles for cohort in cohorts], [3, 4])
        np.testing.assert_array_equal(selected, np.arange(10))

    def test_partial_copies_of_one_name_fail_without_explicit_permission(self):
        trajectory = _SyntheticTrajectory()
        args = getargs_polystat(
            ["-s", "run.tpr", "-f", "run.xtc", "-o", "out.xvg"]
        )
        inconsistent = np.asarray([0, 1, 3, 4, 5], dtype=np.int64)

        with self.assertRaisesRegex(ValueError, "heterogeneous copies"):
            _build_cohorts(trajectory, [("partial-A", inconsistent)], args)

    def test_one_thousand_equivalent_proteins_form_one_linear_size_cohort(self):
        trajectory = _ManyCopyTrajectory(copies=1000, length=3)
        args = getargs_polystat(
            ["-s", "run.tpr", "-f", "run.xtc", "-o", "out.xvg"]
        )

        cohorts, molecule_atoms, selected, bonds = _build_cohorts(
            trajectory,
            [("PROT", np.arange(3000, dtype=np.int64))],
            args,
        )

        self.assertEqual(len(cohorts), 1)
        self.assertEqual(cohorts[0].chain_ids.size, 1000)
        self.assertEqual(len(molecule_atoms), 1000)
        self.assertEqual(selected.size, 3000)
        self.assertEqual(bonds.shape, (2000, 2))


class PolystatCommandTests(unittest.TestCase):
    def test_streaming_outputs_keep_types_separate_and_reconstruct_pbc(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            summary = directory / "polystat.xvg"
            internal = directory / "internal.csv"
            persistence = directory / "persistence.xvg"
            tangent = directory / "tangent.csv"
            molecules = directory / "molecules.csv"
            args = getargs_polystat(
                [
                    "-s",
                    "run.tpr",
                    "-f",
                    "run.xtc",
                    "-o",
                    str(summary),
                    "--select",
                    "mol ALL",
                    "-i",
                    str(internal),
                    "-p",
                    str(persistence),
                    "--tangent-output",
                    str(tangent),
                    "--molecule-output",
                    str(molecules),
                ]
            )

            with patch(
                "dropps.analysis.polystat.trajectory_class",
                _SyntheticTrajectory,
            ):
                polystat(args)

            data_lines = [
                line
                for line in summary.read_text(encoding="utf-8").splitlines()
                if line and not line.startswith(("#", "@"))
            ]
            self.assertEqual(len(data_lines), 2)
            first = np.fromstring(data_lines[0], sep="\t")
            self.assertEqual(first.size, 21)
            self.assertAlmostEqual(first[1], 0.4)
            self.assertAlmostEqual(first[2], 0.4)
            self.assertAlmostEqual(first[3], 0.0)
            self.assertAlmostEqual(first[11], 0.9)

            with molecules.open(newline="", encoding="utf-8") as handle:
                molecule_rows = list(csv.DictReader(handle))
            self.assertEqual(len(molecule_rows), 6)
            self.assertEqual({row["cohort"] for row in molecule_rows}, {"A", "B"})

            with internal.open(newline="", encoding="utf-8") as handle:
                internal_rows = list(csv.DictReader(handle))
            self.assertEqual(len(internal_rows), 5)
            self.assertEqual(
                [row["lag_steps"] for row in internal_rows if row["cohort"] == "A"],
                ["1", "2"],
            )
            a_lag_two = next(
                row
                for row in internal_rows
                if row["cohort"] == "A" and row["lag_steps"] == "2"
            )
            self.assertAlmostEqual(float(a_lag_two["mean_r2_nm2"]), 0.16)

            with tangent.open(newline="", encoding="utf-8") as handle:
                tangent_rows = list(csv.DictReader(handle))
            self.assertEqual(len(tangent_rows), 5)
            self.assertTrue(
                all(abs(float(row["mean_cosine"]) - 1.0) < 1.0e-12 for row in tangent_rows)
            )
            persistence_data = np.loadtxt(persistence, comments=("#", "@"))
            self.assertEqual(persistence_data.shape, (2, 5))
            self.assertTrue(np.all(np.isnan(persistence_data[:, 1:])))


if __name__ == "__main__":
    unittest.main()
