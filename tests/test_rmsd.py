import csv
import pickle
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import openmm
import openmm.app
from openmm import unit
from MDAnalysis.lib.mdamath import triclinic_vectors

from dropps.analysis.image_interaction_core import build_unwrap_plan
from dropps.analysis.rmsd import (
    _output_paths,
    _split_indices_by_chain,
    _unique_labels,
    _unwrap_selected_positions,
    getargs_rmsd,
    rmsd,
)


class RMSDArgumentTests(unittest.TestCase):
    def _parse_quietly(self, argv):
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            return getargs_rmsd(argv)

    def test_repeated_selection_expressions_are_supported(self):
        args = self._parse_quietly(
            [
                "-s",
                "run.tpr",
                "-f",
                "run.xtc",
                "-o",
                "rmsd.xvg",
                "--select",
                "(mol PROT) & (resid 1-20)",
                "--select",
                "(mol RNA) & (name BB)",
            ]
        )

        self.assertEqual(
            args.selection_expressions,
            ["(mol PROT) & (resid 1-20)", "(mol RNA) & (name BB)"],
        )
        self.assertIsNone(args.selection_groups)
        self.assertEqual(args.fit_mode, "molecule")
        self.assertEqual(args.output_mode, "molecule")

    def test_fit_and_output_modes_are_independently_selectable(self):
        for fit_mode in ("none", "molecule", "selection"):
            for output_mode in ("molecule", "selection"):
                with self.subTest(fit_mode=fit_mode, output_mode=output_mode):
                    args = self._parse_quietly(
                        [
                            "-s",
                            "run.tpr",
                            "-f",
                            "run.xtc",
                            "-o",
                            "rmsd.xvg",
                            "--fit-mode",
                            fit_mode,
                            "--output-mode",
                            output_mode,
                        ]
                    )
                    self.assertEqual(args.fit_mode, fit_mode)
                    self.assertEqual(args.output_mode, output_mode)

    def test_group_and_expression_sources_are_mutually_exclusive(self):
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as raised:
                getargs_rmsd(
                    [
                        "-s",
                        "run.tpr",
                        "-f",
                        "run.xtc",
                        "-o",
                        "rmsd.xvg",
                        "-sel",
                        "1",
                        "--select",
                        "mol PROT",
                    ]
                )

        self.assertEqual(raised.exception.code, 2)

    def test_duplicate_selection_labels_are_disambiguated(self):
        self.assertEqual(
            _unique_labels(["protein", "rna", "protein"]),
            ["protein", "rna", "protein#2"],
        )

    def test_selected_atoms_are_partitioned_by_molecule_in_one_pass(self):
        groups = _split_indices_by_chain(
            np.asarray([0, 2, 3, 4, 7]),
            np.asarray([0, 0, 0, 1, 1, 2, 2, 2]),
        )

        self.assertEqual([chain_id for chain_id, _ in groups], [0, 1, 2])
        np.testing.assert_array_equal(groups[0][1], [0, 2])
        np.testing.assert_array_equal(groups[1][1], [3, 4])
        np.testing.assert_array_equal(groups[2][1], [7])

    def test_companion_csv_path_and_summary_only_mode(self):
        paths = _output_paths("result", summary_only=False)
        self.assertEqual(paths["summary"], Path("result.xvg"))
        self.assertEqual(paths["details"], Path("result.csv"))
        self.assertEqual(
            _output_paths("result", summary_only=True),
            {"summary": Path("result.xvg")},
        )


class RMSDPeriodicUnwrapTests(unittest.TestCase):
    def test_partial_selection_uses_full_bonded_molecule_in_triclinic_box(self):
        dimensions = np.asarray([10.0, 9.0, 8.0, 90.0, 90.0, 60.0])
        cell = np.asarray(triclinic_vectors(dimensions, dtype=np.float64))
        fractional = np.asarray(
            [
                [0.9, 0.2, 0.3],
                [0.0, 0.2, 0.3],
                [0.1, 0.2, 0.3],
            ]
        )
        positions = fractional @ cell
        plan = build_unwrap_plan(
            molecule_atom_indices=([0, 1, 2],),
            selected_atom_indices=[0, 2],
            bonds=[[0, 1], [1, 2]],
        )

        unwrapped_nm = _unwrap_selected_positions(positions, plan, dimensions)

        expected_fractional = np.asarray([[0.9, 0.2, 0.3], [1.1, 0.2, 0.3]])
        np.testing.assert_allclose(
            unwrapped_nm,
            expected_fractional @ cell / 10.0,
            atol=1.0e-12,
        )


class RMSDCommandTests(unittest.TestCase):
    def test_two_molecules_are_selected_once_and_fitted_independently(self):
        box_length_nm = 4.0
        box = np.diag([box_length_nm] * 3)
        system = openmm.System()
        for _ in range(6):
            system.addParticle(100.0)
        system.setDefaultPeriodicBoxVectors(
            *(openmm.Vec3(*row) * unit.nanometer for row in box)
        )

        topology = openmm.app.Topology()
        helium = openmm.app.Element.getBySymbol("He")
        atom_records_by_molecule = []
        for molecule_id in range(2):
            chain = topology.addChain()
            residue = topology.addResidue("MOL", chain)
            topology_atoms = []
            atom_records = []
            for atom_id in range(3):
                topology_atoms.append(topology.addAtom(f"B{atom_id}", helium, residue))
                atom_records.append(
                    SimpleNamespace(
                        charge=0.0,
                        mass=100.0,
                        name=f"B{atom_id}",
                        abbr="B",
                        residueid=atom_id + 1,
                        residuename="MOL",
                    )
                )
            topology.addBond(topology_atoms[0], topology_atoms[1])
            topology.addBond(topology_atoms[1], topology_atoms[2])
            atom_records_by_molecule.append(
                SimpleNamespace(molecule_name="MOL", atoms=atom_records)
            )
        topology.setPeriodicBoxVectors(
            [openmm.Vec3(*row) * unit.nanometer for row in box]
        )

        reference = np.asarray(
            [
                [[3.8, 1.0, 1.0], [4.0, 1.0, 1.0], [4.0, 1.2, 1.0]],
                [[1.0, 2.0, 2.0], [1.2, 2.0, 2.0], [1.2, 2.2, 2.0]],
            ]
        )
        rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        rigid = np.empty_like(reference)
        translations = np.asarray([[0.5, 0.3, 0.0], [-0.4, 0.2, 0.1]])
        for molecule_id in range(2):
            rigid[molecule_id] = (
                (reference[molecule_id] - reference[molecule_id, 0]) @ rotation
                + reference[molecule_id, 0]
                + translations[molecule_id]
            )
        deformed = rigid.copy()
        deformed[1, 2, 1] += 0.25
        frames = [reference, rigid, deformed]

        initial_positions = np.mod(reference.reshape(-1, 3), box_length_nm)
        runtime = {
            "parameters": {"vdwtype": "none"},
            "mdsystem": system,
            "mdtopology": topology,
            "positions": initial_positions * unit.nanometer,
            "pdb_raw": "",
            "ITP_list": atom_records_by_molecule,
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            tpr_path = directory / "molecules.tpr"
            xtc_path = directory / "molecules.xtc"
            with tpr_path.open("wb") as handle:
                pickle.dump(runtime, handle)

            integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
            simulation = openmm.app.Simulation(
                topology,
                system,
                integrator,
                openmm.Platform.getPlatformByName("Reference"),
            )
            reporter = openmm.app.XTCReporter(str(xtc_path), 1)
            for frame_index, frame in enumerate(frames):
                wrapped = np.mod(frame.reshape(-1, 3), box_length_nm)
                simulation.context.setPositions(wrapped * unit.nanometer)
                simulation.context.setTime(frame_index * unit.picoseconds)
                simulation.currentStep = frame_index
                reporter.report(
                    simulation,
                    simulation.context.getState(getPositions=True),
                )
            del reporter
            del simulation
            del integrator

            results = {}
            for fit_mode in ("none", "molecule", "selection"):
                for output_mode in ("molecule", "selection"):
                    output_path = directory / f"rmsd_{fit_mode}_{output_mode}.xvg"
                    args = getargs_rmsd(
                        [
                            "-s",
                            str(tpr_path),
                            "-f",
                            str(xtc_path),
                            "-o",
                            str(output_path),
                            "--select",
                            "mol MOL",
                            "--fit-mode",
                            fit_mode,
                            "--output-mode",
                            output_mode,
                        ]
                    )
                    rmsd(args)

                    with output_path.with_suffix(".csv").open(
                        encoding="utf-8"
                    ) as handle:
                        rows = list(csv.DictReader(handle))
                    data_lines = [
                        line
                        for line in output_path.read_text(encoding="utf-8").splitlines()
                        if line and not line.startswith(("#", "@"))
                    ]
                    results[(fit_mode, output_mode)] = (rows, data_lines)

                    expected_rows = 6 if output_mode == "molecule" else 3
                    expected_columns = 3 if output_mode == "molecule" else 2
                    self.assertEqual(len(rows), expected_rows)
                    self.assertEqual(len(data_lines), 3)
                    self.assertTrue(
                        all(
                            len(line.split()) == expected_columns for line in data_lines
                        )
                    )
                    self.assertEqual({row["fit_mode"] for row in rows}, {fit_mode})
                    self.assertEqual(
                        {row["output_mode"] for row in rows},
                        {output_mode},
                    )

            molecule_rows, _ = results[("molecule", "molecule")]
            self.assertEqual(
                {row["chain_id"] for row in molecule_rows},
                {"0", "1"},
            )
            molecule_fit_by_frame = {
                frame_index: [
                    float(row["rmsd_nm"])
                    for row in molecule_rows
                    if int(row["frame"]) == frame_index
                ]
                for frame_index in range(3)
            }
            np.testing.assert_allclose(molecule_fit_by_frame[0], 0.0, atol=1.0e-6)
            np.testing.assert_allclose(molecule_fit_by_frame[1], 0.0, atol=1.0e-5)
            self.assertGreater(max(molecule_fit_by_frame[2]), 0.01)

            no_fit_rows, _ = results[("none", "molecule")]
            no_fit_frame_one = [
                float(row["rmsd_nm"]) for row in no_fit_rows if int(row["frame"]) == 1
            ]
            self.assertTrue(all(value > 0.1 for value in no_fit_frame_one))

            pooled_molecule_fit_rows, _ = results[("molecule", "selection")]
            self.assertAlmostEqual(
                float(pooled_molecule_fit_rows[1]["rmsd_nm"]),
                0.0,
                places=5,
            )
            self.assertGreater(
                float(pooled_molecule_fit_rows[2]["rmsd_nm"]),
                0.01,
            )

            selection_fit_rows, _ = results[("selection", "selection")]
            self.assertGreater(float(selection_fit_rows[1]["rmsd_nm"]), 0.01)


if __name__ == "__main__":
    unittest.main()
