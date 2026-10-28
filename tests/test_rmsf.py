import csv
import pickle
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import MDAnalysis as mda
import numpy as np
import openmm
import openmm.app
from openmm import unit
from MDAnalysis.lib.mdamath import triclinic_vectors

from dropps.analysis.image_interaction_core import build_unwrap_plan
from dropps.analysis.rmsf import (
    _align_positions,
    _output_paths,
    _unwrap_selected_positions,
    getargs_rmsf,
    rmsf,
)


class RMSFArgumentTests(unittest.TestCase):
    def _parse_quietly(self, argv):
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            return getargs_rmsf(argv)

    def test_gromacs_style_residue_alias_and_mass_defaults(self):
        args = self._parse_quietly(
            [
                "-s",
                "run.tpr",
                "-f",
                "run.xtc",
                "-o",
                "rmsf.xvg",
                "-res",
            ]
        )

        self.assertEqual(args.grouping, "residue")
        self.assertEqual(args.fit_mode, "molecule")
        self.assertEqual(args.fit_weighting, "mass")
        self.assertEqual(args.group_weighting, "mass")

    def test_repeated_expressions_and_optional_gmx_style_outputs(self):
        args = self._parse_quietly(
            [
                "-s",
                "run.tpr",
                "-f",
                "run.xtc",
                "-o",
                "rmsf",
                "-od",
                "deviation",
                "-ox",
                "average",
                "-oq",
                "reference",
                "--select",
                "mol PROT",
                "--select",
                "mol RNA",
            ]
        )

        self.assertEqual(args.selection_expressions, ["mol PROT", "mol RNA"])
        paths = _output_paths(args)
        self.assertEqual(paths["rmsf"], Path("rmsf.xvg"))
        self.assertEqual(paths["details"], Path("rmsf.csv"))
        self.assertEqual(paths["deviation"], Path("deviation.xvg"))
        self.assertEqual(paths["average"], Path("average.pdb"))
        self.assertEqual(paths["reference"], Path("reference.pdb"))

    def test_grouping_forms_are_mutually_exclusive(self):
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as raised:
                getargs_rmsf(
                    [
                        "-s",
                        "run.tpr",
                        "-f",
                        "run.xtc",
                        "-o",
                        "rmsf.xvg",
                        "--grouping",
                        "atom",
                        "-res",
                    ]
                )

        self.assertEqual(raised.exception.code, 2)


class RMSFPeriodicUnwrapTests(unittest.TestCase):
    def test_partial_selection_reconstructs_complete_triclinic_molecule(self):
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


class RMSFFittingTests(unittest.TestCase):
    @staticmethod
    def _plan(reference):
        atom_count = reference.shape[1]
        indices = np.arange(reference.shape[0] * atom_count, dtype=np.int64).reshape(
            reference.shape[0], atom_count
        )
        return {
            "reference": reference.reshape(-1, 3),
            "fit_weights": np.ones(reference.shape[0] * atom_count),
            "batches": [
                {
                    "position_indices": indices,
                    "flat_indices": indices,
                    "weights": np.ones(indices.shape),
                }
            ],
        }

    def test_selection_fit_removes_one_common_rigid_transform(self):
        reference = np.asarray(
            [
                [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.3, 0.0]],
                [[1.0, 1.0, 0.0], [1.2, 1.0, 0.0], [1.0, 1.3, 0.0]],
            ]
        )
        rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        mobile = reference.reshape(-1, 3) @ rotation + [2.0, 1.0, 0.5]
        dimensions = np.asarray([100.0, 100.0, 100.0, 90.0, 90.0, 90.0])

        aligned = _align_positions(
            mobile,
            self._plan(reference),
            dimensions,
            "selection",
        )

        np.testing.assert_allclose(aligned, reference.reshape(-1, 3), atol=1e-12)

    def test_molecule_fit_handles_independent_copy_transforms(self):
        reference = np.asarray(
            [
                [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.3, 0.0]],
                [[1.0, 1.0, 0.0], [1.2, 1.0, 0.0], [1.0, 1.3, 0.0]],
            ]
        )
        rotations = np.asarray(
            [
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            ]
        )
        translations = np.asarray([[2.0, 1.0, 0.5], [-1.0, 2.0, -0.2]])
        mobile = np.einsum("mai,mij->maj", reference, rotations)
        mobile += translations[:, None, :]
        dimensions = np.asarray([100.0, 100.0, 100.0, 90.0, 90.0, 90.0])

        aligned = _align_positions(
            mobile.reshape(-1, 3),
            self._plan(reference),
            dimensions,
            "molecule",
        )

        np.testing.assert_allclose(aligned, reference.reshape(-1, 3), atol=1e-12)


class RMSFCommandTests(unittest.TestCase):
    def test_molecule_fitting_profiles_copies_and_writes_b_factor_structures(self):
        box_length_nm = 5.0
        box = np.diag([box_length_nm] * 3)
        system = openmm.System()
        for _ in range(6):
            system.addParticle(100.0)
        system.setDefaultPeriodicBoxVectors(
            *(openmm.Vec3(*row) * unit.nanometer for row in box)
        )

        topology = openmm.app.Topology()
        helium = openmm.app.Element.getBySymbol("He")
        molecule_records = []
        for _ in range(2):
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
                        residueid=1,
                        residuename="MOL",
                    )
                )
            topology.addBond(topology_atoms[0], topology_atoms[1])
            topology.addBond(topology_atoms[1], topology_atoms[2])
            molecule_records.append(
                SimpleNamespace(molecule_name="MOL", atoms=atom_records)
            )
        topology.setPeriodicBoxVectors(
            [openmm.Vec3(*row) * unit.nanometer for row in box]
        )

        reference = np.asarray(
            [
                [[4.8, 1.0, 1.0], [5.0, 1.0, 1.0], [5.0, 1.2, 1.0]],
                [[1.0, 3.0, 2.0], [1.2, 3.0, 2.0], [1.2, 3.2, 2.0]],
            ]
        )
        rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        rigid = np.empty_like(reference)
        translations = np.asarray([[0.4, 0.3, 0.0], [-0.3, 0.2, 0.1]])
        for molecule_id in range(2):
            rigid[molecule_id] = (
                (reference[molecule_id] - reference[molecule_id, 0]) @ rotation
                + reference[molecule_id, 0]
                + translations[molecule_id]
            )
        deformed_up = rigid.copy()
        deformed_up[1, 2, 1] += 0.30
        deformed_down = rigid.copy()
        deformed_down[1, 2, 1] -= 0.20
        frames = [reference, rigid, deformed_up, deformed_down]

        runtime = {
            "parameters": {"vdwtype": "none"},
            "mdsystem": system,
            "mdtopology": topology,
            "positions": np.mod(reference.reshape(-1, 3), box_length_nm)
            * unit.nanometer,
            "pdb_raw": "",
            "ITP_list": molecule_records,
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            tpr_path = directory / "molecules.tpr"
            xtc_path = directory / "molecules.xtc"
            output_path = directory / "rmsf.xvg"
            deviation_path = directory / "deviation.xvg"
            average_path = directory / "average.pdb"
            reference_path = directory / "reference.pdb"
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
                simulation.context.setPositions(
                    np.mod(frame.reshape(-1, 3), box_length_nm) * unit.nanometer
                )
                simulation.context.setTime(frame_index * unit.picoseconds)
                simulation.currentStep = frame_index
                reporter.report(
                    simulation,
                    simulation.context.getState(getPositions=True),
                )
            del reporter
            del simulation
            del integrator

            args = getargs_rmsf(
                [
                    "-s",
                    str(tpr_path),
                    "-f",
                    str(xtc_path),
                    "-o",
                    str(output_path),
                    "-od",
                    str(deviation_path),
                    "-ox",
                    str(average_path),
                    "-oq",
                    str(reference_path),
                    "--select",
                    "mol MOL",
                ]
            )
            rmsf(args)

            with output_path.with_suffix(".csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 6)
            molecule_zero = [row for row in rows if row["molecule_id"] == "0"]
            molecule_one = [row for row in rows if row["molecule_id"] == "1"]
            self.assertTrue(
                all(float(row["rmsf_nm"]) < 2.0e-4 for row in molecule_zero)
            )
            self.assertGreater(
                max(float(row["rmsf_nm"]) for row in molecule_one),
                0.02,
            )

            data_lines = [
                line
                for line in output_path.read_text(encoding="utf-8").splitlines()
                if line and not line.startswith(("#", "@", "&"))
            ]
            self.assertEqual(len(data_lines), 3)
            self.assertTrue(deviation_path.is_file())

            average_universe = mda.Universe(str(average_path))
            reference_universe = mda.Universe(str(reference_path))
            self.assertEqual(len(average_universe.atoms), 6)
            self.assertEqual(len(reference_universe.atoms), 6)
            self.assertGreater(float(np.max(average_universe.atoms.tempfactors)), 0.0)

            residue_output = directory / "rmsf-residue.xvg"
            residue_args = getargs_rmsf(
                [
                    "-s",
                    str(tpr_path),
                    "-f",
                    str(xtc_path),
                    "-o",
                    str(residue_output),
                    "--select",
                    "mol MOL",
                    "-res",
                ]
            )
            rmsf(residue_args)
            with residue_output.with_suffix(".csv").open(encoding="utf-8") as handle:
                residue_rows = list(csv.DictReader(handle))
            self.assertEqual(len(residue_rows), 2)
            self.assertTrue(all(row["residue_id"] == "1" for row in residue_rows))


if __name__ == "__main__":
    unittest.main()
