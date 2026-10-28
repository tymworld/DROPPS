import importlib
import pickle
import tempfile
import unittest
import warnings
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import MDAnalysis as mda
import numpy as np
import openmm
import openmm.app
from openmm import unit
from openmm.unit import nanometer


trjconv_module = importlib.import_module("dropps.commands.trjconv")


class _FakeIndex:
    @staticmethod
    def print_all():
        return None


class _FakeTrajectory:
    def __init__(self):
        universe = mda.Universe.empty(
            2,
            n_residues=2,
            atom_resindex=[0, 1],
            trajectory=True,
        )
        universe.add_TopologyAttr("names", ["A", "B"])
        universe.add_TopologyAttr("types", ["C", "C"])
        universe.add_TopologyAttr("elements", ["C", "C"])
        universe.add_TopologyAttr("resnames", ["AAA", "BBB"])
        universe.add_TopologyAttr("resids", [1, 2])
        universe.add_TopologyAttr("segids", ["A"])
        universe.add_TopologyAttr("chainIDs", ["A", "A"])
        universe.add_TopologyAttr("masses", [1.0, 1.0])
        universe.add_TopologyAttr("bonds", [(0, 1)])
        coordinates = np.asarray(
            [
                [[9.0, 1.0, 1.0], [0.0, 1.0, 1.0]],
                [[9.5, 1.0, 1.0], [0.5, 1.0, 1.0]],
                [[8.0, 1.0, 1.0], [9.0, 1.0, 1.0]],
            ],
            dtype=np.float32,
        )
        dimensions = np.tile(
            np.asarray([10.0, 10.0, 10.0, 90.0, 90.0, 90.0]),
            (coordinates.shape[0], 1),
        )
        universe.load_new(
            coordinates,
            format=mda.coordinates.memory.MemoryReader,
            dimensions=dimensions,
            dt=1000.0,
        )

        self.Universe = universe
        self.id2chainID = [0, 0]
        self.id2resID = [1, 2]
        self.index = _FakeIndex()
        self.tpr = SimpleNamespace(
            positions=np.asarray(coordinates[0], dtype=np.float64)
            / 10.0
            * nanometer
        )

    def getSelection(self, expression, help="selection"):
        if expression in {"group 1", "1"}:
            return self.Universe.atoms[[0]], "first"
        return self.Universe.atoms, "System"


class TrjconvIntegrationTests(unittest.TestCase):
    def _run(self, argv):
        args = trjconv_module.getargs_trjconv(argv)
        fake = _FakeTrajectory()
        with patch.object(
            trjconv_module,
            "trajectory_class",
            return_value=fake,
        ):
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = trjconv_module._run_trjconv(args)
        self.assertEqual(result, 0)
        return fake

    def test_equal_times_write_nearest_frame_and_make_molecule_whole(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "frame.pdb"
            self._run(
                [
                    "-s",
                    "run.tpr",
                    "-f",
                    "run.xtc",
                    "-o",
                    str(output),
                    "-sel",
                    "group 0",
                    "-b",
                    "1.5",
                    "-e",
                    "1.5",
                    "-pbc",
                    "whole",
                    "--conect",
                ]
            )

            written = mda.Universe(str(output))
            np.testing.assert_allclose(
                written.atoms.positions[1] - written.atoms.positions[0],
                [1.0, 0.0, 0.0],
                atol=1.0e-6,
            )
            self.assertIn("CONECT", output.read_text(encoding="utf-8"))

    def test_selected_atom_count_and_times_are_preserved_in_xtc(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "subset.xtc"
            self._run(
                [
                    "-s",
                    "run.tpr",
                    "-f",
                    "run.xtc",
                    "-o",
                    str(output),
                    "-sel",
                    "group 1",
                    "-b",
                    "0",
                    "-e",
                    "2",
                    "-dt",
                    "1",
                ]
            )

            reader = mda.coordinates.XTC.XTCReader(str(output))
            try:
                self.assertEqual(reader.n_atoms, 1)
                self.assertEqual(reader.n_frames, 3)
                np.testing.assert_allclose(
                    [timestep.time for timestep in reader],
                    [0.0, 1000.0, 2000.0],
                )
            finally:
                reader.close()

    def test_separate_pdb_output_uses_numbered_files(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "frame.pdb"
            self._run(
                [
                    "-s",
                    "run.tpr",
                    "-f",
                    "run.xtc",
                    "-o",
                    str(output),
                    "-sel",
                    "group 0",
                    "-b",
                    "0",
                    "-e",
                    "2",
                    "--separate",
                    "--zero-pad",
                    "3",
                ]
            )

            self.assertTrue((Path(directory) / "frame_000.pdb").is_file())
            self.assertTrue((Path(directory) / "frame_001.pdb").is_file())
            self.assertTrue((Path(directory) / "frame_002.pdb").is_file())

    def test_real_tpr_xtc_and_pdb_input_round_trip(self):
        topology = openmm.app.Topology()
        chain = topology.addChain("A")
        residue = topology.addResidue("MOL", chain)
        helium = openmm.app.Element.getBySymbol("He")
        atom_a = topology.addAtom("A", helium, residue)
        atom_b = topology.addAtom("B", helium, residue)
        topology.addBond(atom_a, atom_b)
        box_nm = np.diag([1.0, 1.0, 1.0])
        topology.setPeriodicBoxVectors(
            [openmm.Vec3(*row) * unit.nanometer for row in box_nm]
        )

        system = openmm.System()
        system.addParticle(10.0)
        system.addParticle(10.0)
        system.setDefaultPeriodicBoxVectors(
            *(openmm.Vec3(*row) * unit.nanometer for row in box_nm)
        )
        atom_records = [
            SimpleNamespace(
                charge=0.0,
                mass=10.0,
                name=name,
                abbr="B",
                residueid=index + 1,
                residuename="MOL",
            )
            for index, name in enumerate(("A", "B"))
        ]
        initial_nm = np.asarray([[0.9, 0.1, 0.1], [0.0, 0.1, 0.1]])
        runtime = {
            "parameters": {"vdwtype": "none"},
            "mdsystem": system,
            "mdtopology": topology,
            "positions": initial_nm * unit.nanometer,
            "pdb_raw": "",
            "ITP_list": [
                SimpleNamespace(molecule_name="MOL", atoms=atom_records)
            ],
        }

        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            tpr_path = directory / "run.tpr"
            xtc_path = directory / "run.xtc"
            pdb_path = directory / "frame.pdb"
            roundtrip_path = directory / "roundtrip.xtc"
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
            for frame_index, coordinates in enumerate(
                (initial_nm, initial_nm + [0.02, 0.0, 0.0])
            ):
                simulation.context.setPositions(coordinates * unit.nanometer)
                simulation.context.setTime(frame_index * unit.picoseconds)
                simulation.currentStep = frame_index
                reporter.report(
                    simulation,
                    simulation.context.getState(getPositions=True),
                )
            del reporter
            del simulation
            del integrator

            arguments = trjconv_module.getargs_trjconv(
                [
                    "-s",
                    str(tpr_path),
                    "-f",
                    str(xtc_path),
                    "-o",
                    str(pdb_path),
                    "-sel",
                    "group 0",
                    "-b",
                    "0.001",
                    "-e",
                    "0.001",
                    "-tu",
                    "ps",
                    "-pbc",
                    "whole",
                ]
            )
            first_stderr = StringIO()
            with redirect_stdout(StringIO()), redirect_stderr(first_stderr):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.assertEqual(
                        trjconv_module.trjconv(arguments),
                        0,
                        first_stderr.getvalue(),
                    )

            pdb_universe = mda.Universe(str(pdb_path))
            np.testing.assert_allclose(
                pdb_universe.atoms.positions[1] - pdb_universe.atoms.positions[0],
                [1.0, 0.0, 0.0],
                atol=2.0e-3,
            )

            pdb_arguments = trjconv_module.getargs_trjconv(
                [
                    "-s",
                    str(tpr_path),
                    "-f",
                    str(pdb_path),
                    "-o",
                    str(roundtrip_path),
                    "-sel",
                    "group 0",
                ]
            )
            second_stderr = StringIO()
            with redirect_stdout(StringIO()), redirect_stderr(second_stderr):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.assertEqual(
                        trjconv_module.trjconv(pdb_arguments),
                        0,
                        second_stderr.getvalue(),
                    )
            reader = mda.coordinates.XTC.XTCReader(str(roundtrip_path))
            try:
                self.assertEqual(reader.n_atoms, 2)
                self.assertEqual(reader.n_frames, 1)
            finally:
                reader.close()


if __name__ == "__main__":
    unittest.main()
