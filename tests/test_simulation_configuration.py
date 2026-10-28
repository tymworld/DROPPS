import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import openmm
from openmm import unit

from dropps.fileio.itp_reader import Atomtype_AH_DH
from dropps.fileio.pdb_reader import read_pdb, write_pdb
from dropps.hp.constants import k0
from dropps.share.build_system import _merge_atomtypes, build_system
from dropps.share.parameters import parameters_dict_template


class PDBBoxValidationTests(unittest.TestCase):
    def test_tilted_box_is_rejected_instead_of_silently_orthogonalized(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "tilted.pdb"
            path.write_text(
                "CRYST1   40.000   40.000   40.000  90.00  90.00  60.00 P 1           1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "orthorhombic"):
                read_pdb(path)

    def test_missing_periodic_box_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "missing-box.pdb"
            path.write_text("END\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no CRYST1"):
                read_pdb(path)


class ForceFieldConfigurationTests(unittest.TestCase):
    def test_conflicting_duplicate_ah_atomtype_is_rejected(self):
        first = SimpleNamespace(
            molecule_name="FIRST",
            atomtypes=[Atomtype_AH_DH("A", "ALA", 0.5, 0.6, 0.0, 0.0, 0.0)],
        )
        second = SimpleNamespace(
            molecule_name="SECOND",
            atomtypes=[Atomtype_AH_DH("A", "ALA", 0.6, 0.6, 0.0, 0.0, 0.0)],
        )
        with self.assertRaisesRegex(ValueError, "Conflicting definition"):
            _merge_atomtypes([first, second])

    def test_temperature_dependent_forces_use_shared_context_temperature(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            itp_path = directory / "mol.itp"
            top_path = directory / "system.top"
            pdb_path = directory / "system.pdb"
            itp_path.write_text(
                """
[ function-type ]
Ashbaugh-Hatch Debye-Huckel
[ nonbond-global ]
1.0
[ coulomb-global ]
temperature-dependent 0 70 0.02 0 0
[ moleculetype ]
MOL 1
[ atomtypes ]
A ALA 0.5 0.5 0.0 0.001 0.0
[ atoms ]
1 A A1 MOL 1 100.0 1.0
2 A A2 MOL 2 100.0 -1.0
""".strip()
                + "\n",
                encoding="utf-8",
            )
            top_path.write_text(
                "[ itp files ]\nmol.itp\n[ system ]\nMOL 1\n",
                encoding="utf-8",
            )
            write_pdb(
                pdb_path,
                [4.0, 4.0, 4.0],
                ["ATOM", "ATOM"],
                [1, 2],
                ["A1", "A2"],
                ["MOL", "MOL"],
                ["A", "A"],
                [1, 2],
                [1.0, 1.8],
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
                [0.0, 0.0],
                ["He", "He"],
                [2],
            )
            parameters = deepcopy(parameters_dict_template)
            parameters.update(
                {
                    "initial_temperature": 200.0,
                    "production_temperature": 300.0,
                    "cutoff_lj": 1.0,
                    "cutoff_coul": 1.0,
                    "pcoulp": False,
                }
            )
            system, _topology, positions, _itp = build_system(
                pdb_path, top_path, parameters
            )
            nonbonded_forces = [
                force
                for force in system.getForces()
                if isinstance(force, openmm.CustomNonbondedForce)
            ]
            self.assertEqual(len(nonbonded_forces), 2)
            for force in nonbonded_forces:
                names = {
                    force.getGlobalParameterName(index)
                    for index in range(force.getNumGlobalParameters())
                }
                self.assertIn("temperature", names)
            self.assertEqual(
                nonbonded_forces[0].getPerParticleParameterName(1),
                "lambda_base",
            )

            integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
            context = openmm.Context(
                system,
                integrator,
                openmm.Platform.getPlatformByName("Reference"),
            )
            try:
                context.setPositions(positions)
                energy_300 = context.getState(getEnergy=True).getPotentialEnergy()
                context.setParameter("temperature", 200.0)
                energy_200 = context.getState(getEnergy=True).getPotentialEnergy()
                self.assertNotEqual(energy_300, energy_200)
            finally:
                del context
                del integrator

    def test_coulomb_constant_matches_standard_openmm_units(self):
        self.assertAlmostEqual(k0, 138.935456, places=6)


if __name__ == "__main__":
    unittest.main()
