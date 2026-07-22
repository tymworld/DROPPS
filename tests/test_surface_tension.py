import unittest
import pickle
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import openmm
import openmm.app
from openmm import unit

from dropps.analysis.surface_tension import (
    create_energy_context,
    evaluate_frame_surface_tension,
    getargs_surface_tension,
    surface_tension as run_surface_tension,
)
from dropps.analysis.surface_tension_core import (
    KJ_PER_MOL_NM2_TO_MN_PER_M,
    block_statistics,
    deform_configuration,
    surface_tension_from_energy_pair,
    volume_preserving_deformation,
)


class SurfaceTensionCoreTests(unittest.TestCase):
    def test_deformation_preserves_volume_for_every_axis(self):
        box = np.diag([4.0, 5.0, 12.0])
        positions = np.asarray([[1.0, 2.0, 3.0], [3.0, 4.0, 8.0]])
        for axis in ("x", "y", "z"):
            deformation = volume_preserving_deformation(axis, 7.0e-4)
            self.assertAlmostEqual(np.linalg.det(deformation), 1.0, places=13)
            _, deformed_box = deform_configuration(positions, box, axis, 7.0e-4)
            self.assertAlmostEqual(
                np.linalg.det(deformed_box), np.linalg.det(box), places=11
            )

    def test_energy_derivative_is_converted_to_surface_tension(self):
        box = np.diag([4.0, 5.0, 12.0])
        strain = 2.0e-4
        expected_gamma = 3.25
        expected_kj_per_mol_nm2 = expected_gamma / KJ_PER_MOL_NM2_TO_MN_PER_M
        derivative = -expected_kj_per_mol_nm2 * 2 * np.linalg.det(box) / box[2, 2]
        center = 800.0
        energy_plus = center + derivative * strain
        energy_minus = center - derivative * strain
        actual = surface_tension_from_energy_pair(
            energy_plus, energy_minus, strain, box
        )
        self.assertAlmostEqual(actual, expected_gamma, places=9)

    def test_block_standard_error_uses_contiguous_block_means(self):
        statistics = block_statistics(np.asarray([1.0, 1.0, 3.0, 3.0]), 2)
        self.assertEqual(statistics["blocks"], 2)
        self.assertAlmostEqual(statistics["mean"], 2.0)
        self.assertAlmostEqual(statistics["sem"], 1.0)
        np.testing.assert_allclose(statistics["block_means"], [1.0, 3.0])


class SurfaceTensionOpenMMTests(unittest.TestCase):
    def test_finite_difference_matches_harmonic_bond_derivative(self):
        system = openmm.System()
        system.addParticle(100.0)
        system.addParticle(100.0)
        bond = openmm.HarmonicBondForce()
        bond.setUsesPeriodicBoundaryConditions(True)
        spring = 200.0
        equilibrium_distance = 0.8
        actual_distance = 1.0
        bond.addBond(
            0,
            1,
            equilibrium_distance * unit.nanometer,
            spring * unit.kilojoule_per_mole / unit.nanometer**2,
        )
        system.addForce(bond)

        box = np.diag([4.0, 5.0, 12.0])
        system.setDefaultPeriodicBoxVectors(
            *(openmm.Vec3(*row) * unit.nanometer for row in box)
        )
        positions = np.asarray([[2.0, 2.5, 5.0], [2.0, 2.5, 6.0]])
        context, integrator, _, _ = create_energy_context(
            system, platform_name="Reference"
        )
        try:
            actual = evaluate_frame_surface_tension(
                context,
                positions,
                box,
                axis="z",
                strain=1.0e-5,
                interfaces=2,
            )
        finally:
            del context
            del integrator

        derivative = spring * (actual_distance - equilibrium_distance) * actual_distance
        expected = (
            -box[2, 2]
            * derivative
            / (2 * np.linalg.det(box))
            * KJ_PER_MOL_NM2_TO_MN_PER_M
        )
        self.assertAlmostEqual(actual, expected, places=7)

    def test_tpr_xtc_analysis_writes_time_series_and_summary(self):
        system = openmm.System()
        system.addParticle(100.0)
        system.addParticle(100.0)
        bond = openmm.HarmonicBondForce()
        bond.setUsesPeriodicBoundaryConditions(True)
        bond.addBond(
            0,
            1,
            0.8 * unit.nanometer,
            200.0 * unit.kilojoule_per_mole / unit.nanometer**2,
        )
        system.addForce(bond)
        box = np.diag([4.0, 5.0, 12.0])
        system.setDefaultPeriodicBoxVectors(
            *(openmm.Vec3(*row) * unit.nanometer for row in box)
        )

        topology = openmm.app.Topology()
        chain = topology.addChain()
        residue_a = topology.addResidue("A", chain)
        residue_b = topology.addResidue("B", chain)
        atom_a = topology.addAtom("A", openmm.app.Element.getBySymbol("He"), residue_a)
        atom_b = topology.addAtom("B", openmm.app.Element.getBySymbol("He"), residue_b)
        topology.addBond(atom_a, atom_b)
        topology.setPeriodicBoxVectors(
            [openmm.Vec3(*row) * unit.nanometer for row in box]
        )
        positions = np.asarray([[2.0, 2.5, 5.0], [2.0, 2.5, 6.0]])

        atom_records = [
            SimpleNamespace(
                charge=0.0,
                mass=100.0,
                name="A",
                abbr="A",
                residueid=1,
                residuename="A",
            ),
            SimpleNamespace(
                charge=0.0,
                mass=100.0,
                name="B",
                abbr="B",
                residueid=2,
                residuename="B",
            ),
        ]
        runtime = {
            "parameters": {
                "vdwtype": "none",
                "pcoulp": False,
                "bondtype": "bond",
            },
            "mdsystem": system,
            "mdtopology": topology,
            "positions": positions * unit.nanometer,
            "pdb_raw": "",
            "ITP_list": [SimpleNamespace(molecule_name="slab", atoms=atom_records)],
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            tpr_path = directory / "slab.tpr"
            xtc_path = directory / "slab.xtc"
            output_path = directory / "gamma.xvg"
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
            for step, distance in enumerate((1.0, 1.01, 0.99)):
                frame_positions = positions.copy()
                frame_positions[1, 2] = frame_positions[0, 2] + distance
                simulation.context.setPositions(frame_positions * unit.nanometer)
                simulation.currentStep = step
                reporter.report(
                    simulation,
                    simulation.context.getState(getPositions=True),
                )
            del reporter
            del simulation
            del integrator

            args = getargs_surface_tension(
                [
                    "-s",
                    str(tpr_path),
                    "-f",
                    str(xtc_path),
                    "-o",
                    str(output_path),
                    "--platform",
                    "Reference",
                    "--strain",
                    "1e-4",
                    "2e-4",
                    "--blocks",
                    "2",
                ]
            )
            run_surface_tension(args)

            output = output_path.read_text(encoding="utf-8")
            self.assertIn("mean_mN_per_m=", output)
            self.assertIn("block_sem_mN_per_m=", output)
            self.assertIn("strain_mean_spread_mN_per_m=", output)
            data_lines = [
                line
                for line in output.splitlines()
                if line and not line.startswith(("#", "@"))
            ]
            self.assertEqual(len(data_lines), 3)
            self.assertTrue(all(len(line.split()) == 5 for line in data_lines))


if __name__ == "__main__":
    unittest.main()
