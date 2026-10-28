import tempfile
import unittest
import pickle
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import openmm
from openmm import unit

from dropps.share.openmm_energy import create_energy_context
from dropps.share.parameters import parameters_dict_template
from dropps.share.stress_tensor_reporter import (
    StressTensorReporter,
    configurational_pressure_tensor,
    kinetic_pressure_tensor,
    pressure_tensor,
    synchronized_velocities,
)
from dropps.share.energy_reporter import EnergyReporter, read_edr
from dropps.commands.mdrun import mdrun


def harmonic_bond_system():
    system = openmm.System()
    system.addParticle(100.0)
    system.addParticle(100.0)
    bond = openmm.HarmonicBondForce()
    bond.addBond(
        0,
        1,
        0.5 * unit.nanometer,
        100.0 * unit.kilojoule_per_mole / unit.nanometer**2,
    )
    system.addForce(bond)
    box = np.diag([4.0, 4.0, 4.0])
    system.setDefaultPeriodicBoxVectors(
        *(openmm.Vec3(*row) * unit.nanometer for row in box)
    )
    return system, box


class StressTensorCoreTests(unittest.TestCase):
    def test_half_step_velocity_is_synchronized_with_current_force(self):
        masses = np.asarray([2.0, 4.0])
        velocities = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        forces = np.asarray([[4.0, 0.0, 0.0], [0.0, -8.0, 0.0]])
        actual = synchronized_velocities(
            masses, velocities, forces, timestep_ps=0.2
        )
        np.testing.assert_allclose(
            actual,
            [[1.2, 0.0, 0.0], [0.0, 0.8, 0.0]],
        )

    def test_synchronized_velocity_respects_distance_constraints(self):
        actual = synchronized_velocities(
            [1.0, 1.0],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            timestep_ps=0.1,
            positions_nm=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            box_vectors_nm=np.diag([4.0, 4.0, 4.0]),
            constraints=((0, 1),),
        )
        np.testing.assert_allclose(actual[:, 0], [0.5, 0.5])

    def test_configurational_tensor_matches_harmonic_bond_virial(self):
        system, box = harmonic_bond_system()
        displacement = np.asarray([0.6, 0.8, 0.0])
        positions = np.asarray([[1.0, 1.0, 1.0], [1.6, 1.8, 1.0]])
        context, integrator, _, _ = create_energy_context(
            system, platform_name="Reference"
        )
        try:
            actual = configurational_pressure_tensor(
                context, positions, box, strain=1.0e-5
            )
        finally:
            del context
            del integrator

        distance = np.linalg.norm(displacement)
        force_on_second = -100.0 * (distance - 0.5) * displacement / distance
        expected = np.outer(displacement, force_on_second) / np.linalg.det(box)
        np.testing.assert_allclose(actual, expected, rtol=2.0e-8, atol=2.0e-10)

    def test_total_tensor_adds_kinetic_dyadic(self):
        system, box = harmonic_bond_system()
        positions = np.asarray([[1.0, 1.0, 1.0], [1.6, 1.8, 1.0]])
        velocities = np.asarray([[0.1, -0.2, 0.3], [0.0, 0.1, -0.1]])
        masses = np.asarray([100.0, 100.0])
        volume = np.linalg.det(box)
        expected_kinetic = (
            np.einsum("i,ia,ib->ab", masses, velocities, velocities) / volume
        )
        np.testing.assert_allclose(
            kinetic_pressure_tensor(masses, velocities, volume), expected_kinetic
        )

        context, integrator, _, _ = create_energy_context(
            system, platform_name="Reference"
        )
        try:
            configurational = configurational_pressure_tensor(
                context, positions, box, strain=1.0e-5
            )
            total = pressure_tensor(
                context,
                masses,
                positions,
                velocities,
                box,
                strain=1.0e-5,
            )
        finally:
            del context
            del integrator
        np.testing.assert_allclose(total, configurational + expected_kinetic)


class StressTensorReporterTests(unittest.TestCase):
    def test_energy_reporter_can_store_selected_pressure_terms(self):
        system, _box = harmonic_bond_system()
        positions = np.asarray([[1.0, 1.0, 1.0], [1.6, 1.8, 1.0]])
        velocities = np.asarray([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
        main_integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
        context = openmm.Context(
            system,
            main_integrator,
            openmm.Platform.getPlatformByName("Reference"),
        )
        context.setPositions(positions * unit.nanometer)
        context.setVelocities(velocities * unit.nanometer / unit.picosecond)
        simulation = SimpleNamespace(
            system=system, currentStep=10, integrator=main_integrator
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "run.edr"
            reporter = EnergyReporter(
                output,
                report_interval=10,
                fields=("pressure", "Pxx", "Pxy"),
                stress_strain=1.0e-5,
                stress_platform="Reference",
            )
            try:
                state = context.getState(
                    getPositions=True,
                    getVelocities=True,
                    getForces=True,
                    getEnergy=True,
                )
                reporter.report(simulation, state)
            finally:
                reporter.close()
                del context
                del main_integrator

            data = read_edr(output)
            self.assertEqual(data.header.fields, ("pressure", "Pxx", "Pxy"))
            self.assertEqual(data.steps.tolist(), [10])
            self.assertTrue(np.all(np.isfinite(data.values["Pxx"])))

    def test_reporter_streams_nine_numeric_columns(self):
        system, box = harmonic_bond_system()
        positions = np.asarray([[1.0, 1.0, 1.0], [1.6, 1.8, 1.0]])
        velocities = np.asarray([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
        main_integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
        context = openmm.Context(
            system,
            main_integrator,
            openmm.Platform.getPlatformByName("Reference"),
        )
        context.setPositions(positions * unit.nanometer)
        context.setVelocities(velocities * unit.nanometer / unit.picosecond)
        simulation = SimpleNamespace(
            system=system, currentStep=10, integrator=main_integrator
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "run.stress.xvg"
            reporter = StressTensorReporter(
                output,
                report_interval=10,
                strain=1.0e-5,
                temperature=300.0,
                platform="Reference",
            )
            try:
                state = context.getState(
                    getPositions=True, getVelocities=True, getForces=True
                )
                reporter.report(simulation, state)
            finally:
                reporter.close()
                del context
                del main_integrator

            contents = output.read_text(encoding="utf-8")
            self.assertIn("temperature_K=300", contents)
            self.assertIn("Pxx Pyy Pzz Pxy Pxz Pyz", contents)
            data_lines = [
                line
                for line in contents.splitlines()
                if line and not line.startswith(("#", "@"))
            ]
            self.assertEqual(len(data_lines), 1)
            self.assertEqual(len(data_lines[0].split()), 9)

    def test_mdrun_records_only_requested_production_samples(self):
        system, box = harmonic_bond_system()
        topology = openmm.app.Topology()
        chain = topology.addChain("A")
        residue = topology.addResidue("MOL", chain)
        atom_a = topology.addAtom("A", openmm.app.Element.getBySymbol("He"), residue)
        atom_b = topology.addAtom("B", openmm.app.Element.getBySymbol("He"), residue)
        topology.addBond(atom_a, atom_b)
        topology.setPeriodicBoxVectors(
            [openmm.Vec3(*row) * unit.nanometer for row in box]
        )
        positions = np.asarray([[1.0, 1.0, 1.0], [1.6, 1.8, 1.0]])

        parameters = deepcopy(parameters_dict_template)
        parameters.update(
            {
                "nsteps": 2,
                "dt": 0.001,
                "minimize": False,
                "comm_mode": "none",
                "pcoulp": False,
                "gen_vel": True,
                "initial_temperature": 300.0,
                "production_temperature": 300.0,
                "nst_xout": 0,
                "nst_cp": 0,
                "nst_screenlog": 0,
                "nst_filelog": 0,
                "nst_stress": 1,
                "stress_strain": 1.0e-5,
                "stress_platform": "Reference",
            }
        )
        pdb_raw = SimpleNamespace(
            record_names=["ATOM", "ATOM"],
            serial_numbers=[1, 2],
            atom_names=["A", "B"],
            residue_names=["MOL", "MOL"],
            chain_IDs=["A", "A"],
            residue_sequence_numbers=[1, 1],
            occupancys=[1.0, 1.0],
            bfactors=[0.0, 0.0],
            elements=["He", "He"],
            molecule_length_list=[2],
        )
        runtime = {
            "parameters": parameters,
            "mdsystem": system,
            "mdtopology": topology,
            "positions": positions * unit.nanometer,
            "pdb_raw": pdb_raw,
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            run_input = directory / "run.tpr"
            output_prefix = directory / "run"
            with run_input.open("wb") as handle:
                pickle.dump(runtime, handle)
            mdrun(
                SimpleNamespace(
                    run_input=str(run_input),
                    output_prefix=str(output_prefix),
                    checkpoint=None,
                )
            )

            stress_output = Path(f"{output_prefix}.stress.xvg")
            contents = stress_output.read_text(encoding="utf-8")
            self.assertIn("pressure_coupling=False", contents)
            data_lines = [
                line
                for line in contents.splitlines()
                if line and not line.startswith(("#", "@"))
            ]
            self.assertEqual(len(data_lines), 2)


if __name__ == "__main__":
    unittest.main()
