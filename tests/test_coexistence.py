import csv
import pickle
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import openmm
import openmm.app
from openmm import unit

from dropps.analysis.coexistence import getargs_coexistence, coexistence


class CoexistenceCommandTests(unittest.TestCase):
    def test_slab_trajectory_writes_profile_fit_and_phase_summary(self):
        rng = np.random.default_rng(1215)
        particle_count = 240
        box_lengths = np.asarray([4.0, 4.0, 20.0])
        box = np.diag(box_lengths)

        system = openmm.System()
        for _ in range(particle_count):
            system.addParticle(100.0)
        system.setDefaultPeriodicBoxVectors(
            *(openmm.Vec3(*row) * unit.nanometer for row in box)
        )

        topology = openmm.app.Topology()
        chain = topology.addChain()
        helium = openmm.app.Element.getBySymbol("He")
        for atom_index in range(particle_count):
            residue = topology.addResidue("SLB", chain)
            topology.addAtom(f"B{atom_index}", helium, residue)
        topology.setPeriodicBoxVectors(
            [openmm.Vec3(*row) * unit.nanometer for row in box]
        )

        dense_count = 192
        dilute_count = particle_count - dense_count
        dense_z = np.linspace(7.05, 12.95, dense_count)
        dilute_z = np.concatenate(
            (
                np.linspace(0.15, 6.85, dilute_count // 2),
                np.linspace(13.15, 19.85, dilute_count - dilute_count // 2),
            )
        )
        initial_positions = np.column_stack(
            (
                rng.uniform(0.0, box_lengths[0], particle_count),
                rng.uniform(0.0, box_lengths[1], particle_count),
                np.concatenate((dense_z, dilute_z)),
            )
        )

        atom_records = [
            SimpleNamespace(
                charge=0.0,
                mass=100.0,
                name=f"B{atom_index}",
                abbr="B",
                residueid=atom_index + 1,
                residuename="SLB",
            )
            for atom_index in range(particle_count)
        ]
        runtime = {
            "parameters": {"vdwtype": "none"},
            "mdsystem": system,
            "mdtopology": topology,
            "positions": initial_positions * unit.nanometer,
            "pdb_raw": "",
            "ITP_list": [SimpleNamespace(molecule_name="slab", atoms=atom_records)],
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            tpr_path = directory / "slab.tpr"
            xtc_path = directory / "slab.xtc"
            output_path = directory / "coexistence.xvg"
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
            for frame_index, shift in enumerate(np.linspace(-2.0, 2.0, 15)):
                positions = initial_positions.copy()
                positions[:, 2] = np.mod(
                    positions[:, 2]
                    + shift
                    + rng.normal(scale=0.025, size=particle_count),
                    box_lengths[2],
                )
                simulation.context.setPositions(positions * unit.nanometer)
                simulation.context.setTime(frame_index * unit.picoseconds)
                simulation.currentStep = frame_index
                reporter.report(
                    simulation,
                    simulation.context.getState(getPositions=True),
                )
            del reporter
            del simulation
            del integrator

            args = getargs_coexistence(
                [
                    "-s",
                    str(tpr_path),
                    "-f",
                    str(xtc_path),
                    "-o",
                    str(output_path),
                    "-ref",
                    "0",
                    "-tp",
                    "number",
                    "--bin-width",
                    "0.5",
                    "--blocks",
                    "3",
                ]
            )
            coexistence(args)

            profile_text = output_path.read_text(encoding="utf-8")
            self.assertIn("interface_width_10_90_nm=", profile_text)
            self.assertIn("fit_r_squared=", profile_text)
            data_lines = [
                line
                for line in profile_text.splitlines()
                if line and not line.startswith(("#", "@"))
            ]
            self.assertGreater(len(data_lines), 12)
            self.assertTrue(all(len(line.split()) == 3 for line in data_lines))

            summary_path = output_path.with_suffix(".csv")
            summary_lines = [
                line
                for line in summary_path.read_text(encoding="utf-8").splitlines()
                if line and not line.startswith("#")
            ]
            rows = list(csv.DictReader(summary_lines))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["group"], "group0")
            self.assertGreater(float(rows[0]["dense_to_dilute_ratio"]), 4.0)
            self.assertGreater(float(rows[0]["dense_mean"]), float(rows[0]["dilute_mean"]))

            block_output_path = directory / "coexistence_block.xvg"
            block_args = getargs_coexistence(
                [
                    "-s",
                    str(tpr_path),
                    "-f",
                    str(xtc_path),
                    "-o",
                    str(block_output_path),
                    "-ref",
                    "0",
                    "-tp",
                    "number",
                    "--bin-width",
                    "0.5",
                    "--blocks",
                    "3",
                    "--center-mode",
                    "block",
                ]
            )
            coexistence(block_args)
            block_profile = block_output_path.read_text(encoding="utf-8")
            self.assertIn("center_mode=block", block_profile)
            self.assertIn("centering_groups=3", block_profile)

            none_output_path = directory / "coexistence_none.xvg"
            none_args = getargs_coexistence(
                [
                    "-s",
                    str(tpr_path),
                    "-f",
                    str(xtc_path),
                    "-o",
                    str(none_output_path),
                    "-ref",
                    "0",
                    "-tp",
                    "number",
                    "--bin-width",
                    "0.5",
                    "--blocks",
                    "3",
                    "--center-mode",
                    "none",
                    "--plateau-buffer",
                    "0.5",
                ]
            )
            coexistence(none_args)
            none_profile = none_output_path.read_text(encoding="utf-8")
            self.assertIn("center_mode=none", none_profile)
            self.assertIn("centering_groups=0", none_profile)


if __name__ == "__main__":
    unittest.main()
