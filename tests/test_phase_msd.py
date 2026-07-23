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

from dropps.analysis.phase_msd import getargs_phase_msd, phase_msd


class PhaseMsdCommandTests(unittest.TestCase):
    def test_planar_phase_msd_removes_collective_phase_translation(self):
        rng = np.random.default_rng(731)
        particle_count = 240
        target_count = 8
        box_lengths = np.asarray([4.0, 4.0, 20.0])
        box = np.diag(box_lengths)

        system = openmm.System()
        for _ in range(particle_count):
            system.addParticle(100.0)
        system.setDefaultPeriodicBoxVectors(
            *(openmm.Vec3(*row) * unit.nanometer for row in box)
        )

        topology = openmm.app.Topology()
        helium = openmm.app.Element.getBySymbol("He")
        itp_list = []
        for atom_index in range(particle_count):
            chain = topology.addChain()
            residue = topology.addResidue("SLB", chain)
            topology.addAtom(f"B{atom_index}", helium, residue)
            atom = SimpleNamespace(
                charge=0.0,
                mass=100.0,
                name=f"B{atom_index}",
                abbr="B",
                residueid=1,
                residuename="SLB",
            )
            itp_list.append(SimpleNamespace(molecule_name="slab", atoms=[atom]))
        topology.setPeriodicBoxVectors(
            [openmm.Vec3(*row) * unit.nanometer for row in box]
        )

        target_z = np.asarray([9.0, 9.5, 10.5, 11.0, 2.0, 3.0, 17.0, 18.0])
        dense_background = np.linspace(7.05, 12.95, 184)
        dilute_background = np.concatenate(
            (np.linspace(0.15, 6.85, 24), np.linspace(13.15, 19.85, 24))
        )
        initial_positions = np.column_stack(
            (
                rng.uniform(0.0, box_lengths[0], particle_count),
                rng.uniform(0.0, box_lengths[1], particle_count),
                np.concatenate((target_z, dense_background, dilute_background)),
            )
        )
        runtime = {
            "parameters": {"vdwtype": "none"},
            "mdsystem": system,
            "mdtopology": topology,
            "positions": initial_positions * unit.nanometer,
            "pdb_raw": "",
            "ITP_list": itp_list,
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            tpr_path = directory / "slab.tpr"
            xtc_path = directory / "slab.xtc"
            ndx_path = directory / "phase_msd.ndx"
            output_path = directory / "phase_msd.xvg"
            with tpr_path.open("wb") as handle:
                pickle.dump(runtime, handle)
            ndx_path.write_text(
                "[ reference ]\n"
                + " ".join(str(index) for index in range(particle_count))
                + "\n[ targets ]\n"
                + " ".join(str(index) for index in range(target_count))
                + "\n",
                encoding="utf-8",
            )

            integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
            simulation = openmm.app.Simulation(
                topology,
                system,
                integrator,
                openmm.Platform.getPlatformByName("Reference"),
            )
            reporter = openmm.app.XTCReporter(str(xtc_path), 1)
            for frame_index in range(12):
                positions = initial_positions.copy()
                positions[:target_count, 0] = np.mod(
                    positions[:target_count, 0] + 0.35 * frame_index,
                    box_lengths[0],
                )
                positions[target_count:, 2] = np.mod(
                    positions[target_count:, 2]
                    + rng.normal(scale=0.015, size=particle_count - target_count),
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

            args = getargs_phase_msd(
                [
                    "-s",
                    str(tpr_path),
                    "-f",
                    str(xtc_path),
                    "-n",
                    str(ndx_path),
                    "-o",
                    str(output_path),
                    "-ref",
                    "0",
                    "-sel",
                    "1",
                    "-t",
                    "xy",
                    "--bin-width",
                    "0.5",
                    "--blocks",
                    "3",
                ]
            )
            phase_msd(args)

            csv_path = output_path.with_suffix(".csv")
            profile_path = output_path.with_name("phase_msd.profile.xvg")
            self.assertTrue(output_path.is_file())
            self.assertTrue(csv_path.is_file())
            self.assertTrue(profile_path.is_file())
            self.assertIn("intrinsic_msd", output_path.read_text(encoding="utf-8"))
            self.assertIn(
                "dense_cutoff_nm=", profile_path.read_text(encoding="utf-8")
            )

            with csv_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            combinations = {
                (row["group"], row["phase"], row["conditioning"])
                for row in rows
            }
            self.assertEqual(
                combinations,
                {
                    ("group1", "dense", "continuous"),
                    ("group1", "dense", "intermittent"),
                    ("group1", "dilute", "continuous"),
                    ("group1", "dilute", "intermittent"),
                },
            )
            lag_one = [
                row
                for row in rows
                if row["phase"] == "dense"
                and row["conditioning"] == "continuous"
                and row["lag_frames"] == "1"
            ][0]
            raw_msd = float(lag_one["raw_msd_nm2"])
            intrinsic_msd = float(lag_one["intrinsic_msd_nm2"])
            translation_msd = float(lag_one["translation_msd_nm2"])
            self.assertGreater(raw_msd, 0.1)
            self.assertLess(intrinsic_msd, 1.0e-8)
            self.assertAlmostEqual(raw_msd, translation_msd, places=7)
            self.assertEqual(int(lag_one["mean_particles_per_origin"]), 4)


if __name__ == "__main__":
    unittest.main()
