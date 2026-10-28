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

from dropps.analysis.gyrate import getargs_gyrate, gyrate


class GyrateCommandTests(unittest.TestCase):
    def test_phase_output_separates_dense_and_dilute_molecular_rg(self):
        rng = np.random.default_rng(812)
        target_molecules = 8
        target_atoms = 2 * target_molecules
        background_atoms = 224
        particle_count = target_atoms + background_atoms
        box_lengths = np.asarray([6.0, 6.0, 20.0])
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
        positions = []
        target_z = (9.0, 9.5, 10.5, 11.0, 2.0, 3.0, 17.0, 18.0)
        for molecule_id, center_z in enumerate(target_z):
            chain = topology.addChain()
            residue = topology.addResidue("PROT", chain)
            atoms = [
                topology.addAtom(f"B{molecule_id}_0", helium, residue),
                topology.addAtom(f"B{molecule_id}_1", helium, residue),
            ]
            topology.addBond(*atoms)
            half_separation = 0.2 if molecule_id < 4 else 0.6
            positions.extend(
                (
                    [3.0 - half_separation, 3.0, center_z],
                    [3.0 + half_separation, 3.0, center_z],
                )
            )
            records = [
                SimpleNamespace(
                    charge=0.0,
                    mass=100.0,
                    name=f"B{molecule_id}_{bead}",
                    abbr="B",
                    residueid=1,
                    residuename="PROT",
                )
                for bead in range(2)
            ]
            itp_list.append(SimpleNamespace(molecule_name="PROT", atoms=records))

        dense_background = np.linspace(7.05, 12.95, 180)
        dilute_background = np.concatenate(
            (np.linspace(0.15, 6.85, 22), np.linspace(13.15, 19.85, 22))
        )
        for atom_id, z_coordinate in enumerate(
            np.concatenate((dense_background, dilute_background))
        ):
            chain = topology.addChain()
            residue = topology.addResidue("SLB", chain)
            topology.addAtom(f"S{atom_id}", helium, residue)
            positions.append(
                [
                    rng.uniform(0.0, box_lengths[0]),
                    rng.uniform(0.0, box_lengths[1]),
                    z_coordinate,
                ]
            )
            record = SimpleNamespace(
                charge=0.0,
                mass=100.0,
                name=f"S{atom_id}",
                abbr="S",
                residueid=1,
                residuename="SLB",
            )
            itp_list.append(SimpleNamespace(molecule_name="SLB", atoms=[record]))

        positions = np.asarray(positions, dtype=float)
        topology.setPeriodicBoxVectors(
            [openmm.Vec3(*row) * unit.nanometer for row in box]
        )
        runtime = {
            "parameters": {"vdwtype": "none"},
            "mdsystem": system,
            "mdtopology": topology,
            "positions": positions * unit.nanometer,
            "pdb_raw": "",
            "ITP_list": itp_list,
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            tpr_path = directory / "slab.tpr"
            xtc_path = directory / "slab.xtc"
            ndx_path = directory / "gyrate.ndx"
            output_path = directory / "phase-rg.xvg"
            with tpr_path.open("wb") as handle:
                pickle.dump(runtime, handle)
            ndx_path.write_text(
                "[ reference ]\n"
                + " ".join(str(index) for index in range(particle_count))
                + "\n[ targets ]\n"
                + " ".join(str(index) for index in range(target_atoms))
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
            for frame_index in range(8):
                frame_positions = positions.copy()
                frame_positions[target_atoms:, 2] = np.mod(
                    frame_positions[target_atoms:, 2]
                    + rng.normal(scale=0.01, size=background_atoms),
                    box_lengths[2],
                )
                simulation.context.setPositions(frame_positions * unit.nanometer)
                simulation.context.setTime(frame_index * unit.picoseconds)
                simulation.currentStep = frame_index
                reporter.report(
                    simulation,
                    simulation.context.getState(getPositions=True),
                )
            del reporter
            del simulation
            del integrator

            args = getargs_gyrate(
                [
                    "-s",
                    str(tpr_path),
                    "-f",
                    str(xtc_path),
                    "-n",
                    str(ndx_path),
                    "-op",
                    str(output_path),
                    "-ref",
                    "0",
                    "-sel",
                    "1",
                    "--density-bin-width",
                    "0.5",
                    "--blocks",
                    "2",
                    "-bw",
                    "0.05",
                ]
            )
            gyrate(args)

            expected_paths = {
                "mean": output_path,
                "observations": directory / "phase-rg.observations.csv",
                "summary": directory / "phase-rg.summary.csv",
                "histogram": directory / "phase-rg.histogram.xvg",
                "profile": directory / "phase-rg.profile.xvg",
                "values": directory / "phase-rg.group01-group1.values.csv",
            }
            self.assertTrue(all(path.is_file() for path in expected_paths.values()))
            mean_text = expected_paths["mean"].read_text(encoding="utf-8")
            self.assertIn('legend "group1 dense"', mean_text)
            self.assertIn('legend "group1 dilute"', mean_text)
            profile_text = expected_paths["profile"].read_text(encoding="utf-8")
            self.assertIn("dense_cutoff_nm=", profile_text)
            self.assertIn("dilute_cutoff_nm=", profile_text)

            with expected_paths["summary"].open(encoding="utf-8", newline="") as handle:
                summary = {row["phase"]: row for row in csv.DictReader(handle)}
            self.assertEqual(set(summary), {"dense", "dilute"})
            self.assertAlmostEqual(float(summary["dense"]["mean_rg_nm"]), 0.2, places=2)
            self.assertAlmostEqual(
                float(summary["dilute"]["mean_rg_nm"]), 0.6, places=2
            )
            self.assertEqual(int(summary["dense"]["observations"]), 32)
            self.assertEqual(int(summary["dilute"]["observations"]), 32)

            with expected_paths["observations"].open(
                encoding="utf-8", newline=""
            ) as handle:
                observations = list(csv.DictReader(handle))
            self.assertEqual(len(observations), 8 * target_molecules)
            self.assertEqual(
                {row["phase"] for row in observations}, {"dense", "dilute"}
            )

            with expected_paths["values"].open(encoding="utf-8", newline="") as handle:
                phase_values = list(csv.DictReader(handle))
            dense_values = [
                float(row["dense_rg_nm"]) for row in phase_values if row["dense_rg_nm"]
            ]
            dilute_values = [
                float(row["dilute_rg_nm"])
                for row in phase_values
                if row["dilute_rg_nm"]
            ]
            self.assertEqual(len(dense_values), 32)
            self.assertEqual(len(dilute_values), 32)
            self.assertAlmostEqual(float(np.mean(dense_values)), 0.2, places=2)
            self.assertAlmostEqual(float(np.mean(dilute_values)), 0.6, places=2)


if __name__ == "__main__":
    unittest.main()
