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

from dropps.analysis.exchange import exchange, getargs_exchange


class ExchangeCommandTests(unittest.TestCase):
    def test_planar_slab_tracks_confirmed_exchange_and_residence(self):
        rng = np.random.default_rng(2246)
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
        helium = openmm.app.Element.getBySymbol("He")
        atom_records = []
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
            atom_records.append(atom)
            itp_list.append(SimpleNamespace(molecule_name="slab", atoms=[atom]))
        topology.setPeriodicBoxVectors(
            [openmm.Vec3(*row) * unit.nanometer for row in box]
        )

        dense_background = np.linspace(7.05, 12.95, 192)
        dilute_background = np.concatenate(
            (np.linspace(0.15, 6.85, 23), np.linspace(13.15, 19.85, 23))
        )
        initial_positions = np.column_stack(
            (
                rng.uniform(0.0, box_lengths[0], particle_count),
                rng.uniform(0.0, box_lengths[1], particle_count),
                np.concatenate(([10.0, 3.0], dense_background, dilute_background)),
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

        tracer_dense_to_dilute = [10.0, 10.0, 10.0, 8.0, 7.5, 7.0, 6.5, 3.0, 3.0, 3.0, 3.0, 3.0]
        tracer_dilute_to_dense = list(reversed(tracer_dense_to_dilute))

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            tpr_path = directory / "slab.tpr"
            xtc_path = directory / "slab.xtc"
            ndx_path = directory / "exchange.ndx"
            output_prefix = directory / "exchange"
            with tpr_path.open("wb") as handle:
                pickle.dump(runtime, handle)
            ndx_path.write_text(
                "[ reference ]\n"
                + " ".join(str(index) for index in range(particle_count))
                + "\n[ tracers ]\n0 1\n",
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
            for frame_index, (first_z, second_z) in enumerate(
                zip(tracer_dense_to_dilute, tracer_dilute_to_dense)
            ):
                positions = initial_positions.copy()
                positions[0, 2] = first_z
                positions[1, 2] = second_z
                positions[2:, 2] = np.mod(
                    positions[2:, 2]
                    + rng.normal(scale=0.015, size=particle_count - 2),
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

            args = getargs_exchange(
                [
                    "-s",
                    str(tpr_path),
                    "-f",
                    str(xtc_path),
                    "-n",
                    str(ndx_path),
                    "-o",
                    str(output_prefix),
                    "-ref",
                    "0",
                    "-sel",
                    "1",
                    "--bin-width",
                    "0.5",
                    "--blocks",
                    "3",
                    "--center-mode",
                    "frame",
                ]
            )
            exchange(args)

            expected_paths = [
                Path(f"{output_prefix}.profile.xvg"),
                Path(f"{output_prefix}.states.csv"),
                Path(f"{output_prefix}.events.csv"),
                Path(f"{output_prefix}.residence.csv"),
                Path(f"{output_prefix}.survival.csv"),
                Path(f"{output_prefix}.summary.csv"),
            ]
            self.assertTrue(all(path.is_file() for path in expected_paths))

            profile = expected_paths[0].read_text(encoding="utf-8")
            self.assertIn("dense_cutoff_nm=", profile)
            self.assertIn("dilute_cutoff_nm=", profile)
            self.assertIn("center_mode=frame", profile)

            with expected_paths[2].open(encoding="utf-8", newline="") as handle:
                event_rows = list(csv.DictReader(handle))
            self.assertEqual(len(event_rows), 2)
            directions = {
                (row["source_phase"], row["destination_phase"])
                for row in event_rows
            }
            self.assertEqual(
                directions, {("dense", "dilute"), ("dilute", "dense")}
            )

            with expected_paths[1].open(encoding="utf-8", newline="") as handle:
                state_rows = list(csv.DictReader(handle))
            self.assertGreaterEqual(len(state_rows), 20)
            self.assertEqual(len(state_rows) % 2, 0)
            self.assertTrue(
                {"dense", "interface", "dilute"}.issubset(
                    {row["phase"] for row in state_rows}
                )
            )

            with expected_paths[5].open(encoding="utf-8", newline="") as handle:
                summary_rows = list(csv.DictReader(handle))
            self.assertEqual(len(summary_rows), 3)
            self.assertTrue(all(row["group"] == "group1" for row in summary_rows))
            bulk_exits = {
                row["phase"]: int(row["exits_to_opposite_bulk"])
                for row in summary_rows
            }
            self.assertEqual(bulk_exits["dense"], 1)
            self.assertEqual(bulk_exits["dilute"], 1)


if __name__ == "__main__":
    unittest.main()
