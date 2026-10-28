import pickle
import shutil
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import openmm
from openmm import unit
from MDAnalysis.lib.formats.libmdaxdr import XTCFile

from dropps.commands.mdrun import getargs_mdrun, mdrun
from dropps.share.parameters import parameters_dict_template
from dropps.share.energy_reporter import read_edr


def _runtime(nsteps, checkpoint_interval, **parameter_overrides):
    system = openmm.System()
    atom_count = 10
    for _ in range(atom_count):
        system.addParticle(100.0)
    box = np.diag([4.0, 4.0, 4.0])
    system.setDefaultPeriodicBoxVectors(
        *(openmm.Vec3(*row) * unit.nanometer for row in box)
    )

    topology = openmm.app.Topology()
    chain = topology.addChain("A")
    residue = topology.addResidue("MOL", chain)
    for index in range(atom_count):
        topology.addAtom(
            f"A{index}", openmm.app.Element.getBySymbol("He"), residue
        )
    topology.setPeriodicBoxVectors(
        [openmm.Vec3(*row) * unit.nanometer for row in box]
    )

    parameters = deepcopy(parameters_dict_template)
    parameters.update(
        {
            "nsteps": nsteps,
            "dt": 0.001,
            "minimize": False,
            "comm_mode": "none",
            "pcoulp": False,
            "gen_vel": True,
            "initial_temperature": 300.0,
            "production_temperature": 300.0,
            "nst_xout": 1,
            "nst_cp": checkpoint_interval,
            "nst_screenlog": 0,
            "nst_filelog": 1,
            "nst_energy": 1,
            "energy_grps": "potentialEnergy,temperature,boxX,boxY,boxZ,volume",
            "nst_stress": 0,
        }
    )
    parameters.update(parameter_overrides)
    pdb_raw = SimpleNamespace(
        record_names=["ATOM"] * atom_count,
        serial_numbers=list(range(1, atom_count + 1)),
        atom_names=[f"A{index}" for index in range(atom_count)],
        residue_names=["MOL"] * atom_count,
        chain_IDs=["A"] * atom_count,
        residue_sequence_numbers=[1] * atom_count,
        occupancys=[1.0] * atom_count,
        bfactors=[0.0] * atom_count,
        elements=["He"] * atom_count,
        molecule_length_list=[atom_count],
    )
    return {
        "parameters": parameters,
        "mdsystem": system,
        "mdtopology": topology,
        "positions": np.asarray(
            [[1.0 + 0.1 * index, 1.0, 1.0] for index in range(atom_count)]
        ) * unit.nanometer,
        "pdb_raw": pdb_raw,
    }


class MdrunResumeTests(unittest.TestCase):
    def test_fresh_run_rejects_missing_initial_velocities(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            run_input = directory / "run.tpr"
            with run_input.open("wb") as handle:
                pickle.dump(
                    _runtime(nsteps=0, checkpoint_interval=0, gen_vel=False),
                    handle,
                )
            with self.assertRaisesRegex(ValueError, "do not contain velocities"):
                mdrun(
                    SimpleNamespace(
                        run_input=str(run_input),
                        output_prefix=str(directory / "run"),
                        checkpoint=None,
                    )
                )

    def test_checkpoint_argument_can_use_output_prefix(self):
        args = getargs_mdrun(["-s", "run.tpr", "-o", "run", "-cpi"])
        self.assertEqual(args.checkpoint, "auto")

    def test_warming_steps_are_excluded_from_production_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            run_input = directory / "run.tpr"
            output_prefix = directory / "run"

            runtime = _runtime(
                nsteps=2,
                checkpoint_interval=0,
                initial_temperature=299.0,
                production_temperature=300.0,
                # With dt=0.001 ps this gives one integration step per K.
                warming_speed=1_000_000,
            )
            with run_input.open("wb") as handle:
                pickle.dump(runtime, handle)

            mdrun(
                SimpleNamespace(
                    run_input=str(run_input),
                    output_prefix=str(output_prefix),
                    checkpoint=None,
                )
            )

            with XTCFile(f"{output_prefix}.xtc", "r") as trajectory:
                trajectory_steps = [frame.step for frame in trajectory]
                trajectory_times = [frame.time for frame in trajectory]
            self.assertEqual(trajectory_steps, [0, 1, 2])
            np.testing.assert_allclose(trajectory_times, [0.0, 0.001, 0.002])

            log_lines = Path(f"{output_prefix}.log").read_text(
                encoding="utf-8"
            ).splitlines()
            logged_steps = []
            for line in log_lines:
                token = line.split(",", 1)[0]
                try:
                    logged_steps.append(int(token))
                except ValueError:
                    pass
            self.assertEqual(logged_steps, [1, 2])

            energy_data = read_edr(f"{output_prefix}.edr")
            np.testing.assert_array_equal(energy_data.steps, [1, 2])
            np.testing.assert_allclose(energy_data.times_ps, [0.001, 0.002])
            self.assertFalse(Path(f"{output_prefix}.warming.xtc").exists())

    def test_optional_warming_trajectory_writes_one_frame_per_kelvin(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            run_input = directory / "run.tpr"
            output_prefix = directory / "run"

            runtime = _runtime(
                nsteps=0,
                checkpoint_interval=0,
                initial_temperature=298.0,
                production_temperature=300.0,
                warming_speed=1_000_000,
                keep_warming_trajectory=True,
            )
            with run_input.open("wb") as handle:
                pickle.dump(runtime, handle)

            mdrun(
                SimpleNamespace(
                    run_input=str(run_input),
                    output_prefix=str(output_prefix),
                    checkpoint=None,
                )
            )

            with XTCFile(
                f"{output_prefix}.warming.xtc", "r"
            ) as trajectory:
                warming_steps = [frame.step for frame in trajectory]
                warming_times = [frame.time for frame in trajectory]
            self.assertEqual(warming_steps, [1, 2])
            np.testing.assert_allclose(
                warming_times,
                [0.001, 0.002],
            )

            with XTCFile(f"{output_prefix}.xtc", "r") as trajectory:
                production_steps = [frame.step for frame in trajectory]
            self.assertEqual(production_steps, [0])

    def test_resume_runs_remaining_steps_and_rolls_back_newer_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            run_input = directory / "run.tpr"
            output_prefix = directory / "run"

            with run_input.open("wb") as handle:
                pickle.dump(_runtime(nsteps=2, checkpoint_interval=1), handle)
            mdrun(
                SimpleNamespace(
                    run_input=str(run_input),
                    output_prefix=str(output_prefix),
                    checkpoint=None,
                )
            )
            retained_checkpoint = directory / "step-2.chk"
            shutil.copyfile(f"{output_prefix}.chk", retained_checkpoint)

            # Advance outputs to step 3 while deliberately retaining the step-2
            # checkpoint, as can happen after an ungraceful process exit.
            with run_input.open("wb") as handle:
                pickle.dump(_runtime(nsteps=3, checkpoint_interval=0), handle)
            mdrun(
                SimpleNamespace(
                    run_input=str(run_input),
                    output_prefix=str(output_prefix),
                    checkpoint="auto",
                )
            )
            shutil.copyfile(retained_checkpoint, f"{output_prefix}.chk")

            with run_input.open("wb") as handle:
                pickle.dump(_runtime(nsteps=4, checkpoint_interval=1), handle)
            mdrun(
                SimpleNamespace(
                    run_input=str(run_input),
                    output_prefix=str(output_prefix),
                    checkpoint="auto",
                )
            )

            with XTCFile(f"{output_prefix}.xtc", "r") as trajectory:
                trajectory_steps = [frame.step for frame in trajectory]
            self.assertEqual(trajectory_steps, [0, 1, 2, 3, 4])

            log_lines = Path(f"{output_prefix}.log").read_text(
                encoding="utf-8"
            ).splitlines()
            logged_steps = []
            for line in log_lines:
                token = line.split(",", 1)[0]
                try:
                    logged_steps.append(int(token))
                except ValueError:
                    pass
            self.assertEqual(logged_steps, [1, 2, 3, 4])

            energy_data = read_edr(f"{output_prefix}.edr")
            np.testing.assert_array_equal(energy_data.steps, [1, 2, 3, 4])
            self.assertEqual(
                energy_data.header.fields,
                (
                    "potentialEnergy",
                    "temperature",
                    "boxX",
                    "boxY",
                    "boxZ",
                    "volume",
                ),
            )
            np.testing.assert_allclose(energy_data.values["volume"], 64.0)

    def test_noappend_writes_part_files_and_keeps_latest_checkpoint_name(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            run_input = directory / "run.tpr"
            output_prefix = directory / "run"

            with run_input.open("wb") as handle:
                pickle.dump(_runtime(nsteps=1, checkpoint_interval=0), handle)
            mdrun(
                SimpleNamespace(
                    run_input=str(run_input),
                    output_prefix=str(output_prefix),
                    checkpoint=None,
                )
            )

            with run_input.open("wb") as handle:
                pickle.dump(_runtime(nsteps=2, checkpoint_interval=0), handle)
            mdrun(
                SimpleNamespace(
                    run_input=str(run_input),
                    output_prefix=str(output_prefix),
                    checkpoint="auto",
                    append=False,
                )
            )

            with XTCFile(f"{output_prefix}.xtc", "r") as trajectory:
                base_steps = [frame.step for frame in trajectory]
            with XTCFile(
                f"{output_prefix}.part0002.xtc", "r"
            ) as trajectory:
                continuation_steps = [frame.step for frame in trajectory]

            self.assertEqual(base_steps, [0, 1])
            self.assertEqual(continuation_steps, [1, 2])
            self.assertTrue(Path(f"{output_prefix}.part0002.log").is_file())
            self.assertTrue(Path(f"{output_prefix}.part0002.edr").is_file())
            self.assertTrue(Path(f"{output_prefix}.part0002.pdb").is_file())
            self.assertTrue(Path(f"{output_prefix}.chk").is_file())
            self.assertTrue(Path(f"{output_prefix}.state.xml").is_file())
            self.assertTrue(Path(f"{output_prefix}.restart.json").is_file())
            self.assertFalse(Path(f"{output_prefix}.part0002.chk").exists())

    def test_invalid_native_checkpoint_falls_back_to_portable_state(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            run_input = directory / "run.tpr"
            output_prefix = directory / "run"

            with run_input.open("wb") as handle:
                pickle.dump(_runtime(nsteps=1, checkpoint_interval=0), handle)
            mdrun(
                SimpleNamespace(
                    run_input=str(run_input),
                    output_prefix=str(output_prefix),
                    checkpoint=None,
                )
            )

            Path(f"{output_prefix}.chk").write_bytes(b"not an OpenMM checkpoint")
            with run_input.open("wb") as handle:
                pickle.dump(_runtime(nsteps=2, checkpoint_interval=0), handle)
            mdrun(
                SimpleNamespace(
                    run_input=str(run_input),
                    output_prefix=str(output_prefix),
                    checkpoint="auto",
                )
            )

            with XTCFile(f"{output_prefix}.xtc", "r") as trajectory:
                trajectory_steps = [frame.step for frame in trajectory]
            self.assertEqual(trajectory_steps, [0, 1, 2])

    def test_nsteps_command_line_override_changes_production_target(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            run_input = directory / "run.tpr"
            output_prefix = directory / "run"
            with run_input.open("wb") as handle:
                pickle.dump(_runtime(nsteps=3, checkpoint_interval=0), handle)

            mdrun(
                SimpleNamespace(
                    run_input=str(run_input),
                    output_prefix=str(output_prefix),
                    checkpoint=None,
                    nsteps=1,
                )
            )

            with XTCFile(f"{output_prefix}.xtc", "r") as trajectory:
                trajectory_steps = [frame.step for frame in trajectory]
            self.assertEqual(trajectory_steps, [0, 1])


if __name__ == "__main__":
    unittest.main()
