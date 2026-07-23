import pickle
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


def _runtime(nsteps, checkpoint_interval):
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
            "filelog_grps": "step,potentialEnergy",
            "nst_stress": 0,
        }
    )
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
    def test_checkpoint_argument_can_use_output_prefix(self):
        args = getargs_mdrun(["-s", "run.tpr", "-o", "run", "-cpi"])
        self.assertEqual(args.checkpoint, "auto")

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


if __name__ == "__main__":
    unittest.main()
