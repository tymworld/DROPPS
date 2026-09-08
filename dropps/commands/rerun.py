"""Re-evaluate configurational observables from a DROPPS TPR and XTC."""

from __future__ import annotations

import os

import numpy as np
from openmm import unit
from tqdm import tqdm

from dropps.fileio.filename_control import validate_extension
from dropps.share.argument_parser import ArgumentParser
from dropps.share.command_class import single_command
from dropps.share.energy_reporter import EDRWriter, RERUN_FIELDS, normalize_field_names
from dropps.share.openmm_energy import (
    clone_energy_system,
    create_energy_context,
    potential_energy_kj_per_mol,
)
from dropps.share.stress_tensor_reporter import configurational_pressure_tensor
from dropps.share.time_selection import select_time_indices
from dropps.share.trajectory import trajectory_class


prog = "rerun"
desc = "Re-evaluate configurational observables from an XTC trajectory."

ANGSTROM_TO_NM = 0.1
_DALTON_PER_NM3_TO_G_PER_ML = 0.00166053906660
_CONFIG_PRESSURE_FIELDS = frozenset(
    (
        "configPressure",
        "configPxx",
        "configPyy",
        "configPzz",
        "configPxy",
        "configPxz",
        "configPyz",
    )
)
_DEFAULT_RERUN_FIELDS = (
    "potentialEnergy",
    "boxX",
    "boxY",
    "boxZ",
    "volume",
    "density",
)


def _select_frame_indices(trajectory, start_ps, end_ps, delta_ps):
    times = np.asarray([float(frame.time) for frame in trajectory], dtype=float)
    return select_time_indices(
        times,
        start_time=start_ps,
        end_time=end_ps,
        delta_time=delta_ps,
        time_unit="ps",
    ).indices


def rerun(args):
    fields = normalize_field_names(
        args.terms if args.terms else _DEFAULT_RERUN_FIELDS,
        available_fields=RERUN_FIELDS,
    )
    trajectory = trajectory_class(args.run_input, trajectory_path=args.input)
    system = trajectory.tpr.mdsystem
    if trajectory.num_atoms() != system.getNumParticles():
        raise ValueError(
            f"Trajectory contains {trajectory.num_atoms()} particles, but the "
            f"run input contains {system.getNumParticles()}."
        )
    needs_config_pressure = bool(_CONFIG_PRESSURE_FIELDS.intersection(fields))
    if needs_config_pressure and system.getNumConstraints() > 0:
        raise ValueError(
            "Configurational pressure rerun does not support constrained bonds "
            "because the constraint virial is unavailable."
        )

    frame_indices = _select_frame_indices(
        trajectory.Universe.trajectory,
        args.start_time,
        args.end_time,
        args.delta_time,
    )
    energy_system = clone_energy_system(system, remove_context_forces=True)
    device = None if args.device.lower() == "auto" else args.device
    threads = None if args.threads <= 0 else args.threads
    context, integrator, platform, properties = create_energy_context(
        energy_system,
        platform_name=args.platform,
        precision=args.precision,
        device=device,
        threads=threads,
    )
    total_mass_dalton = sum(
        system.getParticleMass(index).value_in_unit(unit.dalton)
        for index in range(system.getNumParticles())
    )
    output = validate_extension(args.output, "edr")
    metadata = {
        "source_run_input": os.path.abspath(args.run_input),
        "source_trajectory": os.path.abspath(args.input),
        "rerun": True,
        "contains_velocities": False,
        "pressure_kind": "configurational_only" if needs_config_pressure else None,
        "pressure_strain": args.strain if needs_config_pressure else None,
        "platform": platform,
        "platform_properties": properties,
    }
    writer = EDRWriter(output, fields, RERUN_FIELDS, metadata=metadata)
    dt_ps = float(trajectory.tpr.parameters["dt"])
    try:
        for frame_index in tqdm(frame_indices, desc="## Rerun"):
            timestep = trajectory.Universe.trajectory[frame_index]
            positions = np.asarray(timestep.positions, dtype=float) * ANGSTROM_TO_NM
            box_angstrom = timestep.triclinic_dimensions
            if box_angstrom is None:
                raise ValueError(f"Trajectory frame {frame_index} has no periodic box.")
            box = np.asarray(box_angstrom, dtype=float) * ANGSTROM_TO_NM
            lengths = np.linalg.norm(box, axis=1)
            volume = abs(float(np.linalg.det(box)))
            if volume <= 0.0:
                raise ValueError(
                    f"Trajectory frame {frame_index} has non-positive box volume."
                )
            values = {
                "boxX": lengths[0],
                "boxY": lengths[1],
                "boxZ": lengths[2],
                "volume": volume,
                "density": total_mass_dalton * _DALTON_PER_NM3_TO_G_PER_ML / volume,
            }
            if "potentialEnergy" in fields:
                values["potentialEnergy"] = potential_energy_kj_per_mol(
                    context, positions, box
                )
            if needs_config_pressure:
                tensor = configurational_pressure_tensor(
                    context, positions, box, strain=args.strain
                )
                values.update(
                    {
                        "configPressure": float(np.trace(tensor) / 3.0),
                        "configPxx": tensor[0, 0],
                        "configPyy": tensor[1, 1],
                        "configPzz": tensor[2, 2],
                        "configPxy": tensor[0, 1],
                        "configPxz": tensor[0, 2],
                        "configPyz": tensor[1, 2],
                    }
                )
            time_ps = float(timestep.time)
            step = int(round(time_ps / dt_ps)) if dt_ps > 0.0 else int(frame_index)
            writer.write(step, time_ps, values)
    finally:
        writer.close()
        del context
        del integrator
    print(
        f"## Re-evaluated {len(frame_indices)} frames and wrote "
        f"{os.path.abspath(output)}."
    )
    if needs_config_pressure:
        print(
            "## The configP* terms exclude kinetic and constraint virials; they "
            "must not be used as a full pressure tensor for Green-Kubo viscosity."
        )


def getargs_rerun(argv):
    parser = ArgumentParser(prog=prog, description=desc)
    parser.add_argument(
        "-s", "--run-input", required=True, help="Input DROPPS run file (.tpr)."
    )
    parser.add_argument(
        "-f", "--input", required=True, help="Input coordinate trajectory (.xtc)."
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Output rerun energy file (.edr)."
    )
    parser.add_argument(
        "--terms",
        nargs="+",
        help=(
            "Configurational terms to calculate. Defaults to potential energy, "
            "box lengths, volume, and density."
        ),
    )
    parser.add_argument(
        "-b", "--start-time", type=float, help="First trajectory time, in ps."
    )
    parser.add_argument(
        "-e", "--end-time", type=float, help="Last trajectory time, in ps."
    )
    parser.add_argument(
        "-dt", "--delta-time", type=float, help="Approximate frame interval, in ps."
    )
    parser.add_argument(
        "--strain",
        type=float,
        default=3.0e-4,
        help="Finite-difference strain for configurational pressure terms.",
    )
    parser.add_argument(
        "--platform", default="auto", help="OpenMM platform name, or auto."
    )
    parser.add_argument(
        "--precision",
        choices=("single", "mixed", "double"),
        default="double",
        help="OpenMM energy-evaluation precision.",
    )
    parser.add_argument(
        "--device", default="auto", help="CUDA/OpenCL device index, or auto."
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="CPU threads; 0 uses the OpenMM default.",
    )
    return parser.parse_args(argv)


rerun_commands = single_command(prog, getargs_rerun, rerun, desc)
