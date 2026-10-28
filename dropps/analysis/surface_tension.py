"""Surface tension of a planar slab from a DROPPS TPR and XTC trajectory."""

from __future__ import annotations

from dropps.share.argument_parser import ArgumentParser
from pathlib import Path

import numpy as np
from tqdm import tqdm

from dropps.analysis.density_core import (
    ANGSTROM_TO_NM,
    PS_TO_NS,
    select_frame_indices,
)
from dropps.analysis.surface_tension_core import (
    block_statistics,
    deform_configuration,
    require_orthorhombic_box,
    running_average,
    surface_tension_from_energy_pair,
)
from dropps.fileio.filename_control import validate_extension
from dropps.share.command_class import single_command
from dropps.share.openmm_energy import (
    create_energy_context,
    potential_energy_kj_per_mol,
)
from dropps.share.trajectory import trajectory_class


prog = "surftension"
desc = (
    "Calculate planar-interface surface tension from a DROPPS TPR and XTC trajectory."
)

def evaluate_frame_surface_tension(
    context,
    positions_nm,
    box_vectors_nm,
    axis,
    strain,
    interfaces,
):
    """Evaluate one trajectory frame by central finite differences."""

    positions_plus, box_plus = deform_configuration(
        positions_nm, box_vectors_nm, axis, strain
    )
    positions_minus, box_minus = deform_configuration(
        positions_nm, box_vectors_nm, axis, -strain
    )
    energy_plus = potential_energy_kj_per_mol(context, positions_plus, box_plus)
    energy_minus = potential_energy_kj_per_mol(context, positions_minus, box_minus)
    return surface_tension_from_energy_pair(
        energy_plus,
        energy_minus,
        strain,
        box_vectors_nm,
        axis=axis,
        interfaces=interfaces,
    )


def _select_frame_indices(trajectory, start_ns, end_ns, delta_ns):
    return select_frame_indices(
        trajectory,
        start_ns,
        end_ns,
        delta_ns,
        minimum_frames=2,
    )


def _write_xvg(output_path, times_ns, gamma_by_strain, statistics, metadata):
    strains = list(gamma_by_strain)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("# DROPPS planar-interface surface tension\n")
        handle.write("# Method: volume-preserving central finite difference\n")
        handle.write(
            f"# interface_normal={metadata['axis']} "
            f"interfaces={metadata['interfaces']} "
            f"platform={metadata['platform']}\n"
        )
        handle.write(
            "# The equilibrium-averaged kinetic pressure anisotropy is assumed "
            "to cancel.\n"
        )
        for strain in strains:
            stat = statistics[strain]
            handle.write(
                f"# strain={strain:.8g} mean_mN_per_m={stat['mean']:.10g} "
                f"block_sem_mN_per_m={stat['sem']:.10g} "
                f"blocks={stat['blocks']} frames={len(times_ns)}\n"
            )
        if len(strains) > 1:
            means = np.asarray(
                [statistics[strain]["mean"] for strain in strains], dtype=float
            )
            spread = float(np.ptp(means))
            relative_spread = spread / max(float(np.max(np.abs(means))), 1.0e-12)
            handle.write(
                f"# strain_mean_spread_mN_per_m={spread:.10g} "
                f"relative_spread={relative_spread:.10g}\n"
            )
        handle.write('@    title "Planar-interface surface tension"\n')
        handle.write('@    xaxis  label "Time (ns)"\n')
        handle.write('@    yaxis  label "Surface tension (mN/m)"\n')
        handle.write("@TYPE xy\n")
        series_id = 0
        for strain in strains:
            handle.write(f'@ s{series_id} legend "gamma, strain={strain:.6g}"\n')
            series_id += 1
            handle.write(f'@ s{series_id} legend "running mean, strain={strain:.6g}"\n')
            series_id += 1

        running = {
            strain: running_average(gamma_by_strain[strain]) for strain in strains
        }
        for frame_index, time_ns in enumerate(times_ns):
            fields = [f"{time_ns:.8f}"]
            for strain in strains:
                fields.extend(
                    [
                        f"{gamma_by_strain[strain][frame_index]:.10g}",
                        f"{running[strain][frame_index]:.10g}",
                    ]
                )
            handle.write("\t".join(fields) + "\n")


def getargs_surface_tension(argv):
    parser = ArgumentParser(prog=prog, description=desc)
    parser.add_argument(
        "-s",
        "--run-input",
        required=True,
        help="Input DROPPS run file (.tpr) containing the OpenMM system and topology.",
    )
    parser.add_argument(
        "-f",
        "--input",
        required=True,
        help="Input equilibrated planar-slab trajectory file (.xtc).",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output surface-tension time series (.xvg); the extension is added if omitted.",
    )
    parser.add_argument(
        "-b",
        "--start-time",
        type=float,
        help="First trajectory time to analyze, in ns.",
    )
    parser.add_argument(
        "-e",
        "--end-time",
        type=float,
        help="Last trajectory time to analyze, in ns.",
    )
    parser.add_argument(
        "-dt",
        "--delta-time",
        type=float,
        help="Approximate interval between analyzed frames, in ns.",
    )
    parser.add_argument(
        "--axis",
        choices=("x", "y", "z"),
        default="z",
        help="Interface-normal axis.",
    )
    parser.add_argument(
        "--strain",
        type=float,
        nargs="+",
        default=[3.0e-4],
        help=(
            "One or more positive finite-difference strains. Supplying several "
            "values provides a convergence check."
        ),
    )
    parser.add_argument(
        "--interfaces",
        type=int,
        default=2,
        help="Number of planar interfaces in the periodic cell.",
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=5,
        help="Number of contiguous blocks used for the standard error.",
    )
    parser.add_argument(
        "--platform",
        default="auto",
        help="OpenMM platform name, or auto for automatic selection.",
    )
    parser.add_argument(
        "--precision",
        choices=("single", "mixed", "double"),
        default="double",
        help="GPU/OpenCL energy precision.",
    )
    parser.add_argument(
        "--device",
        help="OpenMM GPU/OpenCL device index.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        help="Number of OpenMM CPU platform threads.",
    )
    return parser.parse_args(argv)


def surface_tension(args):
    strains = []
    for strain in args.strain:
        if not np.isfinite(strain) or strain <= 0:
            raise ValueError("Every finite-difference strain must be positive.")
        if strain not in strains:
            strains.append(strain)
    if args.interfaces <= 0:
        raise ValueError("Number of interfaces must be positive.")
    if args.blocks <= 0:
        raise ValueError("Number of blocks must be positive.")
    if args.threads is not None and args.threads <= 0:
        raise ValueError("Number of CPU threads must be positive.")

    trajectory = trajectory_class(args.run_input, trajectory_path=args.input)
    tpr = trajectory.tpr
    system = tpr.mdsystem
    if system.getNumConstraints() > 0:
        raise RuntimeError(
            "Finite-strain analysis does not yet include constraint virials. "
            "Use a trajectory generated with bondtype=bond."
        )
    if tpr.parameters.get("pcoulp", False):
        print(
            "## WARNING: This TPR has pressure coupling enabled. A fixed-box "
            "NVT production trajectory is recommended for slab surface tension."
        )
    for interaction, interaction_key, disabled_value, shift_key in (
        ("van der Waals", "vdwtype", "none", "shift_lj"),
        ("electrostatic", "coulombtype", "no", "shift_coul"),
    ):
        interaction_enabled = (
            str(tpr.parameters.get(interaction_key, "")).lower() != disabled_value
        )
        if interaction_enabled and tpr.parameters.get(shift_key) is False:
            print(
                f"## WARNING: The {interaction} potential is not shifted to "
                "zero at its cutoff. Verify strain convergence carefully because "
                "pair crossings can make finite differences noisy."
            )

    if trajectory.num_atoms() != system.getNumParticles():
        raise RuntimeError(
            f"Trajectory contains {trajectory.num_atoms()} particles, but the "
            f"TPR system contains {system.getNumParticles()}."
        )

    frame_indices, actual_interval_ns = _select_frame_indices(
        trajectory.Universe.trajectory,
        args.start_time,
        args.end_time,
        args.delta_time,
    )
    context, integrator, platform, properties = create_energy_context(
        system,
        platform_name=args.platform,
        precision=args.precision,
        device=args.device,
        threads=args.threads,
    )
    print(
        f"## Evaluating {len(frame_indices)} frames with median interval "
        f"{actual_interval_ns:g} ns on OpenMM {platform}."
    )
    if properties:
        print(f"## OpenMM platform properties: {properties}")

    times_ns = []
    gamma_lists = {strain: [] for strain in strains}
    try:
        for frame_index in tqdm(frame_indices, desc="## Surface tension"):
            timestep = trajectory.Universe.trajectory[frame_index]
            positions_nm = np.asarray(timestep.positions, dtype=float) * ANGSTROM_TO_NM
            box_angstrom = timestep.triclinic_dimensions
            if box_angstrom is None:
                raise RuntimeError(
                    f"Frame {frame_index} does not contain periodic box vectors."
                )
            box_nm = np.asarray(box_angstrom, dtype=float) * ANGSTROM_TO_NM
            require_orthorhombic_box(box_nm)
            times_ns.append(float(timestep.time) * PS_TO_NS)

            for strain in strains:
                gamma_lists[strain].append(
                    evaluate_frame_surface_tension(
                        context,
                        positions_nm,
                        box_nm,
                        args.axis,
                        strain,
                        args.interfaces,
                    )
                )
    finally:
        del context
        del integrator

    times_ns = np.asarray(times_ns, dtype=float)
    gamma_by_strain = {
        strain: np.asarray(values, dtype=float)
        for strain, values in gamma_lists.items()
    }
    statistics = {
        strain: block_statistics(values, args.blocks)
        for strain, values in gamma_by_strain.items()
    }

    output_path = validate_extension(args.output, "xvg")
    _write_xvg(
        output_path,
        times_ns,
        gamma_by_strain,
        statistics,
        {
            "axis": args.axis,
            "interfaces": args.interfaces,
            "platform": platform,
        },
    )

    print(f"## Surface-tension time series written to {Path(output_path).resolve()}.")
    for strain in strains:
        stat = statistics[strain]
        print(
            f"## strain={strain:.6g}: gamma = {stat['mean']:.8g} +/- "
            f"{stat['sem']:.4g} mN/m "
            f"(block SEM, {stat['blocks']} blocks)."
        )
    if len(strains) > 1:
        means = np.asarray(
            [statistics[strain]["mean"] for strain in strains], dtype=float
        )
        spread = float(np.ptp(means))
        relative_spread = spread / max(float(np.max(np.abs(means))), 1.0e-12)
        print(
            f"## Strain convergence: mean spread = {spread:.6g} mN/m "
            f"(relative spread {relative_spread:.3%})."
        )


surface_tension_commands = single_command(
    "surftension", getargs_surface_tension, surface_tension, desc
)
