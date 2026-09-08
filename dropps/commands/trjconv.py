"""Select, transform, and convert DROPPS trajectories."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from MDAnalysis.coordinates.PDB import PDBWriter
from MDAnalysis.coordinates.XTC import XTCWriter
from MDAnalysis.exceptions import NoDataError
from openmm.unit import nanometer
from tqdm import tqdm

from dropps.commands.trjconv_core import (
    TIME_UNIT_TO_PS,
    FitState,
    NoJumpState,
    build_whole_plan,
    center_coordinates,
    center_dense_phase,
    make_whole,
    select_time_indices,
    validate_cell,
    wrap_atoms,
    wrap_groups,
)
from dropps.fileio.pdb_reader import conect_lines_from_bond_dict
from dropps.share.argument_parser import ArgumentParser
from dropps.share.command_class import single_command
from dropps.share.trajectory import trajectory_class


prog = "trjconv"
desc = "Select, transform, and convert a DROPPS trajectory."
ANGSTROM_PER_NM = 10.0
SUPPORTED_EXTENSIONS = {".xtc", ".pdb"}


def getargs_trjconv(argv):
    parser = ArgumentParser(
        prog=prog,
        description=desc,
        epilog=(
            "Frame selection is time-based. Set -b and -e to the same value "
            "to write the saved frame nearest that time. Transform order is: "
            "make whole/no-jump, center, pack into the box, fit, translate."
        ),
    )
    parser.add_argument(
        "-s",
        "--run-input",
        required=True,
        help="Input DROPPS run file (.tpr) containing topology and coordinates.",
    )
    parser.add_argument(
        "-f",
        "--input",
        required=True,
        help="Input trajectory or structure file (.xtc or .pdb).",
    )
    parser.add_argument(
        "-n",
        "--index",
        help="Optional index file (.ndx) defining additional atom groups.",
    )
    parser.add_argument(
        "-sel",
        "--select",
        dest="selection",
        section="input",
        metavar="EXPR",
        help=(
            "DROPPS selection expression for atoms to write; when omitted, "
            "prompt interactively."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output trajectory or structure file (.xtc or .pdb).",
    )
    parser.add_argument(
        "-b",
        "--start-time",
        type=float,
        help="First trajectory time to write.",
    )
    parser.add_argument(
        "-e",
        "--end-time",
        type=float,
        help=(
            "Last trajectory time to write; set equal to -b to select the "
            "nearest saved frame."
        ),
    )
    parser.add_argument(
        "-dt",
        "--delta-time",
        type=float,
        help="Time interval between output frames, using nearest saved frames.",
    )
    parser.add_argument(
        "-tu",
        "--time-unit",
        choices=tuple(TIME_UNIT_TO_PS),
        default="ns",
        help="Unit used by -b, -e, and -dt.",
    )
    parser.add_argument(
        "-pbc",
        "--pbc",
        choices=("none", "whole", "atom", "res", "mol", "nojump"),
        default="none",
        help=(
            "Periodic-boundary treatment: whole reconstructs bonded molecules; "
            "atom packs atoms; res/mol pack residue or whole-molecule centers; "
            "nojump removes temporal box jumps."
        ),
    )
    parser.add_argument(
        "--center",
        choices=("none", "geometry", "mass", "dense"),
        default="none",
        help="Center the selected group geometrically, by mass, or as a dense slab.",
    )
    parser.add_argument(
        "--center-select",
        section="input",
        metavar="EXPR",
        help="Selection used for centering; defaults to the output selection.",
    )
    parser.add_argument(
        "--center-axis",
        choices=("x", "y", "z", "xyz"),
        help=(
            "Axes to center; defaults to xyz for geometry/mass and z for dense "
            "centering."
        ),
    )
    parser.add_argument(
        "-dpt",
        "--dense-phase-threshold",
        type=float,
        default=0.5,
        help="Dense bins must exceed this fraction of the maximum mass density.",
    )
    parser.add_argument(
        "--density-bin-width",
        type=float,
        default=0.05,
        help="Dense-phase histogram bin width in nm.",
    )
    parser.add_argument(
        "-fit",
        "--fit",
        choices=(
            "none",
            "translation",
            "transxy",
            "rot+trans",
            "rotxy+transxy",
            "progressive",
        ),
        default="none",
        help="Fit coordinates to the TPR or first processed frame.",
    )
    parser.add_argument(
        "--fit-select",
        section="input",
        metavar="EXPR",
        help="Selection used for fitting; defaults to the output selection.",
    )
    parser.add_argument(
        "--fit-reference",
        choices=("tpr", "first"),
        default="tpr",
        help="Reference coordinates for ordinary or progressive fitting.",
    )
    parser.add_argument(
        "--fit-weighting",
        choices=("mass", "uniform"),
        default="mass",
        help="Weights used to determine fit centers and rotations.",
    )
    parser.add_argument(
        "-trans",
        "--translate",
        type=float,
        nargs=3,
        metavar=("DX", "DY", "DZ"),
        help="Final constant Cartesian translation vector in nm.",
    )
    parser.add_argument(
        "-shift",
        "--shift",
        type=float,
        nargs=3,
        metavar=("DX", "DY", "DZ"),
        help="Additional translation in nm multiplied by the input frame index.",
    )
    parser.add_argument(
        "-ndec",
        "--precision",
        type=int,
        default=3,
        help="Number of decimal places used for XTC output precision.",
    )
    parser.add_argument(
        "-sep",
        "--separate",
        action="store_true",
        help="Write each selected frame to a separate numbered PDB file.",
    )
    parser.add_argument(
        "--zero-pad",
        type=int,
        default=6,
        help="Number of digits in filenames created by --separate.",
    )
    parser.add_argument(
        "--conect",
        action="store_true",
        help="Add reindexed topology bonds as CONECT records to PDB output.",
    )
    args = parser.parse_args(argv)
    _validate_cli_args(parser, args)
    return args


def _validate_cli_args(parser, args):
    input_extension = Path(args.input).suffix.lower()
    output_extension = Path(args.output).suffix.lower()
    if input_extension not in SUPPORTED_EXTENSIONS:
        parser.error("input extension must be .xtc or .pdb")
    if output_extension not in SUPPORTED_EXTENSIONS:
        parser.error("output extension must be .xtc or .pdb")
    if (
        Path(args.input).expanduser().resolve()
        == Path(args.output).expanduser().resolve()
    ):
        parser.error("input and output paths must be different")
    if args.delta_time is not None and args.delta_time <= 0.0:
        parser.error("-dt/--delta-time must be greater than zero")
    if args.precision < 0:
        parser.error("-ndec/--precision must not be negative")
    if args.zero_pad < 1:
        parser.error("--zero-pad must be at least 1")
    if args.separate and output_extension != ".pdb":
        parser.error("--separate is available only for PDB output")
    if args.conect and output_extension != ".pdb":
        parser.error("--conect is available only for PDB output")
    if args.center == "dense" and args.center_axis == "xyz":
        parser.error("dense-phase centering requires one axis: x, y, or z")
    if not 0.0 < args.dense_phase_threshold < 1.0:
        parser.error("--dense-phase-threshold must be between zero and one")
    if args.density_bin_width <= 0.0:
        parser.error("--density-bin-width must be greater than zero")


def _indices_by_key(keys):
    groups = {}
    for atom_index, key in enumerate(keys):
        groups.setdefault(key, []).append(atom_index)
    return [np.asarray(indices, dtype=np.int64) for indices in groups.values()]


def _molecule_groups(trajectory):
    return _indices_by_key(int(chain_id) for chain_id in trajectory.id2chainID)


def _residue_groups(trajectory):
    keys = zip(trajectory.id2chainID, trajectory.id2resID)
    return _indices_by_key((int(chain), int(residue)) for chain, residue in keys)


def _topology_bonds(trajectory):
    try:
        return np.asarray(trajectory.Universe.bonds.indices, dtype=np.int64)
    except NoDataError:
        return np.empty((0, 2), dtype=np.int64)


def _tpr_positions_angstrom(trajectory):
    positions = trajectory.tpr.positions
    try:
        values = positions.value_in_unit(nanometer)
        return np.asarray(values, dtype=np.float64) * ANGSTROM_PER_NM
    except AttributeError:
        return (
            np.asarray(
                [
                    [coordinate.value_in_unit(nanometer) for coordinate in row]
                    for row in positions
                ],
                dtype=np.float64,
            )
            * ANGSTROM_PER_NM
        )


def _frame_cell(timestep, operation):
    cell = timestep.triclinic_dimensions
    if cell is None:
        raise ValueError(f"{operation} requires periodic box dimensions.")
    return validate_cell(cell)


def _resolve_selection(trajectory, expression, purpose, *, interactive=False):
    if expression is None:
        if not interactive:
            raise ValueError(f"A selection is required for {purpose}.")
        trajectory.index.print_all()
        selection, _ = trajectory.getSelection_interactive(purpose)
    else:
        selection, _ = trajectory.getSelection(expression, purpose)
    if len(selection) == 0:
        raise ValueError(f"The selection for {purpose} is empty.")
    return selection


def _selected_bond_adjacency(selection_indices, bonds):
    local_index = {
        int(global_index): local
        for local, global_index in enumerate(
            np.asarray(selection_indices, dtype=np.int64)
        )
    }
    adjacency = {}
    for atom_a, atom_b in np.asarray(bonds, dtype=np.int64):
        local_a = local_index.get(int(atom_a))
        local_b = local_index.get(int(atom_b))
        if local_a is None or local_b is None:
            continue
        adjacency.setdefault(local_a, []).append(local_b)
        adjacency.setdefault(local_b, []).append(local_a)
    return adjacency


def _append_conect(path, conect_lines):
    if not conect_lines:
        return
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    insert_at = len(lines)
    for index in range(len(lines) - 1, -1, -1):
        if lines[index] == "END":
            insert_at = index
            break
    lines[insert_at:insert_at] = conect_lines
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class _OutputManager:
    def __init__(self, args, selection, frame_count, bonds):
        self.args = args
        self.selection = selection
        self.frame_count = frame_count
        self.bonds = bonds
        self.extension = Path(args.output).suffix.lower()
        self.writer = None
        self.paths = []
        adjacency = _selected_bond_adjacency(selection.indices, bonds)
        self.conect_lines = conect_lines_from_bond_dict(adjacency)

    def __enter__(self):
        if not self.args.separate:
            self.writer = self._new_writer(Path(self.args.output), self.frame_count > 1)
        return self

    def _new_writer(self, path, multiframe):
        self.paths.append(Path(path))
        if self.extension == ".xtc":
            return XTCWriter(
                str(path),
                n_atoms=len(self.selection),
                precision=self.args.precision,
            )
        return PDBWriter(
            str(path),
            n_atoms=len(self.selection),
            multiframe=multiframe,
            bonds=None,
            reindex=True,
        )

    def write(self, output_index):
        if self.args.separate:
            base = Path(self.args.output)
            suffix = f"{output_index:0{self.args.zero_pad}d}"
            path = base.with_name(f"{base.stem}_{suffix}{base.suffix}")
            writer = self._new_writer(path, False)
            try:
                writer.write(self.selection)
            finally:
                writer.close()
            if self.args.conect:
                _append_conect(path, self.conect_lines)
            return
        self.writer.write(self.selection)

    def __exit__(self, exc_type, exc, traceback):
        if self.writer is not None:
            self.writer.close()
        if exc_type is None and self.args.conect and not self.args.separate:
            _append_conect(self.args.output, self.conect_lines)
        return False


def _apply_center(args, positions, cell, center_indices, masses):
    if args.center == "none":
        return positions
    axis = args.center_axis or ("z" if args.center == "dense" else "xyz")
    if args.center == "dense":
        return center_dense_phase(
            positions,
            cell,
            center_indices,
            masses,
            axis=axis,
            threshold=args.dense_phase_threshold,
            bin_width=args.density_bin_width * ANGSTROM_PER_NM,
        )
    weights = masses if args.center == "mass" else None
    return center_coordinates(positions, cell, center_indices, axis, weights)


def _apply_packing(args, positions, cell, molecule_groups, residue_groups, masses):
    if args.pbc == "atom":
        return wrap_atoms(positions, cell)
    if args.pbc == "res":
        return wrap_groups(positions, cell, residue_groups, masses)
    if args.pbc == "mol":
        return wrap_groups(positions, cell, molecule_groups, masses)
    return positions


def _run_trjconv(args):
    trajectory = trajectory_class(args.run_input, args.index, args.input)
    universe = trajectory.Universe
    frame_times_ps = np.asarray(
        [float(timestep.time) for timestep in universe.trajectory],
        dtype=np.float64,
    )
    time_selection = select_time_indices(
        frame_times_ps,
        args.start_time,
        args.end_time,
        args.delta_time,
        args.time_unit,
    )
    output_indices = time_selection.indices
    if time_selection.single_time:
        actual = time_selection.times_ps[0] / TIME_UNIT_TO_PS[args.time_unit]
        print(
            f"## Equal -b/-e selected input frame {output_indices[0]} at "
            f"{actual:.12g} {args.time_unit}."
        )
    else:
        print(
            f"## Selected {output_indices.size} frame(s), from input frame "
            f"{output_indices[0]} to {output_indices[-1]}."
        )

    output_group = _resolve_selection(
        trajectory,
        args.selection,
        "trajectory output",
        interactive=True,
    )
    center_group = (
        _resolve_selection(trajectory, args.center_select, "centering")
        if args.center_select is not None
        else output_group
    )
    fit_group = (
        _resolve_selection(trajectory, args.fit_select, "trajectory fitting")
        if args.fit_select is not None
        else output_group
    )
    center_indices = np.asarray(center_group.indices, dtype=np.int64)
    fit_indices = np.asarray(fit_group.indices, dtype=np.int64)
    masses = np.asarray(universe.atoms.masses, dtype=np.float64)
    if not np.all(np.isfinite(masses)) or np.any(masses < 0.0):
        raise ValueError("Topology atom masses must be finite and non-negative.")

    molecule_groups = _molecule_groups(trajectory)
    residue_groups = _residue_groups(trajectory)
    bonds = _topology_bonds(trajectory)
    whole_plan = None
    if args.pbc in {"whole", "mol", "nojump"}:
        whole_plan = build_whole_plan(molecule_groups, bonds, universe.atoms.n_atoms)

    requires_cell = args.pbc != "none" or args.center != "none"
    first_timestep = universe.trajectory[0]
    first_cell = (
        _frame_cell(first_timestep, "The requested transformation")
        if requires_cell
        else None
    )
    tpr_reference = _tpr_positions_angstrom(trajectory)
    if tpr_reference.shape != (universe.atoms.n_atoms, 3):
        raise ValueError("TPR coordinate count does not match the trajectory topology.")

    nojump_state = None
    if args.pbc == "nojump":
        nojump_reference = make_whole(tpr_reference, first_cell, whole_plan)
        nojump_state = NoJumpState(nojump_reference, first_cell)

    fit_state = None
    fit_weights = None
    if args.fit != "none":
        fit_weights = (
            masses[fit_indices]
            if args.fit_weighting == "mass"
            else np.ones(fit_indices.size, dtype=np.float64)
        )
        if args.fit_reference == "tpr":
            reference = tpr_reference
            if args.pbc in {"whole", "mol", "nojump"}:
                reference = make_whole(reference, first_cell, whole_plan)
            fit_state = FitState(
                fit_indices,
                reference[fit_indices],
                args.fit,
                fit_weights,
            )
        if args.center != "none":
            print(
                "## WARNING: fitting follows centering and can move the fitted "
                "group away from the box center."
            )

    constant_translation = (
        np.zeros(3, dtype=np.float64)
        if args.translate is None
        else np.asarray(args.translate, dtype=np.float64) * ANGSTROM_PER_NM
    )
    frame_shift = (
        np.zeros(3, dtype=np.float64)
        if args.shift is None
        else np.asarray(args.shift, dtype=np.float64) * ANGSTROM_PER_NM
    )

    output_lookup = {
        int(frame_index): output_index
        for output_index, frame_index in enumerate(output_indices)
    }
    stateful = args.pbc == "nojump" or args.fit == "progressive"
    processing_start = 0 if args.pbc == "nojump" else int(output_indices[0])
    processing_indices = (
        range(processing_start, int(output_indices[-1]) + 1)
        if stateful
        else (int(index) for index in output_indices)
    )

    with _OutputManager(args, output_group, output_indices.size, bonds) as output:
        for frame_index in tqdm(processing_indices, desc="## Converting trajectory"):
            timestep = universe.trajectory[frame_index]
            positions = np.asarray(timestep.positions, dtype=np.float64).copy()
            cell = (
                _frame_cell(timestep, "The requested transformation")
                if requires_cell
                else None
            )

            if args.pbc in {"whole", "mol"}:
                positions = make_whole(positions, cell, whole_plan)
            elif args.pbc == "nojump":
                positions = nojump_state.apply(positions, cell)

            # Frames before -b are traversed only to establish no-jump state.
            if frame_index < int(output_indices[0]):
                continue

            positions = _apply_center(
                args,
                positions,
                cell,
                center_indices,
                masses,
            )
            positions = _apply_packing(
                args,
                positions,
                cell,
                molecule_groups,
                residue_groups,
                masses,
            )

            if args.fit != "none":
                if fit_state is None:
                    fit_state = FitState(
                        fit_indices,
                        positions[fit_indices],
                        args.fit,
                        fit_weights,
                    )
                positions = fit_state.apply(positions)

            positions = positions + constant_translation + frame_index * frame_shift

            output_index = output_lookup.get(frame_index)
            if output_index is None:
                continue
            universe.atoms.positions = positions
            output.write(output_index)

    print(
        f"## Wrote {output_indices.size} frame(s) and {len(output_group)} atom(s) "
        f"per frame to {args.output}."
    )
    return 0


def trjconv(args):
    try:
        return _run_trjconv(args)
    except (OSError, RuntimeError, ValueError, IndexError) as exc:
        print(f"dps trjconv: error: {exc}", file=sys.stderr)
        return 1


trjconv_commands = single_command("trjconv", getargs_trjconv, trjconv, desc)
