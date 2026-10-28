"""Generate a reproducible simulation-methods report from a DROPPS TPR."""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from dropps.fileio.filename_control import validate_extension
from dropps.fileio.tpr_reader import read_tpr
from dropps.share.argument_parser import ArgumentParser
from dropps.share.command_class import single_command


prog = "report-methods"
desc = "Report the system, simulation methods, and provenance stored in a DROPPS TPR."


@dataclass(frozen=True)
class MethodsReport:
    sections: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]
    summary: str


def _parameter(parameters, name, default="unknown"):
    value = parameters.get(name, default)
    if value is None:
        return default
    return value


def _format_number(value):
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _topology_counts(topology):
    chains = list(topology.chains())
    residues = [residue for chain in chains for residue in chain.residues()]
    residue_counts = Counter(residue.name for residue in residues)
    composition = ", ".join(
        f"{name}={count}" for name, count in sorted(residue_counts.items())
    )
    return {
        "atoms": int(topology.getNumAtoms()),
        "chains": len(chains),
        "residues": len(residues),
        "bonds": sum(1 for _ in topology.bonds()),
        "composition": composition or "none",
    }


def _interaction_description(parameters, itp_list):
    lj_types = sorted(
        {
            str(getattr(topology, "function_type_LJ"))
            for topology in itp_list
            if getattr(topology, "function_type_LJ", None) is not None
        }
    )
    coulomb_types = sorted(
        {
            str(getattr(topology, "function_type_Coulomb"))
            for topology in itp_list
            if getattr(topology, "function_type_Coulomb", None) is not None
        }
    )
    lj_scheme = str(_parameter(parameters, "cutoff_scheme_lj"))
    if lj_scheme == "dynamic":
        lj_cutoff = (
            f"{_format_number(_parameter(parameters, 'cutoff_lj_multi'))} times "
            "pair sigma"
        )
    else:
        lj_cutoff = f"{_format_number(_parameter(parameters, 'cutoff_lj'))} nm"
    coulomb_type = str(_parameter(parameters, "coulombtype"))
    electrostatics = "disabled"
    if coulomb_type.lower() not in {"no", "none", "false"}:
        electrostatics = (
            f"{coulomb_type}; cutoff "
            f"{_format_number(_parameter(parameters, 'cutoff_coul'))} nm; "
            f"salt {_format_number(_parameter(parameters, 'salt_conc'))} mol/L"
        )
    return {
        "vdw_model": ", ".join(lj_types) or str(_parameter(parameters, "vdwtype")),
        "vdw_runtime": str(_parameter(parameters, "vdwtype")),
        "lj_scheme": lj_scheme,
        "lj_cutoff": lj_cutoff,
        "shift_lj": _format_number(_parameter(parameters, "shift_lj")),
        "coulomb_model": ", ".join(coulomb_types) or coulomb_type,
        "electrostatics": electrostatics,
        "shift_coul": _format_number(_parameter(parameters, "shift_coul")),
    }


def build_methods_report(content, source_path):
    parameters = content.parameters
    metadata = content.metadata
    counts = _topology_counts(content.mdtopology)
    interactions = _interaction_description(parameters, content.itp_list)

    dt_ps = float(_parameter(parameters, "dt", 0.0))
    nsteps = int(_parameter(parameters, "nsteps", 0))
    duration_ns = dt_ps * nsteps / 1000.0
    minimization_only = str(_parameter(parameters, "integrator")).lower() == "steep"
    pressure_coupling = bool(_parameter(parameters, "pcoulp", False))
    ensemble = "NPT" if pressure_coupling else "NVT"
    if pressure_coupling:
        ensemble_detail = (
            f"NPT; Monte Carlo barostat at "
            f"{_format_number(_parameter(parameters, 'ref_P'))} bar every "
            f"{_format_number(_parameter(parameters, 'tau_P'))} steps"
        )
    else:
        ensemble_detail = "NVT; fixed periodic box"

    molecule_types = sorted(
        {
            str(getattr(topology, "molecule_name"))
            for topology in content.itp_list
            if getattr(topology, "molecule_name", None)
        }
    )
    source_members = sorted(metadata.get("sources", {}))
    source_names = (
        ", ".join(Path(name).name for name in source_members) or "not embedded"
    )

    sections = (
        (
            "System",
            (
                ("Coarse-grained beads", str(counts["atoms"])),
                ("Chains/molecules", str(counts["chains"])),
                ("Residues", str(counts["residues"])),
                ("Topology bonds", str(counts["bonds"])),
                ("Constraints", str(content.mdsystem.getNumConstraints())),
                ("Residue composition", str(counts["composition"])),
                ("Molecule types", ", ".join(molecule_types) or "unknown"),
            ),
        ),
        (
            "Dynamics",
            (
                ("Integrator", str(_parameter(parameters, "integrator"))),
                (
                    "Integrator semantics",
                    str(metadata.get("integrator_semantics", "unknown")),
                ),
                ("Time step", f"{dt_ps:g} ps"),
                ("Production steps", str(nsteps)),
                ("Production duration", f"{duration_ns:g} ns"),
                (
                    "Production temperature",
                    f"{_format_number(_parameter(parameters, 'production_temperature'))} K",
                ),
                (
                    "Langevin friction",
                    f"{_format_number(_parameter(parameters, 'friction'))} ps^-1",
                ),
                ("Ensemble", ensemble_detail),
                ("Bond treatment", str(_parameter(parameters, "bondtype"))),
                ("Center-of-mass removal", str(_parameter(parameters, "comm_mode"))),
                (
                    "Energy minimization",
                    _format_number(_parameter(parameters, "minimize")),
                ),
                (
                    "Initial/warming temperature",
                    f"{_format_number(_parameter(parameters, 'initial_temperature'))} K",
                ),
            ),
        ),
        (
            "Interactions",
            (
                ("Topology van der Waals model", interactions["vdw_model"]),
                ("Runtime van der Waals mode", interactions["vdw_runtime"]),
                ("LJ cutoff scheme", interactions["lj_scheme"]),
                ("LJ cutoff", interactions["lj_cutoff"]),
                ("Shift LJ potential", interactions["shift_lj"]),
                ("Topology electrostatic model", interactions["coulomb_model"]),
                ("Runtime electrostatics", interactions["electrostatics"]),
                ("Shift electrostatic potential", interactions["shift_coul"]),
            ),
        ),
        (
            "Output",
            (
                (
                    "Trajectory interval",
                    f"{_format_number(_parameter(parameters, 'nst_xout'))} steps",
                ),
                (
                    "Energy interval",
                    f"{_format_number(_parameter(parameters, 'nst_energy'))} steps",
                ),
                ("Energy fields", str(_parameter(parameters, "energy_grps"))),
                (
                    "Stress-tensor interval",
                    f"{_format_number(_parameter(parameters, 'nst_stress'))} steps",
                ),
            ),
        ),
        (
            "Software and provenance",
            (
                ("Run input", os.path.abspath(os.fspath(source_path))),
                ("TPR format", f"DROPPS TPR v{content.format_version}"),
                ("DROPPS version", str(metadata.get("dropps_version", "unknown"))),
                ("OpenMM version", str(metadata.get("openmm_version", "unknown"))),
                ("Python version", str(metadata.get("python_version", "unknown"))),
                ("Run ID", content.run_id or "unknown"),
                ("System ID", content.system_id or "unknown"),
                ("Embedded source files", source_names),
                (
                    "Report scope",
                    "TPR configuration; command-line runtime overrides are not recorded",
                ),
            ),
        ),
    )

    if minimization_only:
        sections = tuple(
            (
                "Energy minimization",
                (
                    ("Method", "OpenMM LocalEnergyMinimizer (L-BFGS)"),
                    ("Maximum iterations", str(_parameter(parameters, "max_step"))),
                    ("Force tolerance", f"{_format_number(_parameter(parameters, 'forcetol'))} kJ mol^-1 nm^-1"),
                    ("Bond treatment", str(_parameter(parameters, "bondtype"))),
                ),
            ) if title == "Dynamics" else (
                "Output", (("Files", "Final PDB, checkpoint, portable State and restart metadata; no production trajectory"),)
            ) if title == "Output" else (title, rows)
            for title, rows in sections
        )

    software = (
        f"DROPPS {metadata.get('dropps_version', 'unknown')} with OpenMM "
        f"{metadata.get('openmm_version', 'unknown')}"
    )
    dynamics = (
        f"a {_parameter(parameters, 'integrator')} integrator with a {dt_ps:g} ps "
        f"time step at {_format_number(_parameter(parameters, 'production_temperature'))} K "
        f"and a friction coefficient of "
        f"{_format_number(_parameter(parameters, 'friction'))} ps^-1"
    )
    procedure = (
        "Energy minimization used OpenMM LocalEnergyMinimizer (L-BFGS), "
        f"with force tolerance {_format_number(_parameter(parameters, 'forcetol'))} "
        f"kJ mol^-1 nm^-1 and a maximum of {_parameter(parameters, 'max_step')} iterations "
        "(0 means unlimited). No production dynamics were requested. "
        if minimization_only else
        f"Dynamics used {dynamics}. Production was conducted in the "
        f"{ensemble} ensemble for {nsteps} steps ({duration_ns:g} ns). "
    )
    summary = (
        f"Simulations were performed using {software}. The system contained "
        f"{counts['atoms']} coarse-grained beads in {counts['chains']} "
        f"chains. {procedure}"
        f"Van der Waals interactions used {interactions['vdw_model']} with a "
        f"{interactions['lj_scheme']} cutoff ({interactions['lj_cutoff']}); "
        f"electrostatics were {interactions['electrostatics']}. Bonds were "
        f"treated as {_parameter(parameters, 'bondtype')}."
    )
    return MethodsReport(sections=sections, summary=summary)


def render_text(report):
    lines = ["DROPPS simulation methods report", ""]
    for title, rows in report.sections:
        lines.append(title)
        width = max(len(label) for label, _ in rows)
        lines.extend(f"  {label:<{width}} : {value}" for label, value in rows)
        lines.append("")
    lines.extend(("Methods summary", f"  {report.summary}", ""))
    return "\n".join(lines)


_LATEX_ESCAPES = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "\\": r"\textbackslash{}",
}


def _latex_escape(value):
    return "".join(_LATEX_ESCAPES.get(character, character) for character in str(value))


def render_latex(report):
    lines = [r"\section*{DROPPS simulation methods}"]
    for title, rows in report.sections:
        lines.extend(
            (f"\\subsection*{{{_latex_escape(title)}}}", r"\begin{description}")
        )
        lines.extend(
            f"\\item[{_latex_escape(label)}] {_latex_escape(value)}"
            for label, value in rows
        )
        lines.append(r"\end{description}")
    lines.extend(
        (
            r"\subsection*{Methods summary}",
            _latex_escape(report.summary),
            "",
        )
    )
    return "\n".join(lines)


def _write_text(path, contents):
    output_path = Path(path)
    output_path.write_text(contents, encoding="utf-8")
    return os.path.abspath(output_path)


def report_methods(args):
    content = read_tpr(args.run_input)
    report = build_methods_report(content, args.run_input)
    if args.output is None and args.latex_output is None:
        print(render_text(report), end="")
        return

    written = []
    if args.output is not None:
        output = validate_extension(args.output, "out")
        if Path(output).resolve() == Path(args.run_input).resolve():
            raise ValueError("Report output must not overwrite the run input.")
        written.append(_write_text(output, render_text(report)))
    if args.latex_output is not None:
        latex_output = validate_extension(args.latex_output, "tex")
        if Path(latex_output).resolve() == Path(args.run_input).resolve():
            raise ValueError("LaTeX output must not overwrite the run input.")
        written.append(_write_text(latex_output, render_latex(report)))
    print(f"## Wrote simulation methods report: {', '.join(written)}.")


def getargs_report_methods(argv):
    parser = ArgumentParser(prog=prog, description=desc)
    parser.add_argument(
        "-s",
        "--run-input",
        required=True,
        help="Input DROPPS run file (.tpr) containing the recorded methods.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help=(
            "Output unformatted methods report (.out); when neither output option "
            "is used, print to the terminal."
        ),
    )
    parser.add_argument(
        "-m",
        "--latex-output",
        help="Optional LaTeX-formatted methods report (.tex).",
    )
    return parser.parse_args(argv)


report_methods_commands = single_command(
    "report-methods", getargs_report_methods, report_methods, desc
)
