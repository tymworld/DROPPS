#!/usr/bin/env python3
"""Static validation for the generated DROPPS 1.0 Colab notebook suite."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
EXPECTED = [
    "00_DROPPS_1_0_Quickstart.ipynb",
    "01_DROPPS_1_0_Model_and_System_Builder.ipynb",
    "02_DROPPS_1_0_Simulation.ipynb",
    "03_DROPPS_1_0_Trajectory_Preparation.ipynb",
    "04_DROPPS_1_0_Phase_Separation_Analysis.ipynb",
]
REQUIRED_COMMANDS = {
    "pdb2dps",
    "genelastic",
    "addangle",
    "modifyres",
    "genmesh",
    "editconf",
    "grompp",
    "mdrun",
    "trjconv",
    "make_ndx",
    "check",
    "density",
    "contact",
    "cstat",
    "gyrate",
    "angle",
    "idist",
    "odist",
    "msd",
    "rmsd",
    "assembly",
}
FORBIDDEN = {
    "dropps-0.3.1",
    "pdb2cgps(",
    "from dropps.commands",
    "Namespace(",
}
REMOVED_COMMANDS = {"coexistence", "phase-msd", "surftension"}
PROTOCOL_MARKERS = {
    "00_DROPPS_1_0_Quickstart.ipynb": {
        'sequence = "FWFWFWFWFWFWFWFW"',
    },
    "01_DROPPS_1_0_Model_and_System_Builder.ipynb": {
        'component_a_sequence = "FWFWFWFWFWFWFWFW"',
        "component_a_count = 80",
        "component_b_enabled = False",
        "mesh_x = 5",
        "mesh_y = 5",
        "mesh_z = 5",
        'box_type = "cubic"',
    },
    "02_DROPPS_1_0_Simulation.ipynb": {
        "npt_duration_ns = 300.0",
        "npt_time_step_ps = 0.01",
        "slab_box_multiplier = 10.0",
        "nvt_duration_ns = 3000.0",
        "nvt_time_step_ps = 0.02",
        "lj_cutoff_nm = 1.5",
        "coulomb_cutoff_nm = 1.5",
        "pressure_coupling=True",
        "pressure_coupling=False",
        '"slab_nvt.xtc"',
    },
    "03_DROPPS_1_0_Trajectory_Preparation.ipynb": {
        'run_input_filename = "slab_nvt.tpr"',
        'trajectory_filename = "slab_nvt.xtc"',
        'if tpr_parameters.get("pcoulp")',
    },
    "04_DROPPS_1_0_Phase_Separation_Analysis.ipynb": {
        'run_input_filename = "slab_nvt.tpr"',
        'trajectory_filename = "processed.xtc"',
        'if tpr_parameters.get("pcoulp")',
        'density_axis = "z"',
        '"-o", "density.xvg", "-x", "z"',
        "if not density_ok",
        'output=WORKDIR / "density_z.png"',
    },
}


def main() -> None:
    errors = []
    all_sources = []
    checksums = {}
    checksum_path = HERE / "SHA256SUMS"
    if checksum_path.is_file():
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            digest, filename = line.split(maxsplit=1)
            checksums[filename.strip()] = digest

    for name in EXPECTED:
        path = HERE / name
        if not path.is_file():
            errors.append(f"missing notebook: {name}")
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"invalid JSON in {name}: {exc}")
            continue
        if document.get("nbformat") != 4:
            errors.append(f"{name}: expected nbformat 4")
        if not document.get("cells"):
            errors.append(f"{name}: contains no cells")

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if checksums.get(name) != digest:
            errors.append(f"{name}: SHA256SUMS mismatch")

        notebook_source = []
        form_count = 0
        for index, cell in enumerate(document.get("cells", [])):
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)
            notebook_source.append(source)
            if cell.get("cell_type") == "code":
                form_count += source.count("#@param")
                try:
                    ast.parse(source, filename=f"{name}:cell-{index}")
                except SyntaxError as exc:
                    errors.append(f"{name}:cell-{index}: {exc}")
        if form_count == 0:
            errors.append(f"{name}: no Colab form parameters found")
        joined_source = "\n".join(notebook_source)
        for marker in sorted(PROTOCOL_MARKERS.get(name, set())):
            if marker not in joined_source:
                errors.append(f"{name}: missing protocol marker: {marker}")
        expected_colab_url = (
            "https://colab.research.google.com/github/tymworld/DROPPS/blob/"
            f"v1.0.0/examples/colab/{name}"
        )
        if expected_colab_url not in joined_source:
            errors.append(f"{name}: missing versioned Open in Colab URL")
        release_wheel_url = (
            "https://github.com/tymworld/DROPPS/releases/download/v1.0.0/"
            "dropps-1.0-py3-none-any.whl"
        )
        if release_wheel_url not in joined_source:
            errors.append(f"{name}: missing immutable GitHub release wheel URL")
        all_sources.append(joined_source)

    combined = "\n".join(all_sources)
    for marker in sorted(FORBIDDEN):
        if marker in combined:
            errors.append(f"forbidden legacy marker present: {marker}")
    for command in sorted(REMOVED_COMMANDS):
        if re.search(rf"dps\(\s*['\"]{re.escape(command)}['\"]", combined):
            errors.append(f"removed CLI command invoked: {command}")

    command_pattern = re.compile(
        r"\"(" + "|".join(map(re.escape, REQUIRED_COMMANDS)) + r")\""
    )
    found_commands = set(command_pattern.findall(combined))
    missing_commands = REQUIRED_COMMANDS - found_commands
    if missing_commands:
        errors.append(
            "commands absent from suite: " + ", ".join(sorted(missing_commands))
        )

    if errors:
        raise SystemExit("Notebook validation failed:\n- " + "\n- ".join(errors))
    print(
        f"Validated {len(EXPECTED)} notebooks; Python syntax, form fields, "
        f"checksums, and {len(REQUIRED_COMMANDS)} current CLI commands are present."
    )


if __name__ == "__main__":
    main()
