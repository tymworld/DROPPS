#!/usr/bin/env python3
"""Generate the DROPPS 1.0 Google Colab tutorial suite.

The notebooks intentionally use the public ``dps`` command-line interface.  This
keeps them aligned with the manuscript, the generated CLI reference, and the
portable files that users move between Colab and HPC systems.
"""

from __future__ import annotations

import hashlib
import json
import textwrap
from pathlib import Path


HERE = Path(__file__).resolve().parent


def _source(text: str) -> str:
    return textwrap.dedent(text).strip("\n") + "\n"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _source(text),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(text),
    }


def notebook(cells: list[dict], *, gpu: bool = False) -> dict:
    colab = {"provenance": []}
    if gpu:
        colab["gpuType"] = "T4"
    return {
        "cells": cells,
        "metadata": {
            "colab": colab,
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


INSTALL_CELL = r"""
#@title Install DROPPS 1.0
#@markdown The default route installs the immutable wheel attached to the
#@markdown public GitHub `v1.0.0` release. Upload remains available for offline use.
installation_source = "Install from the DROPPS v1.0.0 GitHub release" #@param ["Install from the DROPPS v1.0.0 GitHub release", "Upload the DROPPS 1.0 wheel", "Install from another wheel URL"]
wheel_url = "https://github.com/tymworld/DROPPS/releases/download/v1.0.0/dropps-1.0-py3-none-any.whl" #@param {type:"string"}
accelerator_dependencies = "auto" #@param ["auto", "cuda12", "cuda13", "latest"]

import importlib.metadata
import re
import shutil
import subprocess
import sys
from pathlib import Path


def detect_install_extra():
    if accelerator_dependencies != "auto":
        return accelerator_dependencies
    if shutil.which("nvidia-smi"):
        probe = subprocess.run(
            ["nvidia-smi"], text=True, capture_output=True, check=False
        ).stdout
        match = re.search(r"CUDA Version:\s*(\d+)", probe)
        if match and int(match.group(1)) >= 13:
            return "cuda13"
        return "cuda12"
    return "latest"


if installation_source.startswith("Upload"):
    try:
        from google.colab import files
    except ImportError as exc:
        raise RuntimeError(
            "This upload form is intended for Google Colab. Set installation_source "
            "to the URL option when running elsewhere."
        ) from exc
    uploaded = files.upload()
    wheel_candidates = [Path(name) for name in uploaded if name.endswith(".whl")]
    if len(wheel_candidates) != 1:
        raise ValueError("Upload exactly one DROPPS 1.0 .whl file.")
    wheel_target = str(wheel_candidates[0].resolve())
else:
    if not wheel_url.strip():
        raise ValueError("Provide the published DROPPS 1.0 wheel URL.")
    wheel_target = wheel_url.strip()

extra = detect_install_extra()
if wheel_target.startswith(("https://", "http://")):
    package_spec = f"dropps[{extra}] @ {wheel_target}"
else:
    package_spec = f"{wheel_target}[{extra}]"
print(f"Installing {package_spec}")
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--upgrade",
        "--upgrade-strategy",
        "only-if-needed",
        package_spec,
    ],
    check=True,
)

version = importlib.metadata.version("dropps")
if version != "1.0":
    raise RuntimeError(f"Expected DROPPS 1.0, but installed {version}.")
subprocess.run(["dps", "--version"], check=True)
"""


HELPERS_CELL = r"""
#@title Shared notebook helpers
import hashlib
import json
import os
import shlex
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COMMAND_LOG = []


def run_command(arguments, *, cwd=None, input_text=None, check=True):
    command = [str(value) for value in arguments]
    print("$", shlex.join(command))
    result = subprocess.run(
        command,
        cwd=None if cwd is None else str(cwd),
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")
    COMMAND_LOG.append(
        {
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "cwd": str(Path(cwd or Path.cwd()).resolve()),
            "command": command,
            "returncode": result.returncode,
        }
    )
    if check and result.returncode:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {shlex.join(command)}"
        )
    return result


def dps(*arguments, cwd=None, input_text=None, check=True):
    return run_command(
        ["dps", *arguments], cwd=cwd, input_text=input_text, check=check
    )


def reset_task_directory(path):
    path = Path(path).resolve()
    if not path.name.startswith("dropps_"):
        raise ValueError(f"Refusing to reset unexpected directory: {path}")
    if path.exists():
        import shutil

        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract_zip(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
        bundle.extractall(destination)


def find_unique(root, basename):
    matches = [path for path in Path(root).rglob(basename) if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one file named {basename!r} below {root}, found {len(matches)}."
        )
    return matches[0]


def make_zip(paths, output_path, *, base=None):
    output_path = Path(output_path)
    base = Path(base or output_path.parent).resolve()
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for source in sorted({Path(path).resolve() for path in paths}):
            if source == output_path.resolve() or not source.is_file():
                continue
            try:
                arcname = source.relative_to(base)
            except ValueError:
                arcname = Path(source.name)
            bundle.write(source, arcname)
    print(f"Wrote {output_path} ({output_path.stat().st_size / 1024:.1f} KiB)")
    return output_path


def download(path):
    try:
        from google.colab import files
    except ImportError:
        print(f"Result available at {Path(path).resolve()}")
    else:
        files.download(str(path))


def optional_time_arguments(start, end, interval):
    arguments = []
    for option, value in (("-b", start), ("-e", end), ("-dt", interval)):
        if str(value).strip():
            arguments.extend([option, str(value).strip()])
    return arguments


def read_xvg(path):
    legends = {}
    metadata = {}
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for raw in stream:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("@"):
                re_module = __import__("re")
                match = re_module.search(r's(\d+)\s+legend\s+"(.*)"', line)
                if match:
                    legends[int(match.group(1)) + 1] = match.group(2)
                for key, pattern in {
                    "title": r'@\s+title\s+"(.*)"',
                    "xlabel": r'@\s+xaxis\s+label\s+"(.*)"',
                    "ylabel": r'@\s+yaxis\s+label\s+"(.*)"',
                }.items():
                    label_match = re_module.search(pattern, line)
                    if label_match:
                        metadata[key] = label_match.group(1)
                continue
            rows.append([float(value) for value in line.split()])
    data = np.asarray(rows, dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"No plottable data in {path}")
    return data, legends, metadata


def plot_xvg(path, *, title=None, output=None):
    data, legends, metadata = read_xvg(path)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for column in range(1, data.shape[1]):
        ax.plot(data[:, 0], data[:, column], label=legends.get(column, f"series {column}"))
    ax.set_title(title or metadata.get("title", Path(path).stem))
    ax.set_xlabel(metadata.get("xlabel", "x / time"))
    ax.set_ylabel(metadata.get("ylabel", "value"))
    if data.shape[1] > 2:
        ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    output_path = Path(output) if output is not None else Path(path).with_suffix(".png")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    print(f"Saved figure: {output_path}")
    plt.show()
    return fig, output_path
"""


UPLOAD_BUNDLE_CELL = r"""
#@title Upload or reuse a workflow bundle
input_source = "Upload ZIP or individual files" #@param ["Upload ZIP or individual files", "Reuse files from an earlier notebook in this runtime"]
reuse_directory = "/content/dropps_simulation" #@param {type:"string"}

import shutil

BASE = Path("/content") if Path("/content").is_dir() else Path.cwd()
INPUT_ROOT = reset_task_directory(BASE / "dropps_uploaded_inputs")

if input_source.startswith("Upload"):
    try:
        from google.colab import files
    except ImportError as exc:
        raise RuntimeError("Upload this bundle in Colab or choose the reuse option.") from exc
    uploaded = files.upload()
    for name in uploaded:
        source = Path(name)
        if source.suffix.lower() == ".zip":
            safe_extract_zip(source, INPUT_ROOT)
        else:
            shutil.copy2(source, INPUT_ROOT / source.name)
else:
    source_root = Path(reuse_directory)
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    for source in source_root.rglob("*"):
        if source.is_file() and source.suffix.lower() in {
            ".pdb", ".itp", ".top", ".mdp", ".tpr", ".xtc", ".ndx", ".chk", ".xml", ".json"
        }:
            target = INPUT_ROOT / source.name
            if target.exists() and sha256(target) != sha256(source):
                raise ValueError(f"Duplicate filename with different content: {source.name}")
            shutil.copy2(source, target)

print("Available files:")
for path in sorted(INPUT_ROOT.rglob("*")):
    if path.is_file():
        print(" ", path.relative_to(INPUT_ROOT), f"({path.stat().st_size / 1024:.1f} KiB)")
"""


def quickstart_notebook() -> dict:
    cells = [
        md(
            r"""
            [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tymworld/DROPPS/blob/v1.0.0/examples/colab/00_DROPPS_1_0_Quickstart.ipynb)

            # DROPPS 1.0 Colab Quickstart

            This notebook performs a complete, deliberately tiny workflow: sequence →
            coarse-grained model → multichain box → portable DROPPS TPR v2 → short
            OpenMM run → validation → density analysis. It is an interface and file-flow
            check, not a phase-separation production simulation.

            Use the focused notebooks for multicomponent systems, structural restraints,
            GPU/restart controls, trajectory transformations, and manuscript-level analyses.
            """
        ),
        md(
            r"""
            ## Runtime

            In Colab choose **Runtime → Change runtime type → GPU** when testing CUDA.
            This quickstart defaults to OpenMM's deterministic `Reference` platform so it
            also works without a GPU.
            """
        ),
        code(INSTALL_CELL),
        code(HELPERS_CELL),
        code(
            r"""
            #@title Configure the tiny demonstration
            sequence = "FWFWFWFWFWFWFWFW" #@param {type:"string"}
            molecule_name = "PEP" #@param {type:"string"}
            forcefield = "HPS" #@param ["HPS", "HPST", "CALVADOS2", "HPSRNA", "MPiPi", "MPiPi_PTM"]
            molecule_count = 4 #@param {type:"integer"}
            random_seed = 1215 #@param {type:"integer"}
            validation_steps = 20 #@param {type:"integer"}
            validation_platform = "Reference" #@param ["Reference", "CPU", "CUDA", "OpenCL", "auto"]

            BASE = Path("/content") if Path("/content").is_dir() else Path.cwd()
            WORKDIR = reset_task_directory(BASE / "dropps_quickstart")
            print("Working directory:", WORKDIR)
            """
        ),
        md("## 1 Build a molecule and a small system"),
        code(
            r"""
            dps(
                "pdb2dps", "-s", sequence, "-ff", forcefield,
                "-oc", f"{molecule_name}.pdb", "-op", f"{molecule_name}.itp",
                "-on", molecule_name, "--seed", random_seed,
                cwd=WORKDIR,
            )
            dps(
                "genmesh", "-f", f"{molecule_name}.pdb", "-p", f"{molecule_name}.itp",
                "-n", molecule_count, "-mesh", 2, 2, 1, "-g", 1,
                "-mx", 4, "-my", 4, "-mz", 4,
                "-oc", "system.pdb", "-op", "system.top", "--seed", random_seed,
                cwd=WORKDIR,
            )
            """
        ),
        md("## 2 Compile a portable run input"),
        code(
            r'''
            mdp = f"""# Tiny DROPPS 1.0 validation run
            integrator = Langevin
            dt = 0.01
            nsteps = {validation_steps}
            friction = 1.0
            seed = {random_seed}
            minimize = True
            forcetol = 100
            max-step = 100
            production-temperature = 300
            gen-vel = True
            initial-temperature = 300
            warming-speed = 1.0
            keep-warming-trajectory = False
            bondtype = bond
            comm-mode = Linear
            nstcomm = 2
            vdwtype = pLJ
            cutoff-scheme-lj = static
            cutoff-lj = 1.0
            cutoff-lj-multi = 3.0
            shift-lj = True
            coulombtype = yukawa
            cutoff-coul = 1.0
            shift-coul = True
            salt-conc = 0.1
            pcoulp = False
            ref-P = 1
            tau-P = 5
            nst-xout = 5
            nst-screenlog = 0
            nst-filelog = 5
            nst-energy = 5
            energy-grps = potentialEnergy,kineticEnergy,totalEnergy,temperature,boxX,boxY,boxZ,volume,density
            nst-stress = 0
            """
            (WORKDIR / "validation.mdp").write_text(mdp, encoding="utf-8")
            dps(
                "grompp", "-f", "system.pdb", "-p", "system.top",
                "-m", "validation.mdp", "-o", "validation.tpr", cwd=WORKDIR,
            )
            '''
        ),
        md("## 3 Run, check, and analyze"),
        code(
            r"""
            dps(
                "mdrun", "-s", "validation.tpr", "-o", "validation",
                "--platform", validation_platform, "-cpt", 0, cwd=WORKDIR,
            )
            dps("check", "-s", "validation.tpr", "-f", "validation.xtc", cwd=WORKDIR)
            dps(
                "density", "-s", "validation.tpr", "-f", "validation.xtc",
                "-o", "density.xvg", "-selfit", 0, "-sel", 0,
                "--center-mode", "none", cwd=WORKDIR,
            )
            plot_xvg(WORKDIR / "density.xvg", title="Tiny-run mass-density profile")
            """
        ),
        md("## 4 Record provenance and download"),
        code(
            r"""
            manifest = {
                "notebook": "00_DROPPS_1_0_Quickstart.ipynb",
                "dropps_version": "1.0",
                "purpose": "interface and file-flow validation only",
                "parameters": {
                    "sequence": sequence,
                    "molecule_name": molecule_name,
                    "forcefield": forcefield,
                    "molecule_count": molecule_count,
                    "random_seed": random_seed,
                    "validation_steps": validation_steps,
                    "validation_platform": validation_platform,
                },
                "commands": COMMAND_LOG,
            }
            (WORKDIR / "manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            outputs = [path for path in WORKDIR.iterdir() if path.is_file()]
            archive = make_zip(outputs, WORKDIR / "DROPPS_1_0_quickstart_results.zip", base=WORKDIR)
            download(archive)
            """
        ),
    ]
    return notebook(cells, gpu=True)


def builder_notebook() -> dict:
    cells = [
        md(
            r"""
            [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tymworld/DROPPS/blob/v1.0.0/examples/colab/01_DROPPS_1_0_Model_and_System_Builder.ipynb)

            # DROPPS 1.0 Molecular and System Builder

            Build a compact **cubic** box containing many copies of one protein for the
            phase-coexistence workflow. Multi-component systems remain available as an
            extension. Optional PTMs, angle restraints, elastic networks, residue
            modifications, and deterministic packing match the construction workflow
            described in the manuscript.

            Do not elongate the initial box here. The simulation notebook first condenses
            this compact system under NPT, then expands one axis tenfold and starts the NVT
            slab-production stage.

            All operations use the public `dps` CLI. Bead IDs in angle and elastic input
            files are **1-based**; DROPPS index groups used for trajectory analysis are
            **0-based** and are handled in later notebooks.
            """
        ),
        code(INSTALL_CELL),
        code(HELPERS_CELL),
        md(
            r"""
            ## 1 Define up to three molecular components

            The default is a lightweight 80-copy, single-protein construction example,
            following the manuscript's 5×5×5 placement grid. Replace the sequence with
            the target protein for a scientific run. Mapping uses Cα coordinates from the
            first model of an uploaded all-atom PDB. Enable Components B/C only for a
            co-phase-separation system and use compatible force-field families.
            """
        ),
        code(
            r"""
            #@title Component A
            component_a_enabled = True #@param {type:"boolean"}
            component_a_name = "SCAFFOLD" #@param {type:"string"}
            component_a_sequence = "FWFWFWFWFWFWFWFW" #@param {type:"string"}
            component_a_count = 80 #@param {type:"integer"}
            component_a_forcefield = "HPS" #@param ["HPS", "HPST", "CALVADOS2", "HPSRNA", "MPiPi", "MPiPi_PTM"]
            component_a_input_pdb = "" #@param {type:"string"}
            component_a_ptms = "" #@param {type:"string"}
            component_a_radius_nm = 2.0 #@param {type:"number"}
            component_a_extension = 0.5 #@param {type:"slider", min:0, max:1, step:0.05}
            component_a_residue_index = 1 #@param {type:"integer"}
            component_a_charge_ntd = False #@param {type:"boolean"}
            component_a_charge_ctd = False #@param {type:"boolean"}
            """
        ),
        code(
            r"""
            #@title Component B
            component_b_enabled = False #@param {type:"boolean"}
            component_b_name = "CLIENT" #@param {type:"string"}
            component_b_sequence = "KKGGEEDD" #@param {type:"string"}
            component_b_count = 4 #@param {type:"integer"}
            component_b_forcefield = "HPS" #@param ["HPS", "HPST", "CALVADOS2", "HPSRNA", "MPiPi", "MPiPi_PTM"]
            component_b_input_pdb = "" #@param {type:"string"}
            component_b_ptms = "" #@param {type:"string"}
            component_b_radius_nm = 2.0 #@param {type:"number"}
            component_b_extension = 0.5 #@param {type:"slider", min:0, max:1, step:0.05}
            component_b_residue_index = 1 #@param {type:"integer"}
            component_b_charge_ntd = False #@param {type:"boolean"}
            component_b_charge_ctd = False #@param {type:"boolean"}
            """
        ),
        code(
            r"""
            #@title Component C
            component_c_enabled = False #@param {type:"boolean"}
            component_c_name = "RNA" #@param {type:"string"}
            component_c_sequence = "AAAAAAAAAA" #@param {type:"string"}
            component_c_count = 2 #@param {type:"integer"}
            component_c_forcefield = "HPSRNA" #@param ["HPS", "HPST", "CALVADOS2", "HPSRNA", "MPiPi", "MPiPi_PTM"]
            component_c_input_pdb = "" #@param {type:"string"}
            component_c_ptms = "" #@param {type:"string"}
            component_c_radius_nm = 2.0 #@param {type:"number"}
            component_c_extension = 0.5 #@param {type:"slider", min:0, max:1, step:0.05}
            component_c_residue_index = 1 #@param {type:"integer"}
            component_c_charge_ntd = False #@param {type:"boolean"}
            component_c_charge_ctd = False #@param {type:"boolean"}
            """
        ),
        code(
            r"""
            #@title Upload optional all-atom PDB reference structures
            upload_reference_pdbs = False #@param {type:"boolean"}
            random_seed = 1215 #@param {type:"integer"}

            BASE = Path("/content") if Path("/content").is_dir() else Path.cwd()
            WORKDIR = reset_task_directory(BASE / "dropps_model_builder")
            UPLOAD_DIR = WORKDIR / "uploads"
            UPLOAD_DIR.mkdir()

            if upload_reference_pdbs:
                try:
                    from google.colab import files
                except ImportError as exc:
                    raise RuntimeError("Upload reference structures in Google Colab.") from exc
                import shutil

                for name in files.upload():
                    source = Path(name)
                    if source.suffix.lower() != ".pdb":
                        raise ValueError(f"Expected PDB input, received {source}")
                    shutil.copy2(source, UPLOAD_DIR / source.name)

            print("Working directory:", WORKDIR)
            """
        ),
        code(
            r"""
            #@title Build monomer structures and self-contained ITP topologies
            import re

            def component(prefix):
                scope = globals()
                return {
                    "enabled": scope[f"component_{prefix}_enabled"],
                    "name": scope[f"component_{prefix}_name"].strip(),
                    "sequence": scope[f"component_{prefix}_sequence"].strip(),
                    "count": int(scope[f"component_{prefix}_count"]),
                    "forcefield": scope[f"component_{prefix}_forcefield"],
                    "input_pdb": scope[f"component_{prefix}_input_pdb"].strip(),
                    "ptms": scope[f"component_{prefix}_ptms"].strip(),
                    "radius": float(scope[f"component_{prefix}_radius_nm"]),
                    "extension": float(scope[f"component_{prefix}_extension"]),
                    "residue_index": int(scope[f"component_{prefix}_residue_index"]),
                    "charge_ntd": scope[f"component_{prefix}_charge_ntd"],
                    "charge_ctd": scope[f"component_{prefix}_charge_ctd"],
                }

            components = [item for item in map(component, "abc") if item["enabled"]]
            if not components:
                raise ValueError("Enable at least one component.")
            if len({item["name"] for item in components}) != len(components):
                raise ValueError("Component names must be unique.")

            for item in components:
                if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", item["name"]):
                    raise ValueError(f"Use a simple alphanumeric molecule name: {item['name']!r}")
                if not item["sequence"]:
                    raise ValueError(f"Sequence is empty for {item['name']}")
                arguments = [
                    "pdb2dps", "-s", item["sequence"], "-ff", item["forcefield"],
                    "-oc", f"{item['name']}.pdb", "-op", f"{item['name']}.itp",
                    "-on", item["name"], "-ri", item["residue_index"],
                    "-r", item["radius"], "-e", item["extension"],
                    "--seed", random_seed,
                ]
                if item["input_pdb"]:
                    source = Path(item["input_pdb"])
                    if not source.is_file():
                        source = UPLOAD_DIR / item["input_pdb"]
                    if not source.is_file():
                        raise FileNotFoundError(source)
                    arguments.extend(["-f", source])
                if item["ptms"]:
                    arguments.extend(["-ptm", *item["ptms"].split()])
                if item["charge_ntd"]:
                    arguments.append("-cNTD")
                if item["charge_ctd"]:
                    arguments.append("-cCTD")
                dps(*arguments, cwd=WORKDIR)
                item["pdb"] = WORKDIR / f"{item['name']}.pdb"
                item["itp"] = WORKDIR / f"{item['name']}.itp"

            print("Built:", ", ".join(item["name"] for item in components))
            """
        ),
        md(
            r"""
            ## 2 Optional topology refinements

            Enter semicolon-separated records. Angle records use
            `CENTER_RESIDUE ANGLE_DEG FORCE_CONSTANT`. Elastic groups use compact
            ranges such as `1-77; 106-177; 192-256`; the notebook expands them into
            the explicit 1-based bead lists required by `genelastic`.
            """
        ),
        code(
            r"""
            #@title Angle restraints with dps addangle
            angle_target = "SCAFFOLD" #@param {type:"string"}
            angle_records = "" #@param {type:"string"}

            if angle_records.strip():
                target = next(item for item in components if item["name"] == angle_target)
                angle_file = WORKDIR / f"{angle_target}_angles.dat"
                angle_file.write_text(
                    "\n".join(part.strip() for part in angle_records.split(";") if part.strip()) + "\n",
                    encoding="utf-8",
                )
                output = WORKDIR / f"{angle_target}_angles.itp"
                dps(
                    "addangle", "-ip", target["itp"], "-op", output,
                    "-al", angle_file, cwd=WORKDIR,
                )
                target["itp"] = output
            else:
                print("No angle restraints requested.")
            """
        ),
        code(
            r"""
            #@title Elastic networks with dps genelastic
            elastic_target = "SCAFFOLD" #@param {type:"string"}
            elastic_groups = "" #@param {type:"string"}
            elastic_force_constant = 5000.0 #@param {type:"number"}
            elastic_lower_nm = 0.5 #@param {type:"number"}
            elastic_upper_nm = 0.9 #@param {type:"number"}

            def expand_integer_ranges(expression):
                values = []
                for token in expression.replace(",", " ").split():
                    if "-" in token:
                        start, stop = map(int, token.split("-", 1))
                        if stop < start:
                            raise ValueError(f"Descending range: {token}")
                        values.extend(range(start, stop + 1))
                    else:
                        values.append(int(token))
                return values

            if elastic_groups.strip():
                target = next(item for item in components if item["name"] == elastic_target)
                lines = []
                for group in elastic_groups.split(";"):
                    if group.strip():
                        lines.append(" ".join(map(str, expand_integer_ranges(group))))
                elastic_file = WORKDIR / f"{elastic_target}_elastic.dat"
                elastic_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
                output = WORKDIR / f"{elastic_target}_elastic.itp"
                dps(
                    "genelastic", "-f", target["pdb"], "-p", target["itp"],
                    "-o", output, "-er", elastic_file,
                    "-ef", elastic_force_constant, "-el", elastic_lower_nm,
                    "-eu", elastic_upper_nm, cwd=WORKDIR,
                )
                target["itp"] = output
            else:
                print("No elastic network requested.")
            """
        ),
        code(
            r"""
            #@title Optional post-build residue modification with dps modifyres
            apply_residue_modifications = False #@param {type:"boolean"}
            modification_target = "SCAFFOLD" #@param {type:"string"}
            residue_modifications = "S3SMP" #@param {type:"string"}
            modification_forcefield = "MPiPi_PTM" #@param ["HPS", "HPST", "CALVADOS2", "HPSRNA", "MPiPi", "MPiPi_PTM"]

            if apply_residue_modifications:
                target = next(item for item in components if item["name"] == modification_target)
                modified_pdb = WORKDIR / f"{modification_target}_modified.pdb"
                modified_itp = WORKDIR / f"{modification_target}_modified.itp"
                dps(
                    "modifyres", "-ip", target["itp"], "-if", target["pdb"],
                    "-op", modified_itp, "-of", modified_pdb,
                    "-ff", modification_forcefield,
                    "-m", *residue_modifications.split(), cwd=WORKDIR,
                )
                target["pdb"], target["itp"] = modified_pdb, modified_itp
            else:
                print("No post-build residue modification requested.")
            """
        ),
        md("## 3 Pack the compact cubic simulation box"),
        code(
            r"""
            #@title System packing parameters
            mesh_x = 5 #@param {type:"integer"}
            mesh_y = 5 #@param {type:"integer"}
            mesh_z = 5 #@param {type:"integer"}
            minimum_gap_nm = 1.0 #@param {type:"number"}
            box_type = "cubic" #@param ["cubic", "anisotropy", "xy"]
            minimum_x_nm = 5.0 #@param {type:"number"}
            minimum_y_nm = 5.0 #@param {type:"number"}
            minimum_z_nm = 5.0 #@param {type:"number"}
            shuffle_components = True #@param {type:"boolean"}
            non_cubic_molecule = False #@param {type:"boolean"}

            total_count = sum(item["count"] for item in components)
            if mesh_x * mesh_y * mesh_z < total_count:
                raise ValueError("The mesh has fewer sites than requested molecules.")

            arguments = ["genmesh", "-f"]
            arguments.extend(item["pdb"] for item in components)
            arguments.append("-p")
            arguments.extend(item["itp"] for item in components)
            arguments.append("-n")
            arguments.extend(item["count"] for item in components)
            arguments.extend(
                [
                    "-mesh", mesh_x, mesh_y, mesh_z,
                    "-g", minimum_gap_nm, "-bt", box_type,
                    "-mx", minimum_x_nm, "-my", minimum_y_nm, "-mz", minimum_z_nm,
                    "--seed", random_seed, "-oc", "system.pdb", "-op", "system.top",
                ]
            )
            if shuffle_components:
                arguments.append("-s")
            if non_cubic_molecule:
                arguments.append("-ncm")
            dps(*arguments, cwd=WORKDIR)
            SYSTEM_PDB = WORKDIR / "system.pdb"
            SYSTEM_TOP = WORKDIR / "system.top"
            """
        ),
        md(
            r"""
            The exported `system.pdb` is deliberately compact and cubic. Its box must be
            elongated only after the NPT stage has produced a condensed `npt.pdb`.
            """
        ),
        md("## 4 Inspect and export the model bundle"),
        code(
            r"""
            #@title Preview bead coordinates
            def pdb_coordinates_nm(path):
                rows = []
                for line in Path(path).read_text(encoding="utf-8").splitlines():
                    if line.startswith(("ATOM  ", "HETATM")):
                        rows.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                return np.asarray(rows) / 10.0

            coordinates = pdb_coordinates_nm(SYSTEM_PDB)
            fig = plt.figure(figsize=(6.5, 5.5))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(*coordinates.T, s=8, alpha=0.7)
            ax.set(xlabel="x (nm)", ylabel="y (nm)", zlabel="z (nm)", title="Initial system")
            fig.tight_layout()
            plt.show()
            """
        ),
        code(
            r"""
            #@title Save manifest and download
            selected_files = [SYSTEM_PDB, SYSTEM_TOP]
            for item in components:
                selected_files.extend([item["pdb"], item["itp"]])
            selected_files = sorted({Path(path).resolve() for path in selected_files})
            manifest = {
                "notebook": "01_DROPPS_1_0_Model_and_System_Builder.ipynb",
                "dropps_version": "1.0",
                "seed": random_seed,
                "components": [
                    {
                        key: value
                        for key, value in item.items()
                        if key not in {"pdb", "itp", "input_pdb"}
                    }
                    | {"pdb": item["pdb"].name, "itp": item["itp"].name}
                    for item in components
                ],
                "system_pdb": SYSTEM_PDB.name,
                "system_top": SYSTEM_TOP.name,
                "files": {path.name: sha256(path) for path in selected_files},
                "commands": COMMAND_LOG,
            }
            manifest_path = WORKDIR / "model_manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            selected_files.append(manifest_path)
            archive = make_zip(
                selected_files, WORKDIR / "DROPPS_1_0_model_bundle.zip", base=WORKDIR
            )
            download(archive)
            """
        ),
    ]
    return notebook(cells)


def simulation_notebook() -> dict:
    cells = [
        md(
            r"""
            [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tymworld/DROPPS/blob/v1.0.0/examples/colab/02_DROPPS_1_0_Simulation.ipynb)

            # DROPPS 1.0 Phase-Coexistence Simulation

            This notebook implements the slab protocol in its required order:

            1. start from many protein monomers in a compact cubic box;
            2. condense/equilibrate the box in the NPT ensemble;
            3. expand the **z box axis tenfold**, recentering the condensed
               configuration in the resulting elongated box;
            4. run the elongated box in the NVT ensemble;
            5. pass only `slab_nvt.tpr` and `slab_nvt.xtc` to trajectory processing and
               phase-coexistence analysis.

            The manuscript benchmark settings are 300 ns NPT at 0.01 ps followed by
            3 μs NVT at 0.02 ps. A short full-workflow mode validates all transitions
            without claiming a physically equilibrated phase-separated state.
            """
        ),
        code(INSTALL_CELL),
        code(HELPERS_CELL),
        code(
            UPLOAD_BUNDLE_CELL.replace(
                'reuse_directory = "/content/dropps_simulation"',
                'reuse_directory = "/content/dropps_model_builder"',
            )
        ),
        md("## 1 Resolve the compact cubic input system"),
        code(
            r"""
            #@title Input filenames
            structure_filename = "system.pdb" #@param {type:"string"}
            topology_filename = "system.top" #@param {type:"string"}
            require_cubic_input = True #@param {type:"boolean"}
            cubic_relative_tolerance = 0.001 #@param {type:"number"}

            BASE = Path("/content") if Path("/content").is_dir() else Path.cwd()
            WORKDIR = reset_task_directory(BASE / "dropps_simulation")
            INPUTS = WORKDIR / "inputs"
            INPUTS.mkdir()

            import shutil

            for source in INPUT_ROOT.rglob("*"):
                if source.is_file() and (
                    source.suffix.lower() in {".pdb", ".itp", ".top", ".ff", ".chk", ".xml", ".json", ".mdp", ".tpr"}
                    or source.name.endswith(".state.xml")
                    or source.name.endswith(".restart.json")
                ):
                    target = INPUTS / source.name
                    if target.exists() and sha256(target) != sha256(source):
                        raise ValueError(f"Conflicting duplicate input: {source.name}")
                    shutil.copy2(source, target)

            STRUCTURE = find_unique(INPUTS, structure_filename)
            TOPOLOGY = find_unique(INPUTS, topology_filename)

            def pdb_box_lengths_nm(path):
                for line in Path(path).read_text(encoding="utf-8").splitlines():
                    if line.startswith("CRYST1"):
                        return np.asarray(
                            [float(line[6:15]), float(line[15:24]), float(line[24:33])]
                        ) / 10.0
                raise ValueError(f"No CRYST1 box record found in {path}")

            compact_box_nm = pdb_box_lengths_nm(STRUCTURE)
            relative_spread = np.ptp(compact_box_nm) / compact_box_nm.mean()
            if require_cubic_input and relative_spread > cubic_relative_tolerance:
                raise ValueError(
                    "The slab protocol must start from a compact cubic box; "
                    f"received {compact_box_nm.tolist()} nm."
                )
            print("Structure:", STRUCTURE)
            print("Topology:", TOPOLOGY)
            print("Compact box (nm):", compact_box_nm)
            """
        ),
        md("## 2 Define the NPT → z×10 → NVT protocol"),
        code(
            r"""
            #@title Stage durations, integration, and temperature
            workflow_action = "Short full-workflow validation" #@param ["Short full-workflow validation", "Run full configured workflow", "Prepare NPT TPR only", "Prepare NVT TPR from uploaded npt.pdb", "Run NVT from uploaded npt.pdb"]
            production_temperature_K = 300.0 #@param {type:"number"}
            npt_time_step_ps = 0.01 #@param {type:"number"}
            npt_duration_ns = 300.0 #@param {type:"number"}
            nvt_time_step_ps = 0.02 #@param {type:"number"}
            nvt_duration_ns = 3000.0 #@param {type:"number"}
            validation_steps_per_stage = 100 #@param {type:"integer"}
            friction_per_ps = 1.0 #@param {type:"number"}
            random_seed = 1215 #@param {type:"integer"}
            minimize_before_npt = True #@param {type:"boolean"}
            minimization_force_tolerance = 100.0 #@param {type:"number"}
            minimization_max_steps = 100000 #@param {type:"integer"}
            slab_axis = "z" #@param ["z", "x", "y"]
            slab_box_multiplier = 10.0 #@param {type:"number"}

            if slab_axis != "z":
                raise ValueError("This Colab phase-coexistence workflow expands the z axis.")
            if slab_box_multiplier != 10.0:
                raise ValueError("This phase-coexistence workflow requires a 10-fold box expansion.")
            if min(npt_time_step_ps, nvt_time_step_ps, npt_duration_ns, nvt_duration_ns) <= 0:
                raise ValueError("Time steps and stage durations must be positive.")

            configured_npt_steps = int(round(npt_duration_ns * 1000 / npt_time_step_ps))
            configured_nvt_steps = int(round(nvt_duration_ns * 1000 / nvt_time_step_ps))
            short_workflow = workflow_action == "Short full-workflow validation"
            npt_steps = validation_steps_per_stage if short_workflow else configured_npt_steps
            nvt_steps = validation_steps_per_stage if short_workflow else configured_nvt_steps

            print(f"NPT: {npt_steps:,} steps at {npt_time_step_ps:g} ps")
            print(f"NVT: {nvt_steps:,} steps at {nvt_time_step_ps:g} ps")
            print(f"Slab conversion: {slab_axis} axis × {slab_box_multiplier:g}")
            """
        ),
        code(
            r"""
            #@title Bond, COM, and nonbonded settings
            bond_treatment = "bond" #@param ["bond", "constraint"]
            center_of_mass_mode = "Linear" #@param ["Linear", "none"]
            center_of_mass_interval = 100 #@param {type:"integer"}
            vdw_type = "pLJ" #@param ["pLJ", "MPiPi"]
            lj_cutoff_scheme = "static" #@param ["static", "dynamic"]
            lj_cutoff_nm = 1.5 #@param {type:"number"}
            lj_sigma_multiplier = 3.0 #@param {type:"number"}
            shift_lj = True #@param {type:"boolean"}
            coulomb_type = "yukawa" #@param ["yukawa", "no"]
            coulomb_cutoff_nm = 1.5 #@param {type:"number"}
            shift_coulomb = True #@param {type:"boolean"}
            salt_concentration_molar = 0.1 #@param {type:"number"}
            """
        ),
        code(
            r"""
            #@title Pressure and output settings
            reference_pressure_bar = 1.0 #@param {type:"number"}
            barostat_attempt_interval = 5 #@param {type:"integer"}
            trajectory_interval_ns = 1.0 #@param {type:"number"}
            screen_log_interval_ns = 10.0 #@param {type:"number"}
            file_log_interval_ns = 1.0 #@param {type:"number"}
            energy_interval_ns = 1.0 #@param {type:"number"}
            energy_groups = "potentialEnergy,kineticEnergy,totalEnergy,temperature,boxX,boxY,boxZ,volume,density" #@param {type:"string"}
            stress_interval_steps = 0 #@param {type:"integer"}
            stress_strain = 0.0003 #@param {type:"number"}
            stress_platform = "auto" #@param ["auto", "CUDA", "OpenCL", "CPU", "Reference"]
            stress_precision = "double" #@param ["single", "mixed", "double"]
            """
        ),
        code(
            r'''
            #@title MDP renderer for both ensembles
            def bool_text(value):
                return "True" if value else "False"

            def interval_steps(interval_ns, dt_ps, total_steps):
                requested = max(1, int(round(interval_ns * 1000 / dt_ps)))
                if short_workflow:
                    return min(requested, max(1, total_steps // 10))
                return requested

            def render_mdp(*, stage, dt_ps, nsteps, pressure_coupling, minimize):
                effective_max_steps = (
                    min(minimization_max_steps, 100)
                    if short_workflow and minimize
                    else minimization_max_steps
                )
                return f"""# DROPPS 1.0 Colab: {stage} stage of NPT -> slab -> NVT
            integrator = Langevin
            dt = {dt_ps}
            nsteps = {nsteps}
            friction = {friction_per_ps}
            seed = {random_seed}
            minimize = {bool_text(minimize)}
            forcetol = {minimization_force_tolerance}
            max-step = {effective_max_steps}
            production-temperature = {production_temperature_K}
            gen-vel = True
            initial-temperature = {production_temperature_K}
            warming-speed = 1.0
            keep-warming-trajectory = False
            bondtype = {bond_treatment}
            comm-mode = {center_of_mass_mode}
            nstcomm = {center_of_mass_interval}
            vdwtype = {vdw_type}
            cutoff-scheme-lj = {lj_cutoff_scheme}
            cutoff-lj = {lj_cutoff_nm}
            cutoff-lj-multi = {lj_sigma_multiplier}
            shift-lj = {bool_text(shift_lj)}
            coulombtype = {coulomb_type}
            cutoff-coul = {coulomb_cutoff_nm}
            shift-coul = {bool_text(shift_coulomb)}
            salt-conc = {salt_concentration_molar}
            pcoulp = {bool_text(pressure_coupling)}
            ref-P = {reference_pressure_bar}
            tau-P = {barostat_attempt_interval}
            nst-xout = {interval_steps(trajectory_interval_ns, dt_ps, nsteps)}
            nst-screenlog = {interval_steps(screen_log_interval_ns, dt_ps, nsteps)}
            nst-filelog = {interval_steps(file_log_interval_ns, dt_ps, nsteps)}
            nst-energy = {interval_steps(energy_interval_ns, dt_ps, nsteps)}
            energy-grps = {energy_groups}
            nst-stress = {stress_interval_steps}
            stress-strain = {stress_strain}
            stress-output = auto
            stress-platform = {stress_platform}
            stress-precision = {stress_precision}
            stress-device = auto
            stress-threads = 0
            """
            '''
        ),
        md("## 3 Runtime, checkpoint, and restart controls"),
        code(
            r"""
            #@title Runtime controls shared by both stages
            compute_platform = "auto" #@param ["auto", "CUDA", "OpenCL", "CPU", "Reference"]
            gpu_device_index = "0" #@param {type:"string"}
            gpu_precision = "mixed" #@param ["single", "mixed", "double"]
            cpu_threads = 0 #@param {type:"integer"}
            checkpoint_minutes = 5.0 #@param {type:"number"}
            keep_numbered_checkpoints = False #@param {type:"boolean"}
            maximum_wall_hours = -1.0 #@param {type:"number"}
            restart_npt_from_checkpoint = False #@param {type:"boolean"}
            npt_checkpoint_filename = "" #@param {type:"string"}
            restart_nvt_from_checkpoint = False #@param {type:"boolean"}
            nvt_checkpoint_filename = "" #@param {type:"string"}
            append_restart_outputs = True #@param {type:"boolean"}

            def run_stage(tpr, prefix, restart=False, checkpoint_filename=""):
                arguments = [
                    "mdrun", "-s", Path(tpr).name, "-o", prefix,
                    "--platform", compute_platform, "-cpt", checkpoint_minutes,
                    "-maxh", maximum_wall_hours,
                ]
                if compute_platform in {"CUDA", "OpenCL"}:
                    arguments.extend(["--precision", gpu_precision])
                    if gpu_device_index.strip():
                        arguments.extend(["-gpu_id", gpu_device_index.strip()])
                elif compute_platform == "CPU" and cpu_threads > 0:
                    arguments.extend(["-nt", cpu_threads])
                if keep_numbered_checkpoints:
                    arguments.append("-cpnum")
                if restart:
                    arguments.append("-cpi")
                    if checkpoint_filename.strip():
                        arguments.append(checkpoint_filename.strip())
                arguments.append("-append" if append_restart_outputs else "-noappend")
                dps(*arguments, cwd=INPUTS)
            """
        ),
        md(
            r"""
            ## 4 Stage 1 — compact cubic-box NPT

            Pressure coupling is mandatory in this stage. The full defaults generate the
            condensed configuration over 300 ns. `Prepare NPT TPR only` is intended for
            transfer to a persistent GPU/HPC resource.
            """
        ),
        code(
            r"""
            #@title Compile and optionally execute NPT
            NPT_MDP = INPUTS / "npt.mdp"
            NPT_TPR = INPUTS / "npt.tpr"
            run_npt_here = workflow_action in {
                "Short full-workflow validation", "Run full configured workflow"
            }
            use_existing_npt = workflow_action in {
                "Prepare NVT TPR from uploaded npt.pdb", "Run NVT from uploaded npt.pdb"
            }

            if not use_existing_npt:
                NPT_MDP.write_text(
                    render_mdp(
                        stage="compact NPT", dt_ps=npt_time_step_ps,
                        nsteps=npt_steps, pressure_coupling=True,
                        minimize=minimize_before_npt,
                    ),
                    encoding="utf-8",
                )
                dps(
                    "grompp", "-f", STRUCTURE.name, "-p", TOPOLOGY.name,
                    "-m", NPT_MDP.name, "-o", NPT_TPR.name, cwd=INPUTS,
                )
                dps("check", "-s", NPT_TPR.name, cwd=INPUTS)

            if run_npt_here:
                run_stage(
                    NPT_TPR, "npt", restart_npt_from_checkpoint,
                    npt_checkpoint_filename,
                )
                dps("check", "-s", NPT_TPR.name, "-f", "npt.xtc", cwd=INPUTS)
                NPT_FINAL = INPUTS / "npt.pdb"
            elif use_existing_npt:
                NPT_FINAL = find_unique(INPUTS, "npt.pdb")
            else:
                NPT_FINAL = None
                print("NPT TPR prepared. Run it to completion before constructing the slab.")
            """
        ),
        md(
            r"""
            ## 5 Stage 2 — tenfold box expansion and elongated-box NVT

            `dps editconf -mz 10` unwraps the expanded axis, increases only that box
            length, and translates the NPT configuration to the center of the elongated
            box. The generated `slab_nvt.mdp` has `pcoulp = False`. Because a PDB does
            not store velocities, the NVT TPR generates new velocities at the same target
            temperature.
            """
        ),
        code(
            r"""
            #@title Build and optionally execute the NVT slab stage
            SLAB_PDB = INPUTS / "slab.pdb"
            NVT_MDP = INPUTS / "slab_nvt.mdp"
            NVT_TPR = INPUTS / "slab_nvt.tpr"
            run_nvt_here = workflow_action in {
                "Short full-workflow validation", "Run full configured workflow",
                "Run NVT from uploaded npt.pdb",
            }

            if NPT_FINAL is not None:
                npt_box_nm = pdb_box_lengths_nm(NPT_FINAL)
                multiplier_option = {"x": "-mx", "y": "-my", "z": "-mz"}[slab_axis]
                dps(
                    "editconf", "-f", NPT_FINAL.name, "-o", SLAB_PDB.name,
                    multiplier_option, slab_box_multiplier, cwd=INPUTS,
                )
                slab_box_nm = pdb_box_lengths_nm(SLAB_PDB)
                axis_index = {"x": 0, "y": 1, "z": 2}[slab_axis]
                ratios = slab_box_nm / npt_box_nm
                expected = np.ones(3)
                expected[axis_index] = slab_box_multiplier
                if not np.allclose(ratios, expected, rtol=2e-3, atol=2e-3):
                    raise RuntimeError(
                        f"Slab-box verification failed: ratios={ratios.tolist()}, "
                        f"expected={expected.tolist()}"
                    )
                print("NPT box (nm):", npt_box_nm)
                print("Slab box (nm):", slab_box_nm)
                print("Verified box-length ratios:", ratios)

                NVT_MDP.write_text(
                    render_mdp(
                        stage="elongated-box NVT", dt_ps=nvt_time_step_ps,
                        nsteps=nvt_steps, pressure_coupling=False, minimize=False,
                    ),
                    encoding="utf-8",
                )
                dps(
                    "grompp", "-f", SLAB_PDB.name, "-p", TOPOLOGY.name,
                    "-m", NVT_MDP.name, "-o", NVT_TPR.name, cwd=INPUTS,
                )
                dps("check", "-s", NVT_TPR.name, cwd=INPUTS)

                if run_nvt_here:
                    run_stage(
                        NVT_TPR, "slab_nvt", restart_nvt_from_checkpoint,
                        nvt_checkpoint_filename,
                    )
                    dps(
                        "check", "-s", NVT_TPR.name, "-f", "slab_nvt.xtc",
                        cwd=INPUTS,
                    )
                    print("Analysis input: slab_nvt.tpr + slab_nvt.xtc")
                else:
                    print("NVT TPR prepared: slab_nvt.tpr")
            """
        ),
        md("## 6 Export the staged simulation and explicit analysis handoff"),
        code(
            r"""
            #@title Save manifest and download simulation bundle
            files_to_export = [path for path in INPUTS.iterdir() if path.is_file()]
            analysis_ready = (INPUTS / "slab_nvt.tpr").is_file() and (INPUTS / "slab_nvt.xtc").is_file()
            manifest = {
                "notebook": "02_DROPPS_1_0_Simulation.ipynb",
                "dropps_version": "1.0",
                "workflow_action": workflow_action,
                "protocol": [
                    "compact cubic box",
                    "NPT condensation/equilibration",
                    f"{slab_axis}-axis x{slab_box_multiplier:g}",
                    "elongated-box NVT production",
                    "analyse NVT trajectory only",
                ],
                "scientific_parameters": {
                    "temperature_K": production_temperature_K,
                    "npt": {
                        "dt_ps": npt_time_step_ps,
                        "configured_duration_ns": npt_duration_ns,
                        "effective_steps": npt_steps,
                        "pressure_coupling": True,
                    },
                    "slab_expansion": {"axis": slab_axis, "multiplier": slab_box_multiplier},
                    "nvt": {
                        "dt_ps": nvt_time_step_ps,
                        "configured_duration_ns": nvt_duration_ns,
                        "effective_steps": nvt_steps,
                        "pressure_coupling": False,
                    },
                    "forcefield_vdw_type": vdw_type,
                    "salt_concentration_molar": salt_concentration_molar,
                    "seed": random_seed,
                },
                "operational_parameters": {
                    "platform": compute_platform,
                    "precision": gpu_precision,
                    "checkpoint_minutes": checkpoint_minutes,
                    "maximum_wall_hours": maximum_wall_hours,
                },
                "analysis_handoff": {
                    "ready": analysis_ready,
                    "run_input": "slab_nvt.tpr",
                    "trajectory": "slab_nvt.xtc",
                    "ensemble": "NVT",
                },
                "files": {path.name: sha256(path) for path in files_to_export},
                "commands": COMMAND_LOG,
            }
            manifest_path = INPUTS / "simulation_manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            files_to_export.append(manifest_path)
            archive = make_zip(
                files_to_export,
                WORKDIR / "DROPPS_1_0_simulation_bundle.zip",
                base=INPUTS,
            )
            download(archive)
            """
        ),
    ]
    return notebook(cells, gpu=True)


def trajectory_notebook() -> dict:
    cells = [
        md(
            r"""
            [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tymworld/DROPPS/blob/v1.0.0/examples/colab/03_DROPPS_1_0_Trajectory_Preparation.ipynb)

            # DROPPS 1.0 Trajectory Preparation and Indexing

            Prepare the **elongated-box NVT trajectory** for analysis. The notebook rejects
            an NPT run input, creates reusable NDX groups through the public `make_ndx`
            interface, and applies PBC reconstruction, centering, fitting, time selection,
            XTC/PDB conversion, and snapshot extraction with `trjconv`.

            Transform order in DROPPS 1.0 is: make whole/no-jump → center → pack → fit →
            translate. Time selections use physical units and choose the nearest saved frame.
            """
        ),
        code(INSTALL_CELL),
        code(HELPERS_CELL),
        code(UPLOAD_BUNDLE_CELL),
        md("## 1 Resolve and validate the run files"),
        code(
            r"""
            #@title Run filenames
            run_input_filename = "slab_nvt.tpr" #@param {type:"string"}
            trajectory_filename = "slab_nvt.xtc" #@param {type:"string"}

            TPR = find_unique(INPUT_ROOT, run_input_filename)
            TRAJECTORY = find_unique(INPUT_ROOT, trajectory_filename)

            import zipfile

            if not zipfile.is_zipfile(TPR):
                raise ValueError("Expected a portable DROPPS 1.0 TPR v2 file.")
            with zipfile.ZipFile(TPR) as archive:
                tpr_parameters = json.loads(archive.read("parameters.json"))
            if tpr_parameters.get("pcoulp"):
                raise ValueError(
                    "Trajectory preparation requires the elongated-box NVT TPR, not the NPT TPR."
                )

            BASE = Path("/content") if Path("/content").is_dir() else Path.cwd()
            WORKDIR = reset_task_directory(BASE / "dropps_trajectory")
            import shutil
            ANALYSIS_TPR = WORKDIR / "slab_nvt.tpr"
            shutil.copy2(TPR, ANALYSIS_TPR)
            dps("check", "-s", TPR, "-f", TRAJECTORY, cwd=WORKDIR)
            print("Verified ensemble: NVT (pcoulp = False)")
            """
        ),
        md(
            r"""
            ## 2 Create an index file

            Commands are separated by semicolons. Initial groups are `System` (group 0)
            followed by one group per molecule type. New selections append groups in order.
            Example: `mol SCAFFOLD; name 3 Scaffold_all; resid 1-4 & mol SCAFFOLD;
            name 4 Scaffold_N; q`. The notebook adds `q` if omitted.
            """
        ),
        code(
            r"""
            #@title Non-interactive make_ndx command script
            index_commands = "q" #@param {type:"string"}
            index_filename = "analysis.ndx" #@param {type:"string"}

            commands = [part.strip() for part in index_commands.split(";") if part.strip()]
            if not commands or commands[-1] != "q":
                commands.append("q")
            NDX = WORKDIR / index_filename
            result = dps(
                "make_ndx", "-s", TPR, "-o", NDX,
                cwd=WORKDIR, input_text="\n".join(commands) + "\n",
                check=False,
            )
            if not NDX.is_file():
                raise RuntimeError(
                    f"make_ndx did not create {NDX}; subprocess status was {result.returncode}."
                )
            if result.returncode:
                print(
                    "Note: DROPPS 1.0 make_ndx writes the requested file and then exits "
                    "through its legacy interactive quit path, which reports status 1."
                )
            dps("check", "-s", TPR, "-f", TRAJECTORY, "-n", NDX, cwd=WORKDIR)
            print(NDX.read_text(encoding="utf-8")[:4000])
            """
        ),
        md("## 3 Convert and transform the trajectory"),
        code(
            r"""
            #@title trjconv settings
            output_filename = "processed.xtc" #@param {type:"string"}
            output_selection = "group 0" #@param {type:"string"}
            start_time = "" #@param {type:"string"}
            end_time = "" #@param {type:"string"}
            sampling_interval = "" #@param {type:"string"}
            time_unit = "ns" #@param ["fs", "ps", "ns", "us", "ms", "s"]
            pbc_mode = "whole" #@param ["none", "whole", "atom", "res", "mol", "nojump"]
            center_mode = "none" #@param ["none", "geometry", "mass", "dense"]
            center_selection = "" #@param {type:"string"}
            center_axis = "xyz" #@param ["x", "y", "z", "xyz"]
            dense_phase_threshold = 0.5 #@param {type:"number"}
            density_bin_width_nm = 0.05 #@param {type:"number"}
            fit_mode = "none" #@param ["none", "translation", "transxy", "rot+trans", "rotxy+transxy", "progressive"]
            fit_selection = "" #@param {type:"string"}
            fit_reference = "tpr" #@param ["tpr", "first"]
            fit_weighting = "mass" #@param ["mass", "uniform"]
            translation_nm = "" #@param {type:"string"}
            per_frame_shift_nm = "" #@param {type:"string"}
            xtc_precision_decimals = 3 #@param {type:"integer"}

            OUTPUT_TRAJECTORY = WORKDIR / output_filename
            arguments = [
                "trjconv", "-s", TPR, "-f", TRAJECTORY, "-n", NDX,
                "-sel", output_selection, "-o", OUTPUT_TRAJECTORY,
                "-tu", time_unit, "-pbc", pbc_mode, "--center", center_mode,
                "--center-axis", center_axis, "-dpt", dense_phase_threshold,
                "--density-bin-width", density_bin_width_nm,
                "-fit", fit_mode, "--fit-reference", fit_reference,
                "--fit-weighting", fit_weighting, "-ndec", xtc_precision_decimals,
                *optional_time_arguments(start_time, end_time, sampling_interval),
            ]
            if center_selection.strip():
                arguments.extend(["--center-select", center_selection.strip()])
            if fit_selection.strip():
                arguments.extend(["--fit-select", fit_selection.strip()])
            if translation_nm.strip():
                vector = translation_nm.replace(",", " ").split()
                if len(vector) != 3:
                    raise ValueError("translation_nm requires three values")
                arguments.extend(["-trans", *vector])
            if per_frame_shift_nm.strip():
                vector = per_frame_shift_nm.replace(",", " ").split()
                if len(vector) != 3:
                    raise ValueError("per_frame_shift_nm requires three values")
                arguments.extend(["-shift", *vector])
            dps(*arguments, cwd=WORKDIR)
            dps("check", "-s", TPR, "-f", OUTPUT_TRAJECTORY, "-n", NDX, cwd=WORKDIR)
            """
        ),
        code(
            r"""
            #@title Extract a PDB snapshot
            extract_snapshot = True #@param {type:"boolean"}
            snapshot_time = 0.0 #@param {type:"number"}
            snapshot_time_unit = "ns" #@param ["fs", "ps", "ns", "us", "ms", "s"]
            snapshot_selection = "group 0" #@param {type:"string"}
            add_conect_records = True #@param {type:"boolean"}

            if extract_snapshot:
                SNAPSHOT = WORKDIR / "snapshot.pdb"
                arguments = [
                    "trjconv", "-s", TPR, "-f", OUTPUT_TRAJECTORY, "-n", NDX,
                    "-sel", snapshot_selection, "-o", SNAPSHOT,
                    "-b", snapshot_time, "-e", snapshot_time, "-tu", snapshot_time_unit,
                    "-pbc", "whole",
                ]
                if add_conect_records:
                    arguments.append("--conect")
                dps(*arguments, cwd=WORKDIR)
                print("Snapshot:", SNAPSHOT)
            """
        ),
        md("## 4 Export the prepared trajectory bundle"),
        code(
            r"""
            #@title Save manifest and download
            manifest = {
                "notebook": "03_DROPPS_1_0_Trajectory_Preparation.ipynb",
                "dropps_version": "1.0",
                "source_tpr_sha256": sha256(TPR),
                "source_trajectory_sha256": sha256(TRAJECTORY),
                "verified_ensemble": "NVT",
                "index_commands": commands,
                "commands": COMMAND_LOG,
            }
            manifest_path = WORKDIR / "trajectory_manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            export_files = [ANALYSIS_TPR, NDX, OUTPUT_TRAJECTORY, manifest_path]
            if extract_snapshot:
                export_files.append(SNAPSHOT)
            archive = make_zip(
                export_files,
                WORKDIR / "DROPPS_1_0_trajectory_bundle.zip",
                base=WORKDIR,
            )
            download(archive)
            """
        ),
    ]
    return notebook(cells)


def analysis_notebook() -> dict:
    cells = [
        md(
            r"""
            [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tymworld/DROPPS/blob/v1.0.0/examples/colab/04_DROPPS_1_0_Phase_Separation_Analysis.ipynb)

            # DROPPS 1.0 Phase Separation Analysis

            Analyse the **elongated-box NVT trajectory**, never the compact-box NPT
            trajectory. The notebook verifies that pressure coupling is disabled before
            running the analysis chain emphasized in the manuscript: density profiles;
            residue contact maps and residue-class statistics; chain size and local angle;
            intra/inter-chain distances; MSD; and spontaneous-condensation assembly size,
            composition, radius of gyration, asphericity, and ellipticity. The notebook
            also exposes the auxiliary 1.0 RMSD command.

            Select groups explicitly to avoid interactive prompts. Default group numbering
            is `System = 0`, followed by molecule types in TOP order. Use the trajectory
            preparation notebook to create and inspect custom NDX groups.
            """
        ),
        code(INSTALL_CELL),
        code(HELPERS_CELL),
        code(
            UPLOAD_BUNDLE_CELL.replace(
                'reuse_directory = "/content/dropps_simulation"',
                'reuse_directory = "/content/dropps_trajectory"',
            )
        ),
        md("## 1 Resolve files and analysis selections"),
        code(
            r"""
            #@title Input filenames and index groups
            run_input_filename = "slab_nvt.tpr" #@param {type:"string"}
            trajectory_filename = "processed.xtc" #@param {type:"string"}
            index_filename = "analysis.ndx" #@param {type:"string"}
            reference_group = 1 #@param {type:"integer"}
            two_component_system = False #@param {type:"boolean"}
            selection_group = 2 #@param {type:"integer"}
            density_center_group = 0 #@param {type:"integer"}
            slab_axis = "z"
            start_time_ns = "" #@param {type:"string"}
            end_time_ns = "" #@param {type:"string"}
            sampling_interval_ns = "" #@param {type:"string"}

            TPR = find_unique(INPUT_ROOT, run_input_filename)
            TRAJECTORY = find_unique(INPUT_ROOT, trajectory_filename)

            import zipfile

            if not zipfile.is_zipfile(TPR):
                raise ValueError("Expected a portable DROPPS 1.0 TPR v2 file.")
            with zipfile.ZipFile(TPR) as archive:
                tpr_parameters = json.loads(archive.read("parameters.json"))
            if tpr_parameters.get("pcoulp"):
                raise ValueError(
                    "Phase-coexistence analysis requires slab_nvt.tpr (pcoulp=False), "
                    "not the compact-box NPT run input."
                )
            try:
                NDX = find_unique(INPUT_ROOT, index_filename)
            except FileNotFoundError:
                NDX = None

            BASE = Path("/content") if Path("/content").is_dir() else Path.cwd()
            WORKDIR = reset_task_directory(BASE / "dropps_analysis")
            if NDX is None:
                NDX = WORKDIR / "analysis.ndx"
                result = dps(
                    "make_ndx", "-s", TPR, "-o", NDX,
                    cwd=WORKDIR, input_text="q\n", check=False,
                )
                if not NDX.is_file():
                    raise RuntimeError(
                        f"make_ndx did not create {NDX}; subprocess status was {result.returncode}."
                    )
            dps("check", "-s", TPR, "-f", TRAJECTORY, "-n", NDX, cwd=WORKDIR)
            print("Verified analysis ensemble: NVT (pcoulp = False)")

            TIME_ARGS = optional_time_arguments(
                start_time_ns, end_time_ns, sampling_interval_ns
            )
            ANALYSIS_STATUS = []

            def run_analysis(label, arguments):
                try:
                    dps(*arguments, cwd=WORKDIR)
                except Exception as exc:
                    ANALYSIS_STATUS.append({"analysis": label, "status": "failed", "error": str(exc)})
                    print(f"[{label}] FAILED: {exc}")
                    return False
                ANALYSIS_STATUS.append({"analysis": label, "status": "complete"})
                return True
            """
        ),
        md(
            r"""
            ## 2 Phase behavior

            `density` can output each component separately while centering on a shared
            reference density. `contact` supports a global cutoff or residue-specific
            σ-scaled cutoff. `cstat` reproduces the residue-class aggregation used for
            interaction interpretation in the manuscript.
            """
        ),
        code(
            r"""
            #@title Required z-axis density profile and PNG figure
            density_axis = "z"
            density_type = "mass" #@param ["mass", "charge"]
            density_bin_width_nm = 0.05 #@param {type:"number"}
            density_center_mode = "frame" #@param ["frame", "block", "none"]
            density_blocks = 5 #@param {type:"integer"}
            dense_phase_threshold = 0.5 #@param {type:"number"}

            if density_axis != "z" or slab_axis != "z":
                raise ValueError("Phase-coexistence density analysis must use the z axis.")
            calculate_groups = [reference_group]
            if two_component_system:
                calculate_groups.append(selection_group)
            arguments = [
                "density", "-s", TPR, "-f", TRAJECTORY, "-n", NDX,
                "-o", "density.xvg", "-x", "z", "-tp", density_type,
                "--bin-width", density_bin_width_nm,
                "-selfit", density_center_group, "-sel", *calculate_groups,
                "--center-mode", density_center_mode, "--blocks", density_blocks,
                "-t", dense_phase_threshold, *TIME_ARGS,
            ]
            density_ok = run_analysis("density", arguments)
            if not density_ok:
                raise RuntimeError("Required z-axis density analysis failed.")
            _, DENSITY_FIGURE = plot_xvg(
                WORKDIR / "density.xvg",
                title=f"{density_type.title()} density along z",
                output=WORKDIR / "density_z.png",
            )
            """
        ),
        code(
            r"""
            #@title Contact maps and contact-number time series
            run_contacts = True #@param {type:"boolean"}
            contact_cutoff_scheme = "global" #@param ["global", "residue"]
            global_contact_cutoff_nm = 0.7 #@param {type:"number"}
            residue_cutoff_multiplier = 1.2 #@param {type:"number"}
            remove_intra_diagonals = 2 #@param {type:"integer"}
            contact_use_pbc = True #@param {type:"boolean"}
            average_intra_chain_maps = True #@param {type:"boolean"}

            contact_ok = False
            if run_contacts:
                arguments = [
                    "contact", "-s", TPR, "-f", TRAJECTORY, "-n", NDX,
                    "-ref", reference_group,
                    "-sel", selection_group if two_component_system else reference_group,
                    "-cs", contact_cutoff_scheme,
                    "-c", global_contact_cutoff_nm, "-cm", residue_cutoff_multiplier,
                    "-rd", remove_intra_diagonals, "-otype", "dat",
                    "-orr", "contact_reference_reference.dat",
                    "-or", "contact_reference_intra.dat",
                    "-otrr", "contact_number_reference_reference.xvg",
                    *TIME_ARGS,
                ]
                if two_component_system:
                    arguments.extend(
                        [
                            "-sel", selection_group,
                            "-ors", "contact_reference_selection.dat",
                            "-otrs", "contact_number_reference_selection.xvg",
                        ]
                    )
                if contact_use_pbc:
                    arguments.append("-pbc")
                if average_intra_chain_maps:
                    arguments.append("-intraavg")
                contact_ok = run_analysis("contact", arguments)

                if contact_ok:
                    for path in sorted(WORKDIR.glob("contact_*.dat")):
                        matrix = np.loadtxt(path)
                        fig, ax = plt.subplots(figsize=(5.2, 4.4))
                        image = ax.imshow(matrix, origin="lower", aspect="auto", cmap="magma")
                        ax.set(title=path.stem, xlabel="selection residue", ylabel="reference residue")
                        fig.colorbar(image, ax=ax, label="contact value")
                        fig.tight_layout()
                        figure_path = WORKDIR / f"{path.stem}.png"
                        fig.savefig(figure_path, dpi=220, bbox_inches="tight")
                        print("Saved figure:", figure_path)
                        plt.show()
            """
        ),
        code(
            r"""
            #@title Residue-class contact statistics with dps cstat
            run_contact_statistics = True #@param {type:"boolean"}
            contact_grouping_scheme = "HPST_SC" #@param ["HPST", "AHCP", "HCP", "HPST_SC", "AHCP_SC", "HCP_SC"]
            contact_aggregation = "sum" #@param ["sum", "average"]

            if run_contact_statistics:
                if not contact_ok:
                    print("Contact statistics skipped because no contact map was generated in this run.")
                else:
                    map_name = (
                        "contact_reference_selection.dat"
                        if two_component_system
                        else "contact_reference_reference.dat"
                    )
                    selected = selection_group if two_component_system else reference_group
                    run_analysis(
                        "cstat",
                        [
                            "cstat", "-m", map_name, "-s", TPR, "-n", NDX,
                            "-ref", reference_group, "-sel", selected,
                            "-o", "contact_statistics.xlsx", "-gr",
                            "-gs", contact_grouping_scheme, "-ag", contact_aggregation,
                        ],
                    )
            """
        ),
        md("## 3 Chain conformation and dynamics"),
        code(
            r"""
            #@title Radius of gyration and backbone angles
            run_gyrate = True #@param {type:"boolean"}
            run_angle = True #@param {type:"boolean"}
            gyrate_histogram_bin_width_nm = 0.1 #@param {type:"number"}
            conformational_use_pbc = True #@param {type:"boolean"}

            if run_gyrate:
                arguments = [
                    "gyrate", "-s", TPR, "-f", TRAJECTORY, "-n", NDX,
                    "-sel", reference_group, "-oa", "gyrate_average.xvg",
                    "-ov", "gyrate_per_chain.xvg", "-oh", "gyrate_histogram.xvg",
                    "-bw", gyrate_histogram_bin_width_nm, *TIME_ARGS,
                ]
                if conformational_use_pbc:
                    arguments.append("-pbc")
                if run_analysis("gyrate", arguments):
                    plot_xvg(WORKDIR / "gyrate_average.xvg", title="Mean chain radius of gyration")

            if run_angle:
                arguments = [
                    "angle", "-s", TPR, "-f", TRAJECTORY, "-n", NDX,
                    "-sel", reference_group, "-ot", "angle_time.xvg",
                    "-or", "angle_by_residue.xvg", "-ors", "angle_statistics.xvg",
                    *TIME_ARGS,
                ]
                if conformational_use_pbc:
                    arguments.append("-pbc")
                if run_analysis("angle", arguments):
                    plot_xvg(WORKDIR / "angle_by_residue.xvg", title="Backbone angle by residue")
            """
        ),
        code(
            r"""
            #@title Intra- and inter-chain distances
            run_intra_chain_distance = False #@param {type:"boolean"}
            intra_pair_group_ids = "" #@param {type:"string"}
            run_inter_chain_distance = False #@param {type:"boolean"}
            inter_reference_group = 0 #@param {type:"integer"}
            inter_selection_group = 0 #@param {type:"integer"}
            distance_use_pbc = True #@param {type:"boolean"}

            if run_intra_chain_distance:
                pair_groups = [int(value) for value in intra_pair_group_ids.replace(",", " ").split()]
                if not pair_groups:
                    raise ValueError("Provide one or more NDX groups containing one bead pair per chain.")
                arguments = [
                    "idist", "-s", TPR, "-f", TRAJECTORY, "-n", NDX,
                    "-sel", *pair_groups, "-ot", "idist_time.xvg",
                    "-op", "idist_pairs.xvg", "-ops", "idist_statistics.xvg",
                    *TIME_ARGS,
                ]
                if distance_use_pbc:
                    arguments.append("-pbc")
                run_analysis("idist", arguments)

            if run_inter_chain_distance:
                arguments = [
                    "odist", "-s", TPR, "-f", TRAJECTORY, "-n", NDX,
                    "-ref", inter_reference_group, "-sel", inter_selection_group,
                    "-oa", "odist_average.xvg", "-ov", "odist_all_pairs.xvg",
                    *TIME_ARGS,
                ]
                if distance_use_pbc:
                    arguments.append("-pbc")
                run_analysis("odist", arguments)
            """
        ),
        code(
            r"""
            #@title Mean-square displacement and RMSD
            run_msd = True #@param {type:"boolean"}
            msd_group = 1 #@param {type:"integer"}
            msd_dimensions = "xyz" #@param ["xyz", "xy", "yz", "xz", "x", "y", "z"]
            run_rmsd = False #@param {type:"boolean"}
            rmsd_selection_expression = "(mol SCAFFOLD)" #@param {type:"string"}
            rmsd_fit_mode = "molecule" #@param ["none", "molecule", "selection"]
            rmsd_output_mode = "molecule" #@param ["molecule", "selection"]
            rmsd_mass_weighted = True #@param {type:"boolean"}

            if run_msd:
                arguments = [
                    "msd", "-s", TPR, "-f", TRAJECTORY, "-n", NDX,
                    "-sel", msd_group, "-t", msd_dimensions,
                    "-o", "msd.xvg", *TIME_ARGS,
                ]
                if run_analysis("msd", arguments):
                    plot_xvg(WORKDIR / "msd.xvg", title="Mean-square displacement")

            if run_rmsd:
                arguments = [
                    "rmsd", "-s", TPR, "-f", TRAJECTORY, "-n", NDX,
                    "-o", "rmsd.xvg", "--select", rmsd_selection_expression,
                    "--fit-mode", rmsd_fit_mode, "--output-mode", rmsd_output_mode,
                    *TIME_ARGS,
                ]
                if rmsd_mass_weighted:
                    arguments.append("--mass-weighted")
                if run_analysis("rmsd", arguments):
                    plot_xvg(WORKDIR / "rmsd.xvg", title="Molecular RMSD")
            """
        ),
        md(
            r"""
            ## 4 Spontaneous-condensation assembly analysis

            A cluster is a maximal set of chains connected through intermolecular bead
            contacts. Tune the bead cutoff and the minimum contact count to the model.
            The molecule-fraction and shape cutoff excludes very small clusters from
            composition and morphology statistics.
            """
        ),
        code(
            r"""
            #@title Cluster formation, composition, and shape
            run_assembly = True #@param {type:"boolean"}
            assembly_reference_group = 0 #@param {type:"integer"}
            assembly_contact_cutoff_nm = 0.7 #@param {type:"number"}
            contacts_to_connect_chains = 5 #@param {type:"integer"}
            large_cluster_size_cutoff = 10 #@param {type:"integer"}
            assembly_use_pbc = True #@param {type:"boolean"}

            if run_assembly:
                selection_groups = [reference_group]
                if two_component_system:
                    selection_groups.append(selection_group)
                arguments = [
                    "assembly", "-s", TPR, "-f", TRAJECTORY, "-n", NDX,
                    "-ref", assembly_reference_group, "-sel", *selection_groups,
                    "-c", assembly_contact_cutoff_nm, "-t", contacts_to_connect_chains,
                    "-mfc", large_cluster_size_cutoff,
                    "-cn", "assembly_cluster_number.xvg",
                    "-cs", "assembly_largest_size.xvg",
                    "-csd", "assembly_size_distribution.xvg",
                    "-mf", "assembly_molecule_fraction.xvg",
                    "-rgl", "assembly_largest_rg.xvg",
                    "-asp", "assembly_asphericity.xvg",
                    "-elp", "assembly_ellipticity.xvg",
                    *TIME_ARGS,
                ]
                if assembly_use_pbc:
                    arguments.append("-pbc")
                if run_analysis("assembly", arguments):
                    plot_xvg(WORKDIR / "assembly_cluster_number.xvg", title="Assembly count")
                    plot_xvg(WORKDIR / "assembly_largest_size.xvg", title="Largest assembly size")
            """
        ),
        md("## 5 Analysis manifest and results archive"),
        code(
            r"""
            #@title Summarize and download
            manifest = {
                "notebook": "04_DROPPS_1_0_Phase_Separation_Analysis.ipynb",
                "dropps_version": "1.0",
                "source_tpr": {"name": TPR.name, "sha256": sha256(TPR)},
                "source_trajectory": {"name": TRAJECTORY.name, "sha256": sha256(TRAJECTORY)},
                "source_index": {"name": NDX.name, "sha256": sha256(NDX)},
                "verified_ensemble": "NVT",
                "slab_axis": slab_axis,
                "groups": {
                    "reference": reference_group,
                    "selection": selection_group if two_component_system else None,
                    "density_center": density_center_group,
                },
                "time_window_ns": {
                    "start": start_time_ns,
                    "end": end_time_ns,
                    "interval": sampling_interval_ns,
                },
                "analysis_status": ANALYSIS_STATUS,
                "commands": COMMAND_LOG,
            }
            manifest_path = WORKDIR / "analysis_manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            export_files = [path for path in WORKDIR.iterdir() if path.is_file()]
            archive = make_zip(
                export_files,
                WORKDIR / "DROPPS_1_0_analysis_results.zip",
                base=WORKDIR,
            )
            print(json.dumps(ANALYSIS_STATUS, indent=2))
            download(archive)
            """
        ),
    ]
    return notebook(cells)


NOTEBOOKS = {
    "00_DROPPS_1_0_Quickstart.ipynb": quickstart_notebook,
    "01_DROPPS_1_0_Model_and_System_Builder.ipynb": builder_notebook,
    "02_DROPPS_1_0_Simulation.ipynb": simulation_notebook,
    "03_DROPPS_1_0_Trajectory_Preparation.ipynb": trajectory_notebook,
    "04_DROPPS_1_0_Phase_Separation_Analysis.ipynb": analysis_notebook,
}


def main() -> None:
    generated = []
    for filename, factory in NOTEBOOKS.items():
        target = HERE / filename
        target.write_text(
            json.dumps(factory(), indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        generated.append(target)

    checksum_lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
        for path in generated
    ]
    (HERE / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print("Generated:")
    for path in generated:
        print(f"  {path.name} ({path.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
