#!/usr/bin/env python3
"""Run a small CUDA simulation, native-checkpoint resume, and trajectory analysis."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import openmm
import MDAnalysis as mda

from dropps import __version__
from dropps.dps import main as dps_main
from scripts.install_dropps import verify_openmm


def run(*args):
    status = dps_main(["dps", *map(str, args)])
    if status:
        raise RuntimeError(f"Command failed ({status}): {args}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    verify_openmm(require_cuda=True)
    run("pdb2dps", "-s", "AGKDEAGKDEAGKDEAGKDE", "-ff", "HPS",
        "-o", work / "chain.pdb", "-op", work / "chain.itp")
    run("genmesh", "-f", work / "chain.pdb", "-p", work / "chain.itp", "-n", "2",
        "-mx", "10", "-my", "10", "-mz", "10", "-o", work / "system.pdb", "-op", work / "system.top")
    mdp = (ROOT / "tests/data/smoke.mdp").read_text()
    mdp = mdp.replace("nsteps = 6", "nsteps = 100").replace("nst-xout = 2", "nst-xout = 10")
    (work / "run.mdp").write_text(mdp)
    run("grompp", "-f", work / "system.pdb", "-p", work / "system.top",
        "-m", work / "run.mdp", "-o", work / "run.tpr")
    common = ["mdrun", "-s", work / "run.tpr", "-o", work / "run",
              "--platform", "CUDA", "--precision", "mixed", "-cpt", "0"]
    run(*common)
    run(*common, "--checkpoint", "--nsteps", "200")
    run("check", "-s", work / "run.tpr", "-f", work / "run.xtc")
    run("msd", "-s", work / "run.tpr", "-f", work / "run.xtc", "-sel", "0", "-o", work / "msd.xvg")
    with mda.coordinates.XTC.XTCReader(str(work / "run.xtc")) as trajectory:
        times = np.array([frame.time for frame in trajectory])
        assert len(times) == 20, times
        assert np.all(np.diff(times) > 0), times
        assert np.isclose(times[-1], 2.0), times
        assert np.isfinite(trajectory[-1].positions).all()
    result = dict(dropps=__version__, python=platform.python_version(), openmm=openmm.__version__,
                  numpy=np.__version__, mdanalysis=mda.__version__, platform="CUDA", precision="mixed",
                  atoms=40, final_step=200, frames=20, final_time_ps=float(times[-1]),
                  native_checkpoint_resume=True, msd=True)
    (work / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
