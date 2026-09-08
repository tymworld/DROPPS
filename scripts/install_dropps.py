#!/usr/bin/env python3
"""Install DROPPS with an OpenMM CUDA extra when NVIDIA hardware is visible."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
CUDA_VERSION_PATTERN = re.compile(r"CUDA Version:\s*(\d+)(?:\.(\d+))?")


@dataclass(frozen=True)
class NvidiaProbe:
    devices: tuple[str, ...]
    max_cuda_version: tuple[int, int] | None
    error: str | None = None

    @property
    def has_gpu(self) -> bool:
        return bool(self.devices)


def parse_cuda_version(output: str) -> tuple[int, int] | None:
    """Parse the maximum CUDA version reported in the nvidia-smi header."""

    match = CUDA_VERSION_PATTERN.search(output)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2) or 0)


def probe_nvidia() -> NvidiaProbe:
    """Return visible NVIDIA devices and the CUDA level supported by the driver."""

    executable = shutil.which("nvidia-smi")
    if executable is None:
        return NvidiaProbe((), None, "nvidia-smi was not found")

    device_result = subprocess.run(
        [executable, "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    if device_result.returncode != 0:
        detail = (device_result.stderr or device_result.stdout).strip()
        return NvidiaProbe(
            (),
            None,
            detail or f"nvidia-smi exited with status {device_result.returncode}",
        )

    devices = tuple(
        line.strip() for line in device_result.stdout.splitlines() if line.strip()
    )
    if not devices:
        return NvidiaProbe((), None, "nvidia-smi reported no visible GPUs")

    version_result = subprocess.run(
        [executable],
        capture_output=True,
        text=True,
        check=False,
    )
    if version_result.returncode != 0:
        detail = (version_result.stderr or version_result.stdout).strip()
        return NvidiaProbe(
            devices,
            None,
            detail or "nvidia-smi could not report driver capabilities",
        )

    max_cuda_version = parse_cuda_version(version_result.stdout)
    error = None
    if max_cuda_version is None:
        error = "could not parse the CUDA version from nvidia-smi"
    return NvidiaProbe(devices, max_cuda_version, error)


def select_install_extra(
    probe: NvidiaProbe,
    requested: str,
    require_cuda: bool = False,
) -> str:
    """Choose a DROPPS CUDA extra or the current CPU/OpenCL stack."""

    if requested == "none":
        if require_cuda:
            raise RuntimeError("--require-cuda cannot be combined with --cuda none")
        return "latest"
    if requested in {"12", "13"}:
        return f"cuda{requested}"

    if not probe.has_gpu:
        if require_cuda:
            raise RuntimeError(
                "No visible NVIDIA GPU was detected"
                + (f": {probe.error}" if probe.error else ".")
            )
        return "latest"
    if probe.max_cuda_version is None:
        raise RuntimeError(
            "An NVIDIA GPU is visible, but its driver compatibility could not be "
            "determined. Use --cuda 12 or --cuda 13 explicitly after checking "
            "the installed NVIDIA driver."
        )

    major, minor = probe.max_cuda_version
    if major >= 13:
        return "cuda13"
    if major >= 12:
        return "cuda12"
    raise RuntimeError(
        f"The NVIDIA driver reports CUDA {major}.{minor} compatibility, but "
        "OpenMM 8.5 CUDA wheels require CUDA 12 or newer. Upgrade the NVIDIA "
        "driver, or pass --cuda none to install without CUDA."
    )


def add_extra(package: str, extra: str | None) -> str:
    if extra is None:
        return package
    if "[" in package or "]" in package:
        raise ValueError(
            "--package must not already contain extras; the installer selects them"
        )
    return f"{package}[{extra}]"


def build_pip_command(
    package: str,
    extra: str | None,
    editable: bool = False,
) -> list[str]:
    command = [sys.executable, "-m", "pip", "install"]
    if editable:
        command.append("--editable")
    command.append(add_extra(package, extra))
    return command


def verify_openmm(require_cuda: bool) -> None:
    """Import OpenMM and, when required, initialize an actual CUDA Context."""

    if require_cuda:
        code = """
import openmm

failures = openmm.Platform.getPluginLoadFailures()
if failures:
    print("OpenMM plugin load warnings:", *failures, sep="\\n  ")
platform = openmm.Platform.getPlatformByName("CUDA")
system = openmm.System()
system.addParticle(1.0)
force = openmm.CustomExternalForce("x*x")
force.addParticle(0, [])
system.addForce(force)
integrator = openmm.VerletIntegrator(0.001)
context = openmm.Context(system, integrator, platform)
context.setPositions([openmm.Vec3(0.0, 0.0, 0.0)])
context.getState(getEnergy=True)
print("CUDA verification passed on OpenMM platform:", context.getPlatform().getName())
del context, integrator
"""
    else:
        code = """
import openmm

platforms = [
    openmm.Platform.getPlatform(index).getName()
    for index in range(openmm.Platform.getNumPlatforms())
]
print("OpenMM verification passed. Available platforms:", ", ".join(platforms))
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Install DROPPS and automatically add the matching OpenMM CUDA "
            "platform when an NVIDIA GPU is visible."
        )
    )
    parser.add_argument(
        "--package",
        default=str(REPO_ROOT),
        help="Package name, source directory, or wheel path to install (default: repository root).",
    )
    parser.add_argument(
        "--cuda",
        choices=("auto", "12", "13", "none"),
        default="auto",
        help="CUDA generation to install; auto detects it with nvidia-smi (default: auto).",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail if auto detection finds no visible NVIDIA GPU.",
    )
    parser.add_argument(
        "--editable",
        action="store_true",
        help="Pass --editable to pip; intended for a local source checkout.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip the post-install OpenMM Context test (not recommended).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected installation command without running it.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    probe = probe_nvidia()

    if probe.has_gpu:
        print("Detected NVIDIA GPU(s):", ", ".join(probe.devices))
        if probe.max_cuda_version is not None:
            major, minor = probe.max_cuda_version
            print(f"NVIDIA driver maximum CUDA compatibility: {major}.{minor}")
    else:
        print(
            "No visible NVIDIA GPU detected; CUDA will not be installed automatically."
        )
        if probe.error:
            print("Detection detail:", probe.error)

    try:
        extra = select_install_extra(probe, args.cuda, args.require_cuda)
        command = build_pip_command(args.package, extra, args.editable)
    except (RuntimeError, ValueError) as exc:
        print(f"Installation aborted: {exc}", file=sys.stderr)
        return 2

    require_cuda = extra.startswith("cuda")
    if not require_cuda:
        print("Selected OpenMM CPU/OpenCL installation.")
    else:
        print(f"Selected OpenMM {extra.upper()} installation.")
    print("Running:", " ".join(command))

    if args.dry_run:
        return 0

    try:
        subprocess.run(command, check=True)
        if not args.skip_verify:
            verify_openmm(require_cuda=require_cuda)
    except subprocess.CalledProcessError as exc:
        print(
            f"Installation or OpenMM verification failed with status {exc.returncode}.",
            file=sys.stderr,
        )
        return exc.returncode or 1

    if require_cuda:
        print(
            "DROPPS CUDA installation verified. Use --platform CUDA for "
            "production runs that must never fall back to CPU."
        )
    else:
        print("DROPPS installation verified without a required CUDA platform.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
