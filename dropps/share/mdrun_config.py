"""Configuration helpers for :mod:`dropps.commands.mdrun`.

The TPR owns scientific simulation settings.  This module resolves the smaller
set of operational overrides that are intentionally configurable at run time:
output locations, hardware selection, run length, checkpoint cadence, and
restart file handling.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class OutputPaths:
    """Resolved output paths for one ``mdrun`` invocation."""

    prefix: str
    log: str
    energy: str
    trajectory: str
    warming_trajectory: str
    checkpoint: str
    final_structure: str
    stress: str

    @property
    def restartable_outputs(self) -> tuple[str, ...]:
        """Outputs that receive a part number with ``--no-append``."""

        return (
            self.log,
            self.energy,
            self.trajectory,
            self.final_structure,
            self.stress,
        )

    def with_part_number(self, part_number: int) -> "OutputPaths":
        """Return paths for a non-appending continuation.

        The checkpoint output deliberately keeps its original name, matching
        GROMACS' continuation behavior and keeping ``-cpi`` without a filename
        useful for the next invocation.
        """

        if part_number < 2:
            raise ValueError("Continuation part numbers must be at least 2.")
        return replace(
            self,
            log=_insert_part_number(self.log, part_number),
            energy=_insert_part_number(self.energy, part_number),
            trajectory=_insert_part_number(self.trajectory, part_number),
            final_structure=_insert_part_number(self.final_structure, part_number),
            stress=_insert_part_number(self.stress, part_number),
        )


@dataclass(frozen=True)
class RuntimeOptions:
    """Validated operational settings resolved from CLI and TPR values."""

    production_steps: int
    random_seed: int
    checkpoint_minutes: float
    max_hours: float | None
    append: bool
    number_checkpoints: bool
    platform: str
    device_index: str | None
    precision: str | None
    threads: int | None

    @property
    def checkpoint_interval_seconds(self) -> float | None:
        if self.checkpoint_minutes == 0:
            return None
        return self.checkpoint_minutes * 60.0

    @property
    def max_run_seconds(self) -> float | None:
        if self.max_hours is None:
            return None
        return self.max_hours * 3600.0 * 0.99


def resolve_output_paths(args: Any, parameters: Mapping[str, Any]) -> OutputPaths:
    """Resolve default-prefix and individual output-file overrides."""

    prefix = os.fspath(args.output_prefix)
    stress_path = _optional_path(args, "stress_output")
    if stress_path is None:
        configured_stress_path = parameters.get("stress_output", "auto")
        configured_stress_path = os.fspath(configured_stress_path)
        if configured_stress_path.lower() == "auto":
            stress_path = prefix + ".stress.xvg"
        else:
            stress_path = configured_stress_path

    return OutputPaths(
        prefix=prefix,
        log=_optional_path(args, "log_output") or prefix + ".log",
        energy=_optional_path(args, "energy_output") or prefix + ".edr",
        trajectory=(_optional_path(args, "trajectory_output") or prefix + ".xtc"),
        warming_trajectory=prefix + ".warming.xtc",
        checkpoint=(_optional_path(args, "checkpoint_output") or prefix + ".chk"),
        final_structure=(_optional_path(args, "final_structure") or prefix + ".pdb"),
        stress=stress_path,
    )


def resolve_runtime_options(args: Any, parameters: Mapping[str, Any]) -> RuntimeOptions:
    """Apply CLI-over-TPR precedence and validate operational overrides."""

    cli_steps = getattr(args, "nsteps", None)
    production_steps = (
        int(parameters["nsteps"]) if cli_steps is None else int(cli_steps)
    )
    if production_steps < 0:
        raise ValueError("nsteps must be zero or greater.")

    cli_seed = getattr(args, "seed", None)
    random_seed = int(parameters.get("seed", 0) if cli_seed is None else cli_seed)
    if random_seed < 0:
        raise ValueError("seed must be zero or greater.")

    checkpoint_minutes = float(getattr(args, "checkpoint_minutes", 5.0))
    if not math.isfinite(checkpoint_minutes) or checkpoint_minutes < 0:
        raise ValueError("cpt must be a finite number that is zero or greater.")

    raw_max_hours = getattr(args, "max_hours", -1.0)
    if raw_max_hours is None or float(raw_max_hours) == -1.0:
        max_hours = None
    else:
        max_hours = float(raw_max_hours)
        if not math.isfinite(max_hours) or max_hours < 0:
            raise ValueError(
                "maxh must be -1 or a finite number that is zero or greater."
            )

    platform = str(getattr(args, "platform", "auto")).strip()
    platform_lookup = {
        "auto": "auto",
        "cuda": "CUDA",
        "opencl": "OpenCL",
        "cpu": "CPU",
        "reference": "Reference",
    }
    try:
        platform = platform_lookup[platform.lower()]
    except KeyError as exc:
        choices = ", ".join(platform_lookup.values())
        raise ValueError(
            f"Unknown OpenMM platform {platform!r}; choose from {choices}."
        ) from exc

    raw_device_index = getattr(args, "device_index", None)
    device_index = None
    if raw_device_index is not None:
        device_index = str(raw_device_index).strip()
        if not device_index:
            raise ValueError("gpu_id must not be empty.")

    raw_precision = getattr(args, "precision", None)
    precision = None if raw_precision is None else str(raw_precision).lower()
    if precision not in {None, "single", "mixed", "double"}:
        raise ValueError("precision must be single, mixed, or double.")

    raw_threads = getattr(args, "threads", None)
    threads = None if raw_threads in {None, 0} else int(raw_threads)
    if threads is not None and threads < 1:
        raise ValueError("nt must be zero or a positive integer.")
    if threads is not None and (device_index is not None or precision is not None):
        raise ValueError(
            "nt cannot be combined with gpu_id or precision because OpenMM "
            "exposes Threads only on the CPU platform."
        )
    if threads is not None and platform not in {"auto", "CPU"}:
        raise ValueError("nt requires platform auto or CPU.")
    if (device_index is not None or precision is not None) and platform not in {
        "auto",
        "CUDA",
        "OpenCL",
    }:
        raise ValueError("gpu_id and precision require a CUDA or OpenCL platform.")

    return RuntimeOptions(
        production_steps=production_steps,
        random_seed=random_seed,
        checkpoint_minutes=checkpoint_minutes,
        max_hours=max_hours,
        append=bool(getattr(args, "append", True)),
        number_checkpoints=bool(getattr(args, "number_checkpoints", False)),
        platform=platform,
        device_index=device_index,
        precision=precision,
        threads=threads,
    )


def next_part_number(paths: OutputPaths) -> int:
    """Find the first continuation part number not used by any output."""

    part_number = 2
    while any(
        os.path.exists(_insert_part_number(path, part_number))
        for path in paths.restartable_outputs
    ):
        part_number += 1
    return part_number


def numbered_checkpoint_path(path: str, step: int) -> str:
    """Add an unambiguous production-step suffix to a checkpoint path."""

    checkpoint_path = Path(path)
    suffix = "".join(checkpoint_path.suffixes)
    base_name = checkpoint_path.name[: -len(suffix)] if suffix else checkpoint_path.name
    numbered_name = f"{base_name}.step{int(step):012d}{suffix}"
    return os.fspath(checkpoint_path.with_name(numbered_name))


def _optional_path(args: Any, name: str) -> str | None:
    value = getattr(args, name, None)
    return None if value is None else os.fspath(value)


def _insert_part_number(path: str, part_number: int) -> str:
    output_path = Path(path)
    suffix = "".join(output_path.suffixes)
    base_name = output_path.name[: -len(suffix)] if suffix else output_path.name
    part_name = f"{base_name}.part{part_number:04d}{suffix}"
    return os.fspath(output_path.with_name(part_name))
