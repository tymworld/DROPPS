"""Helpers for evaluating OpenMM potential energies in an isolated Context."""

from __future__ import annotations

import openmm
from openmm import unit


def _platform_properties(platform, precision, device, threads):
    property_names = set(platform.getPropertyNames())
    properties = {}
    if "Precision" in property_names:
        properties["Precision"] = precision
    if "DeviceIndex" in property_names and device is not None:
        properties["DeviceIndex"] = str(device)
    if "OpenCLDeviceIndex" in property_names and device is not None:
        properties["OpenCLDeviceIndex"] = str(device)
    if "OpenCLPrecision" in property_names:
        properties["OpenCLPrecision"] = precision
    if "Threads" in property_names and threads is not None:
        properties["Threads"] = str(threads)
    return properties


def clone_energy_system(system, remove_context_forces=False):
    """Clone a System and optionally remove forces irrelevant to energy evaluation."""

    cloned = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(system))
    if remove_context_forces:
        for index in reversed(range(cloned.getNumForces())):
            force = cloned.getForce(index)
            if isinstance(force, (openmm.CMMotionRemover, openmm.MonteCarloBarostat)):
                cloned.removeForce(index)
    return cloned


def create_energy_context(
    system,
    platform_name="auto",
    precision="double",
    device=None,
    threads=None,
):
    """Create an OpenMM Context, falling back across platforms for ``auto``."""

    available = {
        openmm.Platform.getPlatform(index).getName(): openmm.Platform.getPlatform(index)
        for index in range(openmm.Platform.getNumPlatforms())
    }
    if platform_name.lower() == "auto":
        candidates = [
            name for name in ("CUDA", "OpenCL", "CPU", "Reference") if name in available
        ]
    else:
        matching = [name for name in available if name.lower() == platform_name.lower()]
        if not matching:
            raise ValueError(
                f"OpenMM platform '{platform_name}' is unavailable. "
                f"Available platforms: {', '.join(available)}."
            )
        candidates = matching

    failures = []
    for candidate in candidates:
        platform = available[candidate]
        properties = _platform_properties(platform, precision, device, threads)
        integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
        try:
            context = openmm.Context(system, integrator, platform, properties)
        except Exception as exc:
            failures.append(f"{candidate}: {exc}")
            if platform_name.lower() != "auto":
                raise RuntimeError(
                    f"Failed to create an OpenMM Context on {candidate}: {exc}"
                ) from exc
            continue
        return context, integrator, candidate, properties

    raise RuntimeError(
        "Could not create an OpenMM energy-evaluation Context. " + " | ".join(failures)
    )


def potential_energy_kj_per_mol(context, positions_nm, box_vectors_nm):
    """Evaluate potential energy for one explicitly supplied configuration."""

    import numpy as np

    box = np.asarray(box_vectors_nm, dtype=float)
    vectors = [openmm.Vec3(*row) * unit.nanometer for row in box]
    context.setPeriodicBoxVectors(*vectors)
    context.setPositions(np.asarray(positions_nm, dtype=float) * unit.nanometer)
    state = context.getState(getEnergy=True)
    return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
