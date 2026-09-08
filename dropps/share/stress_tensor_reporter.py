"""OpenMM reporter for the instantaneous microscopic pressure tensor."""

from __future__ import annotations

import os

import numpy as np
from openmm import unit

from dropps.share.openmm_energy import (
    clone_energy_system,
    create_energy_context,
    potential_energy_kj_per_mol,
)


PRESSURE_COMPONENTS = ("Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz")


def synchronized_velocities(
    masses_dalton,
    velocities_nm_per_ps,
    forces_kj_per_mol_nm,
    timestep_ps,
    *,
    positions_nm=None,
    box_vectors_nm=None,
    constraints=(),
):
    """Convert leapfrog half-step velocities to the position time slice."""

    masses = np.asarray(masses_dalton, dtype=float)
    velocities = np.asarray(velocities_nm_per_ps, dtype=float)
    forces = np.asarray(forces_kj_per_mol_nm, dtype=float)
    if velocities.shape != (len(masses), 3) or forces.shape != velocities.shape:
        raise ValueError("Velocity/force arrays do not match the particle masses.")
    synchronized = velocities.copy()
    movable = masses > 0.0
    synchronized[movable] += (
        0.5 * float(timestep_ps) * forces[movable] / masses[movable, None]
    )

    constraints = tuple(constraints)
    if not constraints:
        return synchronized
    if positions_nm is None or box_vectors_nm is None:
        raise ValueError(
            "Positions and box vectors are required to synchronize constrained velocities."
        )

    positions = np.asarray(positions_nm, dtype=float)
    box = np.asarray(box_vectors_nm, dtype=float)
    inverse_box = np.linalg.inv(box)
    inverse_masses = np.divide(1.0, masses, out=np.zeros_like(masses), where=movable)
    constraint_vectors = []
    for first, second in constraints:
        displacement = positions[second] - positions[first]
        fractional = displacement @ inverse_box
        displacement -= np.round(fractional) @ box
        inverse_mass_sum = inverse_masses[first] + inverse_masses[second]
        denominator = float(np.dot(displacement, displacement)) * inverse_mass_sum
        if denominator <= 0.0:
            raise ValueError("Constraint connects particles with no movable mass.")
        constraint_vectors.append((first, second, displacement, denominator))

    # Sparse Gauss-Seidel/RATTLE velocity projection avoids allocating the
    # dense constraint matrix, which would be prohibitive for long polymers.
    tolerance = 1.0e-12
    for _iteration in range(200):
        maximum_violation = 0.0
        for first, second, displacement, denominator in constraint_vectors:
            violation = float(
                np.dot(displacement, synchronized[second] - synchronized[first])
            )
            maximum_violation = max(maximum_violation, abs(violation))
            multiplier = violation / denominator
            synchronized[first] += multiplier * inverse_masses[first] * displacement
            synchronized[second] -= multiplier * inverse_masses[second] * displacement
        if maximum_violation <= tolerance:
            return synchronized
    raise ValueError("Constrained velocity synchronization did not converge.")


def _require_orthorhombic_box(box_vectors_nm, tolerance=1.0e-8):
    box = np.asarray(box_vectors_nm, dtype=float)
    if box.shape != (3, 3):
        raise ValueError("Periodic box vectors must form a 3 by 3 matrix.")
    off_diagonal = box - np.diag(np.diag(box))
    if not np.allclose(off_diagonal, 0.0, atol=tolerance, rtol=0.0):
        raise ValueError(
            "Runtime pressure-tensor recording currently requires an "
            "orthorhombic simulation box."
        )
    if np.any(np.diag(box) <= 0.0):
        raise ValueError("Periodic box lengths must be positive.")
    return box


def deformation_matrix(component, strain):
    """Return the deformation used to differentiate one pressure component."""

    if component in ("xx", "yy", "zz"):
        axis = "xyz".index(component[0])
        matrix = np.eye(3)
        matrix[axis, axis] = np.exp(strain)
        return matrix

    if component not in ("xy", "xz", "yz"):
        raise ValueError(f"Unknown pressure-tensor component: {component}")
    first, second = ("xyz".index(axis) for axis in component)
    matrix = np.eye(3)
    # This lower-triangular-box-compatible simple shear has determinant one.
    # For rotationally invariant force fields the microscopic virial is
    # symmetric, so dU/d(strain) yields the corresponding symmetric component.
    matrix[first, second] = strain
    return matrix


def deform_configuration(positions_nm, box_vectors_nm, component, strain):
    """Apply a diagonal or simple shear deformation at fixed fractional positions."""

    transform = deformation_matrix(component, strain)
    positions = np.asarray(positions_nm, dtype=float) @ transform.T
    box = np.asarray(box_vectors_nm, dtype=float) @ transform.T
    return positions, box


def kinetic_pressure_tensor(masses_dalton, velocities_nm_per_ps, volume_nm3):
    """Return the kinetic pressure tensor in kJ mol^-1 nm^-3."""

    masses = np.asarray(masses_dalton, dtype=float)
    velocities = np.asarray(velocities_nm_per_ps, dtype=float)
    if velocities.shape != (len(masses), 3):
        raise ValueError("Velocity array shape does not match the particle masses.")
    if volume_nm3 <= 0.0:
        raise ValueError("Periodic box volume must be positive.")
    return np.einsum("i,ia,ib->ab", masses, velocities, velocities) / volume_nm3


def configurational_pressure_tensor(
    context,
    positions_nm,
    box_vectors_nm,
    strain=3.0e-4,
):
    """Return the symmetric configurational pressure tensor by energy derivatives."""

    if not 0.0 < strain < 0.02:
        raise ValueError("Finite-difference strain must be between 0 and 0.02.")
    box = _require_orthorhombic_box(box_vectors_nm)
    volume = abs(float(np.linalg.det(box)))
    pressure = np.zeros((3, 3), dtype=float)
    component_axes = {
        "xx": (0, 0),
        "yy": (1, 1),
        "zz": (2, 2),
        "xy": (0, 1),
        "xz": (0, 2),
        "yz": (1, 2),
    }
    for component, (first, second) in component_axes.items():
        positions_plus, box_plus = deform_configuration(
            positions_nm, box, component, strain
        )
        positions_minus, box_minus = deform_configuration(
            positions_nm, box, component, -strain
        )
        energy_plus = potential_energy_kj_per_mol(context, positions_plus, box_plus)
        energy_minus = potential_energy_kj_per_mol(context, positions_minus, box_minus)
        value = -(energy_plus - energy_minus) / (2.0 * strain * volume)
        pressure[first, second] = value
        pressure[second, first] = value
    return pressure


def pressure_tensor(
    context,
    masses_dalton,
    positions_nm,
    velocities_nm_per_ps,
    box_vectors_nm,
    strain=3.0e-4,
):
    """Return kinetic + configurational pressure in kJ mol^-1 nm^-3."""

    box = _require_orthorhombic_box(box_vectors_nm)
    volume = abs(float(np.linalg.det(box)))
    kinetic = kinetic_pressure_tensor(masses_dalton, velocities_nm_per_ps, volume)
    configurational = configurational_pressure_tensor(
        context, positions_nm, box, strain=strain
    )
    return kinetic + configurational


class StressTensorReporter:
    """Stream the six independent components of the pressure tensor to an XVG file."""

    def __init__(
        self,
        file,
        report_interval,
        strain=3.0e-4,
        temperature=None,
        pressure_coupling=None,
        platform="auto",
        precision="double",
        device=None,
        threads=None,
        append=False,
    ):
        if int(report_interval) <= 0:
            raise ValueError("Stress report interval must be positive.")
        self._file_name = os.fspath(file)
        self._report_interval = int(report_interval)
        self._strain = float(strain)
        self._temperature = temperature
        self._pressure_coupling = pressure_coupling
        self._platform_name = platform
        self._precision = precision
        self._device = device
        self._threads = threads
        self._append = bool(append)
        self._context = None
        self._integrator = None
        self._masses = None
        self._stream = None
        self._platform_used = None

    def describeNextReport(self, simulation):
        steps = self._report_interval - simulation.currentStep % self._report_interval
        return steps, True, True, True, False

    def _initialize(self, simulation):
        if simulation.system.getNumConstraints() > 0:
            raise ValueError(
                "Pressure-tensor recording does not support constrained bonds: "
                "their constraint virial is unavailable to the finite-strain estimator."
            )
        energy_system = clone_energy_system(
            simulation.system, remove_context_forces=True
        )
        (
            self._context,
            self._integrator,
            self._platform_used,
            properties,
        ) = create_energy_context(
            energy_system,
            platform_name=self._platform_name,
            precision=self._precision,
            device=self._device,
            threads=self._threads,
        )
        self._masses = np.asarray(
            [
                simulation.system.getParticleMass(index).value_in_unit(unit.dalton)
                for index in range(simulation.system.getNumParticles())
            ],
            dtype=float,
        )

        has_existing_data = (
            self._append
            and os.path.exists(self._file_name)
            and os.path.getsize(self._file_name) > 0
        )
        mode = "a" if has_existing_data else "w"
        self._stream = open(self._file_name, mode, encoding="utf-8")
        if not has_existing_data:
            self._write_header(properties)

    def _write_header(self, properties):
        temperature = (
            "unknown"
            if self._temperature is None
            else f"{float(self._temperature):.10g}"
        )
        pressure_coupling = (
            "unknown"
            if self._pressure_coupling is None
            else str(bool(self._pressure_coupling))
        )
        self._stream.write("# DROPPS instantaneous microscopic pressure tensor\n")
        self._stream.write(
            "# Method: kinetic dyadic + configurational central finite-strain derivative\n"
        )
        self._stream.write(
            "# pressure_unit=kJ_mol^-1_nm^-3 "
            f"temperature_K={temperature} strain={self._strain:.10g} "
            f"pressure_coupling={pressure_coupling} "
            f"platform={self._platform_used} properties={properties}\n"
        )
        self._stream.write(
            "# columns: step time_ps volume_nm3 Pxx Pyy Pzz Pxy Pxz Pyz\n"
        )
        self._stream.write('@ title "Instantaneous microscopic pressure tensor"\n')
        self._stream.write('@ xaxis label "Time (ps)"\n')
        self._stream.write('@ yaxis label "Pressure (kJ mol^-1 nm^-3)"\n')
        self._stream.write("@TYPE xy\n")

    def report(self, simulation, state):
        if self._context is None:
            self._initialize(simulation)

        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        velocities = state.getVelocities(asNumpy=True).value_in_unit(
            unit.nanometer / unit.picosecond
        )
        forces = state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        )
        velocities = synchronized_velocities(
            self._masses,
            velocities,
            forces,
            simulation.integrator.getStepSize().value_in_unit(unit.picosecond),
        )
        box_vectors = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(
            unit.nanometer
        )
        volume = abs(float(np.linalg.det(box_vectors)))
        tensor = pressure_tensor(
            self._context,
            self._masses,
            positions,
            velocities,
            box_vectors,
            strain=self._strain,
        )
        time_ps = state.getTime().value_in_unit(unit.picosecond)
        values = (
            simulation.currentStep,
            time_ps,
            volume,
            tensor[0, 0],
            tensor[1, 1],
            tensor[2, 2],
            tensor[0, 1],
            tensor[0, 2],
            tensor[1, 2],
        )
        self._stream.write(
            f"{values[0]:d} " + " ".join(f"{value:.12g}" for value in values[1:]) + "\n"
        )
        self._stream.flush()

    def close(self):
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        self._context = None
        self._integrator = None

    def __del__(self):
        self.close()
