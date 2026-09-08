"""DROPPS EDR I/O and an OpenMM reporter for thermodynamic observables.

DROPPS ``.edr`` files deliberately use a documented CSV representation.  They
are not byte-compatible with GROMACS EDR files; the extension denotes their
role in the simulation workflow rather than their binary encoding.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass

import numpy as np
from openmm import unit

from dropps.share.openmm_energy import clone_energy_system, create_energy_context
from dropps.share.stress_tensor_reporter import (
    pressure_tensor,
    synchronized_velocities,
)


EDR_MAGIC = "# DROPPS EDR CSV"
EDR_FORMAT_VERSION = 1
PRESSURE_FIELDS = frozenset(("pressure", "Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz"))


@dataclass(frozen=True)
class EnergyField:
    name: str
    unit: str
    description: str


ENERGY_FIELDS = {
    field.name: field
    for field in (
        EnergyField("potentialEnergy", "kJ/mol", "Potential energy"),
        EnergyField("kineticEnergy", "kJ/mol", "Kinetic energy"),
        EnergyField("totalEnergy", "kJ/mol", "Total energy"),
        EnergyField("temperature", "K", "Instantaneous temperature"),
        EnergyField("boxX", "nm", "Length of periodic box vector a"),
        EnergyField("boxY", "nm", "Length of periodic box vector b"),
        EnergyField("boxZ", "nm", "Length of periodic box vector c"),
        EnergyField("volume", "nm^3", "Periodic box volume"),
        EnergyField("density", "g/mL", "System mass density"),
        EnergyField("pressure", "kJ mol^-1 nm^-3", "Instantaneous isotropic pressure"),
        EnergyField("Pxx", "kJ mol^-1 nm^-3", "Pressure tensor xx component"),
        EnergyField("Pyy", "kJ mol^-1 nm^-3", "Pressure tensor yy component"),
        EnergyField("Pzz", "kJ mol^-1 nm^-3", "Pressure tensor zz component"),
        EnergyField("Pxy", "kJ mol^-1 nm^-3", "Pressure tensor xy component"),
        EnergyField("Pxz", "kJ mol^-1 nm^-3", "Pressure tensor xz component"),
        EnergyField("Pyz", "kJ mol^-1 nm^-3", "Pressure tensor yz component"),
    )
}

RERUN_FIELDS = {
    field.name: field
    for field in (
        ENERGY_FIELDS["potentialEnergy"],
        ENERGY_FIELDS["boxX"],
        ENERGY_FIELDS["boxY"],
        ENERGY_FIELDS["boxZ"],
        ENERGY_FIELDS["volume"],
        ENERGY_FIELDS["density"],
        EnergyField(
            "configPressure",
            "kJ mol^-1 nm^-3",
            "Configurational isotropic pressure",
        ),
        EnergyField("configPxx", "kJ mol^-1 nm^-3", "Configurational pressure xx"),
        EnergyField("configPyy", "kJ mol^-1 nm^-3", "Configurational pressure yy"),
        EnergyField("configPzz", "kJ mol^-1 nm^-3", "Configurational pressure zz"),
        EnergyField("configPxy", "kJ mol^-1 nm^-3", "Configurational pressure xy"),
        EnergyField("configPxz", "kJ mol^-1 nm^-3", "Configurational pressure xz"),
        EnergyField("configPyz", "kJ mol^-1 nm^-3", "Configurational pressure yz"),
    )
}

DEFAULT_ENERGY_FIELDS = (
    "potentialEnergy",
    "kineticEnergy",
    "totalEnergy",
    "temperature",
    "boxX",
    "boxY",
    "boxZ",
    "volume",
    "density",
)

_ENERGY_VALUE_FIELDS = frozenset(
    ("potentialEnergy", "kineticEnergy", "totalEnergy", "temperature")
)
_KINETIC_VALUE_FIELDS = frozenset(("kineticEnergy", "totalEnergy", "temperature"))
_DALTON_PER_NM3_TO_G_PER_ML = 0.00166053906660
_R_KJ_PER_MOL_K = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(
    unit.kilojoule_per_mole / unit.kelvin
)


@dataclass(frozen=True)
class EDRHeader:
    fields: tuple[str, ...]
    units: dict[str, str]
    metadata: dict[str, object]


@dataclass(frozen=True)
class EDRData:
    header: EDRHeader
    steps: np.ndarray
    times_ps: np.ndarray
    values: dict[str, np.ndarray]


def normalize_field_names(raw_fields, available_fields=ENERGY_FIELDS):
    """Return canonical, de-duplicated field names in the requested order."""

    if isinstance(raw_fields, str):
        tokens = raw_fields.split(",")
    else:
        tokens = [part for token in raw_fields for part in str(token).split(",")]
    canonical_by_lower = {name.lower(): name for name in available_fields}
    normalized = []
    for token in tokens:
        name = str(token).strip()
        if not name:
            continue
        if name.lower() in ("step", "time", "time_ps"):
            continue
        canonical = canonical_by_lower.get(name.lower())
        if canonical is None:
            choices = ", ".join(available_fields)
            raise ValueError(
                f"Unknown energy term '{name}'. Available terms: {choices}."
            )
        if canonical not in normalized:
            normalized.append(canonical)
    if not normalized:
        raise ValueError("At least one energy term must be selected.")
    return tuple(normalized)


def read_edr_header(path):
    """Read and validate a DROPPS CSV-EDR header without loading its data."""

    metadata = {}
    header_fields = None
    saw_magic = False
    try:
        with open(path, "r", encoding="utf-8", newline="") as stream:
            for line in stream:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped == EDR_MAGIC:
                    saw_magic = True
                    continue
                if stripped.startswith("# format_version="):
                    version = int(stripped.split("=", 1)[1])
                    if version != EDR_FORMAT_VERSION:
                        raise ValueError(
                            f"Unsupported DROPPS EDR format version {version}."
                        )
                    continue
                if stripped.startswith("# metadata="):
                    metadata = json.loads(stripped.split("=", 1)[1])
                    continue
                if stripped.startswith("#"):
                    continue
                header_fields = next(csv.reader([line]))
                break
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"'{path}' is not a DROPPS CSV-EDR file. GROMACS binary EDR files "
            "are not supported."
        ) from exc

    if not saw_magic:
        raise ValueError(
            f"'{path}' is not a DROPPS CSV-EDR file (missing format marker)."
        )
    if header_fields is None or header_fields[:2] != ["step", "time_ps"]:
        raise ValueError("DROPPS EDR must begin with step,time_ps columns.")
    fields = tuple(header_fields[2:])
    if len(fields) != len(set(fields)):
        raise ValueError("DROPPS EDR contains duplicate field names.")
    units = dict(metadata.get("units", {}))
    return EDRHeader(fields=fields, units=units, metadata=metadata)


def read_edr(path):
    """Load a DROPPS EDR table and validate every numeric row."""

    header = read_edr_header(path)
    rows = []
    found_columns = False
    expected_columns = 2 + len(header.fields)
    with open(path, "r", encoding="utf-8", newline="") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            columns = next(csv.reader([line]))
            if not found_columns:
                found_columns = True
                continue
            if len(columns) != expected_columns:
                raise ValueError(
                    f"{path}:{line_number}: expected {expected_columns} columns, "
                    f"found {len(columns)}."
                )
            try:
                step = int(columns[0])
                numeric = [float(value) for value in columns[1:]]
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: EDR data row is not numeric."
                ) from exc
            if not np.all(np.isfinite(numeric)):
                raise ValueError(f"{path}:{line_number}: EDR row is not finite.")
            rows.append((step, *numeric))

    if rows:
        steps = np.asarray([row[0] for row in rows], dtype=np.int64)
        table = np.asarray([row[1:] for row in rows], dtype=float)
        times = table[:, 0]
        values = {name: table[:, index + 1] for index, name in enumerate(header.fields)}
    else:
        steps = np.asarray([], dtype=np.int64)
        times = np.asarray([], dtype=float)
        values = {name: np.asarray([], dtype=float) for name in header.fields}
    if len(steps) > 1 and np.any(np.diff(steps) <= 0):
        raise ValueError("DROPPS EDR steps must be strictly increasing.")
    if len(times) > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError("DROPPS EDR times must be strictly increasing.")
    return EDRData(header=header, steps=steps, times_ps=times, values=values)


class EDRWriter:
    """Write or append a versioned DROPPS CSV-EDR stream."""

    def __init__(self, path, fields, field_specs, metadata=None, append=False):
        self.path = os.fspath(path)
        self.fields = tuple(fields)
        self.units = {name: field_specs[name].unit for name in self.fields}
        self._stream = None
        self._writer = None

        has_existing_data = (
            append and os.path.isfile(self.path) and os.path.getsize(self.path) > 0
        )
        if has_existing_data:
            existing = read_edr_header(self.path)
            if existing.fields != self.fields:
                raise ValueError(
                    "Cannot append to DROPPS EDR with a different field schema: "
                    f"existing={','.join(existing.fields)}; "
                    f"requested={','.join(self.fields)}."
                )
            self._stream = open(
                self.path, "a", encoding="utf-8", newline="", buffering=1
            )
        else:
            self._stream = open(
                self.path, "w", encoding="utf-8", newline="", buffering=1
            )
            stored_metadata = dict(metadata or {})
            stored_metadata["units"] = self.units
            self._stream.write(EDR_MAGIC + "\n")
            self._stream.write(f"# format_version={EDR_FORMAT_VERSION}\n")
            self._stream.write(
                "# metadata="
                + json.dumps(stored_metadata, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
            csv.writer(self._stream).writerow(("step", "time_ps", *self.fields))
        self._writer = csv.writer(self._stream)

    def write(self, step, time_ps, values):
        row = [int(step), f"{float(time_ps):.12g}"]
        row.extend(f"{float(values[name]):.12g}" for name in self.fields)
        self._writer.writerow(row)
        self._stream.flush()

    def close(self):
        if self._stream is not None:
            self._stream.close()
            self._stream = None
            self._writer = None

    def __del__(self):
        self.close()


class EnergyReporter:
    """Record selected thermodynamic quantities to a DROPPS CSV-EDR file."""

    def __init__(
        self,
        file,
        report_interval,
        fields=DEFAULT_ENERGY_FIELDS,
        append=False,
        temperature=None,
        pressure_coupling=None,
        stress_strain=3.0e-4,
        stress_platform="auto",
        stress_precision="double",
        stress_device=None,
        stress_threads=None,
    ):
        if int(report_interval) <= 0:
            raise ValueError("Energy report interval must be positive.")
        self._file_name = os.fspath(file)
        self._report_interval = int(report_interval)
        self.fields = normalize_field_names(fields)
        self._append = bool(append)
        if (
            self._append
            and os.path.isfile(self._file_name)
            and os.path.getsize(self._file_name) > 0
        ):
            existing = read_edr_header(self._file_name)
            if existing.fields != self.fields:
                raise ValueError(
                    "Cannot append to DROPPS EDR with a different field schema: "
                    f"existing={','.join(existing.fields)}; "
                    f"requested={','.join(self.fields)}."
                )
        self._temperature = temperature
        self._pressure_coupling = pressure_coupling
        self._stress_strain = float(stress_strain)
        self._stress_platform = stress_platform
        self._stress_precision = stress_precision
        self._stress_device = stress_device
        self._stress_threads = stress_threads
        self._writer = None
        self._masses = None
        self._total_mass_dalton = None
        self._degrees_of_freedom = None
        self._constraints = ()
        self._pressure_context = None
        self._pressure_integrator = None

    @property
    def records_pressure(self):
        return bool(PRESSURE_FIELDS.intersection(self.fields))

    def describeNextReport(self, simulation):
        steps = self._report_interval - simulation.currentStep % self._report_interval
        needs_pressure = self.records_pressure
        needs_energy = bool(_ENERGY_VALUE_FIELDS.intersection(self.fields))
        needs_kinetic = bool(_KINETIC_VALUE_FIELDS.intersection(self.fields))
        needs_positions = needs_pressure or (
            needs_kinetic and simulation.system.getNumConstraints() > 0
        )
        return (
            steps,
            needs_positions,
            needs_pressure or needs_kinetic,
            needs_pressure or needs_kinetic,
            needs_energy,
        )

    def _initialize(self, simulation):
        self._masses = np.asarray(
            [
                simulation.system.getParticleMass(index).value_in_unit(unit.dalton)
                for index in range(simulation.system.getNumParticles())
            ],
            dtype=float,
        )
        self._total_mass_dalton = float(np.sum(self._masses))
        self._constraints = tuple(
            simulation.system.getConstraintParameters(index)[:2]
            for index in range(simulation.system.getNumConstraints())
        )
        if "temperature" in self.fields:
            self._degrees_of_freedom = int(3 * np.count_nonzero(self._masses > 0.0))
            for constraint_index in range(simulation.system.getNumConstraints()):
                first, second, _distance = simulation.system.getConstraintParameters(
                    constraint_index
                )
                if self._masses[first] > 0.0 or self._masses[second] > 0.0:
                    self._degrees_of_freedom -= 1
            for force in simulation.system.getForces():
                if force.__class__.__name__ == "CMMotionRemover":
                    self._degrees_of_freedom -= 3
                    break
            if self._degrees_of_freedom <= 0:
                raise ValueError("System has no positive kinetic degrees of freedom.")

        pressure_metadata = {}
        if self.records_pressure:
            if simulation.system.getNumConstraints() > 0:
                raise ValueError(
                    "Pressure terms in energy-grps do not support constrained "
                    "bonds because their constraint virial is unavailable."
                )
            energy_system = clone_energy_system(
                simulation.system, remove_context_forces=True
            )
            (
                self._pressure_context,
                self._pressure_integrator,
                platform_used,
                properties,
            ) = create_energy_context(
                energy_system,
                platform_name=self._stress_platform,
                precision=self._stress_precision,
                device=self._stress_device,
                threads=self._stress_threads,
            )
            pressure_metadata = {
                "pressure_method": "kinetic+configurational_finite_strain",
                "pressure_strain": self._stress_strain,
                "pressure_platform": platform_used,
                "pressure_platform_properties": properties,
            }

        metadata = {
            "production_temperature_K": self._temperature,
            "pressure_coupling": self._pressure_coupling,
            "kinetic_method": "position-synchronized-leapfrog-velocity",
            **pressure_metadata,
        }
        self._writer = EDRWriter(
            self._file_name,
            self.fields,
            ENERGY_FIELDS,
            metadata=metadata,
            append=self._append,
        )

    def report(self, simulation, state):
        if self._writer is None:
            self._initialize(simulation)

        time_ps = state.getTime().value_in_unit(unit.picosecond)
        box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
        box = np.asarray(box, dtype=float)
        lengths = np.linalg.norm(box, axis=1)
        volume = abs(float(np.linalg.det(box)))
        if volume <= 0.0:
            raise ValueError("Periodic box volume must be positive.")

        values = {
            "boxX": lengths[0],
            "boxY": lengths[1],
            "boxZ": lengths[2],
            "volume": volume,
            "density": self._total_mass_dalton * _DALTON_PER_NM3_TO_G_PER_ML / volume,
        }
        synchronized = None
        if _KINETIC_VALUE_FIELDS.intersection(self.fields) or self.records_pressure:
            velocities = state.getVelocities(asNumpy=True).value_in_unit(
                unit.nanometer / unit.picosecond
            )
            forces = state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer
            )
            positions = None
            if self.records_pressure or self._constraints:
                positions = state.getPositions(asNumpy=True).value_in_unit(
                    unit.nanometer
                )
            synchronized = synchronized_velocities(
                self._masses,
                velocities,
                forces,
                simulation.integrator.getStepSize().value_in_unit(unit.picosecond),
                positions_nm=positions,
                box_vectors_nm=box,
                constraints=self._constraints,
            )

        if _ENERGY_VALUE_FIELDS.intersection(self.fields):
            potential = state.getPotentialEnergy().value_in_unit(
                unit.kilojoule_per_mole
            )
            kinetic = (
                0.5 * np.einsum("i,ia,ia->", self._masses, synchronized, synchronized)
                if synchronized is not None
                else state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
            )
            values.update(
                {
                    "potentialEnergy": potential,
                    "kineticEnergy": kinetic,
                    "totalEnergy": potential + kinetic,
                }
            )
            if "temperature" in self.fields:
                values["temperature"] = (
                    2.0 * kinetic / (self._degrees_of_freedom * _R_KJ_PER_MOL_K)
                )
        if self.records_pressure:
            positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            tensor = pressure_tensor(
                self._pressure_context,
                self._masses,
                positions,
                synchronized,
                box,
                strain=self._stress_strain,
            )
            values.update(
                {
                    "pressure": float(np.trace(tensor) / 3.0),
                    "Pxx": tensor[0, 0],
                    "Pyy": tensor[1, 1],
                    "Pzz": tensor[2, 2],
                    "Pxy": tensor[0, 1],
                    "Pxz": tensor[0, 2],
                    "Pyz": tensor[1, 2],
                }
            )
        self._writer.write(simulation.currentStep, time_ps, values)

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        self._pressure_context = None
        self._pressure_integrator = None

    def __del__(self):
        self.close()
