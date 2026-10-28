"""Versioned DROPPS run-input files and legacy pickle compatibility.

TPR v1 files were ordinary Python pickles containing live OpenMM and DROPPS
objects.  TPR v2 is a ZIP container whose OpenMM ``System`` is serialized as
portable XML and whose remaining data is stored in JSON/NumPy formats.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import os
import pickle
import platform
import tempfile
import zipfile
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import openmm
import openmm.app
from openmm import unit

from dropps.fileio.pdb_reader import PDBData


TPR_FORMAT = "dropps-tpr"
TPR_FORMAT_VERSION = 2
_MANIFEST_NAME = "manifest.json"
_MEMBER_NAMES = {
    "system": "system.xml",
    "topology": "topology.json",
    "positions": "positions.npy",
    "parameters": "parameters.json",
    "pdb": "pdb.json",
    "itp": "itp.json",
}
_IDENTITY_IGNORED_PARAMETERS = {
    "nsteps",
    "seed",
    "nst_cp",
    "nst_energy",
    "energy_grps",
    "nst_stress",
    "stress_output",
    "stress_platform",
    "stress_precision",
    "stress_device",
    "stress_threads",
    "nst_screenlog",
    "nst_filelog",
    "screenlog_grps",
    "filelog_grps",
    "nst_xout",
}


class TPRFormatError(ValueError):
    """Raised when a DROPPS run-input file is malformed or unsupported."""


@dataclass
class TPRContent:
    parameters: dict[str, Any]
    mdsystem: openmm.System
    mdtopology: openmm.app.Topology
    positions: Any
    pdb_raw: Any
    itp_list: list[Any]
    metadata: dict[str, Any]

    @property
    def run_id(self) -> str:
        return str(self.metadata.get("run_id", ""))

    @property
    def format_version(self) -> int:
        return int(self.metadata.get("format_version", 1))

    @property
    def system_id(self) -> str:
        return str(self.metadata.get("system_id", self.run_id))

    def as_runtime_dict(self) -> dict[str, Any]:
        return {
            "parameters": self.parameters,
            "mdsystem": self.mdsystem,
            "mdtopology": self.mdtopology,
            "positions": self.positions,
            "pdb_raw": self.pdb_raw,
            "ITP_list": self.itp_list,
        }


# Preserve the historical public name used by a few callers.
tpr_content = TPRContent


def _package_version() -> str:
    try:
        return importlib.metadata.version("dropps")
    except importlib.metadata.PackageNotFoundError:
        return "development"


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _json_safe(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if is_dataclass(value):
        return _json_safe(asdict(value))
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _run_id(members: Mapping[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name in sorted(members):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(members[name])
        digest.update(b"\0")
    return digest.hexdigest()


def _system_id(
    system_xml: bytes,
    topology_json: bytes,
    itp_json: bytes,
    parameters: Mapping[str, Any],
) -> str:
    scientific_parameters = {
        key: value
        for key, value in parameters.items()
        if key not in _IDENTITY_IGNORED_PARAMETERS
    }
    return _run_id(
        {
            "system.xml": system_xml,
            "topology.json": topology_json,
            "itp.json": itp_json,
            "scientific-parameters.json": _json_bytes(scientific_parameters),
        }
    )


def _positions_to_nm(positions: Any) -> np.ndarray:
    if unit.is_quantity(positions):
        values = positions.value_in_unit(unit.nanometer)
        return np.asarray(values, dtype=np.float64)

    rows = []
    for position in positions:
        rows.append(
            [
                coordinate.value_in_unit(unit.nanometer)
                if unit.is_quantity(coordinate)
                else float(coordinate)
                for coordinate in position
            ]
        )
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise TPRFormatError("Positions must have shape (particle_count, 3).")
    return values


def _positions_bytes(positions: Any) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, _positions_to_nm(positions), allow_pickle=False)
    return buffer.getvalue()


def _vector_nm(vector: Any) -> list[float]:
    return [
        coordinate.value_in_unit(unit.nanometer)
        if unit.is_quantity(coordinate)
        else float(coordinate)
        for coordinate in vector
    ]


def _topology_to_dict(topology: openmm.app.Topology) -> dict[str, Any]:
    chains = []
    atom_indices: dict[Any, int] = {}
    for chain in topology.chains():
        residues = []
        for residue in chain.residues():
            atoms = []
            for atom in residue.atoms():
                atom_indices[atom] = atom.index
                atoms.append(
                    {
                        "name": atom.name,
                        "id": atom.id,
                        "element": (
                            None
                            if atom.element is None
                            else int(atom.element.atomic_number)
                        ),
                        "formal_charge": getattr(atom, "formalCharge", None),
                    }
                )
            residues.append(
                {
                    "name": residue.name,
                    "id": residue.id,
                    "insertion_code": getattr(residue, "insertionCode", ""),
                    "atoms": atoms,
                }
            )
        chains.append({"id": chain.id, "residues": residues})

    bonds = []
    for bond in topology.bonds():
        bond_type = getattr(bond, "type", None)
        bonds.append(
            {
                "atom1": atom_indices[bond[0]],
                "atom2": atom_indices[bond[1]],
                "type": None if bond_type is None else str(bond_type),
                "order": getattr(bond, "order", None),
            }
        )

    box = topology.getPeriodicBoxVectors()
    return {
        "chains": chains,
        "bonds": bonds,
        "periodic_box_vectors_nm": (
            None if box is None else [_vector_nm(vector) for vector in box]
        ),
    }


def _topology_from_dict(data: Mapping[str, Any]) -> openmm.app.Topology:
    topology = openmm.app.Topology()
    atoms = []
    for chain_data in data.get("chains", []):
        chain = topology.addChain(chain_data.get("id"))
        for residue_data in chain_data.get("residues", []):
            residue = topology.addResidue(
                residue_data["name"],
                chain,
                residue_data.get("id"),
                residue_data.get("insertion_code", ""),
            )
            for atom_data in residue_data.get("atoms", []):
                atomic_number = atom_data.get("element")
                element = (
                    None
                    if atomic_number is None
                    else openmm.app.Element.getByAtomicNumber(int(atomic_number))
                )
                atom_arguments = (
                    atom_data["name"],
                    element,
                    residue,
                    atom_data.get("id"),
                )
                formal_charge = atom_data.get("formal_charge")
                try:
                    # OpenMM 8.2 added the formalCharge argument.  Omitting it
                    # on 8.1 keeps the portable topology readable there too.
                    atom = topology.addAtom(*atom_arguments, formal_charge)
                except TypeError:
                    atom = topology.addAtom(*atom_arguments)
                atoms.append(atom)

    for bond_data in data.get("bonds", []):
        topology.addBond(
            atoms[int(bond_data["atom1"])],
            atoms[int(bond_data["atom2"])],
            bond_data.get("type"),
            bond_data.get("order"),
        )

    box = data.get("periodic_box_vectors_nm")
    if box is not None:
        topology.setPeriodicBoxVectors(
            [openmm.Vec3(*vector) * unit.nanometer for vector in box]
        )
    return topology


def _pdb_to_dict(pdb_raw: Any) -> dict[str, Any]:
    if isinstance(pdb_raw, str):
        return {"kind": "text", "value": pdb_raw}
    if is_dataclass(pdb_raw):
        fields = asdict(pdb_raw)
        kind = type(pdb_raw).__name__
    elif hasattr(pdb_raw, "__dict__"):
        fields = vars(pdb_raw)
        kind = type(pdb_raw).__name__
    else:
        raise TPRFormatError(
            f"Unsupported PDB metadata type: {type(pdb_raw).__name__}."
        )
    return {"kind": kind, "fields": _json_safe(fields)}


def _pdb_from_dict(data: Mapping[str, Any]) -> Any:
    if data.get("kind") == "text":
        return data.get("value", "")
    fields = dict(data.get("fields", {}))
    required = set(PDBData.__dataclass_fields__)
    if required.issubset(fields):
        return PDBData(**{name: fields[name] for name in required})
    return SimpleNamespace(**fields)


def _quantity_value(value: Any, expected_unit: Any) -> float:
    if unit.is_quantity(value):
        return float(value.value_in_unit(expected_unit))
    return float(value)


def _object_fields(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TPRFormatError(f"Cannot serialize {type(value).__name__} as TPR metadata.")


def _itp_to_dict(topology: Any) -> dict[str, Any]:
    atomtypes = [
        _json_safe(_object_fields(item)) for item in getattr(topology, "atomtypes", [])
    ]
    atoms = [
        _json_safe(_object_fields(item)) for item in getattr(topology, "atoms", [])
    ]
    bonds = [
        {
            "a1": int(bond.a1),
            "a2": int(bond.a2),
            "r0_nm": _quantity_value(bond.r0, unit.nanometer),
            "k_kj_mol_nm2": _quantity_value(
                bond.k, unit.kilojoule_per_mole / unit.nanometer**2
            ),
        }
        for bond in (getattr(topology, "bonds", None) or [])
    ]
    angles = [
        {
            "a1": int(angle.a1),
            "a2": int(angle.a2),
            "a3": int(angle.a3),
            "theta_degree": _quantity_value(angle.theta_in_degree, unit.degree),
            "k_kj_mol_rad2": _quantity_value(
                angle.k, unit.kilojoule_per_mole / unit.radian**2
            ),
        }
        for angle in (getattr(topology, "angles", None) or [])
    ]

    fields = {}
    for name in (
        "molecule_name",
        "nrexcl",
        "function_type_LJ",
        "function_type_Coulomb",
        "relative_permittivity_mode",
        "relative_permittivity",
        "relative_permittivity_coeffs",
        "simulation_settings",
        "typelist",
        "type2sigma",
        "type2mylambda",
        "epsilon",
        "types2sigma",
        "types2mu",
        "types2epsilon",
        "nonbond_params",
    ):
        if hasattr(topology, name):
            fields[name] = _json_safe(getattr(topology, name))
    return {
        "fields": fields,
        "atomtypes": atomtypes,
        "atoms": atoms,
        "bonds": bonds,
        "angles": angles,
    }


def _itp_from_dict(data: Mapping[str, Any]) -> SimpleNamespace:
    fields = dict(data.get("fields", {}))
    atomtypes = [SimpleNamespace(**item) for item in data.get("atomtypes", [])]
    atoms = [SimpleNamespace(**item) for item in data.get("atoms", [])]
    bonds = [
        SimpleNamespace(
            a1=int(item["a1"]),
            a2=int(item["a2"]),
            r0=float(item["r0_nm"]) * unit.nanometer,
            k=float(item["k_kj_mol_nm2"]) * unit.kilojoule_per_mole / unit.nanometer**2,
        )
        for item in data.get("bonds", [])
    ]
    angles = [
        SimpleNamespace(
            a1=int(item["a1"]),
            a2=int(item["a2"]),
            a3=int(item["a3"]),
            theta_in_degree=float(item["theta_degree"]) * unit.degree,
            k=float(item["k_kj_mol_rad2"]) * unit.kilojoule_per_mole / unit.radian**2,
        )
        for item in data.get("angles", [])
    ]
    topology = SimpleNamespace(**fields)
    topology.atomtypes = atomtypes
    topology.atoms = atoms
    topology.bonds = bonds or None
    topology.angles = angles or None
    return _normalize_itp(topology)


def _normalize_itp(topology: Any) -> Any:
    """Fill fields omitted by historical DROPPS ITP objects."""

    if not hasattr(topology, "function_type_LJ"):
        topology.function_type_LJ = "Ashbaugh-Hatch"
    if not hasattr(topology, "function_type_Coulomb"):
        topology.function_type_Coulomb = "Debye-Huckel"
    if not hasattr(topology, "relative_permittivity_mode"):
        topology.relative_permittivity_mode = "constant"
    if not hasattr(topology, "relative_permittivity"):
        topology.relative_permittivity = 80.0
    if not hasattr(topology, "relative_permittivity_coeffs"):
        topology.relative_permittivity_coeffs = None
    if not hasattr(topology, "simulation_settings"):
        topology.simulation_settings = []
    if not hasattr(topology, "bonds"):
        topology.bonds = None
    if not hasattr(topology, "angles"):
        topology.angles = None
    if not hasattr(topology, "atomtypes"):
        topology.atomtypes = []
    if not hasattr(topology, "atoms"):
        topology.atoms = []
    if not hasattr(topology, "typelist"):
        topology.typelist = [item.abbr for item in topology.atomtypes]

    if topology.function_type_LJ == "Ashbaugh-Hatch":
        if not hasattr(topology, "type2sigma"):
            topology.type2sigma = {
                item.abbr: item.sigma
                for item in topology.atomtypes
                if hasattr(item, "sigma")
            }
        if not hasattr(topology, "type2mylambda"):
            topology.type2mylambda = {
                item.abbr: item.mylambda
                for item in topology.atomtypes
                if hasattr(item, "mylambda")
            }
    return topology


def _integrator_semantics(parameters: Mapping[str, Any], openmm_version: str) -> str:
    if parameters.get("integrator") == "steep":
        return "steep-minimization"
    if parameters.get("integrator") != "Langevin":
        return str(parameters.get("integrator", "unknown"))
    # mdrun explicitly constructs LangevinMiddleIntegrator, so new TPR files
    # have stable semantics independent of OpenMM's LangevinIntegrator alias.
    return "langevin-middle"


def _runtime_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, TPRContent):
        return payload.as_runtime_dict()
    if isinstance(payload, Mapping):
        return dict(payload)
    raise TypeError("TPR payload must be a mapping or TPRContent.")


def _source_members(
    sources: Mapping[str, os.PathLike[str] | str] | None,
) -> dict[str, bytes]:
    if not sources:
        return {}
    result = {}
    used_names = set()
    for label, raw_path in sources.items():
        path = Path(raw_path)
        name = f"sources/{label}/{path.name}"
        suffix = 2
        while name in used_names:
            name = f"sources/{label}/{suffix}_{path.name}"
            suffix += 1
        used_names.add(name)
        result[name] = path.read_bytes()
    return result


def write_tpr(
    tpr_path: os.PathLike[str] | str,
    payload: Any,
    *,
    sources: Mapping[str, os.PathLike[str] | str] | None = None,
) -> dict[str, Any]:
    """Atomically write a portable TPR v2 file and return its manifest."""

    runtime = _runtime_payload(payload)
    required = {"parameters", "mdsystem", "mdtopology", "positions", "pdb_raw"}
    missing = sorted(required.difference(runtime))
    if missing:
        raise TPRFormatError(f"TPR payload is missing: {', '.join(missing)}.")

    system_xml = openmm.XmlSerializer.serialize(runtime["mdsystem"])
    members = {
        _MEMBER_NAMES["system"]: system_xml.encode("utf-8"),
        _MEMBER_NAMES["topology"]: _json_bytes(
            _topology_to_dict(runtime["mdtopology"])
        ),
        _MEMBER_NAMES["positions"]: _positions_bytes(runtime["positions"]),
        _MEMBER_NAMES["parameters"]: _json_bytes(runtime["parameters"]),
        _MEMBER_NAMES["pdb"]: _json_bytes(_pdb_to_dict(runtime["pdb_raw"])),
        _MEMBER_NAMES["itp"]: _json_bytes(
            [_itp_to_dict(item) for item in runtime.get("ITP_list", [])]
        ),
    }
    source_members = _source_members(sources)
    identity_members = dict(members)
    manifest = {
        "format": TPR_FORMAT,
        "format_version": TPR_FORMAT_VERSION,
        "dropps_version": _package_version(),
        "openmm_version": openmm.__version__,
        "python_version": platform.python_version(),
        "integrator_semantics": _integrator_semantics(
            runtime["parameters"], openmm.__version__
        ),
        "run_id": _run_id(identity_members),
        "system_id": _system_id(
            members[_MEMBER_NAMES["system"]],
            members[_MEMBER_NAMES["topology"]],
            members[_MEMBER_NAMES["itp"]],
            runtime["parameters"],
        ),
        "members": {
            name: {"sha256": _sha256(data), "size": len(data)}
            for name, data in sorted(members.items())
        },
        "sources": {
            name: {"sha256": _sha256(data), "size": len(data)}
            for name, data in sorted(source_members.items())
        },
    }
    members.update(source_members)
    members[_MANIFEST_NAME] = _json_bytes(manifest)

    path = os.fspath(tpr_path)
    directory = os.path.dirname(os.path.abspath(path))
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory
    )
    os.close(descriptor)
    try:
        with zipfile.ZipFile(
            temporary_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for name in sorted(members):
                archive.writestr(name, members[name])
        os.replace(temporary_path, path)
    except BaseException:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise
    return manifest


def _read_checked_member(
    archive: zipfile.ZipFile, manifest: Mapping[str, Any], name: str
) -> bytes:
    try:
        data = archive.read(name)
    except KeyError as exc:
        raise TPRFormatError(f"TPR v2 is missing required member {name!r}.") from exc
    expected = manifest.get("members", {}).get(name, {}).get("sha256")
    if expected is None:
        raise TPRFormatError(f"TPR v2 manifest has no checksum for {name!r}.")
    actual = _sha256(data)
    if actual != expected:
        raise TPRFormatError(f"TPR v2 member {name!r} failed checksum validation.")
    return data


def _read_v2(path: os.PathLike[str] | str) -> TPRContent:
    with zipfile.ZipFile(path, "r") as archive:
        try:
            manifest = json.loads(archive.read(_MANIFEST_NAME))
        except (KeyError, json.JSONDecodeError) as exc:
            raise TPRFormatError("TPR v2 has an invalid manifest.") from exc
        if manifest.get("format") != TPR_FORMAT:
            raise TPRFormatError("ZIP file is not a DROPPS TPR.")
        version = int(manifest.get("format_version", 0))
        if version != TPR_FORMAT_VERSION:
            raise TPRFormatError(
                f"Unsupported TPR format version {version}; "
                f"this DROPPS release supports version {TPR_FORMAT_VERSION}."
            )

        blobs = {
            key: _read_checked_member(archive, manifest, name)
            for key, name in _MEMBER_NAMES.items()
        }

    try:
        system = openmm.XmlSerializer.deserialize(blobs["system"].decode("utf-8"))
    except Exception as exc:
        source_version = manifest.get("openmm_version", "unknown")
        raise TPRFormatError(
            f"OpenMM {openmm.__version__} could not deserialize a System "
            f"written by OpenMM {source_version}: {exc}"
        ) from exc

    try:
        positions = np.load(io.BytesIO(blobs["positions"]), allow_pickle=False)
        parameters = json.loads(blobs["parameters"])
        topology = _topology_from_dict(json.loads(blobs["topology"]))
        pdb_raw = _pdb_from_dict(json.loads(blobs["pdb"]))
        itp_list = [_itp_from_dict(item) for item in json.loads(blobs["itp"])]
    except Exception as exc:
        raise TPRFormatError(f"Could not decode TPR v2 data: {exc}") from exc

    if positions.shape != (system.getNumParticles(), 3):
        raise TPRFormatError(
            "TPR position count does not match the serialized OpenMM System."
        )
    if topology.getNumAtoms() != system.getNumParticles():
        raise TPRFormatError(
            "TPR topology atom count does not match the serialized OpenMM System."
        )
    return TPRContent(
        parameters=dict(parameters),
        mdsystem=system,
        mdtopology=topology,
        positions=positions * unit.nanometer,
        pdb_raw=pdb_raw,
        itp_list=itp_list,
        metadata=dict(manifest),
    )


class _LegacyTPRUnpickler(pickle.Unpickler):
    """Restricted loader for trusted historical DROPPS TPR files."""

    _ALLOWED_MODULES = {
        "builtins",
        "collections",
        "copyreg",
        "numpy",
        "openmm",
        "types",
        "_codecs",
        "dropps.fileio.itp_reader",
        "dropps.fileio.pdb_reader",
    }
    _ALLOWED_PREFIXES = ("openmm.", "numpy.")

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) == ("openmm.unit.baseunit", "BaseUnit"):
            # OpenMM 8.2 changed BaseUnit from identity hashing to structural
            # hashing.  An 8.1 pickle restores dictionaries containing a
            # BaseUnit before restoring the object's attributes, so the newer
            # __hash__ raises AttributeError.  Decode into the identity-hashed
            # legacy shell, then normalize the few unit-bearing TPR fields.
            return _LegacyBaseUnit
        if (module, name) == ("dropps.fileio.itp_reader", "Atomtype"):
            raise pickle.UnpicklingError(
                "This TPR predates the Atomtype_AH_DH/Atomtype_WF_DH format "
                "and is intentionally unsupported. Regenerate it with a newer "
                "DROPPS release."
            )
        if module in self._ALLOWED_MODULES or module.startswith(self._ALLOWED_PREFIXES):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Legacy TPR references disallowed global {module}.{name}."
        )


class _LegacyBaseUnit:
    """Identity-hashed shell used only while decoding OpenMM 8.1 pickles."""


def _legacy_quantity_value(value: Any) -> Any:
    if isinstance(value, unit.Quantity):
        return value._value
    return value


def _legacy_positions_in_nm(positions: Any) -> Any:
    raw_positions = _legacy_quantity_value(positions)
    if raw_positions is not positions:
        return np.asarray(raw_positions, dtype=np.float64) * unit.nanometer

    rows = []
    for row in positions:
        raw_row = _legacy_quantity_value(row)
        rows.append(
            [float(_legacy_quantity_value(coordinate)) for coordinate in raw_row]
        )
    return np.asarray(rows, dtype=np.float64) * unit.nanometer


def _normalize_legacy_openmm_units(payload: Mapping[str, Any]) -> None:
    """Replace OpenMM 8.1 pickle units with current OpenMM unit objects."""

    payload["positions"] = _legacy_positions_in_nm(payload["positions"])

    topology = payload["mdtopology"]
    box_vectors = getattr(topology, "_periodicBoxVectors", None)
    if box_vectors is not None:
        normalized_vectors = []
        for vector in box_vectors:
            raw_vector = _legacy_quantity_value(vector)
            normalized_vectors.append(
                openmm.Vec3(
                    *(float(_legacy_quantity_value(value)) for value in raw_vector)
                )
                * unit.nanometer
            )
        topology.setPeriodicBoxVectors(normalized_vectors)

    for itp_topology in payload.get("ITP_list", []):
        for bond in getattr(itp_topology, "bonds", None) or []:
            bond.r0 = float(_legacy_quantity_value(bond.r0)) * unit.nanometer
            bond.k = (
                float(_legacy_quantity_value(bond.k))
                * unit.kilojoule_per_mole
                / unit.nanometer**2
            )
        for angle in getattr(itp_topology, "angles", None) or []:
            angle.theta_in_degree = (
                float(_legacy_quantity_value(angle.theta_in_degree)) * unit.degree
            )
            angle.k = (
                float(_legacy_quantity_value(angle.k))
                * unit.kilojoule_per_mole
                / unit.radian**2
            )


def _legacy_file_id(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_legacy(path: os.PathLike[str] | str) -> TPRContent:
    try:
        with open(path, "rb") as stream:
            payload = _LegacyTPRUnpickler(stream).load()
    except Exception as exc:
        raise TPRFormatError(
            "Could not read legacy pickle TPR. Open it only if it came from a "
            f"trusted source. Root cause: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise TPRFormatError("Legacy TPR root object is not a mapping.")

    required = {"parameters", "mdsystem", "mdtopology", "positions", "pdb_raw"}
    missing = sorted(required.difference(payload))
    if missing:
        raise TPRFormatError(f"Legacy TPR is missing: {', '.join(missing)}.")
    _normalize_legacy_openmm_units(payload)
    itp_list = [_normalize_itp(item) for item in payload.get("ITP_list", [])]
    system_xml = openmm.XmlSerializer.serialize(payload["mdsystem"]).encode("utf-8")
    topology_json = _json_bytes(_topology_to_dict(payload["mdtopology"]))
    itp_json = _json_bytes([_itp_to_dict(item) for item in itp_list])
    legacy_run_id = _legacy_file_id(path)
    return TPRContent(
        parameters=dict(payload["parameters"]),
        mdsystem=payload["mdsystem"],
        mdtopology=payload["mdtopology"],
        positions=payload["positions"],
        pdb_raw=payload["pdb_raw"],
        itp_list=itp_list,
        metadata={
            "format": TPR_FORMAT,
            "format_version": 1,
            "source_format": "legacy-pickle",
            "openmm_version": "8.1.x (inferred from DROPPS dependency)",
            "integrator_semantics": (
                "langevin-legacy"
                if payload["parameters"].get("integrator") == "Langevin"
                else (
                    "steep-minimization"
                    if payload["parameters"].get("integrator") == "steep"
                    else str(payload["parameters"].get("integrator", "unknown"))
                )
            ),
            "run_id": legacy_run_id,
            "system_id": _system_id(
                system_xml,
                topology_json,
                itp_json,
                payload["parameters"],
            ),
        },
    )


def read_tpr(tpr_path: os.PathLike[str] | str) -> TPRContent:
    """Read either a portable TPR v2 or a trusted legacy pickle TPR."""

    path = os.fspath(tpr_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"TPR file not found: {path}")
    if zipfile.is_zipfile(path):
        return _read_v2(path)
    return _read_legacy(path)


def is_portable_tpr(tpr_path: os.PathLike[str] | str) -> bool:
    """Return whether *tpr_path* is a TPR v2 ZIP container."""

    return zipfile.is_zipfile(tpr_path)
