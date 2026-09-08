"""Helpers for producing restart-safe simulation output."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile

import openmm
from openmm import unit
from openmm.app import XTCFile
from openmm.app.internal.xtc_utils import get_xtc_nframes


class RestartableXTCReporter:
    """Write an XTC trajectory whose step numbers remain valid after appending."""

    def __init__(
        self,
        file,
        report_interval,
        append=False,
        enforce_periodic_box=None,
    ):
        if int(report_interval) <= 0:
            raise ValueError("XTC report interval must be positive.")
        self._file_name = os.fspath(file)
        self._report_interval = int(report_interval)
        self._append = bool(append)
        self._enforce_periodic_box = enforce_periodic_box
        self._xtc = None

    def describeNextReport(self, simulation):
        steps = self._report_interval - simulation.currentStep % self._report_interval
        return steps, True, False, False, False, self._enforce_periodic_box

    def report(self, simulation, state):
        if self._xtc is None:
            first_step = simulation.currentStep
            if self._append:
                frame_count = get_xtc_nframes(self._file_name.encode("utf-8"))
                first_step -= frame_count * self._report_interval
                if first_step < 0:
                    raise ValueError(
                        "Existing XTC trajectory is inconsistent with the restart step."
                    )
            self._xtc = XTCFile(
                self._file_name,
                simulation.topology,
                simulation.integrator.getStepSize(),
                first_step,
                self._report_interval,
                self._append,
            )
        self._xtc.writeModel(
            state.getPositions(),
            periodicBoxVectors=state.getPeriodicBoxVectors(),
        )


def truncate_delimited_report(file, checkpoint_step, separator=None):
    """Remove data rows newer than a checkpoint from a text reporter file."""

    path = os.fspath(file)
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False

    with open(path, "r", encoding="utf-8") as stream:
        lines = stream.readlines()

    kept_lines = []
    changed = False
    for line in lines:
        stripped = line.lstrip()
        if not stripped or stripped.startswith(("#", "@")):
            kept_lines.append(line)
            continue
        if separator is not None and separator not in stripped:
            kept_lines.append(line)
            continue
        token = stripped.split(separator, 1)[0] if separator else stripped.split()[0]
        try:
            step = int(token)
        except ValueError:
            kept_lines.append(line)
            continue
        if step <= checkpoint_step:
            kept_lines.append(line)
        else:
            changed = True

    if changed:
        _atomic_write_text(path, kept_lines)
    return changed


def truncate_xtc(file, checkpoint_step):
    """Atomically discard XTC frames whose stored step is after a checkpoint."""

    path = os.fspath(file)
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False

    # Import lazily: this path is used only while resuming an existing run.
    from MDAnalysis.lib.formats.libmdaxdr import XTCFile as XTCReaderWriter

    with XTCReaderWriter(path, "r") as source:
        offsets = source.calc_offsets()
        if len(offsets) == 0:
            return False
        source.seek(len(offsets) - 1)
        last_frame = source.read()
        if last_frame.step <= checkpoint_step:
            return False

    directory = os.path.dirname(os.path.abspath(path))
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".restart", dir=directory
    )
    os.close(descriptor)
    frames_written = 0
    try:
        with (
            XTCReaderWriter(path, "r") as source,
            XTCReaderWriter(temporary_path, "w") as target,
        ):
            for frame in source:
                if frame.step > checkpoint_step:
                    break
                target.write(
                    frame.x,
                    frame.box,
                    frame.step,
                    frame.time,
                    frame.prec,
                )
                frames_written += 1
        if frames_written == 0:
            os.unlink(temporary_path)
            os.unlink(path)
        else:
            os.replace(temporary_path, path)
    except BaseException:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise
    return True


def save_checkpoint_safely(simulation, file):
    """Write a checkpoint atomically without destroying the previous one."""

    path = os.fspath(file)
    directory = os.path.dirname(os.path.abspath(path))
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory
    )
    os.close(descriptor)
    try:
        simulation.saveCheckpoint(temporary_path)
        os.replace(temporary_path, path)
    except BaseException:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise


def _restart_base(path):
    path = os.fspath(path)
    for suffix in (".state.xml", ".restart.json", ".chk"):
        if path.endswith(suffix):
            return path[: -len(suffix)]
    return path


def portable_state_path(checkpoint_file):
    """Return the portable State XML path paired with a checkpoint."""

    return _restart_base(checkpoint_file) + ".state.xml"


def restart_metadata_path(restart_file):
    """Return the restart manifest paired with a checkpoint or State XML."""

    return _restart_base(restart_file) + ".restart.json"


def native_checkpoint_path(restart_file):
    """Return the native checkpoint paired with a State XML or manifest."""

    return _restart_base(restart_file) + ".chk"


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_state_safely(simulation, file):
    """Atomically write an OpenMM portable State XML file."""

    path = os.fspath(file)
    directory = os.path.dirname(os.path.abspath(path))
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory
    )
    os.close(descriptor)
    try:
        simulation.saveState(temporary_path)
        os.replace(temporary_path, path)
    except BaseException:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise


def save_portable_restart_safely(simulation, checkpoint_file, metadata=None):
    """Write the portable State and its manifest for a native checkpoint."""

    checkpoint_file = os.fspath(checkpoint_file)
    state_file = portable_state_path(checkpoint_file)
    manifest_file = restart_metadata_path(checkpoint_file)
    save_state_safely(simulation, state_file)

    context = simulation.context
    platform_object = context.getPlatform()
    properties = {}
    for name in platform_object.getPropertyNames():
        try:
            properties[name] = platform_object.getPropertyValue(context, name)
        except Exception:
            continue

    manifest = dict(metadata or {})
    manifest.update(
        {
            "format": "dropps-restart",
            "format_version": 1,
            "step": int(simulation.currentStep),
            "time_ps": float(context.getTime().value_in_unit(unit.picosecond)),
            "openmm_version": openmm.__version__,
            "platform": platform_object.getName(),
            "platform_properties": properties,
            "checkpoint": os.path.basename(checkpoint_file),
            "checkpoint_sha256": (
                _file_sha256(checkpoint_file)
                if os.path.isfile(checkpoint_file)
                else None
            ),
            "state": os.path.basename(state_file),
            "state_sha256": _file_sha256(state_file),
            "exact_continuation": False,
        }
    )
    _atomic_write_json(manifest_file, manifest)
    return state_file, manifest_file


def load_restart_metadata(restart_file):
    """Load a restart manifest, returning ``None`` when it does not exist."""

    path = restart_metadata_path(restart_file)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("format") != "dropps-restart":
        raise ValueError(f"Invalid DROPPS restart manifest: {path}")
    if int(manifest.get("format_version", 0)) != 1:
        raise ValueError(
            f"Unsupported restart manifest version in {path}: "
            f"{manifest.get('format_version')}."
        )
    return manifest


def _validate_restart_system_id(manifest, expected_system_id, restart_file):
    if expected_system_id is None:
        return
    restart_system_id = manifest.get("tpr_system_id")
    if not restart_system_id:
        raise ValueError(
            f"Restart manifest for {restart_file} does not identify its molecular "
            "system."
        )
    if expected_system_id != restart_system_id:
        raise ValueError(
            "Restart was created for a different molecular system "
            f"(restart {restart_system_id}, input {expected_system_id})."
        )


def _validate_restart_checksum(file, manifest, checksum_key, description):
    expected_checksum = manifest.get(checksum_key)
    if not expected_checksum:
        raise ValueError(
            f"Restart manifest does not contain a {description} checksum: {file}"
        )
    if _file_sha256(file) != expected_checksum:
        raise ValueError(f"{description.capitalize()} checksum failed: {file}")


def validate_native_restart(
    checkpoint_file,
    expected_system_id=None,
    manifest=None,
):
    """Validate a native checkpoint against its paired manifest when present.

    Native checkpoints created before restart manifests were introduced remain
    loadable.  Once a manifest exists, however, its checkpoint hash and system
    identity are mandatory so an interrupted multi-file update cannot combine
    state from different save generations.
    """

    if manifest is None:
        manifest = load_restart_metadata(checkpoint_file)
    if manifest is None:
        return None
    _validate_restart_checksum(
        checkpoint_file,
        manifest,
        "checkpoint_sha256",
        "native checkpoint",
    )
    _validate_restart_system_id(manifest, expected_system_id, checkpoint_file)
    return manifest


def validate_portable_restart(state_file, expected_system_id=None):
    """Validate a portable State against its checksum and TPR identity."""

    manifest = load_restart_metadata(state_file)
    if manifest is None:
        raise FileNotFoundError(
            "Portable State requires its paired restart manifest: "
            f"{restart_metadata_path(state_file)}"
        )
    _validate_restart_checksum(
        state_file,
        manifest,
        "state_sha256",
        "portable State",
    )
    _validate_restart_system_id(manifest, expected_system_id, state_file)
    return manifest


def _atomic_write_text(path, lines):
    directory = os.path.dirname(os.path.abspath(path))
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.writelines(lines)
        os.replace(temporary_path, path)
    except BaseException:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise


def _atomic_write_json(path, value):
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    _atomic_write_text(path, [text])
