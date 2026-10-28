import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from dropps.share.restart import (
    restart_metadata_path,
    validate_native_restart,
    validate_portable_restart,
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_manifest(restart_file, *, checkpoint, state, system_id="system-a"):
    manifest = {
        "format": "dropps-restart",
        "format_version": 1,
        "tpr_system_id": system_id,
        "checkpoint_sha256": _sha256(checkpoint),
        "state_sha256": _sha256(state),
    }
    Path(restart_metadata_path(restart_file)).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return manifest


class RestartValidationTests(unittest.TestCase):
    def test_portable_state_requires_its_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            state = Path(temporary_directory) / "run.state.xml"
            state.write_text("state", encoding="utf-8")

            with self.assertRaisesRegex(FileNotFoundError, "requires"):
                validate_portable_restart(state, expected_system_id="system-a")

    def test_native_checkpoint_hash_detects_a_mixed_save_generation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            checkpoint = directory / "run.chk"
            state = directory / "run.state.xml"
            checkpoint.write_bytes(b"checkpoint-generation-1")
            state.write_bytes(b"state-generation-1")
            manifest = _write_manifest(
                checkpoint,
                checkpoint=checkpoint,
                state=state,
            )

            checkpoint.write_bytes(b"checkpoint-generation-2")
            with self.assertRaisesRegex(ValueError, "checksum failed"):
                validate_native_restart(
                    checkpoint,
                    expected_system_id="system-a",
                    manifest=manifest,
                )

            self.assertEqual(
                validate_portable_restart(
                    state,
                    expected_system_id="system-a",
                ),
                manifest,
            )

    def test_restart_manifest_must_identify_the_expected_system(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            checkpoint = directory / "run.chk"
            state = directory / "run.state.xml"
            checkpoint.write_bytes(b"checkpoint")
            state.write_bytes(b"state")
            _write_manifest(
                checkpoint,
                checkpoint=checkpoint,
                state=state,
                system_id="system-a",
            )

            with self.assertRaisesRegex(ValueError, "different molecular system"):
                validate_portable_restart(state, expected_system_id="system-b")


if __name__ == "__main__":
    unittest.main()
