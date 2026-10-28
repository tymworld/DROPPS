import tempfile
import base64
import hashlib
import unittest
import zipfile
from pathlib import Path

from scripts.prepare_release import (
    RELEASE_VERSION,
    _artifact_paths,
    _forbidden_member,
    validate_wheel,
)


class ReleaseToolingTests(unittest.TestCase):
    def test_release_version_accepts_beta_and_formal_versions(self):
        self.assertIsNotNone(RELEASE_VERSION.fullmatch("0.5.0b4"))
        self.assertIsNotNone(RELEASE_VERSION.fullmatch("0.5.0"))
        self.assertIsNone(RELEASE_VERSION.fullmatch("0.5.0-beta4"))

    def test_generated_and_nas_metadata_are_forbidden(self):
        for name in (
            "dropps/__pycache__/module.pyc",
            "dropps/.DS_Store",
            "dropps/._module.py",
            "dropps/@eaDir/module.py@SynoEAStream",
            "dropps/backups/old.py",
        ):
            with self.subTest(name=name):
                self.assertTrue(_forbidden_member(name))
        self.assertFalse(_forbidden_member("dropps/analysis/density_core.py"))

    def test_wheel_metadata_and_payload_are_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            wheel = Path(temporary) / "dropps-0.5.0b4-py3-none-any.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("dropps/__init__.py", "")
                archive.writestr(
                    "dropps-0.5.0b4.dist-info/METADATA",
                    "Metadata-Version: 2.4\nName: dropps\nVersion: 0.5.0b4\n",
                )
                records = []
                for name in archive.namelist():
                    payload = archive.read(name)
                    digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode("ascii")
                    records.append(f"{name},sha256={digest},{len(payload)}")
                archive.writestr("dropps-0.5.0b4.dist-info/RECORD", "\n".join(records) + "\ndropps-0.5.0b4.dist-info/RECORD,,\n")
            validate_wheel(wheel, "0.5.0b4")

            with zipfile.ZipFile(wheel, "a") as archive:
                archive.writestr("dropps/__pycache__/bad.pyc", b"bad")
            with self.assertRaisesRegex(RuntimeError, "forbidden members"):
                validate_wheel(wheel, "0.5.0b4")

    def test_artifact_set_must_be_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "dropps-0.5.0b4-py3-none-any.whl").touch()
            (directory / "dropps-0.5.0b4.tar.gz").touch()
            _artifact_paths(directory, "0.5.0b4")
            (directory / "unexpected.txt").touch()
            with self.assertRaisesRegex(RuntimeError, "unexpected artifacts"):
                _artifact_paths(directory, "0.5.0b4")


if __name__ == "__main__":
    unittest.main()
