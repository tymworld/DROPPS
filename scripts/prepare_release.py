#!/usr/bin/env python3
"""Build and validate a clean, checksummed DROPPS release bundle."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
RELEASE_VERSION = re.compile(
    r"^[0-9]+\.[0-9]+\.[0-9]+(?:(?:a|b|rc)[0-9]+)?$"
)


def read_project_version() -> str:
    """Read the PEP 621 version without adding a TOML dependency to Python 3.9."""

    match = re.search(
        r'^version\s*=\s*["\']([^"\']+)["\']',
        PYPROJECT.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if match is None:
        raise RuntimeError("Could not find [project] version in pyproject.toml.")
    return match.group(1)


def _run(command: Sequence[str], *, cwd: Path = REPO_ROOT, env=None) -> None:
    print("+", " ".join(os.fspath(part) for part in command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _forbidden_member(name: str) -> bool:
    normalized = name.replace("\\", "/").strip("/")
    parts = tuple(part for part in normalized.split("/") if part)
    if not parts:
        return False
    if any(part in {".DS_Store", "@eaDir", "__pycache__"} for part in parts):
        return True
    if any(part.startswith("._") or part.endswith((".pyc", ".pyo")) for part in parts):
        return True
    return "dropps" in parts and "backups" in parts


def _metadata_fields(text: str) -> dict[str, str]:
    fields = {}
    for line in text.splitlines():
        if not line or line[0].isspace() or ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields.setdefault(key, value.strip())
    return fields


def _validate_members(names: Iterable[str], path: Path) -> None:
    forbidden = sorted(name for name in names if _forbidden_member(name))
    if forbidden:
        preview = ", ".join(forbidden[:5])
        raise RuntimeError(f"{path.name} contains forbidden members: {preview}")


def validate_wheel(path: Path, version: str) -> None:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        _validate_members(names, path)
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise RuntimeError(f"{path.name} has {len(metadata_names)} METADATA files.")
        fields = _metadata_fields(
            archive.read(metadata_names[0]).decode("utf-8", errors="strict")
        )
        if fields.get("Name", "").lower() != "dropps":
            raise RuntimeError(f"{path.name} has unexpected package name metadata.")
        if fields.get("Version") != version:
            raise RuntimeError(f"{path.name} has version {fields.get('Version')!r}.")
        record_name = metadata_names[0].rsplit("/", 1)[0] + "/RECORD"
        if record_name not in names:
            raise RuntimeError(f"{path.name} is missing RECORD.")
        rows = list(csv.reader(io.StringIO(archive.read(record_name).decode("utf-8"))))
        if len(names) != len(set(names)) or len(rows) != len({row[0] for row in rows}):
            raise RuntimeError(f"{path.name} has duplicate archive or RECORD entries.")
        if {row[0] for row in rows} != {name for name in names if not name.endswith("/")}:
            raise RuntimeError(f"{path.name} RECORD does not cover the complete payload.")
        for name, digest, size in rows:
            if name == record_name:
                continue
            payload = archive.read(name)
            expected = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode("ascii")
            if digest != f"sha256={expected}" or size != str(len(payload)):
                raise RuntimeError(f"{path.name} RECORD hash/size mismatch for {name}.")


def validate_sdist(path: Path, version: str) -> None:
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        _validate_members(names, path)
        metadata = [member for member in members if member.name.endswith("/PKG-INFO")]
        if len(metadata) != 1:
            raise RuntimeError(f"{path.name} has {len(metadata)} PKG-INFO files.")
        extracted = archive.extractfile(metadata[0])
        if extracted is None:
            raise RuntimeError(f"Could not read metadata from {path.name}.")
        fields = _metadata_fields(extracted.read().decode("utf-8", errors="strict"))
        if fields.get("Name", "").lower() != "dropps":
            raise RuntimeError(f"{path.name} has unexpected package name metadata.")
        if fields.get("Version") != version:
            raise RuntimeError(f"{path.name} has version {fields.get('Version')!r}.")

        required_suffixes = (
            "/CHANGELOG.md",
            "/docs/architecture.md",
            "/docs/developer-guide.md",
            "/docs/user-guide.md",
            "/docs/command-reference.md",
            "/docs/commands/mdrun.md",
            "/LICENSE",
            "/NOTICE",
            "/CITATION.cff",
            "/tests/test_workflow_smoke.py",
            "/examples/colab/README.md",
            "/scripts/install_dropps.py",
            "/scripts/prepare_release.py",
        )
        missing = [
            suffix
            for suffix in required_suffixes
            if not any(name.endswith(suffix) for name in names)
        ]
        if missing:
            raise RuntimeError(
                f"{path.name} is missing source-distribution files: "
                + ", ".join(missing)
            )


def validate_payload_match(wheel: Path, sdist: Path) -> None:
    """Require wheel and sdist to carry identical runtime package bytes."""

    with zipfile.ZipFile(wheel) as archive:
        wheel_payload = {
            name: archive.read(name)
            for name in archive.namelist()
            if name.startswith("dropps/") and not name.endswith("/")
        }

    with tarfile.open(sdist, "r:gz") as archive:
        sdist_payload = {}
        for member in archive.getmembers():
            marker = "/dropps/"
            if marker not in member.name or not member.isfile():
                continue
            relative = "dropps/" + member.name.split(marker, 1)[1]
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"Could not read {member.name} from {sdist.name}.")
            sdist_payload[relative] = extracted.read()

    wheel_only = sorted(wheel_payload.keys() - sdist_payload.keys())
    sdist_only = sorted(sdist_payload.keys() - wheel_payload.keys())
    changed = sorted(
        name
        for name in wheel_payload.keys() & sdist_payload.keys()
        if wheel_payload[name] != sdist_payload[name]
    )
    if wheel_only or sdist_only or changed:
        raise RuntimeError(
            "Wheel/sdist runtime payload mismatch: "
            f"wheel-only={wheel_only[:5]}, sdist-only={sdist_only[:5]}, "
            f"changed={changed[:5]}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_paths(directory: Path, version: str) -> tuple[Path, Path]:
    wheel = directory / f"dropps-{version}-py3-none-any.whl"
    sdist = directory / f"dropps-{version}.tar.gz"
    missing = [path.name for path in (wheel, sdist) if not path.is_file()]
    if missing:
        raise RuntimeError("Build did not produce expected artifacts: " + ", ".join(missing))
    unexpected = sorted(
        path.name
        for path in directory.iterdir()
        if path.is_file() and path not in {wheel, sdist}
    )
    if unexpected:
        raise RuntimeError("Build produced unexpected artifacts: " + ", ".join(unexpected))
    return wheel, sdist


def copy_release_source(destination: Path) -> None:
    """Stage only source inputs, without filesystem metadata or local reports."""
    destination.mkdir(parents=True)
    inputs = [
        "pyproject.toml", "README.md", "LICENSE", "NOTICE", "LICENSE.md",
        "CITATION.cff", "CHANGELOG.md", "requirements.txt",
        "dropps", "docs", "tests", "scripts", "examples",
    ]
    for name in inputs:
        source = REPO_ROOT / name
        paths = source.rglob("*") if source.is_dir() else [source]
        for path in paths:
            relative = path.relative_to(REPO_ROOT)
            if not path.is_file() or _forbidden_member(relative.as_posix()):
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Bundle destination; defaults to release/<version>.",
    )
    parser.add_argument(
        "--test-python",
        default=sys.executable,
        help="Python executable containing the scientific test dependencies.",
    )
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args(argv)

    version = read_project_version()
    if version != args.expected_version:
        raise RuntimeError(
            f"Project version is {version!r}, expected {args.expected_version!r}."
        )
    if RELEASE_VERSION.fullmatch(version) is None:
        raise RuntimeError(
            f"{version!r} is not a supported x.y.z or PEP 440 prerelease version."
        )

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else REPO_ROOT / "release" / version
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Release destination is not empty: {output_dir}")

    _run([sys.executable, "scripts/update_readme_commands.py", "--check"])
    _run([os.fspath(args.test_python), "scripts/update_command_docs.py", "--check"])
    _run([sys.executable, "examples/colab/validate_notebooks.py"])

    if not args.skip_tests:
        test_env = os.environ.copy()
        test_env["PYTHONDONTWRITEBYTECODE"] = "1"
        _run(
            [
                os.fspath(args.test_python),
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-p",
                "test_*.py",
            ],
            env=test_env,
        )

    with tempfile.TemporaryDirectory(prefix="dropps-release-") as temporary:
        build_dir = Path(temporary) / "artifacts"
        source_dir = Path(temporary) / "source"
        copy_release_source(source_dir)
        _run(
            [
                sys.executable,
                "-m",
                "build",
                "--no-isolation",
                "--outdir",
                os.fspath(build_dir),
            ],
            cwd=source_dir,
        )
        wheel, sdist = _artifact_paths(build_dir, version)
        validate_wheel(wheel, version)
        validate_sdist(sdist, version)
        validate_payload_match(wheel, sdist)

        smoke_env = os.environ.copy()
        smoke_env["PYTHONPATH"] = os.fspath(wheel)
        _run(
            [
                sys.executable,
                "-c",
                (
                    "import dropps; "
                    f"assert dropps.__version__ == {version!r}; "
                    "print(dropps.__version__)"
                ),
            ],
            cwd=build_dir,
            env=smoke_env,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        copied = [output_dir / wheel.name, output_dir / sdist.name]
        for source, destination in zip((wheel, sdist), copied):
            shutil.copy2(source, destination)

    checksum_path = output_dir / "SHA256SUMS"
    checksum_text = "".join(
        f"{_sha256(path)}  {path.name}\n" for path in copied
    )
    checksum_path.write_text(checksum_text, encoding="ascii")
    print(f"Release bundle ready: {output_dir}")
    print(checksum_text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
