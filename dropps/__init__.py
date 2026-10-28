"""DROPPS package metadata."""

import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _get_version() -> str:
    """Return the source-tree version, or installed metadata as a fallback."""

    # When PYTHONPATH points at a checkout, installed metadata may belong to an
    # older build in the same environment. The adjacent pyproject.toml is the
    # authoritative version for that source tree.
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    try:
        match = re.search(
            r'^version\s*=\s*["\']([^"\']+)["\']',
            pyproject.read_text(encoding="utf-8"),
            re.MULTILINE,
        )
    except OSError:
        match = None
    if match is not None:
        return match.group(1)

    try:
        return version("dropps")
    except PackageNotFoundError:
        return "unknown"


__version__ = _get_version()

__all__ = ["__version__"]
