"""Consistent terminal output and opt-in runtime diagnostics for DROPPS."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import re
import shlex
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, TextIO


_DEBUG = False
_START_TIME = time.perf_counter()
_LEVEL_PREFIXES = {
    "debug": "[DEBUG]",
    "info": "[INFO]",
    "success": "[OK]",
    "warning": "[WARNING]",
    "error": "[ERROR]",
}
_LEVEL_COLORS = {
    "debug": "\033[2;36m",
    "info": "\033[1;34m",
    "success": "\033[1;32m",
    "warning": "\033[1;33m",
    "error": "\033[1;31m",
}
_ANSI_RESET = "\033[0m"
_ANSI_PREFIX = re.compile(r"^(?:\x1b\[[0-9;]*m)*")
_LEGACY_PREFIXES = (
    ("## CRITICAL WARNING:", "warning"),
    ("CRITICAL WARNING:", "warning"),
    ("## WARNING:", "warning"),
    ("WARNING:", "warning"),
    ("## ERROR:", "error"),
    ("ERROR:", "error"),
    ("ERROR ", "error"),
    ("Error:", "error"),
    ("WARNING ", "warning"),
    ("Warning:", "warning"),
    ("## ", "info"),
)
_LEGACY_ERROR_LINE = re.compile(r"^dps\s+\S+\s*:\s*error\s*:", re.IGNORECASE)


def configure(*, debug: bool = False) -> None:
    """Configure process-wide CLI diagnostics."""

    global _DEBUG, _START_TIME
    _DEBUG = bool(debug)
    _START_TIME = time.perf_counter()


def debug_enabled() -> bool:
    return _DEBUG


def _color_enabled(stream: TextIO) -> bool:
    mode = os.environ.get("DROPPS_COLOR", "auto").strip().lower()
    if mode not in {"auto", "always", "never"}:
        mode = "auto"
    if mode == "never" or "NO_COLOR" in os.environ:
        return False
    if mode == "always" or os.environ.get("FORCE_COLOR", "") not in {"", "0"}:
        return True
    return bool(getattr(stream, "isatty", lambda: False)()) and os.environ.get(
        "TERM", ""
    ).lower() != "dumb"


def _styled_prefix(level: str, stream: TextIO) -> str:
    prefix = _LEVEL_PREFIXES[level]
    if not _color_enabled(stream):
        return prefix
    return f"{_LEVEL_COLORS[level]}{prefix}{_ANSI_RESET}"


def emit(level: str, message: object, *, stream: TextIO | None = None) -> None:
    """Write one standardized terminal message."""

    if level == "debug" and not _DEBUG:
        return
    if level not in _LEVEL_PREFIXES:
        raise ValueError(f"Unknown terminal message level: {level!r}")
    if stream is None:
        stream = sys.stderr if level in {"debug", "warning", "error"} else sys.stdout
    print(f"{_styled_prefix(level, stream)} {message}", file=stream)


def debug(message: object) -> None:
    emit("debug", message)


def info(message: object) -> None:
    emit("info", message)


def success(message: object) -> None:
    emit("success", message)


def warning(message: object) -> None:
    emit("warning", message)


def error(message: object) -> None:
    emit("error", message)


def _installed_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _runtime_dropps_version() -> str:
    from dropps import __version__

    installed = _installed_version("dropps")
    if installed in {__version__, "not installed"}:
        return __version__
    return f"{__version__} (installed metadata={installed})"


def emit_runtime_context(argv: list[str], command_name: str | None = None) -> None:
    """Report reproducibility-oriented runtime context in debug mode."""

    if not _DEBUG:
        return
    debug("Debug diagnostics enabled.")
    debug(f"Command: {command_name or 'top-level'}")
    debug(f"Arguments: {shlex.join(argv)}")
    debug(f"Working directory: {Path.cwd()}")
    debug(f"Process: pid={os.getpid()}")
    debug(f"Python: {platform.python_version()} ({sys.executable})")
    debug(f"Platform: {platform.platform()}")
    version_items = [("DROPPS", _runtime_dropps_version())]
    version_items.extend(
        (name, _installed_version(distribution))
        for name, distribution in (
            ("OpenMM", "openmm"),
            ("MDAnalysis", "MDAnalysis"),
            ("NumPy", "numpy"),
            ("SciPy", "scipy"),
        )
    )
    versions = ", ".join(f"{name}={version}" for name, version in version_items)
    debug(f"Distributions: {versions}")


def emit_parsed_arguments(arguments: Any) -> None:
    if not _DEBUG:
        return
    values = vars(arguments) if hasattr(arguments, "__dict__") else arguments
    debug(f"Parsed arguments: {values!r}")


_INPUT_ARGUMENTS = frozenset(
    {
        "angle_list",
        "checkpoint",
        "elastic_residues",
        "group_file",
        "index",
        "input",
        "input_pdb",
        "parameter",
        "reference",
        "run_input",
        "sequence",
        "structure",
        "topology",
        "trajectory",
    }
)


def _display_value(value: Any, limit: int = 160) -> str:
    if isinstance(value, Path):
        rendered = str(value)
    elif isinstance(value, str):
        rendered = value
    else:
        rendered = repr(value)
    if len(rendered) > limit:
        return f"{rendered[: limit - 3]}..."
    return rendered


def _argument_summary(arguments: Any) -> tuple[list[str], list[str], list[str]]:
    values = vars(arguments) if hasattr(arguments, "__dict__") else {}
    inputs: list[str] = []
    outputs: list[str] = []
    parameters: list[str] = []
    for name, value in sorted(values.items()):
        if name == "debug" or value is None:
            continue
        item = f"{name.replace('_', '-').lower()}={_display_value(value)}"
        if name in _INPUT_ARGUMENTS or name.startswith("input_"):
            inputs.append(item)
        elif name == "output" or name.startswith("output_") or name.endswith("_output"):
            outputs.append(item)
        else:
            parameters.append(item)
    return inputs, outputs, parameters


def emit_command_start(command_name: str, arguments: Any) -> None:
    """Print the same useful execution summary for every command."""

    info(f"Starting command '{command_name}'.")
    inputs, outputs, parameters = _argument_summary(arguments)
    if inputs:
        info("Inputs: " + ", ".join(inputs))
    if outputs:
        info("Outputs: " + ", ".join(outputs))
    if parameters:
        info("Parameters: " + ", ".join(parameters))


def emit_command_result(command_name: str, status: int) -> None:
    if status == 0:
        success(f"Command '{command_name}' completed successfully.")
    else:
        error(f"Command '{command_name}' failed with status {status}.")


def emit_completion(status: int) -> None:
    if _DEBUG:
        elapsed = time.perf_counter() - _START_TIME
        debug(f"Command completed: status={status}, elapsed={elapsed:.6f} s")


def emit_exception(exc: BaseException) -> None:
    """Print a concise failure, plus a traceback when debug mode is enabled."""

    error(f"{type(exc).__name__}: {exc}")
    if _DEBUG:
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
    else:
        emit(
            "info",
            "Re-run with --debug to display the traceback and runtime context.",
            stream=sys.stderr,
        )


def _normalize_legacy_prefix(text: str) -> tuple[str, str | None]:
    stripped = text.lstrip()
    indentation = text[: len(text) - len(stripped)]
    for prefix, level in _LEGACY_PREFIXES:
        if stripped.startswith(prefix):
            remainder = stripped[len(prefix) :].lstrip()
            return f"{indentation}{_LEVEL_PREFIXES[level]} {remainder}", level
    if _LEGACY_ERROR_LINE.match(stripped):
        return f"{indentation}{_LEVEL_PREFIXES['error']} {stripped}", "error"
    return text, None


def _standard_level(text: str) -> str | None:
    stripped = text.lstrip()
    unstyled = _ANSI_PREFIX.sub("", stripped)
    for level, prefix in _LEVEL_PREFIXES.items():
        if unstyled.startswith(prefix):
            return level
    return None


class _NormalizedStream:
    """Translate legacy line prefixes while preserving file-like behavior."""

    def __init__(
        self,
        stream: TextIO,
        error_stream: TextIO | None = None,
        *,
        normalize_plain: bool = False,
        plain_level: str = "info",
    ) -> None:
        self._stream = stream
        self._error_stream = error_stream
        self._normalize_plain = normalize_plain
        self._plain_level = plain_level
        self._at_line_start = True
        self._line_stream = stream

    def write(self, text: str) -> int:
        if not isinstance(text, str):
            text = str(text)
        written = len(text)
        for chunk in text.splitlines(keepends=True) or [text]:
            content = chunk
            if self._at_line_start and content and not content.startswith("\r"):
                level = _standard_level(content)
                if level is None:
                    content, level = _normalize_legacy_prefix(content)
                if (
                    level is None
                    and self._normalize_plain
                    and content.strip()
                    and not content.lstrip().startswith("[")
                ):
                    indentation = content[: len(content) - len(content.lstrip())]
                    content = (
                        f"{indentation}{_LEVEL_PREFIXES[self._plain_level]} "
                        f"{content.lstrip()}"
                    )
                    level = self._plain_level
                self._line_stream = (
                    self._error_stream
                    if level in {"warning", "error"} and self._error_stream is not None
                    else self._stream
                )
                if (
                    level is not None
                    and not content.lstrip().startswith("\033[")
                    and _color_enabled(self._line_stream)
                ):
                    plain = _LEVEL_PREFIXES[level]
                    content = content.replace(
                        plain,
                        _styled_prefix(level, self._line_stream),
                        1,
                    )
            self._line_stream.write(content)
            if content.endswith("\n"):
                self._at_line_start = True
                self._line_stream = self._stream
            elif content.endswith("\r"):
                # A bare carriage return updates one terminal line in place;
                # the following progress payload is not a new log message.
                self._at_line_start = False
            elif content:
                self._at_line_start = False
        return written

    def flush(self) -> None:
        self._stream.flush()
        if self._error_stream is not None and self._error_stream is not self._stream:
            self._error_stream.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


@contextmanager
def normalized_output(*, normalize_plain: bool = False) -> Iterator[None]:
    """Normalize legacy output emitted by command implementations."""

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = _NormalizedStream(  # type: ignore[assignment]
        old_stdout,
        old_stderr,
        normalize_plain=normalize_plain,
        plain_level="info",
    )
    sys.stderr = _NormalizedStream(  # type: ignore[assignment]
        old_stderr,
        normalize_plain=normalize_plain,
        # stderr is also the conventional transport for tqdm and other
        # progress renderers; the stream itself does not imply warning severity.
        plain_level="info",
    )
    try:
        yield
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
