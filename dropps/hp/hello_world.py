"""DROPPS command-line startup banner."""

from __future__ import annotations

from os import getcwd
from pathlib import Path
from random import choice
from sys import executable
from textwrap import wrap

from dropps import __version__
from dropps.hp.quotes import QUOTES


package_name = "DROPPS"
package_abbr = "DPS"
package_description = (
    "Distributed Rapid Operation Platform for Phase-separation Simulations"
)
developer_name = "Yiming Tang"
developer_affiliation = "Fudan University"
developer_email = "ymtang@fudan.edu.cn"

_BANNER_WIDTH = 80
_CONTENT_WIDTH = _BANNER_WIDTH - 2
_INTRODUCTIONS = (
    "DROPPS quote",
    "A thought for this run",
    "Simulation fuel",
    "A spark between timesteps",
    "Wisdom for the trajectory",
)


def _random_reminder():
    introduction = choice(_INTRODUCTIONS)
    theme, quote, author = choice(QUOTES)
    return f'{introduction} [{theme}]: "{quote}" — {author}'


def _border(left, fill, right):
    return left + fill * _CONTENT_WIDTH + right


def _box_line(text="", *, align="left"):
    if align == "center":
        content = text.center(_CONTENT_WIDTH)
    else:
        content = text.ljust(_CONTENT_WIDTH)
    return f"│{content}│"


def _field_lines(label, value):
    prefix = f"  {label:<9}"
    continuation = " " * len(prefix)
    available = _CONTENT_WIDTH - len(prefix) - 2
    value_lines = wrap(str(value), width=available, break_long_words=False) or [""]
    lines = [f"{prefix}{value_lines[0]}  "]
    lines.extend(f"{continuation}{line}  " for line in value_lines[1:])
    return [_box_line(line) for line in lines]


def _text_lines(text):
    lines = wrap(text, width=_CONTENT_WIDTH - 4, break_long_words=False) or [""]
    return [_box_line(f"  {line}  ") for line in lines]


def print_hello(entry, cmd):
    """Print the startup banner for both help and executable commands."""

    launcher_name = Path(entry).name or "dps"
    command_line = " ".join(part for part in (launcher_name, cmd) if part)

    lines = [
        _border("╭", "─", "╮"),
        _box_line(f"{package_name} · {package_abbr} {__version__}", align="center"),
        _box_line(package_description, align="center"),
        _box_line(
            f"Developed by {developer_name} · {developer_affiliation}",
            align="center",
        ),
        _box_line(developer_email, align="center"),
        _border("├", "─", "┤"),
    ]
    lines.extend(_field_lines("Command", command_line))
    lines.extend(_field_lines("Python", executable))
    lines.extend(_field_lines("Launcher", entry))
    lines.extend(_field_lines("Workdir", getcwd()))
    lines.extend(
        (
            _border("├", "─", "┤"),
            _box_line("  Help"),
            _box_line("    dps help commands    List commands by task"),
            _box_line("    dps help COMMAND     Explain one command"),
            _box_line("    dps COMMAND -h       Show detailed options and examples"),
            _border("├", "─", "┤"),
        )
    )
    lines.extend(_text_lines(_random_reminder()))
    lines.append(_border("╰", "─", "╯"))

    print("\n".join(lines), end="\n\n")
