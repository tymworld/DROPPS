#!/usr/bin/env python3
"""Update README command catalog from source code.

This script parses command registration in dropps/share/all_commands.py and then
extracts command names, descriptions, and required arguments from each module
without importing runtime dependencies.
"""

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ALL_COMMANDS_FILE = REPO_ROOT / "dropps" / "share" / "all_commands.py"
README_FILE = REPO_ROOT / "README.md"

BEGIN_MARKER = "<!-- BEGIN AUTO COMMANDS -->"
END_MARKER = "<!-- END AUTO COMMANDS -->"

SECTION_MAP = {
    "commands_modelling": "Modeling",
    "commands_simulation": "Simulation",
    "commands_analysis": "Analysis",
}

CANONICAL_LONG_OPTION = re.compile(r"^--[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
CANONICAL_SHORT_OPTION = re.compile(r"^-[a-z][a-z0-9]?$")


def _is_documented_option(option: str) -> bool:
    """Keep standard public aliases and omit compatibility-only spellings."""

    return bool(
        CANONICAL_SHORT_OPTION.fullmatch(option)
        or CANONICAL_LONG_OPTION.fullmatch(option)
    )


@dataclass
class ArgInfo:
    options: list[str]
    required: bool
    exclusive_group: str | None = None


@dataclass
class CommandInfo:
    name: str
    desc: str
    required_args: list[ArgInfo]


def _parse_all_commands() -> dict[str, list[Path]]:
    tree = ast.parse(ALL_COMMANDS_FILE.read_text(encoding="utf-8"), filename=str(ALL_COMMANDS_FILE))
    result: dict[str, list[Path]] = {k: [] for k in SECTION_MAP}

    imported_commands: dict[str, Path] = {}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        chunks = node.module.split(".")
        if len(chunks) != 3 or chunks[:2] not in (
            ["dropps", "commands"],
            ["dropps", "analysis"],
        ):
            continue
        module_file = REPO_ROOT.joinpath(*chunks).with_suffix(".py")
        for alias in node.names:
            imported_commands[alias.asname or alias.name] = module_file

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue

        section_name = node.targets[0].id
        if section_name not in SECTION_MAP:
            continue

        if not isinstance(node.value, ast.List):
            continue

        for elt in node.value.elts:
            if isinstance(elt, ast.Name) and elt.id in imported_commands:
                result[section_name].append(imported_commands[elt.id])
                continue

            # Backwards compatibility with the original registry layout:
            # dropps.commands.module.symbol or dropps.analysis.module.symbol.
            if not isinstance(elt, ast.Attribute):
                continue
            path_chunks: list[str] = []
            cur: Any = elt
            while isinstance(cur, ast.Attribute):
                path_chunks.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                path_chunks.append(cur.id)
            path_chunks = list(reversed(path_chunks))

            if len(path_chunks) < 4:
                continue
            if path_chunks[0] != "dropps" or path_chunks[1] not in {"commands", "analysis"}:
                continue

            module_file = REPO_ROOT / "dropps" / path_chunks[1] / f"{path_chunks[2]}.py"
            result[section_name].append(module_file)

    return result


def _get_str_constant(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _extract_parser_required_args(
    func_def: ast.FunctionDef,
    module_tree: ast.Module,
    module_file: Path,
    seen: set[tuple[Path, str]] | None = None,
) -> list[ArgInfo]:
    if seen is None:
        seen = set()
    key = (module_file.resolve(), func_def.name)
    if key in seen:
        return []
    seen.add(key)

    required_args: list[ArgInfo] = []
    required_exclusive_groups: set[str] = set()

    for node in ast.walk(func_def):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        call = node.value
        if not isinstance(call.func, ast.Attribute):
            continue
        if call.func.attr != "add_mutually_exclusive_group":
            continue
        if any(
            kw.arg == "required"
            and isinstance(kw.value, ast.Constant)
            and kw.value.value is True
            for kw in call.keywords
        ):
            required_exclusive_groups.add(node.targets[0].id)

    for node in ast.walk(func_def):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue

        options: list[str] = []
        for arg in node.args:
            text = _get_str_constant(arg)
            if (
                text is not None
                and text.startswith("-")
                and _is_documented_option(text)
            ):
                options.append(text)

        if len(options) == 0:
            continue

        short_options = [
            option for option in options if CANONICAL_SHORT_OPTION.fullmatch(option)
        ]
        long_options = [
            option for option in options if CANONICAL_LONG_OPTION.fullmatch(option)
        ]
        options = [*short_options[:1], *long_options]

        required = False
        exclusive_group = None
        for kw in node.keywords:
            if kw.arg == "required" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                required = True
                break

        if (
            isinstance(node.func.value, ast.Name)
            and node.func.value.id in required_exclusive_groups
        ):
            required = True
            exclusive_group = node.func.value.id

        if required:
            required_args.append(
                ArgInfo(
                    options=options,
                    required=True,
                    exclusive_group=exclusive_group,
                )
            )

    imported_functions: dict[str, Path] = {}
    for node in module_tree.body:
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        if not node.module.startswith("dropps."):
            continue
        imported_module = REPO_ROOT.joinpath(*node.module.split(".")).with_suffix(".py")
        for alias in node.names:
            imported_functions[alias.asname or alias.name] = imported_module

    for node in ast.walk(func_def):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        helper_file = imported_functions.get(node.func.id)
        if helper_file is None or not helper_file.exists():
            continue
        helper_tree = ast.parse(
            helper_file.read_text(encoding="utf-8"),
            filename=str(helper_file),
        )
        for helper_node in helper_tree.body:
            if isinstance(helper_node, ast.FunctionDef) and helper_node.name == node.func.id:
                required_args.extend(
                    _extract_parser_required_args(
                        helper_node,
                        helper_tree,
                        helper_file,
                        seen,
                    )
                )
                break

    return required_args


def _parse_module_info(module_file: Path) -> CommandInfo:
    tree = ast.parse(module_file.read_text(encoding="utf-8"), filename=str(module_file))

    desc = ""
    command_name = ""
    getargs_name = ""

    def _resolve_desc_value(value_node: ast.AST) -> str | None:
        text = _get_str_constant(value_node)
        if text is not None:
            return text
        if isinstance(value_node, ast.Name):
            # Resolve desc variable if call passes variable name.
            for n in tree.body:
                if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                    if n.targets[0].id == value_node.id:
                        return _get_str_constant(n.value)
        return None

    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = node.targets[0].id
            if target == "desc":
                value = _get_str_constant(node.value)
                if value is not None:
                    desc = " ".join(value.strip().split())

        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Name) and call.func.id == "single_command" and len(call.args) >= 2:
                name_val = _get_str_constant(call.args[0])
                if name_val:
                    command_name = name_val
                if isinstance(call.args[1], ast.Name):
                    getargs_name = call.args[1].id
                if len(call.args) >= 4:
                    desc_from_call = _resolve_desc_value(call.args[3])
                    if desc_from_call:
                        desc = " ".join(desc_from_call.strip().split())

    required_args: list[ArgInfo] = []
    if getargs_name:
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == getargs_name:
                required_args = _extract_parser_required_args(
                    node,
                    tree,
                    module_file,
                )
                break

    if not command_name:
        command_name = module_file.stem

    return CommandInfo(name=command_name, desc=desc or "(No description)", required_args=required_args)


def _format_required_args(required_args: list[ArgInfo]) -> str:
    if len(required_args) == 0:
        return "None"

    labels: list[str] = []
    emitted_groups: set[str] = set()
    for item in required_args:
        if item.exclusive_group is None:
            labels.append(f"`{'/'.join(item.options)}`")
            continue
        if item.exclusive_group in emitted_groups:
            continue
        emitted_groups.add(item.exclusive_group)
        alternatives = [
            f"`{'/'.join(grouped.options)}`"
            for grouped in required_args
            if grouped.exclusive_group == item.exclusive_group
        ]
        labels.append("one of " + " or ".join(alternatives))
    return ", ".join(labels)


def _render_commands_section(section_modules: dict[str, list[Path]]) -> str:
    help_tree = ast.parse((REPO_ROOT / "dropps/share/command_help.py").read_text(encoding="utf-8"))
    examples = next(
        ast.literal_eval(node.value)
        for node in help_tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "COMMAND_EXAMPLES" for target in node.targets)
    )
    lines: list[str] = []
    lines.append(BEGIN_MARKER)
    lines.append("Every public command is listed below with a minimal example and a link to its complete options, defaults, units, and workflow notes. See the [complete command reference](docs/command-reference.md) for task-based navigation.")
    lines.append("")

    for section_key, title in SECTION_MAP.items():
        lines.append(f"### {title}")
        lines.append("")

        for module_file in section_modules.get(section_key, []):
            info = _parse_module_info(module_file)
            lines.append(f"- `{info.name}`: {info.desc}")
            lines.append(f"  Required args: {_format_required_args(info.required_args)}")
            lines.append(f"  [Full reference](docs/commands/{info.name}.md).")
            example = "dps help commands" if info.name == "help" else examples[info.name][0]
            lines.extend(["", "  ```bash", "  " + example, "  ```", ""])

        lines.append("")

    lines.append(END_MARKER)
    return "\n".join(lines).rstrip() + "\n"


def _update_readme(rendered_section: str) -> None:
    text = README_FILE.read_text(encoding="utf-8")

    if BEGIN_MARKER in text and END_MARKER in text:
        start = text.index(BEGIN_MARKER)
        end = text.index(END_MARKER) + len(END_MARKER)
        new_text = text[:start] + rendered_section + text[end:]
    else:
        anchor = "## Commands"
        idx = text.find(anchor)
        if idx == -1:
            raise RuntimeError("README does not contain '## Commands' section.")

        insert_at = idx + len(anchor)
        new_text = text[:insert_at] + "\n\n" + rendered_section + text[insert_at:]

    README_FILE.write_text(new_text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the generated command catalog differs from README.",
    )
    args = parser.parse_args(argv)

    section_modules = _parse_all_commands()
    rendered = _render_commands_section(section_modules)
    if args.check:
        text = README_FILE.read_text(encoding="utf-8")
        if BEGIN_MARKER not in text or END_MARKER not in text:
            print("README command catalog markers are missing.")
            return 1
        start = text.index(BEGIN_MARKER)
        end = text.index(END_MARKER) + len(END_MARKER)
        current = text[start:end].rstrip() + "\n"
        if current != rendered:
            print(
                "README command catalog is stale; run "
                "'python scripts/update_readme_commands.py'."
            )
            return 1
        print("README command catalog is current.")
        return 0

    _update_readme(rendered)
    print(f"Updated {README_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
