#!/usr/bin/env python3
"""Update README command catalog from source code.

This script parses command registration in dropps/share/all_commands.py and then
extracts command names, descriptions, and required arguments from each module
without importing runtime dependencies.
"""

from __future__ import annotations

import ast
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


@dataclass
class ArgInfo:
    options: list[str]
    required: bool


@dataclass
class CommandInfo:
    name: str
    desc: str
    required_args: list[ArgInfo]


def _parse_all_commands() -> dict[str, list[Path]]:
    tree = ast.parse(ALL_COMMANDS_FILE.read_text(encoding="utf-8"), filename=str(ALL_COMMANDS_FILE))
    result: dict[str, list[Path]] = {k: [] for k in SECTION_MAP}

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
            # Expect: dropps.commands.module.symbol or dropps.analysis.module.symbol
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


def _extract_parser_required_args(func_def: ast.FunctionDef) -> list[ArgInfo]:
    required_args: list[ArgInfo] = []

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
            if text is not None and text.startswith("-"):
                options.append(text)

        if len(options) == 0:
            continue

        required = False
        for kw in node.keywords:
            if kw.arg == "required" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                required = True
                break

        if required:
            required_args.append(ArgInfo(options=options, required=True))

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
                required_args = _extract_parser_required_args(node)
                break

    if not command_name:
        command_name = module_file.stem

    return CommandInfo(name=command_name, desc=desc or "(No description)", required_args=required_args)


def _format_required_args(required_args: list[ArgInfo]) -> str:
    if len(required_args) == 0:
        return "None"
    labels = []
    for item in required_args:
        labels.append("/".join(item.options))
    return ", ".join(f"`{label}`" for label in labels)


def _render_commands_section(section_modules: dict[str, list[Path]]) -> str:
    lines: list[str] = []
    lines.append(BEGIN_MARKER)
    lines.append("_This section is auto-generated from source. Do not edit manually._")
    lines.append("")

    for section_key, title in SECTION_MAP.items():
        lines.append(f"### {title}")
        lines.append("")

        for module_file in section_modules.get(section_key, []):
            info = _parse_module_info(module_file)
            if info.name == "help":
                continue
            lines.append(f"- `{info.name}`: {info.desc}")
            lines.append(f"  Required args: {_format_required_args(info.required_args)}")

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


def main() -> None:
    section_modules = _parse_all_commands()
    rendered = _render_commands_section(section_modules)
    _update_readme(rendered)
    print(f"Updated {README_FILE}")


if __name__ == "__main__":
    main()
