#!/usr/bin/env python3
"""Minimal yq-compatible helper for CI functional test scripts."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml


def format_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def load_document(path: str | None) -> Any:
    if path:
        text = Path(path).read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()
    data = yaml.safe_load(text)
    return data if data is not None else {}


def write_document(path: str, data: Any) -> None:
    Path(path).write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def parse_path(path_expr: str) -> list[str]:
    if not path_expr.startswith("."):
        raise ValueError(f"unsupported path expression: {path_expr}")

    parts: list[str] = []
    index = 0
    length = len(path_expr)
    while index < length:
        if path_expr[index] != ".":
            raise ValueError(f"invalid path expression: {path_expr}")
        index += 1
        if index >= length:
            raise ValueError(f"incomplete path expression: {path_expr}")

        if path_expr[index] == '"':
            end = path_expr.find('"', index + 1)
            if end == -1:
                raise ValueError(f"unterminated quoted key in: {path_expr}")
            parts.append(path_expr[index + 1 : end])
            index = end + 1
        else:
            end = index
            while end < length and path_expr[end] != ".":
                end += 1
            parts.append(path_expr[index:end])
            index = end

    return parts


def get_path(data: Any, path_expr: str) -> Any:
    current = data
    for part in parse_path(path_expr):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def set_path(data: Any, path_expr: str, value: Any) -> None:
    parts = parse_path(path_expr)
    current = data
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def parse_literal(expr: str, data: Any) -> Any:
    expr = expr.strip()
    if expr.startswith("."):
        return get_path(data, expr)
    if expr.startswith('"') and expr.endswith('"'):
        return expr[1:-1]
    if expr == "true":
        return True
    if expr == "false":
        return False
    if re.fullmatch(r"-?\d+", expr):
        return int(expr)
    if re.fullmatch(r"-?\d+\.\d+", expr):
        return float(expr)
    return expr


def print_defaulted_path(data: Any, expr: str) -> bool:
    match = re.fullmatch(r'(\..+?)\s*//\s*(".*"|[^"].*)', expr)
    if not match:
        return False
    value = get_path(data, match.group(1))
    if value is None:
        value = parse_literal(match.group(2), data)
    print(format_value(value))
    return True


def print_simple_path(data: Any, expr: str) -> bool:
    if not expr.startswith("."):
        return False
    print(format_value(get_path(data, expr)))
    return True


def handle_read_query(data: Any, expr: str) -> int:
    if expr == '... comments="" | .ENV_VARS | to_entries | .[] | [.key + "=" + .value] | join(" ")':
        for key, value in (data.get("ENV_VARS") or {}).items():
            print(f"{key}={format_value(value)}")
        return 0

    if expr == '... comments="" | .MODEL_ARGS | to_entries | .[] | with(select(.value == true); .value = "true") | .key + "=" + (select(.value != "") | .value | tostring)':
        for key, value in (data.get("MODEL_ARGS") or {}).items():
            if format_value(value) != "":
                print(f"{key}={format_value(value)}")
        return 0

    if expr == 'explode(.) | ... comments="" | .[env(KEY)] | to_entries | .[] | with(select(.value == true); .value = "true") | .key + ": " + (select(.value != "") | .value | tostring)':
        key = os.environ.get("KEY", "MODEL_ARGS")
        for item_key, value in (data.get(key) or {}).items():
            if format_value(value) != "":
                print(f"{item_key}: {format_value(value)}")
        return 0

    match = re.fullmatch(r'has\("([^"]+)"\)', expr)
    if match:
        print("true" if match.group(1) in data else "false")
        return 0

    if expr == 'keys | .[] | select(test("^MODEL_ARGS(_\\d+)?$"))':
        keys = [key for key in data.keys() if re.fullmatch(r"MODEL_ARGS(_\d+)?", key)]
        print("\n".join(keys))
        return 0

    if print_defaulted_path(data, expr):
        return 0

    if print_simple_path(data, expr):
        return 0

    print(f"unsupported expression: {expr}", file=sys.stderr)
    return 1


def handle_in_place_update(data: Any, path: str, expr: str) -> int:
    match = re.fullmatch(r'(\..+?)\s*=\s*(.+)', expr)
    if not match:
        print(f"unsupported -i expression: {expr}", file=sys.stderr)
        return 1

    set_path(data, match.group(1), parse_literal(match.group(2), data))
    write_document(path, data)
    return 0


def main() -> int:
    args = sys.argv[1:]
    if args == ["--version"]:
        print("yq-compat 0.1")
        return 0

    in_place = False
    if args and args[0] == "-i":
        in_place = True
        args = args[1:]

    if not args:
        print("usage: yq_compat.py [-i] <expression> [file]", file=sys.stderr)
        return 2

    expr = args[0]
    path = args[1] if len(args) > 1 else None
    data = load_document(path)

    if in_place:
        if path is None:
            print("in-place updates require a file path", file=sys.stderr)
            return 2
        return handle_in_place_update(data, path, expr)

    return handle_read_query(data, expr)


if __name__ == "__main__":
    sys.exit(main())
