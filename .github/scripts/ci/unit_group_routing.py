#!/usr/bin/env python3
"""Logical unit-test group routing shared by the planner and its validator.

The catalog decides which logical groups a change affects. Platform configs
decide which groups a platform can actually execute. This module keeps those
two layers separate: a logical group is materialized only when the platform
declares an execution group with the same name.
"""

from __future__ import annotations

import fnmatch
import functools
import hashlib
import pathlib
import re
import subprocess
from dataclasses import dataclass
from typing import Any, Iterable

import yaml

SCHEMA_VERSION = 1
GROUP_NAME = re.compile(r"[A-Za-z0-9_-]+")


class CatalogError(ValueError):
    """Raised when the catalog or a platform config cannot be loaded."""


@dataclass(frozen=True)
class Group:
    """A logical unit-test group and its routing metadata."""

    name: str
    description: str
    paths: tuple[str, ...]
    depends_on: tuple[str, ...]
    resource_class: str
    timeout_minutes: int
    parallel_safe: bool


@dataclass(frozen=True)
class Level:
    """Required groups for one routing level."""

    name: str
    description: str
    required_groups: tuple[str, ...]


@dataclass(frozen=True)
class Catalog:
    """Parsed unit-group catalog."""

    path: pathlib.Path
    data: dict[str, Any]
    groups: dict[str, Group]
    levels: dict[str, Level]
    documentation_paths: tuple[str, ...]
    documentation_suffixes: tuple[str, ...]
    shared_paths: tuple[str, ...]
    default_timeout_minutes: int
    digest: str

    def matches(self, path: str) -> str | None:
        """Return the most specific logical group for a repository path."""
        matches = [
            (len(pattern), group.name)
            for group in self.groups.values()
            for pattern in group.paths
            if _matches(path, pattern)
        ]
        if not matches:
            return None
        # Longest pattern wins, so a specific directory beats a broader one.
        return max(matches)[1]

    def is_documentation(self, path: str) -> bool:
        if path.startswith(self.documentation_paths):
            return True
        return pathlib.PurePosixPath(path).suffix in self.documentation_suffixes

    def is_shared(self, path: str) -> bool:
        return any(_matches(path, pattern) for pattern in self.shared_paths)

    def closure(self, names: Iterable[str]) -> set[str]:
        """Return the dependency closure of the given logical groups."""
        result: set[str] = set()
        stack = list(names)
        while stack:
            name = stack.pop()
            if name in result:
                continue
            result.add(name)
            stack.extend(self.groups[name].depends_on)
        return result

    def required_groups(self, level: str) -> tuple[str, ...]:
        if level not in self.levels:
            raise CatalogError(f"Unknown routing level: {level}")
        return self.levels[level].required_groups

    def validate(self, root: pathlib.Path | None = None) -> list[str]:
        """Return catalog errors; an empty list means the catalog is usable."""
        errors: list[str] = []
        if not self.groups:
            errors.append("Catalog declares no groups")
        for name, group in self.groups.items():
            if not GROUP_NAME.fullmatch(name):
                errors.append(f"Invalid group name: {name!r}")
            if not group.paths:
                errors.append(f"Group {name} declares no paths")
            for pattern in group.paths:
                if not pattern:
                    errors.append(f"Group {name} declares an empty path")
                elif root is not None and not path_exists(root, pattern):
                    errors.append(f"Group {name} path does not exist: {pattern}")
            for dependency in group.depends_on:
                if dependency not in self.groups:
                    errors.append(f"Group {name} depends on unknown group: {dependency}")
                if dependency == name:
                    errors.append(f"Group {name} depends on itself")
        errors.extend(self._cycle_errors())
        for level in self.levels.values():
            for name in level.required_groups:
                if name not in self.groups:
                    errors.append(f"Level {level.name} requires unknown group: {name}")
        if root is not None:
            unit_tests = root / "tests/unit_tests"
            if unit_tests.is_dir():
                for directory in sorted(unit_tests.iterdir()):
                    if not directory.is_dir() or directory.name == "__pycache__":
                        continue
                    probe = f"tests/unit_tests/{directory.name}/"
                    if self.matches(probe) is None:
                        errors.append(f"Unit-test directory has no logical group: {probe}")
        return errors

    def _cycle_errors(self) -> list[str]:
        errors: list[str] = []
        visiting: set[str] = set()
        done: set[str] = set()

        def visit(name: str, stack: list[str]) -> None:
            if name in done:
                return
            if name in visiting:
                cycle = " -> ".join(stack + [name])
                errors.append(f"Dependency cycle: {cycle}")
                return
            visiting.add(name)
            for dependency in self.groups[name].depends_on:
                if dependency in self.groups:
                    visit(dependency, stack + [name])
            visiting.remove(name)
            done.add(name)

        for name in sorted(self.groups):
            visit(name, [])
        return errors


@dataclass(frozen=True)
class PlatformUnit:
    """Unit-test execution capabilities of one platform config."""

    name: str
    path: pathlib.Path
    data: dict[str, Any]
    device_types: tuple[str, ...]
    nproc_per_node: int
    groups: dict[str, dict[str, Any]]


def _matches(path: str, pattern: str) -> bool:
    if pattern.endswith("/"):
        return path.startswith(pattern)
    if any(character in pattern for character in "*?["):
        # Keep globs inside one directory: tests/unit_tests/test_*.py must not
        # match tests/unit_tests/some_directory/test_file.py.
        return fnmatch.fnmatchcase(path, pattern) and path.count("/") == pattern.count("/")
    return path == pattern


@functools.lru_cache(maxsize=8)
def _tracked_paths(root: pathlib.Path) -> tuple[str, ...] | None:
    """Return repository-relative paths tracked by git, or None outside git."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return tuple(path for path in result.stdout.split("\0") if path)


def path_exists(root: pathlib.Path, pattern: str) -> bool:
    # CI validates a clean checkout. Checking the git index here keeps local
    # validation honest when ignored or generated directories are present.
    tracked = _tracked_paths(root)
    if tracked is not None:
        if pattern.endswith("/"):
            return any(path.startswith(pattern) for path in tracked)
        if any(character in pattern for character in "*?["):
            return any(_matches(path, pattern) for path in tracked)
        return pattern in tracked
    if any(character in pattern for character in "*?["):
        return any(root.glob(pattern))
    target = root / pattern
    return target.is_dir() if pattern.endswith("/") else target.exists()


def load_catalog(path: pathlib.Path) -> Catalog:
    """Load and schema-check a catalog file."""
    try:
        raw = path.read_bytes()
        data = yaml.safe_load(raw)
    except (OSError, yaml.YAMLError) as error:
        raise CatalogError(f"Cannot read unit group catalog {path}: {error}") from error
    if not isinstance(data, dict):
        raise CatalogError(f"Unit group catalog must be a mapping: {path}")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise CatalogError(f"Unsupported unit group catalog schema: {data.get('schema_version')!r}")

    global_config = data.get("global") or {}
    if not isinstance(global_config, dict):
        raise CatalogError("global must be a mapping")
    documentation_paths = _string_tuple(global_config, "documentation_paths")
    documentation_suffixes = _string_tuple(global_config, "documentation_suffixes")
    shared_paths = _string_tuple(global_config, "shared_paths")
    default_timeout = global_config.get("default_timeout_minutes", 60)
    if (
        not isinstance(default_timeout, int)
        or isinstance(default_timeout, bool)
        or default_timeout <= 0
    ):
        raise CatalogError("global.default_timeout_minutes must be a positive integer")

    groups: dict[str, Group] = {}
    for name, entry in _mapping(data.get("groups"), "groups").items():
        if not isinstance(name, str) or not GROUP_NAME.fullmatch(name):
            raise CatalogError(f"Invalid group name: {name!r}")
        if not isinstance(entry, dict):
            raise CatalogError(f"Group {name} must be a mapping")
        paths = _string_tuple(entry, "paths", required=True)
        depends_on = _string_tuple(entry, "depends_on")
        resource_class = entry.get("resource_class", "single_device")
        if not isinstance(resource_class, str) or not resource_class:
            raise CatalogError(f"Group {name} has an invalid resource_class")
        timeout = entry.get("timeout_minutes", default_timeout)
        if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout <= 0:
            raise CatalogError(f"Group {name} has an invalid timeout_minutes")
        parallel_safe = entry.get("parallel_safe", False)
        if not isinstance(parallel_safe, bool):
            raise CatalogError(f"Group {name} has an invalid parallel_safe value")
        groups[name] = Group(
            name=name,
            description=str(entry.get("description", "")),
            paths=paths,
            depends_on=depends_on,
            resource_class=resource_class,
            timeout_minutes=timeout,
            parallel_safe=parallel_safe,
        )

    levels: dict[str, Level] = {}
    for name, entry in _mapping(data.get("levels"), "levels").items():
        if not isinstance(name, str) or not name:
            raise CatalogError(f"Invalid level name: {name!r}")
        if not isinstance(entry, dict):
            raise CatalogError(f"Level {name} must be a mapping")
        levels[name] = Level(
            name=name,
            description=str(entry.get("description", "")),
            required_groups=_string_tuple(entry, "required_groups"),
        )

    catalog = Catalog(
        path=path,
        data=data,
        groups=groups,
        levels=levels,
        documentation_paths=documentation_paths,
        documentation_suffixes=documentation_suffixes,
        shared_paths=shared_paths,
        default_timeout_minutes=default_timeout,
        digest=hashlib.sha256(raw).hexdigest(),
    )
    errors = catalog.validate()
    if errors:
        raise CatalogError("; ".join(errors))
    return catalog


def load_platform(config_path: pathlib.Path) -> PlatformUnit:
    """Load the unit-test section of a platform configuration."""
    try:
        data = yaml.safe_load(config_path.read_text())
    except (OSError, yaml.YAMLError) as error:
        raise CatalogError(f"Cannot read platform config {config_path}: {error}") from error
    if not isinstance(data, dict):
        raise CatalogError(f"Platform config must be a mapping: {config_path}")

    unit = (data.get("test_matrix") or {}).get("unit") or {}
    if not isinstance(unit, dict):
        raise CatalogError(f"test_matrix.unit must be a mapping: {config_path}")
    nproc = unit.get("nproc_per_node")
    if not isinstance(nproc, int) or isinstance(nproc, bool) or nproc <= 0:
        raise CatalogError(
            f"test_matrix.unit.nproc_per_node must be a positive integer: {config_path}"
        )

    groups: dict[str, dict[str, Any]] = {}
    entries = unit.get("groups") or []
    if not isinstance(entries, list):
        raise CatalogError(f"test_matrix.unit.groups must be a list: {config_path}")
    for entry in entries:
        if not isinstance(entry, dict):
            raise CatalogError(f"Each unit group must be a mapping: {config_path}")
        name = entry.get("name")
        if not isinstance(name, str) or not GROUP_NAME.fullmatch(name):
            raise CatalogError(f"Invalid unit group name {name!r}: {config_path}")
        if name in groups:
            raise CatalogError(f"Duplicate unit group {name}: {config_path}")
        if not isinstance(entry.get("path"), str) or not entry["path"].strip():
            raise CatalogError(f"Unit group {name} has no test path: {config_path}")
        groups[name] = entry

    device_types = data.get("device_types") or []
    if not isinstance(device_types, list) or any(
        not isinstance(item, str) for item in device_types
    ):
        raise CatalogError(f"device_types must be a string list: {config_path}")

    return PlatformUnit(
        name=config_path.stem,
        path=config_path,
        data=data,
        device_types=tuple(device_types),
        nproc_per_node=nproc,
        groups=groups,
    )


def select_groups(
    catalog: Catalog, changed: list[str], *, selection: str, level: str, unknown_diff: bool = False
) -> dict[str, Any]:
    """Select logical groups and explain every selection.

    Returns selected names, per-group reasons, fallback metadata, and the paths
    that forced a fallback. The caller materializes the result per platform.
    """
    all_groups = set(catalog.groups)
    required = set(catalog.required_groups(level))

    def everything(reason: str, *, fallback: bool, unmapped: list[str] | None = None):
        return {
            "selected": all_groups,
            "reasons": {name: [{"type": "fallback", "reason": reason}] for name in all_groups},
            "required": required,
            "fallback": fallback,
            "fallback_reason": reason if fallback else None,
            "unmapped_paths": unmapped or [],
            "documentation_only": False,
        }

    if selection == "all":
        return {
            "selected": all_groups,
            "reasons": {name: [{"type": "all", "reason": selection}] for name in all_groups},
            "required": required,
            "fallback": False,
            "fallback_reason": None,
            "unmapped_paths": [],
            "documentation_only": False,
        }
    if selection != "affected":
        raise CatalogError(f"Unknown selection: {selection}")
    if unknown_diff:
        return everything("Cannot read changed files", fallback=True)
    if not changed:
        return everything("No changed-file information", fallback=True)

    code_paths = [path for path in changed if not catalog.is_documentation(path)]
    if not code_paths:
        return {
            "selected": set(),
            "reasons": {},
            "required": set(),
            "fallback": False,
            "fallback_reason": None,
            "unmapped_paths": [],
            "documentation_only": True,
        }

    direct: dict[str, list[str]] = {}
    for path in code_paths:
        if catalog.is_shared(path):
            return everything(f"Shared path: {path}", fallback=True)
        name = catalog.matches(path)
        if name is None:
            return everything(f"Unmapped path: {path}", fallback=True, unmapped=[path])
        direct.setdefault(name, []).append(path)

    selected = catalog.closure(direct) | required
    reasons: dict[str, list[dict[str, str]]] = {}
    for name, paths in direct.items():
        reasons.setdefault(name, []).extend(
            {"type": "direct", "path": path} for path in sorted(paths)
        )
    for name in selected:
        if name in direct:
            continue
        sources = sorted(source for source in direct if name in catalog.closure([source]))
        if sources:
            reasons[name] = [{"type": "dependency", "source_group": source} for source in sources]
        elif name in required:
            reasons[name] = [{"type": "required", "level": level}]
        else:
            reasons[name] = [{"type": "dependency"}]
    for name in required:
        if not any(reason["type"] == "required" for reason in reasons.setdefault(name, [])):
            reasons[name].append({"type": "required", "level": level})

    return {
        "selected": selected,
        "reasons": {name: _dedupe(reasons[name]) for name in selected},
        "required": required,
        "fallback": False,
        "fallback_reason": None,
        "unmapped_paths": [],
        "documentation_only": False,
    }


def materialize_groups(
    catalog: Catalog,
    platform: PlatformUnit,
    selected: set[str],
    reasons: dict[str, list[dict[str, str]]],
    required: set[str],
) -> dict[str, list[dict[str, Any]]]:
    """Split logical groups into executable, not-configured, and blocked groups."""
    executable: list[dict[str, Any]] = []
    not_configured: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for name in sorted(selected):
        group = catalog.groups[name]
        entry = {
            "name": name,
            "execution_group": name if name in platform.groups else None,
            "resource_class": group.resource_class,
            "required": name in required,
            "timeout_minutes": group.timeout_minutes,
            "parallel_safe": group.parallel_safe,
            "reasons": reasons[name],
        }
        if name in platform.groups:
            executable.append(entry)
        elif name in required:
            blocked.append(entry)
        else:
            not_configured.append(entry)
    return {"selected_groups": executable, "not_configured": not_configured, "blocked": blocked}


def _dedupe(reasons: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    result = []
    for reason in reasons:
        key = tuple(sorted(reason.items()))
        if key not in seen:
            seen.add(key)
            result.append(reason)
    return result


def _mapping(value: Any, label: str) -> dict[Any, Any]:
    if not isinstance(value, dict) or not value:
        raise CatalogError(f"{label} must be a non-empty mapping")
    return value


def _string_tuple(entry: dict[str, Any], key: str, *, required: bool = False) -> tuple[str, ...]:
    value = entry.get(key, [])
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise CatalogError(f"{key} must be a list of non-empty strings")
    if required and not value:
        raise CatalogError(f"{key} must not be empty")
    return tuple(value)
