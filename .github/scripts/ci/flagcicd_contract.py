#!/usr/bin/env python3
"""Load and validate the repository-facing FlagCICD contract.

The project contract owns stage, level, resource, and logical-group semantics.
Platform YAML remains the adapter for images, runners, setup, and concrete
execution commands.  This module intentionally keeps the legacy catalog
conversion small so the migration can be checked before the planner switches
to the project file as its only editable source.
"""

from __future__ import annotations

import copy
import hashlib
import pathlib
from typing import Any

import yaml

PROJECT_SCHEMA = "flagcicd.project/v1"
PLAN_SCHEMA = 1


class ContractError(ValueError):
    """Raised when the project contract is malformed or inconsistent."""


def _read(path: pathlib.Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        value = yaml.safe_load(raw)
    except (OSError, yaml.YAMLError) as error:
        raise ContractError(f"Cannot read FlagCICD project contract {path}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"FlagCICD project contract must be a mapping: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def load_project(path: pathlib.Path) -> tuple[dict[str, Any], str]:
    """Load the project contract and perform schema-level validation."""
    project, digest = _read(path)
    if project.get("schema_version") != PROJECT_SCHEMA:
        raise ContractError(
            f"Unsupported project contract schema: {project.get('schema_version')!r}"
        )
    if not isinstance(project.get("repository"), str) or not project["repository"].strip():
        raise ContractError("repository must be a non-empty string")
    for key in ("platform_config_glob", "catalog_source"):
        if not isinstance(project.get(key), str) or not project[key].strip():
            raise ContractError(f"{key} must be a non-empty string")
    if project.get("plan_schema") != PLAN_SCHEMA:
        raise ContractError(f"plan_schema must be {PLAN_SCHEMA!r}")
    for key in ("stages", "levels", "resource_classes", "test_groups", "coverage", "gate_policy"):
        if key not in project:
            raise ContractError(f"Project contract is missing {key}")
    if not isinstance(project["stages"], dict) or not project["stages"]:
        raise ContractError("stages must be a non-empty mapping")
    if not isinstance(project["levels"], dict) or not project["levels"]:
        raise ContractError("levels must be a non-empty mapping")
    if not isinstance(project["resource_classes"], dict):
        raise ContractError("resource_classes must be a mapping")
    for key in ("global", "coverage"):
        if not isinstance(project.get(key), dict):
            raise ContractError(f"{key} must be a mapping")
    for name, resource in project["resource_classes"].items():
        if not isinstance(resource, dict):
            raise ContractError(f"resource class {name} must be a mapping")
        if not isinstance(resource.get("exclusive"), bool):
            raise ContractError(f"resource class {name} exclusive must be a boolean")
        devices = resource.get("devices")
        if (
            not (isinstance(devices, int) and not isinstance(devices, bool) and devices >= 0)
            and devices != "nproc_per_node"
        ):
            raise ContractError(
                f"resource class {name} devices must be a non-negative integer " "or nproc_per_node"
            )
    groups = project["test_groups"]
    if not isinstance(groups, list) or not groups:
        raise ContractError("test_groups must be a non-empty list")
    names: set[str] = set()
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("name"), str):
            raise ContractError("every test group must have a string name")
        name = group["name"]
        if name in names:
            raise ContractError(f"duplicate test group: {name}")
        names.add(name)
        status = group.get("status", "active")
        if status not in {"active", "planned", "unsupported"}:
            raise ContractError(f"invalid status for test group {name}: {status!r}")
        if status != "active" and not str(group.get("reason", "")).strip():
            raise ContractError(f"test group {name} needs a reason for status {status}")
        if not isinstance(group.get("paths"), list) or not group["paths"]:
            raise ContractError(f"test group {name} must declare paths")
        resource_class = group.get("resource_class")
        if resource_class not in project["resource_classes"]:
            raise ContractError(
                f"test group {name} references unknown resource class {resource_class!r}"
            )
        if not isinstance(group.get("depends_on", []), list):
            raise ContractError(f"test group {name} depends_on must be a list")
        for dependency in group.get("depends_on", []):
            if dependency not in names and dependency not in {item.get("name") for item in groups}:
                raise ContractError(f"test group {name} depends on unknown group {dependency}")
    for level, config in project["levels"].items():
        if not isinstance(config, dict):
            raise ContractError(f"level {level} must be a mapping")
        status = config.get("status", "active")
        if status not in {"active", "planned"}:
            raise ContractError(f"level {level} has invalid status {status!r}")
        required = config.get("required_groups", [])
        if not isinstance(required, list) or any(item not in names for item in required):
            raise ContractError(f"level {level} has an unknown required group")
    for stage, config in project["stages"].items():
        if not isinstance(config, dict):
            raise ContractError(f"stage {stage} must be a mapping")
        status = config.get("status", "active")
        if status not in {"active", "planned"}:
            raise ContractError(f"stage {stage} has invalid status {status!r}")
        selection = config.get("selection")
        if selection not in {"affected", "configured", "requested", "all"}:
            raise ContractError(f"stage {stage} has invalid selection {selection!r}")
        level = config.get("default_level")
        if level not in project["levels"]:
            raise ContractError(f"stage {stage} references unknown level {level!r}")
        level_status = project["levels"][level].get("status", "active")
        if status == "active" and level_status == "planned":
            raise ContractError(f"active stage {stage} references planned level {level}")
        if "fixed_level" in config and not isinstance(config["fixed_level"], bool):
            raise ContractError(f"stage {stage} fixed_level must be a boolean")
        if status == "planned" and config.get("fixed_level", False) is not True:
            raise ContractError(f"planned stage {stage} must declare fixed_level: true")
        if selection == "configured" and not config.get("full_configured_matrix", False):
            raise ContractError(
                f"stage {stage} uses configured selection without full_configured_matrix"
            )
    if not isinstance(project["gate_policy"], dict):
        raise ContractError("gate_policy must be a mapping")
    for stage in project["stages"]:
        policy = project["gate_policy"].get(stage)
        if not isinstance(policy, dict):
            raise ContractError(f"gate_policy is missing stage {stage}")
        statuses = policy.get("required_plan_status")
        if (
            not isinstance(statuses, list)
            or not statuses
            or any(status not in {"ready", "partial"} for status in statuses)
        ):
            raise ContractError(
                f"gate_policy for {stage} has invalid required_plan_status; "
                "blocked plans cannot be accepted"
            )
        if not isinstance(policy.get("allow_catalog_partial"), bool):
            raise ContractError(f"gate_policy for {stage} allow_catalog_partial must be a boolean")
    return project, digest


def project_to_catalog(project: dict[str, Any]) -> dict[str, Any]:
    """Convert the project contract to the legacy catalog shape during migration."""
    groups: dict[str, dict[str, Any]] = {}
    for entry in project["test_groups"]:
        group = copy.deepcopy(entry)
        group.pop("name", None)
        group.pop("kind", None)
        group.pop("evidence", None)
        groups[entry["name"]] = group
    return {
        "schema_version": 1,
        "global": copy.deepcopy(project.get("global", {})),
        "coverage": copy.deepcopy(project.get("coverage", {})),
        "levels": copy.deepcopy(project["levels"]),
        "groups": groups,
    }


def contract_catalog_errors(project: dict[str, Any], legacy_catalog: dict[str, Any]) -> list[str]:
    """Return semantic differences between the new contract and legacy catalog."""
    expected = project_to_catalog(project)
    errors: list[str] = []
    if legacy_catalog.get("schema_version") != 1:
        errors.append("legacy catalog has an unsupported schema version")
    for section in ("global", "coverage"):
        if expected.get(section) != legacy_catalog.get(section):
            errors.append(f"contract section differs from legacy catalog: {section}")
    legacy_levels = legacy_catalog.get("levels", {})
    project_levels = project.get("levels", {})
    unexpected_legacy_levels = set(legacy_levels) - set(project_levels)
    if unexpected_legacy_levels:
        errors.append(
            "legacy catalog declares unknown levels: " + ", ".join(sorted(unexpected_legacy_levels))
        )
    for name in sorted(set(project_levels) & set(legacy_levels)):
        project_level = project_levels[name]
        legacy_level = legacy_levels[name]
        if project_level.get("required_groups", []) != legacy_level.get("required_groups", []):
            errors.append(f"contract level differs from legacy catalog: {name}")
    missing_active_levels = {
        name
        for name, level in project_levels.items()
        if level.get("status", "active") == "active" and name not in legacy_levels
    }
    if missing_active_levels:
        errors.append(
            "legacy catalog is missing active project levels: "
            + ", ".join(sorted(missing_active_levels))
        )
    expected_groups = expected["groups"]
    actual_groups = legacy_catalog.get("groups", {})
    if set(expected_groups) != set(actual_groups):
        errors.append("contract and legacy catalog declare different group names")
    for name in sorted(set(expected_groups) & set(actual_groups)):
        if expected_groups[name] != actual_groups[name]:
            errors.append(f"contract group differs from legacy catalog: {name}")
    return errors
