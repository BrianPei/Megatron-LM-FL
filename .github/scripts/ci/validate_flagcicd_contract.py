#!/usr/bin/env python3
"""Validate the FlagCICD project contract and its migration catalog."""

from __future__ import annotations

import argparse
import pathlib
import sys

import flagcicd_contract
import unit_group_routing
import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=".flagcicd/project.yaml")
    parser.add_argument("--catalog", default=".github/configs/unit_group_routes.yml")
    args = parser.parse_args()
    root = pathlib.Path.cwd()
    try:
        project, digest = flagcicd_contract.load_project(root / args.project)
        legacy = yaml.safe_load((root / args.catalog).read_text())
        if not isinstance(legacy, dict):
            raise flagcicd_contract.ContractError("legacy catalog must be a mapping")
        errors = flagcicd_contract.contract_catalog_errors(project, legacy)
        catalog = unit_group_routing.load_catalog(root / args.catalog)
        errors.extend(catalog.validate(root))
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1
    print(f"FlagCICD project contract: {project['repository']}")
    print(f"Project hash: sha256:{digest}")
    print(f"Stages: {', '.join(project['stages'])}")
    print(f"Levels: {', '.join(project['levels'])}")
    print(f"Logical groups: {len(project['test_groups'])}")
    if errors:
        for error in errors:
            print(f"::error::{error}", file=sys.stderr)
        return 1
    print("FlagCICD project contract validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
