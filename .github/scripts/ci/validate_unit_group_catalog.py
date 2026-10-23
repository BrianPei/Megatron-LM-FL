#!/usr/bin/env python3
"""Validate the logical unit-group catalog against the repository and platforms."""

from __future__ import annotations

import argparse
import pathlib
import sys

import generate_matrix
import unit_group_routing


def collect_errors(
    root: pathlib.Path,
    configs_dir: pathlib.Path,
    catalog: unit_group_routing.Catalog,
    platforms: dict[str, unit_group_routing.PlatformUnit],
    level: str,
) -> list[str]:
    errors = catalog.validate(root)
    for name, platform in platforms.items():
        for group_name in platform.groups:
            if group_name not in catalog.groups:
                errors.append(
                    f"Platform {name} executes group {group_name!r} that is absent from the catalog"
                )
        for group_name, entry in platform.groups.items():
            for token in entry["path"].split():
                if not unit_group_routing.path_exists(root, token):
                    errors.append(
                        f"Platform {name} group {group_name} path does not exist: {token}"
                    )
    for group_name in catalog.required_groups(level):
        missing = sorted(
            name for name, platform in platforms.items() if group_name not in platform.groups
        )
        if missing:
            errors.append(
                f"Required group {group_name!r} is not executable on: {', '.join(missing)}"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs-dir", default=".github/configs")
    parser.add_argument("--catalog", default=".github/configs/unit_group_routes.yml")
    parser.add_argument("--level", default="L1")
    args = parser.parse_args()

    root = pathlib.Path.cwd()
    configs_dir = root / args.configs_dir
    try:
        catalog = unit_group_routing.load_catalog(root / args.catalog)
        rules = generate_matrix._load_watch_rules(configs_dir)
        platforms = {
            name: unit_group_routing.load_platform(configs_dir / f"{name}.yml")
            for name in sorted(rules)
        }
        errors = collect_errors(root, configs_dir, catalog, platforms, args.level)
    except (unit_group_routing.CatalogError, OSError, ValueError) as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1

    executable: dict[str, list[str]] = {}
    for name, platform in platforms.items():
        for group_name in platform.groups:
            executable.setdefault(group_name, []).append(name)

    print(
        f"Unit group catalog: {len(catalog.groups)} logical groups, "
        f"{len(platforms)} platforms, required level {args.level}"
    )
    for group_name in sorted(catalog.groups):
        owners = executable.get(group_name, [])
        if owners:
            print(f"  {group_name}: {', '.join(owners)}")
        else:
            print(f"  {group_name}: not configured on any platform")

    if errors:
        for error in errors:
            print(f"::error::{error}", file=sys.stderr)
        return 1
    print("Unit group catalog validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
