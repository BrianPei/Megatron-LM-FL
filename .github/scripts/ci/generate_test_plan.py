#!/usr/bin/env python3
"""Generate a reproducible unit-test plan for changed files and platforms.

The planner combines three inputs:

* the logical catalog (.github/configs/unit_group_routes.yml),
* the platform router (generate_matrix.py), and
* each platform's configured unit groups.

Shadow mode renders the proposed plan without changing what CI executes. The
plan is stable for identical source/base SHAs, configs, and changed files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
from typing import Any

import generate_matrix
import unit_group_routing

PLAN_SCHEMA_VERSION = 1


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode()).hexdigest()


def _git(*arguments: str) -> str:
    try:
        result = subprocess.run(["git", *arguments], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def _default_stage(event_name: str) -> str:
    if event_name == "pull_request":
        return "pr"
    if event_name == "schedule":
        return "nightly"
    return "manual"


def _default_selection(event_name: str) -> str:
    return "affected" if event_name == "pull_request" else "all"


def _parse_platforms(value: str, known: set[str]) -> list[str]:
    try:
        platforms = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"--platforms must be a JSON array: {error}") from error
    if not isinstance(platforms, list) or any(
        not isinstance(platform, str) or not platform for platform in platforms
    ):
        raise ValueError("--platforms must be a JSON array of platform names")
    unknown = sorted(set(platforms) - known)
    if unknown:
        raise ValueError(f"Unknown platforms: {', '.join(unknown)}")
    return sorted(set(platforms))


def build_plan(
    *,
    event_name: str,
    stage: str,
    selection: str,
    level: str,
    changed: list[str],
    unknown_diff: bool,
    platforms: list[str],
    catalog: unit_group_routing.Catalog,
    platform_units: dict[str, unit_group_routing.PlatformUnit],
    source_sha: str,
    base_sha: str,
) -> dict[str, Any]:
    """Build the complete plan document, including its stable hash."""
    group_selection = unit_group_routing.select_groups(
        catalog, changed, selection=selection, level=level, unknown_diff=unknown_diff
    )
    required = group_selection["required"]

    platform_plans = []
    for name in platforms:
        platform = platform_units[name]
        materialized = unit_group_routing.materialize_groups(
            catalog, platform, group_selection["selected"], group_selection["reasons"], required
        )
        platform_plans.append(
            {
                "name": name,
                "devices": list(platform.device_types),
                "nproc_per_node": platform.nproc_per_node,
                "configured_groups": sorted(platform.groups),
                **materialized,
            }
        )

    config_hash = _sha256(
        {
            "catalog": catalog.data,
            "platforms": {name: platform_units[name].data for name in sorted(platforms)},
        }
    )
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "event": event_name,
        "stage": stage,
        "selection": selection,
        "level": level,
        "source_sha": source_sha,
        "base_sha": base_sha,
        "changed_files": sorted(set(changed)),
        "catalog_hash": f"sha256:{catalog.digest}",
        "config_hash": config_hash,
        "global_groups": {
            "selected": sorted(group_selection["selected"]),
            "required": sorted(required),
            "fallback": group_selection["fallback"],
            "fallback_reason": group_selection["fallback_reason"],
            "unmapped_paths": group_selection["unmapped_paths"],
            "documentation_only": group_selection["documentation_only"],
        },
        "platforms": platform_plans,
    }
    plan["plan_hash"] = _sha256(plan)
    return plan


def _escape(value: str) -> str:
    return value.replace("<", "&lt;").replace(">", "&gt;").replace("|", "\\|")


def write_summary(path: pathlib.Path, plan: dict[str, Any]) -> None:
    """Append a human-readable shadow plan to the Actions summary."""
    global_groups = plan["global_groups"]
    lines = [
        "### Unit test plan (shadow)",
        "",
        f"- Stage: `{plan['stage']}`",
        f"- Selection: `{plan['selection']}`",
        f"- Level: `{plan['level']}`",
        f"- Changed files: {len(plan['changed_files'])}",
        f"- Fallback: {'yes' if global_groups['fallback'] else 'no'}"
        + (
            f" ({_escape(global_groups['fallback_reason'])})"
            if global_groups["fallback_reason"]
            else ""
        ),
        f"- Plan hash: `{plan['plan_hash']}`",
        "",
    ]
    if global_groups["documentation_only"]:
        lines.extend(["Documentation-only change: no hardware tests proposed.", ""])
    if global_groups["unmapped_paths"]:
        lines.extend(
            [
                "**Unmapped paths (need catalog onboarding):**",
                "",
                *[f"- `{_escape(path)}`" for path in global_groups["unmapped_paths"]],
                "",
            ]
        )
    if plan["platforms"]:
        lines.extend(
            [
                "| Platform | Selected | Configured | Not configured | Blocked | Potential reduction |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for platform in plan["platforms"]:
            selected = (
                ", ".join(
                    group["name"] + (" (required)" if group["required"] else "")
                    for group in platform["selected_groups"]
                )
                or "none"
            )
            not_configured = (
                ", ".join(group["name"] for group in platform["not_configured"]) or "none"
            )
            blocked = ", ".join(group["name"] for group in platform["blocked"]) or "none"
            reduction = len(platform["configured_groups"]) - len(platform["selected_groups"])
            lines.append(
                f"| {platform['name']} | {_escape(selected)} | "
                f"{len(platform['configured_groups'])} | {_escape(not_configured)} | "
                f"{_escape(blocked)} | {reduction} groups |"
            )
        lines.extend(["", "Selected group reasons:", ""])
        for platform in plan["platforms"]:
            for group in platform["selected_groups"]:
                reasons = "; ".join(_reason_text(reason) for reason in group["reasons"])
                lines.append(f"- `{platform['name']}/{group['name']}`: {reasons}")
        lines.append("")
    with path.open("a") as stream:
        stream.write("\n".join(lines) + "\n")


def _reason_text(reason: dict[str, str]) -> str:
    if reason["type"] == "direct":
        return f"direct match `{_escape(reason['path'])}`"
    if reason["type"] == "dependency":
        source = reason.get("source_group")
        return f"dependency of `{source}`" if source else "dependency"
    if reason["type"] == "required":
        return f"required for `{reason['level']}`"
    if reason["type"] == "fallback":
        return f"fallback: {_escape(reason['reason'])}"
    return reason["type"]


def _changed_files(args: argparse.Namespace) -> tuple[list[str], bool]:
    if args.changed_files is not None:
        return [path for path in args.changed_files.splitlines() if path], False
    if args.event_name != "pull_request":
        return [], False
    try:
        return generate_matrix._changed_files(args.base_ref), False
    except subprocess.CalledProcessError as error:
        print(f"::warning::Cannot read diff; proposing all groups: {error}", file=sys.stderr)
        return [], True


def _source_sha(value: str | None) -> str:
    return value or os.environ.get("GITHUB_SHA", "") or _git("rev-parse", "HEAD")


def _base_sha(value: str | None, base_ref: str) -> str:
    return (
        value
        or os.environ.get("GITHUB_BASE_SHA", "")
        or _git("merge-base", f"origin/{base_ref}", "HEAD")
        or _git("rev-parse", f"origin/{base_ref}")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--base-ref", default="main")
    parser.add_argument("--configs-dir", default=".github/configs")
    parser.add_argument("--catalog", default=".github/configs/unit_group_routes.yml")
    parser.add_argument("--changed-files", default=None, help="Newline-separated paths for testing")
    parser.add_argument("--platforms", default=None, help="JSON array of selected platforms")
    parser.add_argument("--selection", choices=("affected", "all"), default=None)
    parser.add_argument("--stage", choices=("pr", "nightly", "manual"), default=None)
    parser.add_argument("--level", default="L1")
    parser.add_argument("--output", default="unit-test-plan.json")
    parser.add_argument("--summary", default=None)
    parser.add_argument("--shadow", action="store_true")
    parser.add_argument("--source-sha", default=None)
    parser.add_argument("--base-sha", default=None)
    args = parser.parse_args()

    root = pathlib.Path.cwd()
    configs_dir = root / args.configs_dir
    stage = args.stage or _default_stage(args.event_name)
    selection = args.selection or _default_selection(args.event_name)

    try:
        catalog = unit_group_routing.load_catalog(root / args.catalog)
        rules = generate_matrix._load_watch_rules(configs_dir)
        changed, unknown_diff = _changed_files(args)
        if selection == "all":
            changed = []
            unknown_diff = False
        if args.platforms is not None:
            platforms = _parse_platforms(args.platforms, set(rules))
        else:
            labels = generate_matrix.event_labels() if args.event_name == "pull_request" else []
            platforms, _ = generate_matrix.select_platforms(rules, changed, args.event_name, labels)
        platform_units = {
            name: unit_group_routing.load_platform(configs_dir / f"{name}.yml")
            for name in platforms
        }
        plan = build_plan(
            event_name=args.event_name,
            stage=stage,
            selection=selection,
            level=args.level,
            changed=changed,
            unknown_diff=unknown_diff,
            platforms=platforms,
            catalog=catalog,
            platform_units=platform_units,
            source_sha=_source_sha(args.source_sha),
            base_sha=_base_sha(args.base_sha, args.base_ref),
        )
    except (unit_group_routing.CatalogError, OSError, ValueError) as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1

    output = root / args.output
    output.write_text(json.dumps(plan, indent=2, sort_keys=False) + "\n")
    if args.summary:
        write_summary(root / args.summary, plan)

    blocked = [
        (platform["name"], group["name"])
        for platform in plan["platforms"]
        for group in platform["blocked"]
    ]
    unmapped = plan["global_groups"]["unmapped_paths"]
    if blocked:
        message = ", ".join(f"{platform}/{group}" for platform, group in blocked)
        if args.shadow:
            print(f"::warning::Required groups lack execution mapping: {message}")
        else:
            print(f"::error::Required groups lack execution mapping: {message}", file=sys.stderr)
            return 1
    if unmapped and not args.shadow:
        print("::error::Unresolved routes: " + ", ".join(unmapped), file=sys.stderr)
        return 1

    print(f"plan_hash={plan['plan_hash']}")
    print(
        "platforms="
        + json.dumps(
            {
                platform["name"]: [group["name"] for group in platform["selected_groups"]]
                for platform in plan["platforms"]
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
