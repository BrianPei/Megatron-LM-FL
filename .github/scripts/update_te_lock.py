#!/usr/bin/env python3
"""
Update TE-FL version lock in docker/versions.yaml

Usage:
    python update_te_lock.py --te-commit <commit> --platform <platform> --artifact <url>
    python update_te_lock.py --te-commit <commit> --all-platforms --artifact-pattern <pattern>
"""

import argparse
import sys
import re
import yaml
from pathlib import Path
from typing import Optional


def err(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def msg(text: str) -> None:
    print(f">>> {text}")


def validate_commit(commit: str) -> bool:
    """Validate TE-FL commit format"""
    return bool(re.match(r'^[0-9a-f]{40}$', commit))


def update_versions_yaml(
    versions_file: Path,
    te_commit: str,
    platform: Optional[str] = None,
    artifact: Optional[str] = None,
    all_platforms: bool = False,
    artifact_pattern: Optional[str] = None
) -> None:
    """Update versions.yaml with new TE-FL version"""

    if not versions_file.exists():
        err(f"versions.yaml not found: {versions_file}")

    # Load current config
    with open(versions_file) as f:
        config = yaml.safe_load(f)

    # Validate commit
    if not validate_commit(te_commit):
        err(f"Invalid TE-FL commit format: {te_commit}")

    # Update TE-FL commit
    old_commit = config.get("te_fl", {}).get("commit", "")
    config["te_fl"]["commit"] = te_commit

    msg(f"Updating TE-FL commit: {old_commit[:8] if old_commit else 'empty'} -> {te_commit[:8]}")

    # Update platform artifacts
    if all_platforms:
        if not artifact_pattern:
            err("--artifact-pattern required when using --all-platforms")

        platforms = list(config.get("platforms", {}).keys())
        msg(f"Updating all platforms: {', '.join(platforms)}")

        for plat in platforms:
            artifact_url = artifact_pattern.replace("{platform}", plat).replace("{commit}", te_commit)
            old_artifact = config["platforms"][plat].get("te_fl_artifact", "")
            config["platforms"][plat]["te_fl_artifact"] = artifact_url

            msg(f"  {plat}: {artifact_url}")

    elif platform:
        if platform not in config.get("platforms", {}):
            err(f"Platform not found in versions.yaml: {platform}")

        if not artifact:
            err("--artifact required when specifying --platform")

        old_artifact = config["platforms"][platform].get("te_fl_artifact", "")
        config["platforms"][platform]["te_fl_artifact"] = artifact

        msg(f"Updated {platform} artifact:")
        msg(f"  Old: {old_artifact if old_artifact else '<empty>'}")
        msg(f"  New: {artifact}")
    else:
        err("Either --platform or --all-platforms must be specified")

    # Write back
    with open(versions_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    msg(f"SUCCESS: Updated {versions_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Update TE-FL version lock in docker/versions.yaml"
    )

    parser.add_argument(
        "--te-commit",
        required=True,
        help="TE-FL commit SHA (40 hex chars)"
    )

    parser.add_argument(
        "--platform",
        help="Platform to update (e.g., musa, metax, ascend)"
    )

    parser.add_argument(
        "--artifact",
        help="TE-FL artifact URL or path for the specified platform"
    )

    parser.add_argument(
        "--all-platforms",
        action="store_true",
        help="Update all platforms"
    )

    parser.add_argument(
        "--artifact-pattern",
        help="Artifact URL pattern with {platform} and {commit} placeholders"
    )

    parser.add_argument(
        "--versions-file",
        default="docker/versions.yaml",
        help="Path to versions.yaml (default: docker/versions.yaml)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show changes without writing"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.platform and not args.all_platforms:
        parser.error("Either --platform or --all-platforms must be specified")

    if args.platform and args.all_platforms:
        parser.error("Cannot use both --platform and --all-platforms")

    if args.platform and not args.artifact:
        parser.error("--artifact required when using --platform")

    if args.all_platforms and not args.artifact_pattern:
        parser.error("--artifact-pattern required when using --all-platforms")

    versions_file = Path(args.versions_file)

    if args.dry_run:
        msg("DRY RUN MODE - no changes will be written")

        with open(versions_file) as f:
            config = yaml.safe_load(f)

        msg(f"Would update TE-FL commit to: {args.te_commit[:8]}")

        if args.all_platforms:
            platforms = list(config.get("platforms", {}).keys())
            for plat in platforms:
                artifact_url = args.artifact_pattern.replace("{platform}", plat).replace("{commit}", args.te_commit)
                msg(f"  {plat}: {artifact_url}")
        else:
            msg(f"  {args.platform}: {args.artifact}")

        return

    # Perform update
    update_versions_yaml(
        versions_file=versions_file,
        te_commit=args.te_commit,
        platform=args.platform,
        artifact=args.artifact,
        all_platforms=args.all_platforms,
        artifact_pattern=args.artifact_pattern
    )


if __name__ == "__main__":
    main()
