#!/usr/bin/env python3
"""Check a prepared TE wheel without importing or installing its runtime."""

import argparse
import re
import subprocess
import sys
import zipfile
from email.parser import Parser
from pathlib import Path


def validate_wheel(wheel_dir: Path, source_dir: Path, expected_commit: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        raise ValueError("Expected a full TE-FL source commit")
    actual_commit = subprocess.check_output(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != expected_commit:
        raise ValueError("TE-FL checkout changed after resolving the cache key")
    short_commit = subprocess.check_output(
        ["git", "-C", str(source_dir), "rev-parse", "--short", "HEAD"], text=True
    ).strip()
    base_version = (source_dir / "build_tools" / "VERSION.txt").read_text().strip()
    expected_version = f"{base_version}+{short_commit}"

    wheels = sorted(wheel_dir.glob("transformer_engine*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"Expected exactly one prepared TE-FL wheel, found {len(wheels)}")
    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as archive:
        metadata_files = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_files) != 1:
            raise ValueError("Expected exactly one wheel distribution metadata file")
        metadata = Parser().parsestr(archive.read(metadata_files[0]).decode("utf-8"))
    names = metadata.get_all("Name", [])
    versions = metadata.get_all("Version", [])
    if len(names) != 1 or re.sub(r"[-_.]+", "-", names[0]).lower() != "transformer-engine":
        raise ValueError("Prepared wheel is not the transformer-engine distribution")
    if versions != [expected_version]:
        raise ValueError(
            f"TE-FL wheel version does not match checkout {expected_commit}: "
            f"expected {expected_version}, found {versions} in {wheel.name}"
        )
    if not wheel.name.startswith(f"transformer_engine-{expected_version}-"):
        raise ValueError("TE-FL wheel filename and distribution version do not agree")
    expected_metadata = f"transformer_engine-{expected_version}.dist-info/METADATA"
    if metadata_files != [expected_metadata]:
        raise ValueError("TE-FL wheel metadata directory does not match its version")
    return wheel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-dir", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    try:
        wheel = validate_wheel(args.wheel_dir, args.source_dir, args.expected_commit)
    except (OSError, ValueError, subprocess.CalledProcessError, zipfile.BadZipFile) as exc:
        print(f"::error::TE-FL wheel validation failed: {exc}", file=sys.stderr)
        return 1
    print(f"Prepared TE-FL wheel matches source {args.expected_commit}: {wheel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
