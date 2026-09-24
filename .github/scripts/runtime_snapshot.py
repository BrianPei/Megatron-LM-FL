#!/usr/bin/env python3
"""Prepare unit dependency wheels once, then reuse them in the same CI run."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urldefrag

SCRIPT_DIR = Path(__file__).resolve().parent
INSTALL_FLAGS = {"--ignore-requires-python", "--break-system-packages", "--force-reinstall"}


def run_pip(arguments: list) -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "--disable-pip-version-check", "--no-input", *arguments],
        check=True,
        timeout=900,
    )


def wheel_source(item: dict) -> str:
    download = item["download_info"]
    url, _ = urldefrag(download["url"])
    if "vcs_info" in download:
        vcs = download["vcs_info"]
        url = f"{vcs['vcs']}+{url}@{vcs['commit_id']}"
        if download.get("subdirectory"):
            url += "#subdirectory=" + download["subdirectory"]
    else:
        digest = download.get("archive_info", {}).get("hashes", {}).get("sha256")
        if digest:
            url += "#sha256=" + digest
    return url


def wheel_options(arguments: list) -> list:
    # Forward the platform's index/build settings, not its package specifications.
    parser = argparse.ArgumentParser(add_help=False)
    for option in (
        "--index-url",
        "--extra-index-url",
        "--find-links",
        "--trusted-host",
        "--timeout",
        "--retries",
        "--no-binary",
        "--only-binary",
        "--config-settings",
    ):
        parser.add_argument(option, action="append")
    for option in ("--no-index", "--no-build-isolation", "--ignore-requires-python"):
        parser.add_argument(option, action="store_true")
    options, _ = parser.parse_known_args(arguments)
    result = []
    for name, values in vars(options).items():
        flag = "--" + name.replace("_", "-")
        if values is True:
            result.append(flag)
        elif values:
            for value in values:
                result.extend([flag, value])
    return result


def install_batch(root: Path, batch: dict) -> None:
    run_pip(
        [
            "install",
            "--no-index",
            "--no-deps",
            *batch["args"],
            *(str(root / wheel) for wheel in batch["wheels"]),
        ]
    )


def collect(root: Path, arguments: list) -> None:
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "install.json"
    batches = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    # Retain pip's download cache between resolution and wheel preparation.
    arguments = [arg for arg in arguments if arg != "--no-cache-dir"]
    with tempfile.TemporaryDirectory() as temporary:
        report = Path(temporary) / "report.json"
        run_pip(["install", "--dry-run", "--report", str(report), *arguments])
        items = json.loads(report.read_text())["install"]
    if not items:
        return

    destination = root / str(len(batches))
    destination.mkdir()
    # Build before installing: producer and consumers install these same wheels.
    run_pip(
        [
            "wheel",
            "--no-deps",
            "--wheel-dir",
            str(destination),
            *wheel_options(arguments),
            *(wheel_source(item) for item in items),
        ]
    )
    batch = {
        "args": [arg for arg in arguments if arg in INSTALL_FLAGS],
        "wheels": [path.relative_to(root).as_posix() for path in sorted(destination.glob("*.whl"))],
    }
    # Editable installs replace an image copy even at the same version.
    # Apply that behavior only to their resolved batch, not every dependency.
    if any(item["download_info"].get("dir_info", {}).get("editable") for item in items):
        if "--force-reinstall" not in batch["args"]:
            batch["args"].append("--force-reinstall")
    install_batch(root, batch)
    batches.append(batch)
    manifest_path.write_text(json.dumps(batches))


def prepare(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "install.json"
    if not manifest_path.exists():
        manifest_path.write_text("[]")
    if os.environ.get("TE_FL_CACHE_KEY"):
        destination = root / "te-fl"
        destination.mkdir()
        for wheel in Path(os.environ["TE_FL_WHEEL_DIR"]).glob("transformer_engine*.whl"):
            shutil.copyfile(wheel, destination / wheel.name)
    with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
        stream.write(f"name={os.environ['CI_UNIT_SNAPSHOT_NAME']}\n")


def install(root: Path) -> None:
    for batch in json.loads((root / "install.json").read_text()):
        install_batch(root, batch)
    if os.environ.get("TE_FL_CACHE_KEY"):
        subprocess.run(
            ["bash", str(SCRIPT_DIR / "install_te_fl_runtime.sh")],
            env={
                **os.environ,
                "CI_PYTHON_BIN": sys.executable,
                "TE_FL_WHEEL_DIR": str(root / "te-fl"),
            },
            check=True,
            timeout=900,
        )
    with open(os.environ["GITHUB_ENV"], "a") as stream:
        stream.write("CI_UNIT_SNAPSHOT_READY=true\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("collect", "prepare", "install"))
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path(os.environ.get("CI_UNIT_SNAPSHOT_DIRECTORY", "unit-runtime-snapshot")),
    )
    args, pip_args = parser.parse_known_args()
    root = args.directory.resolve()
    if args.command == "collect":
        collect(root, pip_args[1:] if pip_args[:1] == ["--"] else pip_args)
    elif args.command == "prepare":
        prepare(root)
    else:
        install(root)


if __name__ == "__main__":
    main()
