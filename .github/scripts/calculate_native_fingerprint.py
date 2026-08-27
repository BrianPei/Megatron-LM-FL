#!/usr/bin/env python3
"""Calculate a TE-FL native artifact cache key.

The checked-out TE-FL commit is recorded as provenance but is deliberately not
part of the fingerprint. Python-only changes therefore reuse the existing
native artifact while native source, ABI, toolchain, image, or build-option
changes invalidate it.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import re
from pathlib import Path
from typing import Any


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def output(command: list[str], *, required: bool = False) -> str:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        if required:
            fail(f"required command is unavailable: {command[0]}")
        return "unavailable"
    if result.returncode != 0:
        if required:
            fail(f"command failed: {' '.join(command)}\n{result.stderr.strip()}")
        return "unavailable"
    return result.stdout.strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_string_list(value: str, field: str) -> list[str]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        fail(f"{field} is not valid JSON: {error}")
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        fail(f"{field} must be a JSON array of strings")
    if any(not item for item in parsed):
        fail(f"{field} must not contain empty strings")
    return parsed


def parse_scalar_map(value: str, field: str) -> dict[str, str]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        fail(f"{field} is not valid JSON: {error}")
    if not isinstance(parsed, dict):
        fail(f"{field} must be a JSON object")
    values: dict[str, str] = {}
    for key, item in parsed.items():
        if (
            not isinstance(key, str)
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
            or not isinstance(item, (str, int, float, bool))
        ):
            fail(f"{field} must contain only scalar values")
        values[key] = str(item)
    return values


def hash_configured_paths(
    root: Path, configured_paths: list[str], *, native_only: bool = False
) -> dict[str, Any]:
    native_suffixes = {
        ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".hip", ".proto",
    }

    def included(path: Path) -> bool:
        return not native_only or path.name == "CMakeLists.txt" or path.suffix in native_suffixes

    files: list[Path] = []
    missing: list[str] = []
    for configured_path in configured_paths:
        path = root / configured_path
        if path.is_file():
            if included(path):
                files.append(path)
        elif path.is_dir():
            files.extend(item for item in path.rglob("*") if item.is_file() and included(item))
        else:
            missing.append(configured_path)
    if missing:
        fail(f"configured native inputs do not exist: {', '.join(missing)}")
    if not files:
        fail("configured native inputs contain no native files")

    aggregate = hashlib.sha256()
    records = []
    for path in sorted(set(files)):
        relative_path = path.relative_to(root).as_posix()
        digest = file_sha256(path)
        aggregate.update(relative_path.encode())
        aggregate.update(b"\0")
        aggregate.update(digest.encode())
        records.append({"path": relative_path, "sha256": digest})
    return {"sha256": aggregate.hexdigest(), "files": records}


def submodule_revisions(root: Path) -> list[dict[str, str]]:
    revisions = []
    for line in output(
        ["git", "-C", str(root), "submodule", "status", "--recursive"],
        required=True,
    ).splitlines():
        fields = line.lstrip(" +-U").split()
        if len(fields) >= 2 and len(fields[0]) == 40:
            revisions.append({"path": fields[1], "commit": fields[0]})
    return sorted(revisions, key=lambda item: item["path"])


def module_identity(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        fail(f"native runtime module {module_name!r} cannot be imported: {error}")
    origin = getattr(module, "__file__", None)
    record: dict[str, Any] = {
        "module": module_name,
        "version": str(getattr(module, "__version__", "unknown")),
        "origin": str(origin),
        "sha256": "synthetic-module",
    }
    if origin and Path(origin).is_file():
        record["sha256"] = file_sha256(Path(origin))
    native_files = []
    package_paths = getattr(module, "__path__", ())
    for package_path in package_paths:
        root = Path(package_path)
        for path in root.rglob("*"):
            if path.is_file() and (".so" in path.name or path.suffix in {".pyd", ".dylib"}):
                native_files.append({
                    "path": path.relative_to(root).as_posix(),
                    "sha256": file_sha256(path),
                })
    record["native_files"] = sorted(native_files, key=lambda item: item["path"])
    return record


def runtime_identity(compiler: str) -> dict[str, Any]:
    try:
        import torch
    except Exception as error:
        fail(f"PyTorch runtime cannot be inspected: {error}")
    compiler_path = shutil.which(compiler)
    if not compiler_path:
        fail(f"configured compiler is unavailable: {compiler}")
    package_inventory = output(["dpkg-query", "-W", "-f=${Package}=${Version}\n"])
    if package_inventory == "unavailable":
        package_inventory = output(["rpm", "-qa", "--qf", "%{NAME}=%{VERSION}-%{RELEASE}\n"])
    package_inventory = "\n".join(sorted(package_inventory.splitlines()))
    return {
        "python": sys.version.split()[0],
        "python_soabi": sysconfig.get_config_var("SOABI"),
        "torch": {
            "version": str(torch.__version__),
            "cuda": str(getattr(torch.version, "cuda", None)),
            "hip": str(getattr(torch.version, "hip", None)),
        },
        "compiler": {
            "path": compiler_path,
            "sha256": file_sha256(Path(compiler_path)),
            "version": output([compiler, "--version"], required=True).splitlines()[0],
        },
        "cmake": output(["cmake", "--version"]).splitlines()[0],
        "ninja": output(["ninja", "--version"]),
        "system_packages_sha256": hashlib.sha256(package_inventory.encode()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", required=True)
    parser.add_argument("--mode", choices=("source", "base_image"), required=True)
    parser.add_argument("--base-image-ref", required=True)
    parser.add_argument("--te-fl-repo", required=True, type=Path)
    parser.add_argument("--target-arch", required=True)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--source-paths-json", default="[]")
    parser.add_argument("--build-files-json", default="[]")
    parser.add_argument("--build-env-json", default="{}")
    parser.add_argument("--build-pip-args-json", default="[]")
    parser.add_argument("--native-modules-json", default="[]")
    parser.add_argument("--runtime-modules-json", default="[]")
    parser.add_argument("--require-shared-library", choices=("true", "false"), required=True)
    parser.add_argument("--build-recipe-hash", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if len(args.build_recipe_hash) != 64 or any(
        character not in "0123456789abcdef" for character in args.build_recipe_hash
    ):
        fail("build recipe hash must be a lowercase SHA256 digest")

    repository = args.te_fl_repo.resolve()
    if not (repository / ".git").exists():
        fail(f"TE-FL checkout is not a Git repository: {repository}")
    python_commit = output(["git", "-C", str(repository), "rev-parse", "HEAD"], required=True)
    if len(python_commit) != 40:
        fail(f"invalid TE-FL commit: {python_commit}")

    source_paths = parse_string_list(args.source_paths_json, "source paths")
    build_files = parse_string_list(args.build_files_json, "build files")
    build_environment = parse_scalar_map(args.build_env_json, "build environment")
    build_pip_args = parse_string_list(args.build_pip_args_json, "build pip args")
    native_modules = parse_string_list(args.native_modules_json, "native modules")
    runtime_modules = parse_string_list(args.runtime_modules_json, "runtime modules")
    if not runtime_modules:
        fail("runtime modules must contain at least one ABI-bearing module")

    # Backend-discovery controls must take effect before importing torch or
    # any vendor runtime module for fingerprinting.
    os.environ.update(build_environment)

    components: dict[str, Any] = {
        "schema_version": 1,
        "platform": args.platform,
        "mode": args.mode,
        "base_image_ref": args.base_image_ref,
        "target_arch": args.target_arch,
        "build_environment": build_environment,
        "build_pip_args": build_pip_args,
        "require_shared_library": args.require_shared_library == "true",
        "build_recipe_hash": args.build_recipe_hash,
        "runtime": runtime_identity(args.compiler),
        "runtime_modules": [module_identity(name) for name in runtime_modules],
    }
    if args.mode == "source":
        if not source_paths or not build_files:
            fail("source mode requires native source paths and build files")
        components["native_sources"] = hash_configured_paths(
            repository, source_paths, native_only=True
        )
        components["build_files"] = hash_configured_paths(repository, build_files)
        components["submodules"] = submodule_revisions(repository)
    else:
        if not native_modules:
            fail("base_image mode requires native runtime modules")
        components["base_image_modules"] = [module_identity(name) for name in native_modules]

    canonical = json.dumps(components, sort_keys=True, separators=(",", ":"))
    result = {
        "fingerprint": hashlib.sha256(canonical.encode()).hexdigest(),
        "te_fl_python_commit": python_commit,
        "components": components,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
