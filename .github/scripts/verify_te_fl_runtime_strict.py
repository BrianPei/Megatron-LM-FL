#!/usr/bin/env python3
"""Fail-closed verification of the TE-FL runtime used by a test job."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any


def fail(message: str) -> None:
    raise SystemExit(f"FATAL: {message}")


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        fail(f"runtime manifest is missing: {path}")
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        fail(f"runtime manifest is invalid JSON: {error}")
    required = {
        "schema_version",
        "platform",
        "native_build_commit",
        "te_fl_python_commit",
        "native_source_fingerprint",
        "native_fingerprint",
        "native_mode",
        "base_image_ref",
        "artifact_files",
        "runtime_abi_fingerprint",
        "python_overlay_fingerprint",
    }
    missing = sorted(required - manifest.keys())
    if missing or manifest.get("schema_version") != 2:
        fail(f"runtime manifest schema is invalid; missing={missing}")
    if manifest.get("native_mode") not in {"source", "base_image"}:
        fail(f"runtime manifest native mode is invalid: {manifest.get('native_mode')!r}")
    if not isinstance(manifest.get("base_image_ref"), str) or not manifest["base_image_ref"]:
        fail("runtime manifest base image reference is empty")
    artifact_files = manifest.get("artifact_files")
    if not isinstance(artifact_files, list) or not artifact_files:
        fail("runtime manifest artifact file list is empty")
    for record in artifact_files:
        if not isinstance(record, dict):
            fail("runtime manifest artifact file record is invalid")
        filename = record.get("filename")
        digest = record.get("sha256")
        size = record.get("size")
        if not isinstance(filename, str) or Path(filename).name != filename:
            fail(f"runtime manifest artifact filename is invalid: {filename!r}")
        if not isinstance(digest, str) or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            fail(f"runtime manifest artifact digest is invalid: {filename}")
        if not isinstance(size, int) or size < 0:
            fail(f"runtime manifest artifact size is invalid: {filename}")
    if not isinstance(manifest.get("runtime_environment"), dict):
        fail("runtime manifest environment is invalid")
    return manifest


def verify_identity(manifest: dict[str, Any], args: argparse.Namespace) -> None:
    if manifest["platform"] != args.platform:
        fail(f"platform mismatch: expected={args.platform} actual={manifest['platform']}")
    if manifest["te_fl_python_commit"] != args.expected_commit:
        fail("TE-FL Python commit mismatch")
    if args.expected_fingerprint and manifest["native_fingerprint"] != args.expected_fingerprint:
        fail("TE-FL native fingerprint mismatch")
    for key, value in (
        ("native_build_commit", manifest["native_build_commit"]),
        ("te_fl_python_commit", args.expected_commit),
        ("native_source_fingerprint", manifest["native_source_fingerprint"]),
        ("native_fingerprint", manifest["native_fingerprint"]),
        ("runtime_abi_fingerprint", manifest["runtime_abi_fingerprint"]),
        ("python_overlay_fingerprint", manifest["python_overlay_fingerprint"]),
    ):
        if not isinstance(value, str) or not value:
            fail(f"manifest identity is empty: {key}")
    if manifest["native_source_fingerprint"] != manifest["native_fingerprint"]:
        fail("native fingerprint fields disagree")
    for key in ("native_build_commit", "te_fl_python_commit"):
        value = manifest[key]
        if len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
            fail(f"manifest commit is invalid: {key}")
    for key in (
        "native_source_fingerprint",
        "native_fingerprint",
        "runtime_abi_fingerprint",
        "python_overlay_fingerprint",
    ):
        value = manifest[key]
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            fail(f"manifest fingerprint is invalid: {key}")
    print("manifest identity: PASS")


def parse_environment(value: str) -> dict[str, str]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        fail(f"runtime environment is invalid JSON: {error}")
    if not isinstance(parsed, dict) or not parsed:
        fail("runtime environment must be a non-empty JSON object")
    environment: dict[str, str] = {}
    for key, item in parsed.items():
        if not isinstance(key, str) or not key.isidentifier():
            fail(f"invalid runtime environment variable name: {key!r}")
        if not isinstance(item, (str, int, float, bool)):
            fail(f"runtime environment value must be scalar: {key}")
        environment[key] = str(item)
    return environment


def verify_module(module_name: str, forbidden_root: Path | None) -> Any:
    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        fail(f"cannot import configured TE-FL module {module_name}: {error}")
    origin = getattr(module, "__file__", None)
    if not origin:
        fail(f"configured TE-FL module has no file origin: {module_name}")
    origin_path = Path(origin)
    if not origin_path.is_file():
        fail(f"TE-FL module file does not exist: {origin_path}")
    if forbidden_root is not None:
        try:
            origin_path.resolve().relative_to(forbidden_root.resolve())
        except ValueError:
            pass
        else:
            fail(f"TE-FL module resolved inside a forbidden source tree: {origin_path}")
    print(f"TE-FL module: PASS ({module_name} -> {origin_path})")
    return module


def verify_python_overlay(manifest: dict[str, Any], source_root: Path, module: Any) -> None:
    package_source = source_root / "transformer_engine"
    package_target = Path(module.__file__).parent
    native_suffixes = {
        ".a",
        ".asm",
        ".c",
        ".cc",
        ".cl",
        ".cmake",
        ".cpp",
        ".cu",
        ".cubin",
        ".cuh",
        ".dll",
        ".dylib",
        ".fatbin",
        ".h",
        ".hip",
        ".hpp",
        ".in",
        ".inc",
        ".j2",
        ".jinja",
        ".jinja2",
        ".lib",
        ".metal",
        ".mk",
        ".o",
        ".obj",
        ".proto",
        ".ptx",
        ".pyd",
        ".pyc",
        ".pxd",
        ".pxi",
        ".pyx",
        ".s",
        ".so",
        ".sycl",
        ".template",
        ".tpl",
    }
    native_filenames = {"BUILD", "BUILD.bazel", "CMakeLists.txt", "Makefile", "_build_config.py"}

    def included(path: Path) -> bool:
        if "__pycache__" in path.parts or path.name in native_filenames:
            return False
        return ".so" not in path.name and path.suffix.lower() not in native_suffixes

    def file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    aggregate = hashlib.sha256()
    files = sorted(
        path.relative_to(package_source)
        for path in package_source.rglob("*")
        if path.is_file() and included(path)
    )
    if not files:
        fail("TE-FL Python source contains no overlay files")
    for relative in files:
        source = package_source / relative
        target = package_target / relative
        if not target.is_file():
            fail(f"installed TE-FL Python file is missing: {relative}")
        source_digest = file_sha256(source)
        if file_sha256(target) != source_digest:
            fail(f"installed TE-FL Python file differs: {relative}")
        aggregate.update(relative.as_posix().encode())
        aggregate.update(b"\0")
        aggregate.update(source_digest.encode())
    actual = aggregate.hexdigest()
    if actual != manifest["python_overlay_fingerprint"]:
        fail("TE-FL Python overlay fingerprint mismatch")
    print(f"Python overlay: PASS ({actual})")


def parse_string_list(value: str, field: str) -> list[str]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        fail(f"{field} is invalid JSON: {error}")
    if not isinstance(parsed, list) or not all(isinstance(item, str) and item for item in parsed):
        fail(f"{field} must be a JSON array of non-empty strings")
    return parsed


def bootstrap_runtime(modules: list[str]) -> None:
    for module_name in modules:
        try:
            importlib.import_module(module_name)
        except Exception as error:
            fail(f"cannot import device bootstrap module {module_name}: {error}")
    if modules:
        print(f"device bootstrap: PASS ({', '.join(modules)})")


def device_api(module_name: str) -> tuple[Any, str]:
    parts = module_name.split(".")
    if len(parts) < 2 or parts[0] != "torch":
        fail(f"configured device module must be under torch: {module_name}")
    try:
        api: Any = importlib.import_module("torch")
        for part in parts[1:]:
            api = getattr(api, part)
    except (ImportError, AttributeError) as error:
        fail(f"configured device module is unavailable: {module_name}: {error}")
    return api, parts[-1]


def verify_device(module_name: str) -> str:
    api, device_name = device_api(module_name)
    if not api.is_available() or api.device_count() < 1:
        fail(f"device is unavailable: {device_name}")
    if hasattr(api, "set_device"):
        api.set_device(0)
    print(f"device: PASS ({device_name}, count={api.device_count()})")
    return device_name


def verify_runtime_environment(manifest: dict[str, Any], environment: dict[str, str]) -> None:
    if manifest.get("runtime_environment") != environment:
        fail("runtime environment does not match the provenance manifest")
    for name, value in environment.items():
        if os.environ.get(name) != value:
            fail(f"runtime environment was not established: {name}={value!r}")
    rendered = ", ".join(f"{name}={value}" for name, value in sorted(environment.items()))
    print(f"runtime environment: PASS ({rendered})")


def verify_operator(device_name: str, expected_backend: str) -> None:
    import torch
    from transformer_engine.pytorch import Linear

    device = device_name
    try:
        # TE Linear defaults to CUDA when no device is provided. Construct it
        # directly on the configured device so non-CUDA runtimes never enter
        # torch.cuda initialization before a later .to(device) call.
        layer = Linear(64, 128, device=device)
        inputs = torch.randn(8, 64, device=device, requires_grad=True)
        output = layer(inputs)
        output.sum().backward()
    except Exception as error:
        fail(
            "TE forward/backward failed on the configured device "
            f"(device={device}, expected_backend={expected_backend}): {error}"
        )

    # Do the value check on the host. Some vendor backends dispatch this
    # reduction through their device pipeline and may spend minutes probing
    # unsupported kernels after the TE forward/backward has already completed.
    try:
        output_is_finite = bool(torch.isfinite(output.detach().cpu().float()).all())
        gradient_is_finite = inputs.grad is not None and bool(
            torch.isfinite(inputs.grad.detach().cpu().float()).all()
        )
    except Exception as error:
        fail(f"TE output/gradient host validation failed: {error}")
    if not output_is_finite or not gradient_is_finite:
        fail("TE forward/backward produced non-finite output or gradient")

    try:
        from transformer_engine.plugin.core import get_manager

        manager = get_manager()
        selected_backend = manager.get_selected_impl_id("generic_gemm")
    except Exception as error:
        fail(
            "TE-FL does not expose the configured implementation query "
            f"(expected={expected_backend}): {error}"
        )
    if selected_backend != expected_backend:
        fail("selected backend mismatch: " f"expected={expected_backend} actual={selected_backend}")
    print(f"operator/selected backend: PASS ({selected_backend})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-fingerprint")
    parser.add_argument("--native-module", required=True)
    parser.add_argument("--device-module")
    parser.add_argument("--bootstrap-modules-json", default="[]")
    parser.add_argument("--expected-backend")
    parser.add_argument("--runtime-env-json", required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--forbidden-import-root", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("/etc/flagos/te-fl.json"))
    parser.add_argument(
        "--image-manifest-only",
        action="store_true",
        help="Verify a prebuilt runtime image without repeating device/backend execution",
    )
    args = parser.parse_args()
    if len(args.expected_commit) != 40 or any(
        c not in "0123456789abcdef" for c in args.expected_commit
    ):
        fail("expected TE-FL commit must be a full lowercase SHA")

    manifest = load_manifest(args.manifest)
    verify_identity(manifest, args)
    runtime_environment = parse_environment(args.runtime_env_json)
    bootstrap_modules = parse_string_list(args.bootstrap_modules_json, "bootstrap modules")
    verify_runtime_environment(manifest, runtime_environment)
    bootstrap_runtime(bootstrap_modules)
    forbidden_root = args.forbidden_import_root or args.source_root
    module = verify_module(args.native_module, forbidden_root)
    if args.image_manifest_only:
        print("TE-FL runtime image verification: PASS")
        return 0
    for name in ("source_root", "device_module", "expected_backend"):
        if getattr(args, name) is None:
            fail(f"--{name.replace('_', '-')} is required for strict runtime verification")
    verify_python_overlay(manifest, args.source_root, module)
    device_name = verify_device(args.device_module)
    verify_operator(device_name, args.expected_backend)
    print("TE-FL runtime verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
