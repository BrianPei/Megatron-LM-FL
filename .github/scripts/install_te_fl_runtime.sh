#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/set_env_common.sh"

required=(PLATFORM TE_FL_SOURCE TE_FL_PYTHON_COMMIT TE_FL_NATIVE_DIR TE_FL_NATIVE_FINGERPRINT TE_FL_RUNTIME_ENV_JSON TE_FL_INSTALL_PIP_ARGS_JSON TE_FL_BASE_IMAGE_REF)
for name in "${required[@]}"; do
  if [ -z "${!name:-}" ]; then
    echo "::error::$name is required" >&2
    exit 1
  fi
done

manifest="$TE_FL_NATIVE_DIR/manifest.json"
checksums="$TE_FL_NATIVE_DIR/checksums.txt"
test -f "$manifest"
test -f "$checksums"

manifest_env=$(mktemp)
python3 - "$manifest" "$manifest_env" <<'PY'
import json
import shlex
import sys
from pathlib import Path

manifest = json.load(open(sys.argv[1]))
if manifest.get("schema_version") != 2:
    raise SystemExit(f"unsupported artifact manifest schema: {manifest.get('schema_version')!r}")
if manifest.get("artifact_type") != "te-fl-native-runtime":
    raise SystemExit(f"unsupported artifact type: {manifest.get('artifact_type')!r}")
keys = (
    "platform",
    "native_source_fingerprint",
    "native_fingerprint",
    "native_build_commit",
    "base_image_ref",
    "mode",
)
lines = []
for key in keys:
    value = manifest.get(key)
    if not isinstance(value, str) or not value:
        raise SystemExit(f"artifact manifest field is missing: {key}")
    lines.append(f"TE_FL_ARTIFACT_{key.upper()}={shlex.quote(value)}")
Path(sys.argv[2]).write_text("\n".join(lines) + "\n")
PY
source "$manifest_env"
rm -f "$manifest_env"

test "$TE_FL_ARTIFACT_PLATFORM" = "$PLATFORM"
test "$TE_FL_ARTIFACT_NATIVE_SOURCE_FINGERPRINT" = "$TE_FL_NATIVE_FINGERPRINT"
test "$TE_FL_ARTIFACT_NATIVE_FINGERPRINT" = "$TE_FL_NATIVE_FINGERPRINT"
test "$TE_FL_ARTIFACT_BASE_IMAGE_REF" = "$TE_FL_BASE_IMAGE_REF"
if ! [[ "$TE_FL_ARTIFACT_NATIVE_BUILD_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "::error::artifact native build commit is invalid" >&2
  exit 1
fi
test "$(git -C "$TE_FL_SOURCE" rev-parse HEAD)" = "$TE_FL_PYTHON_COMMIT"
(cd "$TE_FL_NATIVE_DIR" && sha256sum -c checksums.txt)

native_mode=$TE_FL_ARTIFACT_MODE
ci_apply_env_json "$TE_FL_RUNTIME_ENV_JSON"

runtime_abi_fingerprint_file=$(mktemp)
python3 - \
  "$TE_FL_NATIVE_DIR/fingerprint.json" \
  "$TE_FL_BASE_IMAGE_REF" \
  "$TE_FL_ARTIFACT_NATIVE_BUILD_COMMIT" \
  "$PLATFORM" \
  "$native_mode" \
  "$runtime_abi_fingerprint_file" <<'PY'
import hashlib
import importlib
import json
import pathlib
import sys
import sysconfig


def fail(message: str) -> None:
    raise SystemExit(f"runtime ABI mismatch: {message}")


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_identity(module_name: str) -> dict:
    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        fail(f"cannot import runtime module {module_name!r}: {error}")
    origin = getattr(module, "__file__", None)
    record = {
        "module": module_name,
        "version": str(getattr(module, "__version__", "unknown")),
        "sha256": "synthetic-module",
    }
    if origin and pathlib.Path(origin).is_file():
        record["sha256"] = file_sha256(pathlib.Path(origin))
    native_files = []
    for package_path in getattr(module, "__path__", ()):
        root = pathlib.Path(package_path)
        for path in root.rglob("*"):
            if path.is_file() and (".so" in path.name or path.suffix in {".pyd", ".dylib"}):
                native_files.append({
                    "path": path.relative_to(root).as_posix(),
                    "sha256": file_sha256(path),
                })
    record["native_files"] = sorted(native_files, key=lambda item: item["path"])
    return record


def module_summary(record: dict) -> dict:
    native_files = record.get("native_files", [])
    native_digest = hashlib.sha256(
        json.dumps(native_files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "version": record.get("version"),
        "module_sha256": record.get("sha256"),
        "native_file_count": len(native_files),
        "native_files_sha256": native_digest,
    }


fingerprint = json.load(open(sys.argv[1]))
components = fingerprint.get("components", {})
expected_fingerprint = __import__("os").environ["TE_FL_NATIVE_FINGERPRINT"]
canonical = json.dumps(components, sort_keys=True, separators=(",", ":"))
if hashlib.sha256(canonical.encode()).hexdigest() != expected_fingerprint:
    fail("fingerprint components do not reproduce the requested artifact key")
if fingerprint.get("fingerprint") != expected_fingerprint:
    fail("fingerprint document does not match the requested artifact")
if components.get("base_image_ref") != sys.argv[2]:
    fail("base image reference differs from the native build environment")
if fingerprint.get("te_fl_python_commit") != sys.argv[3]:
    fail("native build commit differs from fingerprint provenance")
if components.get("platform") != sys.argv[4]:
    fail("platform differs from fingerprint provenance")
if components.get("mode") != sys.argv[5]:
    fail("native mode differs from fingerprint provenance")

expected_runtime = components.get("runtime", {})
try:
    import torch
except Exception as error:
    fail(f"cannot import torch: {error}")

actual_runtime = {
    "python": sys.version.split()[0],
    "python_soabi": sysconfig.get_config_var("SOABI"),
    "torch": {
        "version": str(torch.__version__),
        "cuda": str(getattr(torch.version, "cuda", None)),
        "hip": str(getattr(torch.version, "hip", None)),
    },
}
for key in ("python", "python_soabi", "torch"):
    if actual_runtime[key] != expected_runtime.get(key):
        fail(f"{key} differs: expected={expected_runtime.get(key)!r} actual={actual_runtime[key]!r}")

actual_modules = []
for expected in components.get("runtime_modules", []):
    expected_contract = {
        key: expected.get(key)
        for key in ("module", "version", "sha256", "native_files")
    }
    actual_contract = module_identity(expected_contract["module"])
    if actual_contract != expected_contract:
        fail(
            f"runtime module differs: {expected_contract['module']}; "
            f"expected={module_summary(expected_contract)!r} "
            f"actual={module_summary(actual_contract)!r}"
        )
    actual_modules.append(actual_contract)

contract = {
    "python": actual_runtime["python"],
    "python_soabi": actual_runtime["python_soabi"],
    "torch": actual_runtime["torch"],
    "runtime_modules": actual_modules,
}
canonical = json.dumps(contract, sort_keys=True, separators=(",", ":"))
digest = hashlib.sha256(canonical.encode()).hexdigest()
pathlib.Path(sys.argv[6]).write_text(digest + "\n")
print(f"runtime ABI: PASS ({digest})")
PY
TE_FL_RUNTIME_ABI_FINGERPRINT=$(<"$runtime_abi_fingerprint_file")
rm -f "$runtime_abi_fingerprint_file"
export TE_FL_RUNTIME_ABI_FINGERPRINT

install_pip_args=()
install_pip_args_file=$(mktemp)
python3 - "$TE_FL_INSTALL_PIP_ARGS_JSON" "$install_pip_args_file" <<'PY'
import json
import sys
from pathlib import Path

values = json.loads(sys.argv[1])
if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
    raise SystemExit("install pip args must be a JSON array of strings")
if any(not value or any(character in value for character in ("\n", "\r")) for value in values):
    raise SystemExit("install pip args must contain non-empty single-line strings")
Path(sys.argv[2]).write_text("".join(f"{value}\n" for value in values))
PY
while IFS= read -r arg; do
  install_pip_args+=("$arg")
done < "$install_pip_args_file"
rm -f "$install_pip_args_file"
if [ "$native_mode" = source ]; then
  wheels=()
  wheel_list_file=$(mktemp)
  python3 - "$manifest" "$TE_FL_NATIVE_DIR" "$wheel_list_file" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.load(open(sys.argv[1]))
artifact_root = Path(sys.argv[2])
wheels = []
for record in manifest.get("files", []):
    filename = record.get("filename")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise SystemExit(f"unsafe artifact filename: {filename!r}")
    if filename.endswith(".whl"):
        wheels.append(str(artifact_root / filename))
Path(sys.argv[3]).write_text("".join(f"{wheel}\n" for wheel in wheels))
PY
  while IFS= read -r wheel; do wheels+=("$wheel"); done < "$wheel_list_file"
  rm -f "$wheel_list_file"
  if [ "${#wheels[@]}" -eq 0 ]; then
    echo "::error::source-mode artifact contains no wheel" >&2
    exit 1
  fi
  # Source-built TE-FL provides its own native extension. Remove the complete
  # NVIDIA TE distribution set first so TE-FL does not detect mixed metadata
  # or a stale transformer_engine_torch extension from the base image.
  python3 -m pip uninstall -y \
    transformer-engine transformer-engine-torch \
    transformer-engine-cu11 transformer-engine-cu12 transformer-engine-cu13 \
    >/dev/null 2>&1 || true
  if [ "${#install_pip_args[@]}" -eq 0 ]; then
    python3 -m pip install --force-reinstall --no-deps --no-cache-dir \
      "${wheels[@]}"
  else
    python3 -m pip install --force-reinstall --no-deps --no-cache-dir \
      "${install_pip_args[@]}" "${wheels[@]}"
  fi
elif [ "$native_mode" != base_image ]; then
  echo "::error::Unsupported TE-FL native mode: $native_mode" >&2
  exit 1
fi

# Copy Python and non-native package data from the current checkout. An editable
# pip install would run build_ext and defeat native artifact reuse.
python_overlay_fingerprint_file=$(mktemp)
export TE_FL_PYTHON_OVERLAY_FINGERPRINT_FILE="$python_overlay_fingerprint_file"
python3 - "$TE_FL_SOURCE" <<'PY'
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

source_root = Path(sys.argv[1]) / "transformer_engine"
if not source_root.is_dir():
    raise SystemExit(f"TE-FL Python package is missing: {source_root}")
spec = importlib.util.find_spec("transformer_engine")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit("installed TE-FL Python package is not importable")
target_root = Path(next(iter(spec.submodule_search_locations)))
native_suffixes = {
    ".a", ".asm", ".c", ".cc", ".cl", ".cmake", ".cpp", ".cu", ".cubin", ".cuh",
    ".dll", ".dylib", ".fatbin", ".h", ".hip", ".hpp", ".in", ".inc",
    ".j2", ".jinja", ".jinja2", ".lib", ".metal", ".mk", ".o", ".obj",
    ".proto", ".ptx", ".pyd", ".pyc", ".pxd", ".pxi", ".pyx", ".s",
    ".so", ".sycl", ".template", ".tpl",
}
native_filenames = {"BUILD", "BUILD.bazel", "CMakeLists.txt", "Makefile", "_build_config.py"}


def is_python_package_file(path: Path) -> bool:
    if (
        "__pycache__" in path.parts
        or path.name in native_filenames
    ):
        return False
    if ".so" in path.name or path.suffix.lower() in native_suffixes:
        return False
    return True


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


source_files = {
    source.relative_to(source_root)
    for source in source_root.rglob("*")
    if source.is_file() and is_python_package_file(source)
}
if not source_files:
    raise SystemExit("TE-FL Python overlay contains no package files")

# Remove only files owned by the cached TE-FL source commit. Vendor packages may
# contribute additional modules under the same namespace and must be preserved.
old_file_list = Path(os.environ["TE_FL_NATIVE_DIR"]) / "python-source-files.json"
if not old_file_list.is_file():
    raise SystemExit(f"cached Python source manifest is missing: {old_file_list}")
old_files = json.loads(old_file_list.read_text())
if not isinstance(old_files, list) or not all(isinstance(name, str) for name in old_files):
    raise SystemExit("cached Python source manifest must be a JSON array of strings")
for relative_name in old_files:
    relative = Path(relative_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise SystemExit(f"unsafe cached Python source path: {relative_name}")
    if relative not in source_files:
        target = target_root / relative
        if target.is_file():
            target.unlink()
copied = 0
aggregate = hashlib.sha256()
for relative in sorted(source_files):
    source = source_root / relative
    target = target_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    source_digest = file_sha256(source)
    if file_sha256(target) != source_digest:
        raise SystemExit(f"TE-FL Python overlay copy mismatch: {relative}")
    aggregate.update(relative.as_posix().encode())
    aggregate.update(b"\0")
    aggregate.update(source_digest.encode())
    copied += 1
Path(os.environ["TE_FL_PYTHON_OVERLAY_FINGERPRINT_FILE"]).write_text(
    aggregate.hexdigest() + "\n"
)
print(f"TE-FL Python overlay installed: {copied} files")
PY
TE_FL_PYTHON_OVERLAY_FINGERPRINT=$(<"$python_overlay_fingerprint_file")
rm -f "$python_overlay_fingerprint_file"
export TE_FL_PYTHON_OVERLAY_FINGERPRINT

runtime_manifest="${TE_FL_RUNTIME_MANIFEST:-/etc/flagos/te-fl.json}"
mkdir -p "$(dirname "$runtime_manifest")"
export TE_FL_ARTIFACT_MANIFEST="$manifest"
python3 - "$runtime_manifest" <<'PY'
import json
import os
import pathlib
import sys
artifact = json.load(open(os.environ["TE_FL_ARTIFACT_MANIFEST"]))
runtime = {
    "schema_version": 2,
    "platform": os.environ["PLATFORM"],
    "native_build_commit": artifact["native_build_commit"],
    "te_fl_python_commit": os.environ["TE_FL_PYTHON_COMMIT"],
    "native_source_fingerprint": artifact["native_source_fingerprint"],
    "native_fingerprint": artifact["native_fingerprint"],
    "runtime_abi_fingerprint": os.environ["TE_FL_RUNTIME_ABI_FINGERPRINT"],
    "python_overlay_fingerprint": os.environ["TE_FL_PYTHON_OVERLAY_FINGERPRINT"],
    "native_mode": artifact["mode"],
    "base_image_ref": artifact["base_image_ref"],
    "artifact_files": artifact["files"],
    "runtime_environment": {
        key: str(value)
        for key, value in json.loads(os.environ["TE_FL_RUNTIME_ENV_JSON"]).items()
    },
}
pathlib.Path(sys.argv[1]).write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
PY

cat "$runtime_manifest"
