#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/set_env_common.sh"

required=(PLATFORM TE_FL_SOURCE TE_FL_PYTHON_COMMIT TE_FL_NATIVE_DIR TE_FL_NATIVE_FINGERPRINT TE_FL_RUNTIME_ENV_JSON TE_FL_INSTALL_PIP_ARGS_JSON)
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

read_manifest() {
  python3 - "$manifest" "$1" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1]))
for part in sys.argv[2].split("."):
    value = value[part]
print(value)
PY
}

test "$(read_manifest platform)" = "$PLATFORM"
test "$(read_manifest native_source_fingerprint)" = "$TE_FL_NATIVE_FINGERPRINT"
test "$(read_manifest native_fingerprint)" = "$TE_FL_NATIVE_FINGERPRINT"
test -n "$(read_manifest te_fl_python_commit)"
test "$(git -C "$TE_FL_SOURCE" rev-parse HEAD)" = "$TE_FL_PYTHON_COMMIT"
(cd "$TE_FL_NATIVE_DIR" && sha256sum -c checksums.txt)

native_mode=$(read_manifest mode)
ci_apply_env_json "$TE_FL_RUNTIME_ENV_JSON"

install_pip_args=()
while IFS= read -r arg; do
  install_pip_args+=("$arg")
done < <(python3 - "$TE_FL_INSTALL_PIP_ARGS_JSON" <<'PY'
import json
import sys

values = json.loads(sys.argv[1])
if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
    raise SystemExit("install pip args must be a JSON array of strings")
if any(not value for value in values):
    raise SystemExit("install pip args must not contain empty strings")
sys.stdout.write("".join(f"{value}\n" for value in values))
PY
)
if [ "$native_mode" = source ]; then
  wheels=()
  while IFS= read -r wheel; do wheels+=("$wheel"); done < <(
    python3 - "$manifest" "$TE_FL_NATIVE_DIR" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.load(open(sys.argv[1]))
artifact_root = Path(sys.argv[2])
for record in manifest.get("files", []):
    filename = record.get("filename")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise SystemExit(f"unsafe artifact filename: {filename!r}")
    if filename.endswith(".whl"):
        print(artifact_root / filename)
PY
  )
  if [ "${#wheels[@]}" -eq 0 ]; then
    echo "::error::source-mode artifact contains no wheel" >&2
    exit 1
  fi
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

# Copy only Python/package metadata from the current checkout. An editable pip
# install would run build_ext and defeat native artifact reuse.
python3 - "$TE_FL_SOURCE" <<'PY'
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
allowed_suffixes = {".py", ".pyi", ".json", ".yaml", ".yml", ".toml", ".txt", ".typed"}
source_files = {
    source.relative_to(source_root)
    for source in source_root.rglob("*")
    if source.is_file() and source.suffix in allowed_suffixes
}
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
for source in source_root.rglob("*"):
    if not source.is_file() or source.suffix not in allowed_suffixes:
        continue
    target = target_root / source.relative_to(source_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    copied += 1
if copied == 0:
    raise SystemExit("TE-FL Python overlay contains no package files")
print(f"TE-FL Python overlay installed: {copied} files")
PY

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
    "schema_version": 1,
    "platform": os.environ["PLATFORM"],
    "te_fl_python_commit": os.environ["TE_FL_PYTHON_COMMIT"],
    "native_source_fingerprint": artifact["native_source_fingerprint"],
    "native_fingerprint": artifact["native_fingerprint"],
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
