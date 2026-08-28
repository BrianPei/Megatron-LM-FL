#!/usr/bin/env bash

set -euo pipefail

required=(PLATFORM NATIVE_MODE BASE_IMAGE_REF TE_FL_REPO TE_FL_COMMIT NATIVE_FINGERPRINT OUTPUT_DIR BUILD_ENV_JSON BUILD_PIP_ARGS_JSON NATIVE_MODULES_JSON FINGERPRINT_MANIFEST REQUIRE_SHARED_LIBRARY)
for name in "${required[@]}"; do
  if [ -z "${!name:-}" ]; then
    echo "::error::$name is required" >&2
    exit 1
  fi
done

if ! [[ "$TE_FL_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "::error::TE_FL_COMMIT must be a full commit SHA" >&2
  exit 1
fi
if [ "$(git -C "$TE_FL_REPO" rev-parse HEAD)" != "$TE_FL_COMMIT" ]; then
  echo "::error::TE-FL checkout does not match TE_FL_COMMIT" >&2
  exit 1
fi

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

build_env_file="$OUTPUT_DIR/.build-env.sh"
python3 - "$BUILD_ENV_JSON" "$build_env_file" <<'PY'
import json
import shlex
import sys
from pathlib import Path

values = json.loads(sys.argv[1])
if not isinstance(values, dict):
    raise SystemExit("build environment must be a JSON object")
lines = []
for key, value in values.items():
    if not isinstance(key, str) or not key or not key.replace("_", "").isascii() or not key.replace("_", "").isalnum() or key[0].isdigit():
        raise SystemExit(f"invalid environment variable name: {key}")
    text = str(value)
    if any(character in text for character in ("\n", "\r", "\t")):
        raise SystemExit(f"environment value contains unsupported control characters: {key}")
    lines.append(f"export {key}={shlex.quote(text)}")
Path(sys.argv[2]).write_text("\n".join(lines) + "\n")
PY
source "$build_env_file"

build_pip_args=()
build_pip_args_file="$OUTPUT_DIR/.build-pip-args"
python3 - "$BUILD_PIP_ARGS_JSON" "$build_pip_args_file" <<'PY'
import json
import sys
from pathlib import Path

values = json.loads(sys.argv[1])
if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
    raise SystemExit("build pip args must be a JSON array of strings")
if any(not value or any(character in value for character in ("\n", "\r")) for value in values):
    raise SystemExit("build pip args must contain non-empty single-line strings")
Path(sys.argv[2]).write_text("".join(f"{value}\n" for value in values))
PY
while IFS= read -r arg; do
  build_pip_args+=("$arg")
done < "$build_pip_args_file"

if [ -n "${TARGET_ARCH:-}" ]; then
  export TE_FL_TARGET_ARCH="$TARGET_ARCH"
fi

artifact_files=()
cp "$FINGERPRINT_MANIFEST" "$OUTPUT_DIR/fingerprint.json"
artifact_files+=("$OUTPUT_DIR/fingerprint.json")
python3 - "$TE_FL_REPO" "$OUTPUT_DIR/python-source-files.json" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "transformer_engine"
native_suffixes = {
    ".a", ".asm", ".c", ".cc", ".cl", ".cmake", ".cpp", ".cu", ".cubin", ".cuh",
    ".dll", ".dylib", ".fatbin", ".h", ".hip", ".hpp", ".in", ".inc",
    ".j2", ".jinja", ".jinja2", ".lib", ".metal", ".mk", ".o", ".obj",
    ".proto", ".ptx", ".pyd", ".pyc", ".pxd", ".pxi", ".pyx", ".s",
    ".so", ".sycl", ".template", ".tpl",
}
native_filenames = {
    "BUILD", "BUILD.bazel", "CMakeLists.txt", "Makefile", "_build_config.py",
}


def is_python_package_file(path: Path) -> bool:
    if "__pycache__" in path.parts or path.name in native_filenames:
        return False
    return ".so" not in path.name and path.suffix.lower() not in native_suffixes


files = sorted(
    path.relative_to(root).as_posix()
    for path in root.rglob("*")
    if path.is_file() and is_python_package_file(path)
)
if not files:
    raise SystemExit("TE-FL Python source file list is empty")
Path(sys.argv[2]).write_text(json.dumps(files, indent=2) + "\n")
PY
artifact_files+=("$OUTPUT_DIR/python-source-files.json")
if [ "$NATIVE_MODE" = source ]; then
  if [ "${#build_pip_args[@]}" -eq 0 ]; then
    python3 -m pip wheel --no-deps --no-build-isolation \
      --wheel-dir "$OUTPUT_DIR" "$TE_FL_REPO"
  else
    python3 -m pip wheel --no-deps --no-build-isolation \
      --wheel-dir "$OUTPUT_DIR" "${build_pip_args[@]}" "$TE_FL_REPO"
  fi
  wheels=()
  while IFS= read -r wheel; do wheels+=("$wheel"); done \
    < <(find "$OUTPUT_DIR" -maxdepth 1 -type f -name 'transformer_engine*.whl' -print | sort)
  if [ "${#wheels[@]}" -eq 0 ]; then
    echo "::error::TE-FL build produced no transformer_engine wheel" >&2
    exit 1
  fi
  artifact_files+=("${wheels[@]}")
  if [ "$REQUIRE_SHARED_LIBRARY" = true ]; then
    python3 - "${wheels[@]}" <<'PY'
import sys
import zipfile

for name in sys.argv[1:]:
    with zipfile.ZipFile(name) as wheel:
        if any(item.endswith((".so", ".pyd")) for item in wheel.namelist()):
            break
else:
    raise SystemExit("native wheel contains no shared library")
PY
  elif [ "$REQUIRE_SHARED_LIBRARY" != false ]; then
    echo "::error::REQUIRE_SHARED_LIBRARY must be true or false" >&2
    exit 1
  fi
elif [ "$NATIVE_MODE" = base_image ]; then
  python3 - "$NATIVE_MODULES_JSON" "$OUTPUT_DIR/base-image-modules.json" <<'PY'
import hashlib
import importlib
import json
import pathlib
import sys

records = []
for name in json.loads(sys.argv[1]):
    module = importlib.import_module(name)
    origin = getattr(module, "__file__", None)
    record = {"module": name, "version": str(getattr(module, "__version__", "unknown")), "origin": origin}
    record["sha256"] = hashlib.sha256(pathlib.Path(origin).read_bytes()).hexdigest() if origin and pathlib.Path(origin).is_file() else "synthetic-module"
    records.append(record)
pathlib.Path(sys.argv[2]).write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
PY
  artifact_files+=("$OUTPUT_DIR/base-image-modules.json")
else
  echo "::error::Unsupported native mode: $NATIVE_MODE" >&2
  exit 1
fi

artifact_files_json="$OUTPUT_DIR/.artifact-files.json"
python3 - "$artifact_files_json" "${artifact_files[@]}" <<'PY'
import hashlib
import json
import pathlib
import sys

pathlib.Path(sys.argv[1]).write_text(json.dumps([{"filename": pathlib.Path(value).name, "sha256": hashlib.sha256(pathlib.Path(value).read_bytes()).hexdigest(), "size": pathlib.Path(value).stat().st_size} for value in sys.argv[2:]]))
PY
export ARTIFACT_FILES_JSON
ARTIFACT_FILES_JSON=$(<"$artifact_files_json")

python3 - "$OUTPUT_DIR/manifest.json" <<'PY'
import json
import os
import pathlib
import platform
import sys

manifest = {
    "schema_version": 2,
    "artifact_type": "te-fl-native-runtime",
    "platform": os.environ["PLATFORM"],
    "mode": os.environ["NATIVE_MODE"],
    "base_image_ref": os.environ["BASE_IMAGE_REF"],
    # This is the commit that produced the cached wheel. Runtime Python may
    # come from a newer commit with the same native fingerprint.
    "native_build_commit": os.environ["TE_FL_COMMIT"],
    "native_source_fingerprint": os.environ["NATIVE_FINGERPRINT"],
    "native_fingerprint": os.environ["NATIVE_FINGERPRINT"],
    "files": json.loads(os.environ["ARTIFACT_FILES_JSON"]),
    "python": platform.python_version(),
}
pathlib.Path(sys.argv[1]).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY

(
  cd "$OUTPUT_DIR"
  : > checksums.txt
  for file in "${artifact_files[@]}"; do sha256sum "$(basename "$file")" >> checksums.txt; done
  sha256sum manifest.json >> checksums.txt
)

cat "$OUTPUT_DIR/manifest.json"
cat "$OUTPUT_DIR/checksums.txt"
rm -f "$build_env_file" "$build_pip_args_file" "$artifact_files_json"
