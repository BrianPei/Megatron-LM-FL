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

build_env_exports=$(python3 - "$BUILD_ENV_JSON" <<'PY'
import json
import shlex
import sys

values = json.loads(sys.argv[1])
if not isinstance(values, dict):
    raise SystemExit("build environment must be a JSON object")
for key, value in values.items():
    if not isinstance(key, str) or not key or not key.replace("_", "").isascii() or not key.replace("_", "").isalnum() or key[0].isdigit():
        raise SystemExit(f"invalid environment variable name: {key}")
    print(f"export {key}={shlex.quote(str(value))}")
PY
)

build_pip_args=()
while IFS= read -r arg; do
  build_pip_args+=("$arg")
done < <(python3 - "$BUILD_PIP_ARGS_JSON" <<'PY'
import json
import sys

values = json.loads(sys.argv[1])
if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
    raise SystemExit("build pip args must be a JSON array of strings")
if any(not value for value in values):
    raise SystemExit("build pip args must not contain empty strings")
sys.stdout.write("".join(f"{value}\n" for value in values))
PY
)
eval "$build_env_exports"

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
suffixes = {".py", ".pyi", ".json", ".yaml", ".yml", ".toml", ".txt", ".typed"}
files = sorted(
    path.relative_to(root).as_posix()
    for path in root.rglob("*")
    if path.is_file() and path.suffix in suffixes
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

export ARTIFACT_FILES_JSON
ARTIFACT_FILES_JSON=$(python3 - "${artifact_files[@]}" <<'PY'
import hashlib
import json
import pathlib
import sys

print(json.dumps([{"filename": pathlib.Path(value).name, "sha256": hashlib.sha256(pathlib.Path(value).read_bytes()).hexdigest(), "size": pathlib.Path(value).stat().st_size} for value in sys.argv[1:]]))
PY
)

python3 - "$OUTPUT_DIR/manifest.json" <<'PY'
import json
import os
import pathlib
import platform
import sys

manifest = {
    "schema_version": 1,
    "artifact_type": "te-fl-native-runtime",
    "platform": os.environ["PLATFORM"],
    "mode": os.environ["NATIVE_MODE"],
    "base_image_ref": os.environ["BASE_IMAGE_REF"],
    # Python provenance is separate from the native identity. This allows a
    # Python-only TE-FL commit to reuse the native artifact.
    "te_fl_python_commit": os.environ["TE_FL_COMMIT"],
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
