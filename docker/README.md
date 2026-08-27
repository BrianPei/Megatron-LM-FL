# TE-FL Unified CI Integration

This directory contains the unified CI/CD infrastructure for building and managing TransformerEngine-FL (TE-FL) across all hardware platforms.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  versions.yaml                                                   │
│  Single source of truth for TE-FL versions and platform configs │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  build.sh                                                        │
│  Unified build script for all platforms                         │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  Platform Dockerfiles                                            │
│  ├─ cuda/Dockerfile          (NVIDIA CUDA)                      │
│  ├─ musa/Dockerfile          (Moore Threads MUSA)               │
│  ├─ metax/Dockerfile         (MetaX MACA)                       │
│  ├─ ascend/Dockerfile        (Huawei Ascend NPU)                │
│  ├─ hygon/Dockerfile         (Hygon DCU)                        │
│  ├─ kunlunxin/Dockerfile     (Kunlunxin XPU)                    │
│  └─ enflame/Dockerfile       (Enflame GCU)                      │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  CI Images (harbor.baai.ac.cn/megatron-ci/<platform>:<tag>)    │
│  Fixed TE-FL version + test dependencies                        │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  Megatron-LM-FL Tests                                           │
│  Unit tests / Functional tests / Benchmarks                     │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. versions.yaml

Central version lock file containing:
- TE-FL repository and commit SHA
- Per-platform configuration:
  - Base images
  - PyTorch/Python versions
  - Vendor SDK versions
  - TE-FL artifact URLs

**Critical invariants:**
- `te_fl.commit` MUST be a 40-char hex SHA
- All platform `base_image` values MUST be non-empty
- `te_fl_artifact` URLs SHOULD use digest references (`@sha256:...`)

### 2. Platform Dockerfiles

Each platform has a multi-stage Dockerfile:

**Stage 1: platform-runtime**
- Base image + build tools
- No TE-FL yet

**Stage 2: te-fl-native**
- Installs TE-FL from artifact (preferred) or source
- Sets provenance environment variables:
  - `TE_FL_COMMIT`
  - `TE_FL_PLATFORM`
  - `TE_FL_REPOSITORY`

**Stage 3: ci**
- Test dependencies (pytest, nltk, zarr, tensorstore)
- Smoke test execution
- Target for CI workflows

**Stage 4: dev**
- Development tools (ipython, jupyter, vim, tmux)
- Optional, not used in CI

### 3. Build Scripts

**build.sh**
- Reads `versions.yaml`
- Validates configuration
- Builds platform-specific image
- Generates build metadata
- Tags with descriptive name: `<platform>-te<commit>-<target>-<timestamp>`

**Common scripts:**
- `common/install_te_fl.sh` - TE-FL installation logic
- `common/smoke_test.sh` - Post-install verification

### 4. CI Workflows

**.github/workflows/build_images.yml**
- Triggers:
  - PR changes to `docker/**`
  - Push to main/ci-all
  - Manual dispatch
  - TE-FL update events
- Detects changed platforms
- Builds images per platform
- Runs smoke tests
- Pushes stable images on main
- Creates candidate images on PR

**.github/workflows/monitor_te_fl.yml**
- Scheduled: every 6 hours
- Checks TE-FL repository for new commits
- Creates update PR automatically
- Triggers image rebuilds

### 5. Update Scripts

**.github/scripts/update_te_lock.py**
```bash
# Update single platform
python update_te_lock.py \
  --te-commit abc123...def \
  --platform musa \
  --artifact harbor.baai.ac.cn/megatron-artifacts/te-fl-musa:abc123

# Update all platforms
python update_te_lock.py \
  --te-commit abc123...def \
  --all-platforms \
  --artifact-pattern "harbor.baai.ac.cn/megatron-artifacts/te-fl-{platform}:{commit}"
```

**.github/scripts/verify_te_provenance.sh**
- Runs at start of every test job
- Verifies `TE_FL_COMMIT` and `TE_FL_PLATFORM` are set
- Checks TE-FL imports
- Fails fast if provenance incomplete

## Usage

### Building Images Locally

```bash
# Build MUSA CI image
cd docker
PLATFORM=musa TARGET=ci bash build.sh

# Build MetaX dev image without cache
PLATFORM=metax TARGET=dev NO_CACHE=1 bash build.sh

# Build with custom artifact
PLATFORM=ascend TARGET=ci bash build.sh \
  --build-arg TE_FL_ARTIFACT=https://example.com/te-fl-ascend.whl
```

### Updating TE-FL Version

**Option 1: Automatic (recommended)**
- Monitor workflow runs every 6 hours
- Creates PR when new TE-FL commit detected
- Review and merge PR

**Option 2: Manual**
```bash
# 1. Update versions.yaml
python .github/scripts/update_te_lock.py \
  --te-commit <new-commit-sha> \
  --platform musa \
  --artifact <artifact-url>

# 2. Commit and push
git add docker/versions.yaml
git commit -m "chore: update TE-FL to <short-sha> for MUSA"
git push

# 3. CI will build and test automatically
```

### Testing New TE-FL Version

```bash
# Trigger build for specific platforms
gh workflow run build_images.yml \
  -f platforms=musa,metax \
  -f force_rebuild=true

# Monitor build progress
gh run watch

# Check smoke test results
gh run view --log
```

## Validation Requirements

Every image MUST pass these checks:

1. **Build-time:**
   - Dockerfile builds successfully
   - TE-FL installs without errors
   - Smoke test passes

2. **Smoke test:**
   - `import transformer_engine` succeeds
   - Native module (`transformer_engine_torch`) loads
   - `TE_FL_COMMIT` and `TE_FL_PLATFORM` set
   - Basic operator (`Linear`) instantiates

3. **Runtime (in Megatron tests):**
   - Provenance verification passes
   - Unit tests run
   - Functional tests run

## Troubleshooting

### Build fails with "TE_FL_ARTIFACT not specified"

**Cause:** `versions.yaml` has empty `te_fl_artifact` for the platform.

**Fix:**
```bash
python .github/scripts/update_te_lock.py \
  --te-commit <current-commit> \
  --platform <platform> \
  --artifact <valid-artifact-url>
```

### Smoke test fails with import error

**Cause:** TE-FL not properly installed or native module missing.

**Debug:**
```bash
docker run --rm -it <image> bash
python3 -c "import transformer_engine; print(transformer_engine.__file__)"
python3 -c "import transformer_engine_torch"
```

### CI fails with "TE-FL provenance incomplete"

**Cause:** Image missing `TE_FL_COMMIT` or `TE_FL_PLATFORM` environment variables.

**Fix:** Rebuild image using `build.sh` (not manual `docker build`)

### Different TE-FL versions across platforms

**Expected behavior.** Each platform uses independently versioned TE-FL.

**To synchronize:**
```bash
# Get target commit
TARGET_COMMIT=abc123...def

# Update all platforms
for platform in cuda musa metax ascend hygon kunlunxin enflame; do
  python .github/scripts/update_te_lock.py \
    --te-commit "$TARGET_COMMIT" \
    --platform "$platform" \
    --artifact "harbor.baai.ac.cn/megatron-artifacts/te-fl-${platform}:${TARGET_COMMIT}"
done
```

## Migration Notes

### From Old Setup

**Old (per-platform manual):**
- Each platform config directly specified `ci_image`
- No TE-FL version tracking
- Manual image builds
- Inconsistent TE-FL sources

**New (unified):**
- `versions.yaml` as single source of truth
- Automated builds via `build.sh`
- TE-FL provenance tracking
- Automatic updates via monitor workflow

### Backward Compatibility

- Existing `ci_image` in `.github/configs/<platform>.yml` still works
- New images coexist with old images
- Gradual migration: update one platform at a time
- `verify_te_provenance.sh` warns but doesn't fail on old images (initially)

## Reference

- TE-FL Repository: https://github.com/FlagOpen/TransformerEngine-FL
- Harbor Registry: harbor.baai.ac.cn/megatron-ci
- Artifact Registry: harbor.baai.ac.cn/megatron-artifacts
