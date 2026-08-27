# TE-FL Runtime Integration Verification - Pilot Implementation

## Overview

This implementation provides minimal integration verification that allows Megatron-LM-FL test workflows to:
1. Consume an externally-built TE-FL image by digest
2. Verify at runtime that TE-FL is actually being used
3. Test with a specific TE-FL commit expectation

## What Was Changed

### 1. Modified Workflows

**`.github/workflows/all_tests_common.yml`**
- Added `image_override` input (optional, must be `image@sha256:digest` format)
- Added `expected_te_fl_commit` input (optional, 40 hex chars)
- Modified config job to validate and resolve image (override takes precedence)
- Passes resolved image to both unit and functional test workflows
- Passes expected commit for runtime verification

**`.github/workflows/unit_tests_common.yml`**
- Added `expected_te_fl_commit` input
- Added `MEGATRON_CI_IMAGE` env var to setup step
- Added "Verify TE-FL runtime" step after platform setup
- Verification runs before unit tests execute

**`.github/workflows/functional_tests_common.yml`**
- Added `expected_te_fl_commit` input
- Added `MEGATRON_CI_IMAGE` env var to setup step
- Added "Verify TE-FL runtime" step after platform setup
- Verification runs before functional tests execute

**`.github/workflows/all_tests_musa.yml` (pilot platform)**
- Added `workflow_dispatch` inputs:
  - `image_override`: Optional immutable image reference
  - `expected_te_fl_commit`: Expected TE-FL commit SHA
  - `run_unit_tests`: Boolean to control unit test execution
  - `run_functional_tests`: Boolean to control functional test execution
- Passes all inputs through to common workflow

### 2. New Runtime Verification Script

**`.github/scripts/verify_te_runtime.sh`**

Runs inside the test container after platform setup to verify:

1. **Resolved container image** - reports the input image reference
2. **transformer_engine installation** - path and version
3. **Native extension presence** - verifies native module loads
4. **TE-FL commit match** - compares actual vs expected (if provided)
5. **Platform implementation registration** - checks TE_FL_PLATFORM env var
6. **Operator backend selection** - verifies Linear operator instantiation
7. **Platform-specific checks** - CUDA version, MUSA availability, etc.

The script:
- Exits 0 on success (all checks pass)
- Exits 1 on failure (missing TE, wrong commit, etc.)
- Reports detailed information for each check
- Is skipped gracefully if the file doesn't exist (backward compatibility)

## How to Use

### Testing with FlagScale-Built Image

To test a validated FlagScale image with known TE-FL commit:

```bash
gh workflow run all_tests_musa.yml \
  -f image_override='harbor.baai.ac.cn/flagscale/megatron-musa@sha256:<64-hex-digest>' \
  -f expected_te_fl_commit='<40-hex-commit-sha>' \
  -f run_unit_tests=true \
  -f run_functional_tests=true
```

**Example with real values:**
```bash
gh workflow run all_tests_musa.yml \
  -f image_override='harbor.baai.ac.cn/flagscale/megatron-musa@sha256:abc123...def456' \
  -f expected_te_fl_commit='1234567890abcdef1234567890abcdef12345678' \
  -f run_unit_tests=true \
  -f run_functional_tests=false
```

### Testing Without Override (Use Platform Config)

```bash
gh workflow run all_tests_musa.yml \
  -f run_unit_tests=true \
  -f run_functional_tests=true
```

This uses the `ci_image` from `.github/configs/musa.yml`.

### Expected Verification Output

When the workflow runs, the "Verify TE-FL runtime" step will output:

```
>>> TE-FL Runtime Verification
===========================================
>>> 1. Resolved Container Image
   Input image: harbor.baai.ac.cn/flagscale/megatron-musa@sha256:...
   Hostname: runner-xyz
>>> 2. transformer_engine Installation
   Path: /usr/local/lib/python3.10/site-packages/transformer_engine/__init__.py
   Version: 1.0.0
>>> 3. Native Extension
   Module: transformer_engine_torch
   Path: /usr/local/lib/python3.10/site-packages/transformer_engine_torch.so
>>> 4. TE-FL Commit Verification
   Expected commit: 1234567890abcdef1234567890abcdef12345678
   Actual commit: 1234567890abcdef1234567890abcdef12345678 (MATCH)
>>> 5. Platform Implementation Registration
   TE_FL_PLATFORM: musa
   Has TE operators: True
   Torch CUDA available: True
   Device count: 8
   Device name: Moore Threads MTT S5000
>>> 6. Operator Backend Selection
   Linear operator imported successfully
   Linear layer instantiated: in_features=4, out_features=8
   Layer type: transformer_engine.pytorch.Linear
>>> 7. Platform-Specific Checks
   Platform: musa
   MUSA available: True
===========================================
>>> SUCCESS: TE-FL runtime verification passed
```

## Verification Failure Scenarios

The script will **fail** if:

1. `transformer_engine` cannot be imported
2. Native extension (`transformer_engine_torch` or `transformer_engine.pytorch`) is missing
3. Expected commit is provided but doesn't match `TE_FL_COMMIT` env var
4. `TE_FL_COMMIT` env var is not set when expected commit is provided
5. Linear operator cannot be imported or instantiated

## Image Requirements

For a FlagScale image to pass verification:

1. **Must have** `transformer_engine` installed and importable
2. **Must have** native extension (`transformer_engine_torch` or `transformer_engine.pytorch`)
3. **Should have** `TE_FL_COMMIT` environment variable set (for commit verification)
4. **Should have** `TE_FL_PLATFORM` environment variable set (for platform reporting)
5. **Must allow** Linear operator instantiation

## Validation Status

### Local Static Validation ✅

- [x] Shell script syntax (`bash -n`)
- [x] YAML syntax (Python yaml.safe_load)
- [x] Git whitespace check
- [x] Executable permissions

### What Was NOT Implemented

Per the requirement to provide a minimal pilot:

- ❌ No TE-FL source compilation
- ❌ No Docker image building in Megatron repo
- ❌ No cross-repository changes
- ❌ No automatic image promotion
- ❌ No modification to megatron/ core code
- ❌ No push (changes are local only)

### Hardware Validation Blocker

**Cannot execute runtime verification without:**
1. A FlagScale-built image with TE-FL installed
2. Access to MUSA hardware runner
3. The image reference and commit SHA

**Next step to unblock:** Provide a validated FlagScale image reference and its TE-FL commit.

## Files Changed

```
M .github/workflows/all_tests_common.yml       (added image_override and expected_te_fl_commit inputs)
M .github/workflows/unit_tests_common.yml      (added verification step)
M .github/workflows/functional_tests_common.yml (added verification step)
M .github/workflows/all_tests_musa.yml         (added workflow_dispatch inputs - PILOT)
A .github/scripts/verify_te_runtime.sh         (new runtime verification script)
A PILOT_IMPLEMENTATION.md                      (this document)
```

## Exact Command for FlagScale Team

When you have a validated image, run:

```bash
gh workflow run all_tests_musa.yml \
  -f image_override='<your-image>@sha256:<digest>' \
  -f expected_te_fl_commit='<40-char-sha>' \
  -f run_unit_tests=true \
  -f run_functional_tests=false
```

Replace:
- `<your-image>@sha256:<digest>` with the full immutable image reference
- `<40-char-sha>` with the TE-FL commit SHA used in that image

The workflow will:
1. Use your provided image instead of the config default
2. Run platform setup
3. Verify TE-FL is installed and matches the expected commit
4. Run unit tests (or functional tests, as configured)

If verification fails, the workflow will fail before running tests, showing you exactly what's missing or mismatched.
