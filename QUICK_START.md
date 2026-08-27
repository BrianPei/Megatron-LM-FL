# TE-FL Runtime Integration - Quick Reference

## What Changed

4 workflows modified + 2 new scripts + 3 docs = **Minimal pilot for testing FlagScale images**

## Test Command

```bash
gh workflow run all_tests_musa.yml \
  -f image_override='harbor.baai.ac.cn/flagscale/megatron-musa@sha256:YOUR_DIGEST_HERE' \
  -f expected_te_fl_commit='YOUR_40_CHAR_COMMIT_SHA_HERE' \
  -f run_unit_tests=true \
  -f run_functional_tests=false
```

## What Happens

1. Workflow validates image format (must be `@sha256:digest`)
2. Uses your image instead of default config
3. Runs platform setup
4. **Verification step runs** → checks TE-FL installation, version, commit, native extension
5. If verification passes → unit tests execute
6. If verification fails → workflow fails before tests, showing what's wrong

## Files to Review

- `IMPLEMENTATION_REPORT.md` - Complete details
- `PILOT_IMPLEMENTATION.md` - Usage guide with examples
- `.github/scripts/verify_te_runtime.sh` - The verification logic
- `.github/workflows/all_tests_common.yml` - Image resolution
- `.github/workflows/all_tests_musa.yml` - Pilot with dispatch inputs

## Validation

Run: `bash validate_pilot.sh`

Expected: All 6 checks pass

## Status

✅ Implementation complete
✅ Local validation passed
⏸️ Hardware execution blocked (need FlagScale image reference)
📝 Not committed/pushed per requirements
