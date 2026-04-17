# Description

Simplifies and consolidates the coverage report generation logic in the CI workflow, reducing redundant steps and dependencies.

## Type of change

- [x] New feature (non-breaking change which adds functionality)
- [x] Infra/Build change (changes to CI/CD workflows or build scripts)
- [ ] Code refactoring
- [ ] Documentation change
- [ ] Bug fix
- [ ] Breaking change

## Changes

- Merged `Generate Coverage Report` into the `Execute Tests` step — coverage `combine` and `json` generation now run inline after `bash test.sh`, following the same pattern as Megatron-LM-FL
- Coverage collection is gated on `test_type == 'unittest'` to avoid running for lint/debug groups, and `pip install` is done only once
- Removed `fetch-depth: 0` from checkout steps (not required for unit test runs)
- Removed unused/leftover scripts from the repository

## Checklist

- [x] I have read and followed the contributing guidelines
- [x] The functionality is complete
- [x] I have commented my code, particularly in coverage report uploading steps
- [x] I have made corresponding changes to the documentation
- [x] My changes generate no new warnings
- [x] I have added/updated tests that prove my feature works on Cuda and Metax platform
- [x] New and existing unit tests pass locally on Cuda and Metax platform
