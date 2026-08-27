# TE-FL Incremental Runtime

Megatron-LM-FL resolves TransformerEngine-FL (TE-FL) for every hardware test
run. The default ref is `main`; CI resolves it once to a full commit and passes
that immutable commit to every unit, functional, and benchmark job.

## Build Model

The runtime has two identities:

- `te_fl_python_commit`: the resolved TE-FL commit used by the test.
- `native_fingerprint`: the native build and ABI inputs used by the cached
  wheel.

The fingerprint deliberately excludes ordinary TE-FL Python files. It includes
native source files, build files and environment, target architecture,
compiler/tool versions, Python SOABI, Torch versions, and hashes of configured
platform runtime modules and their shared libraries. Runtime installation and
verification script changes do not invalidate the native artifact.

A Python-only TE-FL change therefore reuses the native artifact and overlays the
current Python package files. A native source, build configuration, toolchain,
ABI, or platform runtime change produces a different cache key and rebuilds the
wheel.

## CI Flow

`.github/workflows/all_tests_common.yml` calls
`.github/workflows/prepare_te_fl.yml` on the platform runner and test image.
The prepare job resolves TE-FL, calculates the fingerprint, restores or builds
the native artifact, and validates its manifest and checksums.

Every unit and functional matrix job restores the same cache key, installs the
wheel, overlays Python files from the resolved TE-FL commit, and runs strict
runtime verification before tests. Benchmark cases are entries in the existing
functional training matrix, so they use the same runtime path.

`.github/workflows/te_fl_daily.yml` runs the prepare path on the default branch
without tests. It detects independent TE-FL `main` updates and creates a
default-branch cache entry that later pull requests can restore.

## Verification

The strict verifier checks the runtime manifest, installed TE-FL module, device
availability, backend selection environment, TE `Linear` forward/backward, and
the implementation selected for `generic_gemm` by the TE-FL manager.

If the installed TE-FL version cannot report the selected implementation,
verification fails. The real device forward/backward must complete before the
implementation ID is accepted.

## Scope

This implementation changes CI configuration, workflows, and installation
scripts only. It does not modify Megatron model or training source code.

Local syntax checks do not prove that every vendor image can build the latest
TE-FL commit. The first hardware CI run for each platform is the required
evidence for native build, cache restore, device execution, and backend ID.
