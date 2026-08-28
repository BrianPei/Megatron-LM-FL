# TE-FL Runtime Delivery

Megatron-LM-FL supports two TE-FL delivery modes. Platform configuration is the
single switch between them.

## Delivery Modes

### Prebuilt image (`image`)

This is the steady-state path. Unit, functional, and benchmark jobs start from
one immutable Harbor image that already contains the platform runtime and a
verified TE-FL installation.

Normal test jobs do not check out TransformerEngine-FL, compile a wheel,
download a native artifact, or reinstall TE-FL. Each job only validates the
image manifest, required runtime environment, bootstrap modules, and installed
TE-FL import before running the existing tests.

The image must be pinned by digest and contain `/etc/flagos/te-fl.json`. Its
configured TE-FL commit must match `te_fl_python_commit` in that manifest.
Strict device execution and backend selection are acceptance checks for image
publication, so they are not repeated in every test matrix job.

### Workflow artifact (`artifact`)

This is the compatibility and fallback path. The prepare job resolves TE-FL,
calculates a native fingerprint, restores or builds the native artifact,
installs the current Python overlay, and runs strict device/backend verification.
It then uploads the validated native directory as a short-lived workflow
artifact. Unit and functional jobs download and install that artifact before
running tests.

The native fingerprint excludes ordinary TE-FL Python files and includes
native sources, build files, toolchain, target architecture, Python/Torch ABI,
base image, and vendor runtime modules. Python-only changes reuse native output;
C++/CUDA/header/build or ABI changes generate a new fingerprint and rebuild.

GitHub cache accelerates native builds across runs. The workflow artifact is
the deterministic delivery mechanism between jobs in the same run.

## Common Flow

`.github/workflows/all_tests_common.yml` reads the platform configuration and
resolves three values:

- delivery mode (`artifact` or `image`)
- whether a TE-FL prepare job is required
- the immutable image used by test jobs

In artifact mode, the configured `ci_image` is both the build and test base
image. In image mode, tests use `te_fl.delivery.runtime_image`; the prepare job
is skipped for normal test runs.

`.github/workflows/te_fl_daily.yml` explicitly invokes the prepare path without
tests. It continues to resolve and validate TE-FL `main` independently of
normal image-mode test consumption. When the resolved commit differs from the
commit recorded for the configured runtime image, the daily run fails with an
explicit stale-image error after validating the latest runtime.

## Verification Boundary

Image publication must prove the full runtime contract on real hardware:

- manifest identity and TE-FL commit
- Python/Torch/vendor runtime ABI
- installed Python overlay
- device availability
- TE `Linear` forward and backward
- selected backend implementation ID

Normal image-mode tests recheck the immutable image identity and imports in the
actual test container. The package must resolve outside the checked-out
Megatron workspace, preventing a source tree from shadowing the image runtime.

## Scope

This implementation changes CI configuration, workflows, and environment
scripts only. It does not modify Megatron model or training source code. All
platforms remain in artifact mode until a verified Harbor image digest and its
matching 40-character TE-FL commit are added to that platform configuration.
