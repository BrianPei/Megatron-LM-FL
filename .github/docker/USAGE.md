# TE-FL CI Usage

## Normal Runs

Platform workflows call `all_tests_common.yml`. No TE-FL input is required:

```yaml
jobs:
  run_tests:
    uses: ./.github/workflows/all_tests_common.yml
    with:
      platform: musa
      run_unit_tests: true
      run_functional_tests: true
```

The common workflow resolves TE-FL `main` and uses the resolved commit for all
test jobs in that run.

The scheduled `te_fl_daily.yml` workflow resolves the same ref and warms native
cache entries on the default branch. It can also be started manually when a
TE-FL native change needs immediate validation.

## Pin A Revision

Set `te_fl_ref` to a branch or full commit when reproducing a failure.
`expected_te_fl_commit` is an optional assertion; when set, resolution must
produce exactly that full commit.

## Platform Configuration

Each `.github/configs/<platform>.yml` contains a `te_fl` block:

```yaml
te_fl:
  ref: main
  native:
    mode: source
    source_paths: [transformer_engine]
    build_files: [setup.py, pyproject.toml, CMakeLists.txt, build_tools]
    runtime_modules: [torch]
    require_shared_library: true
    target_arch: sm_80
    build_pip_args: []
    environment:
      NVTE_FRAMEWORK: pytorch
    compiler: gcc
  runtime:
    expected_backend: vendor.cuda
    device_module: torch.cuda
    bootstrap_modules: []
    native_module: transformer_engine
    environment:
      TE_FL_PREFER: vendor
    install_pip_args: []
  ```

`native.environment` is applied before fingerprint imports and native wheel
builds. Include `TE_FL_SKIP_CUDA: "1"` for non-NVIDIA platforms when TE-FL
must not probe the CUDA vendor backend; do not set it for CUDA.

`build_pip_args` are used only when a cache miss builds the native wheel.
`install_pip_args` are used by unit and functional jobs when installing that
cached wheel. Keep these lists platform-specific when the image's Python
version requires pip compatibility flags.

`runtime_modules` must name importable modules that carry the platform ABI.
Their module files and package shared libraries participate in the fingerprint.
`bootstrap_modules` are imported before `transformer_engine` in the strict
runtime verifier so vendor torch extensions can establish their device APIs.

Set `require_shared_library: true` only when the TE-FL wheel itself must contain
a native extension. Vendor plugin platforms may use `false` when native kernels
come from the vendor runtime, but they still must pass device execution and
actual backend verification.

## Cache Behavior

The cache key is `te-fl-native-<platform>-<native_fingerprint>`.

- Python-only TE-FL change: same fingerprint, cache hit, Python overlay update.
- C++/CUDA/header/build-file change: new fingerprint, wheel rebuild.
- Torch/vendor runtime/toolchain change: new fingerprint, wheel rebuild.
- Corrupt or mismatched artifact: checksum or manifest failure before tests.

## Hardware Acceptance

For each platform, confirm one cache-miss run and one cache-hit rerun. Evidence
must include the resolved TE-FL commit, fingerprint, artifact manifest, runtime
verification output, and the existing unit/functional/benchmark result.
