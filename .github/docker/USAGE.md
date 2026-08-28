# TE-FL CI Usage

## Platform Configuration

Each `.github/configs/<platform>.yml` owns its TE-FL build, runtime, and delivery
contract:

```yaml
te_fl:
  ref: main
  delivery:
    mode: artifact
    runtime_image: ""
    expected_commit: ""
  native:
    mode: source
    source_paths: [transformer_engine]
    build_files: [setup.py, pyproject.toml, build_tools]
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

Non-CUDA platforms that must disable CUDA probing set
`TE_FL_SKIP_CUDA: "1"` in the configured TE-FL environment. The setting is
applied before TE-FL is imported or built.

## Artifact Mode

`mode: artifact` preserves the incremental workflow:

1. Resolve `te_fl.ref` to one full commit.
2. Restore or build `te-fl-native-<platform>-<fingerprint>`.
3. Install the current Python overlay and run strict hardware verification.
4. Upload the native directory once for downstream jobs in that run.
5. Each unit and functional job downloads and installs it.

`expected_te_fl_commit` can assert the resolved revision. `image_override`
overrides `ci_image` and must use `image@sha256:<64 lowercase hex>`.

GitHub caches are branch-scoped build accelerators. A PR cache mainly helps
reruns of that PR; shared entries come from default-branch or daily runs.

After a platform switches to image mode, normal PR tests no longer query
TE-FL `main`. The daily prepare run still resolves and validates `main`; it
fails if that commit differs from the commit recorded for the runtime image.
That failure is the signal to publish and pin a replacement image.

## Switch To Image Mode

After the platform image has passed strict device/backend verification and is
published to Harbor, update only its delivery block:

```yaml
delivery:
  mode: image
  runtime_image: harbor.baai.ac.cn/example/megatron-te-fl@sha256:<64-hex-digest>
  expected_commit: <40-hex-te-fl-commit>
```

The digest and commit are both mandatory. A tag-only image is rejected.
`expected_commit` must match `/etc/flagos/te-fl.json` in the image.

In image mode, `image_override` overrides `runtime_image`; it does not replace
the artifact build base image. `expected_te_fl_commit` is an optional runtime
assertion and overrides the configured expected commit for that manual run.

Expected normal-run logs must show that these artifact steps are skipped:

- `te_fl_prepare`
- `Checkout TransformerEngine-FL`
- `Download TE-FL native artifact`
- `Install TE-FL runtime`

The job still runs `Verify prebuilt TE-FL runtime image`, then the existing
unit or functional test command.

## Image Publication Contract

Before changing a platform to image mode, confirm the image contains:

- the platform Torch/vendor runtime and TE-FL runtime dependencies
- the intended TE-FL native build and Python overlay
- `/etc/flagos/te-fl.json` with schema version 2
- the runtime environment recorded in the platform configuration
- all configured bootstrap modules

The publication run must record a successful strict verifier result on real
hardware. For MUSA, the image must also import `onnxscript` and `torchada`, and
must not contain NVIDIA `flash-attn`, `flash-attn-3`, or `flash-attn-4`
distributions.

## Acceptance And Rollback

For the first image of each platform:

1. Run unit tests with the digest-pinned image.
2. Confirm manifest/import verification passes in every container.
3. Confirm no TE-FL artifact is downloaded or installed by test jobs.
4. Run the existing functional/benchmark matrix.
5. Keep the image digest and TE-FL commit together in the platform config.

If the image fails, restore `mode: artifact` and clear `runtime_image` and
`expected_commit`. The incremental artifact path remains available while the
image is rebuilt.
