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
    compiler: nvcc
  runtime:
    expected_backend: default.flagos
    device_module: torch.cuda
    bootstrap_modules: []
    native_module: transformer_engine
    environment:
      TE_FL_PREFER: flagos
    install_pip_args: []
```

Non-CUDA platforms that must disable CUDA probing set
`TE_FL_SKIP_CUDA: "1"` in the configured TE-FL environment. The setting is
applied before TE-FL is imported or built.

## Artifact Mode

`mode: artifact` preserves the incremental workflow:

1. Resolve `te_fl.ref` to one full commit.
2. Restore or build `te-fl-native-<platform>-<fingerprint>`.
3. Install the current Python overlay and run strict hardware verification once.
4. Upload the native directory once for downstream jobs in that run.
5. Each unit and functional job downloads and installs it. This is the
   compatibility path and is not the steady-state image path.

`expected_te_fl_commit` can assert the resolved revision. `image_override`
overrides `ci_image` and must use `image@sha256:<64 lowercase hex>`.

GitHub caches are branch-scoped build accelerators. A PR cache mainly helps
reruns of that PR; shared entries come from default-branch or daily runs.

In image mode, the normal PR still runs one `te_fl_prepare` job to query and
strictly validate the latest TE-FL `main`. That job is a watcher; its temporary
Python overlay and native cache are not passed to the test matrix. The watcher
also performs a lightweight manifest check for the fixed image. Unit and
functional jobs consume only the configured digest-pinned runtime image and do
not install or verify TE-FL again. If the resolved commit differs from the
commit recorded for the runtime image, `te_fl_gate` stops the expensive test
matrix and reports a stale-image failure.

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

In image mode, `image_override` replaces the configured runtime image for both
the prepare watcher and test jobs. In artifact mode, it replaces `ci_image` for
the prepare and test jobs. `expected_te_fl_commit` is an optional assertion and
overrides the configured expected commit for that manual run.

Expected image-mode test-job logs must show that these TE-FL steps are skipped:

- `Checkout TransformerEngine-FL`
- `Download TE-FL native artifact`
- `Install TE-FL runtime`
- strict TE-FL runtime verification

The job starts from the configured runtime image and runs the existing unit or
functional test command. The `te_fl_prepare` and `te_fl_gate` jobs are visible
once per workflow and own the latest-TE-FL watch and freshness decision.

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

1. Run a dedicated image acceptance check on real hardware for the candidate digest.
2. Confirm manifest/import/device/backend verification passes for the image itself.
3. Run the prepare watcher against the same digest and confirm the latest TE-FL overlay passes.
4. Run unit tests with the same digest-pinned image.
5. Confirm no TE-FL artifact is downloaded or installed by image-mode test jobs.
6. Run the existing functional/benchmark matrix.
7. Keep the image digest and TE-FL commit together in the platform config.

If the image fails, restore `mode: artifact` and clear `runtime_image` and
`expected_commit`. The incremental artifact path remains available while the
image is rebuilt.
