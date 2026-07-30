Megatron-LM Unit Test Authoring Guide
=====================================

This document explains how to add and integrate unit tests under `tests/unit_tests` so that they are easy to run locally and correctly discovered and executed by CI.

Directory layout and naming conventions
---------------------------------------

- **Location**: All unit tests live in `tests/unit_tests` and its subdirectories, for example:
  - `tests/unit_tests/test_basic.py`
  - `tests/unit_tests/transformer/test_attention.py`
  - `tests/unit_tests/models/test_gpt_model.py`
- **File names**:
  - Use the `test_xxx.py` naming convention (so that `pytest` can auto-discover them).
  - For an existing module, prefer adding tests in the corresponding subdirectory instead of dropping everything into the root.
- **Test names**:
  - Function names must start with `test_`, for example:

    ```python
    def test_my_feature():
        ...
    ```

  - Or use a class whose name starts with `Test`, and define `test_` methods inside:

    ```python
    class TestMyFeature:
        def test_case_1(self):
            ...
    ```

Workflow config and `.github/configs`
-------------------------------------

Unit-test GitHub workflows are parameterized by platform configs under
`.github/configs/`, for example `cuda.yml`, `metax.yml`, and `hygon.yml`.
Each config defines its image, runner labels, container settings, device
matrix, unit groups, pytest arguments, and platform-specific exclusions.

The complete configuration contract and platform setup flow are documented in
[`CI_TESTING_GUIDE.md`](CI_TESTING_GUIDE.md#platform-configuration-contract).

Use `test_matrix.unit.ignored_tests` only for a narrow, documented platform
limitation. A complete file path is converted to `--ignore`; a pytest node ID
containing `::` is converted to `--deselect`. Prefer fixing a test or using an
appropriate repository marker when the limitation is not platform-specific.

Running unit tests locally
--------------------------

For a CI-equivalent run, use the configured container, platform setup script,
and `tests/test_utils/runners/run_ci_unit_tests.sh` as documented in
[`CI_TESTING_GUIDE.md`](CI_TESTING_GUIDE.md#run-a-unit-test-group-with-ci-parity).

After that setup succeeds, this shorter command is useful for debugging a
single file:

```bash
python3 -m torch.distributed.run --nproc_per_node=8 \
  -m pytest -v tests/unit_tests/xxx.py -p no:randomly
```

The direct command does not apply the platform ignore list, group arguments,
coverage settings, or hard timeout. Run the CI runner before submission.

When in doubt, find a similar existing test in the tree and follow the same style and patterns.
