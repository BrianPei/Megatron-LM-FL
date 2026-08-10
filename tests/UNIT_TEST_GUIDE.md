# Unit Test Authoring Guide

This document covers unit-test code conventions. For workflow architecture,
platform configuration, container setup, and CI-equivalent commands, see
[`CI_TESTING_GUIDE.md`](CI_TESTING_GUIDE.md).

## Location and discovery

- Put tests under `tests/unit_tests/` in the directory that owns the behavior.
- Name files `test_*.py`, functions and methods `test_*`, and test classes
  `Test*` so pytest discovers them.
- Prefer extending the nearest existing test module instead of creating a new
  root-level test file for unrelated behavior.

Example:

```python
class TestMyFeature:
    def test_expected_behavior(self):
        assert run_feature() == "expected"
```

## CI group selection

Each platform declares unit groups under
`.github/configs/<platform>.yml:test_matrix.unit.groups`. A group has a unique
name and one or more repository-relative test paths:

```yaml
test_matrix:
  unit:
    nproc_per_node: 8
    groups:
      - name: models
        path: tests/unit_tests/models/
        description: Model tests
```

The common workflow expands top-level `device_types` and these groups. The
single `test_matrix.unit.nproc_per_node` value is passed to every group as
`CI_NPROC_PER_NODE`. Do not add another process-count or device field inside a
group.

Before adding a new group, verify that an existing path does not already
collect the test. Update every platform config on which the new test is
expected to run.

## Distributed test requirements

CI launches each group with `torch.distributed.run`, so every rank collects and
executes the same pytest selection. Distributed tests must:

- enter collectives and process-group creation in the same order on all ranks;
- destroy groups in fixture teardown, including exception paths;
- avoid rank-local skips after other ranks have entered a collective;
- use bounded timeouts for network, subprocess, and rendezvous operations;
- avoid multiplying large worker pools by the number of torchrun ranks; and
- use temporary paths whose ownership and synchronization are explicit.

A failure on one rank often appears on the others as a later rendezvous or
collective timeout. Diagnose the first rank-specific exception, not the final
`ChildFailedError` or SIGTERM cascade.

## Platform exclusions

`test_matrix.unit.ignored_tests` accepts either a complete test file or a
pytest node ID:

```yaml
ignored_tests:
  - tests/unit_tests/vendor_only/test_kernel.py
  - tests/unit_tests/test_optimizer.py::test_vendor_specific_path
```

The runner converts complete files to `--ignore` and node IDs to `--deselect`.
Use an exclusion only for a demonstrated unsupported backend boundary or a
tracked infrastructure limitation. Keep it as narrow as the shared failure
boundary and add a short evidence-based comment. Do not change Megatron core
behavior merely to hide a platform CI failure.

## Running tests

Use the platform container and setup described in
[`CI_TESTING_GUIDE.md`](CI_TESTING_GUIDE.md#running-tests-on-a-development-machine).
For a quick debug loop after setup:

```bash
python3 -m torch.distributed.run --nproc_per_node=8 \
  -m pytest -v tests/unit_tests/models/test_gpt_model.py
```

The process count must match `test_matrix.unit.nproc_per_node` for the target
platform. A direct pytest command does not apply the platform ignore list,
extra pytest arguments, coverage settings, or hard timeout. Run
`tests/test_utils/runners/run_ci_unit_tests.sh` before submitting a change.
