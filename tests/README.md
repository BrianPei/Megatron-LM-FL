# Megatron-LM Tests

## CI and Local Execution

See [`CI_TESTING_GUIDE.md`](CI_TESTING_GUIDE.md) for the configuration-driven
GitHub Actions architecture, adding unit/functional/benchmark cases, platform
integration, and CI-equivalent local container commands.

See [`UNIT_TEST_GUIDE.md`](UNIT_TEST_GUIDE.md) for unit-test naming and
organization conventions.

## Updating Functional Test Golden Values

Golden values must come from a complete, reviewed run on the declared target
environment and hardware. The regular and benchmark formats, local reproduction
commands, and review requirements are documented in
[`CI_TESTING_GUIDE.md`](CI_TESTING_GUIDE.md#adding-a-functional-test).

Do not refresh golden values solely to make an unexplained regression pass.
Performance values must be recorded on the target accelerator rather than
copied from another platform.
