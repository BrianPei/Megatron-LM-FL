# Multi-platform CI

`ci.yml` runs lint, selects platforms, executes their pipelines, and reports the
required `CI Complete` gate. PRs target `main` or `devops`. Common pre-commit and
helper checks run once before the platform matrix, including requested tests.
All manual platform entrypoints reuse `_platform_pipeline.yml`, preserving image
and test-scope inputs. Every entrypoint runs common checks once before invoking
the platform pipeline; platform workflows contain no duplicate lint job. The shared test gate reports the platform result.

## Platform selection

- `paths_watch` accepts exact files and directory prefixes ending in `/`.
- Vendor-only changes run the affected platforms; shared or unclassified paths
  select all platforms, and the unit plan then blocks unknown paths until they
  are onboarded.
- Documentation (`docs/`, `.md`, `.rst`) skips hardware tests but still reports
  the gate. Documentation-only PRs never start hardware jobs, even when a
  `platform:<name>` label is present.
- Missing diffs/watch rules select all platforms; invalid configs fail.
- Renames include both old and new paths. Push/schedule/manual CI runs select all.
- Labels `platform:<config-name>` override path routing, matching TE, but only
  for PRs that contain at least one code change. Multiple labels take their
  union; `platform:all` selects every platform. Removing all platform labels
  restores automatic routing. Narrow shared-code changes after review.
- A functional case shared by several platforms must appear in all their watch lists.

## Unit group routing

`.github/configs/unit_group_routes.yml` defines logical groups, their source and
test paths, dependencies, resource classes, status (`active`, `planned`, or
`unsupported`), and the groups required by each level. Platform configs remain
authoritative: each declares the execution groups it runs and must declare every
other active group in `unsupported_groups` with a reason. A platform may also
declare an active group unsupported when its execution group exists but does not
collect every test path of that group; the validator rejects a declaration only
when the execution group already covers all of them. A group may set an
`execution_group` alias (for example, `plugin` executes as `core`); the alias
must target an existing active catalog group, and a required level is checked
against that effective execution group. Catalog path ownership must be
unambiguous: the validator fails when a tracked path matches more than one
logical group.
`generate_test_plan.py` combines the catalog with the platform router and each
selected platform's config, writes `unit-test-plan.json` and an Actions summary,
and emits the per-platform execution matrix consumed by `ci.yml`.

The plan is the affected-only gate for pull requests. It records
`direct_groups` (logical groups matched directly by a changed path),
`direct_gaps` (direct groups no selected platform actually executes),
`fallback_excluded` (active groups a planned-group fallback could not run), and a
`plan_status` of `ready`, `partial`, or `blocked`:

- `ready`: every selected direct group executes and no planned/unsupported
  fallback was involved. Documentation-only plans are `ready` with no platforms.
- `partial`: the plan executes, but it relied on a shared/fallback selection,
  `selection=all`, or a planned/unsupported/fallback-excluded group. A partial
  plan is not proof of a complete logical-group regression.
- `blocked`: no unit matrix is emitted and `generate-matrix` fails.

Blocking conditions:

- A directly affected group is `planned` or declared unsupported on every
  selected platform. Planned fallback is allowed only when the planned group is
  not the direct match.
- A changed path is unknown or matches more than one logical group (route
  conflict). Unknown paths still produce a diagnostic full fallback plan, but
  the plan is `blocked` and the fallback is not executed until catalog
  onboarding is completed.
- A missing required group, an unresolved/blocked group, or a platform with no
  executable groups.
- A shared path falls back to every logical group and executes every active
  group the platform can run. This is allowed but marked `partial`.

A selected planned group is not executed. The platform instead runs every
active group it can execute, so a known gap cannot narrow coverage; groups it
cannot run are listed under `fallback_excluded`.

Plan identity is carried end to end: `ci.yml` exports `plan_hash` and
`expected_unit_groups`, `_platform_pipeline.yml` and `all_tests_common.yml`
forward them, and `run_unit_groups.py` writes them to
`coverage-report/unit-results.json` together with `executed_groups`,
`missing_groups`, `unexpected_groups`, and `status`. The per-platform workflow
verifies this attestation, and `ci-complete` re-verifies that every planned
`(platform, device)` reports the same `plan_hash`, executes exactly its expected
groups, and has `status=passed`.

`resource_class`, `timeout_minutes`, and `parallel_safe` are currently advisory
metadata only. A platform still runs one unit job per platform/device, groups
execute sequentially, the group timeout is fixed by the runner, and there is no
exclusive lock or group-level parallel scheduling.

Coverage sources in the catalog (`megatron/plugin/`, `megatron/core/`) are the
unit coverage boundary; this is core/plugin coverage, not full Megatron
unit-group coverage. Artifact names such as `coverage-<platform>-<device>` are
transport labels and make no claim about logical-group completeness.
`validate_unit_group_catalog.py` checks path existence against the git index,
dependency acyclicity, platform group names, alias targets, owner and
unsupported declarations, required-group execution through aliases, tracked-path
overlaps between logical groups, and coverage of every `tests/unit_tests/`
directory.

## Prepared wheels

Preparation uploads a platform/attempt-specific wheel artifact, even on a cache
hit. Consumers download that exact artifact from the same run; missing artifacts
fail the job. Test interfaces require a nonempty artifact name, with no separate
preparation switch. Only preparation uses the cross-run cache; consumers have no cache
fallback or local backup. Cache-save failure cannot block artifact delivery.

## Retry policy

Platform configs explicitly enable `test_retry: {transient_retries: 1}`. Omission
means zero retries. Unit groups and functional cases can override the policy with
`transient_retries: 0` or `1`; `n_repeat` remains training repetitions.

The shared runner `.github/scripts/run_unit_groups.py` retries recognized network
failures once, with complete pytest JUnit reports confirming network failures.
Each unit rank/phase has a separate report; functional validations are numbered.
Missing, malformed, or incomplete reports disable automatic retries. Assertions,
golden/loss mismatches, invalid imports/configuration,
OOM, timeout and cancellation are not retried. The unit dataset downloader retries
network failures and HTTP 502/503/504 once, including interrupted streams, with
partial archives replaced and bounded connection/read timeouts. HTTP 404 and
extraction failures are not retried. Setup yq downloads allow two attempts.
Failed coverage/training diagnostics are archived before retrying; checkpoint
directories and model weight files are discarded instead of uploaded.
Artifacts `unit-retry-*` / `functional-retry-*`
retain attempts; summaries distinguish `PASS`, `FLAKY_PASS`, and `FAIL`.
Successful unit groups remain reusable when rerunning failed jobs.
Each attempt allows 60 minutes; functional steps allow 125 minutes for two attempts.

## Comment commands

`ci-requested.yml` handles both PR comments and manual requests:

```text
/ci run ascend
/ci run cuda,ascend
/test ascend
/ci run all
/ci rerun failed
```

Only repository writers/maintainers/admins can request tests on open PRs targeting
`main` or `devops`. Exact commands are required; comment text is never shell code.
The control/report jobs run default-branch code. Test jobs run the exact PR head
SHA, including fork PRs, with read-only permissions and no inherited secrets.
Requested tests publish a separate `CI Requested Tests` check on that SHA; they
never replace `CI Complete`. Old results cannot pass a newer commit.
`/ci rerun failed` restarts only the latest failed automatic CI run for the current
PR commit, preserving the original gate and successful unit-group checkpoints.

Comment handling activates only after these files reach the default branch.
Platforms accept commas/spaces and are sorted/deduplicated, matching TE.
Identical PR/SHA/platform-set requests reuse the running check. Validation is
serialized per PR, and platform execution is serialized per PR/SHA/platform.
Both locks use `queue: max` so distinct pending requests are not replaced.
Normal cancellation completes the check. A force-cancelled reporter may leave a
check pending; the next identical request closes that stale check before starting.
Do not make `CI Requested Tests` a required merge check.

## Coverage reports

Unit coverage (`megatron/plugin/*`, `megatron/core/*`) is combined into
`coverage-report/coverage-<platform>-<device>-all.json` and uploaded as a GitHub
artifact and to FlagCICD via `flagos-ai/FlagOps/actions/post-pytest-report@v2`.

Functional training tests retain their existing execution, retry logs and test
artifacts. Performance smoke report generation and upload are not included.

Configure `FLAGCICD_API_TOKEN` and runner access to
`https://flagcicd.flagos.net/flagcicd/api/v1/cicd/metrics/`. Automatic/manual entrypoints
forward this named secret through the reusable workflows. Fork PRs and requested
tests (`/ci run`) receive no token and retain GitHub artifacts. Backend upload
is skipped with a notice when the token or combined report is missing, and is
limited to two minutes otherwise. A failed test may still upload its report if
one was generated. Upload failures never override test results. Verify live upload
and backend parsing in a repository CI run with the token configured.

## Validation

```sh
python -m pip install PyYAML==6.0.3 coverage==7.6.1 pytest==8.3.5 requests==2.32.5 click==8.1.8
python .github/scripts/ci/validate_unit_group_catalog.py
python -m unittest discover -s .github/scripts/ci -p 'test_*.py'
python -m unittest discover -s .github/scripts/tests -p 'test_*.py'
```

Before requiring `CI Complete`, verify real Actions runs for docs-only, vendor-only,
mixed/shared changes, failure, cancellation, label changes and comment commands.
Use a test base already containing this CI; the PR introducing these CI files
selects all platforms. Include a fork PR, an unauthorized comment, a push during
requested tests, and a transient failure followed by success. Intentional failures
belong in disposable branches. Local tests do not prove hardware/Actions behavior.
