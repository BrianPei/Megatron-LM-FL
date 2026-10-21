# Multi-platform CI

`ci.yml` runs lint, selects platforms, executes their pipelines, and reports the
required `CI Complete` gate. PRs target `main` or `devops`. Common pre-commit and
helper checks run once before the platform matrix, including requested tests.
All manual platform entrypoints reuse `_platform_pipeline.yml`, preserving image
and test-scope inputs. Every entrypoint runs common checks once before invoking
the platform pipeline; platform workflows contain no duplicate lint job. The shared test gate reports the platform result.

## L0-L4: training risk and declared platform capabilities

| Level | Goal | Registered execution |
| --- | --- | --- |
| L0 | Valid configuration and plan | Changed-file lint, CI helper tests, manifest/schema, routing, entry paths, resource/dependency contracts |
| L1 | Correct components | Hardware-backed unit groups for models, training, plugins, data and distributed components |
| L2 | Usable training pipeline | Small-model short training, loss/golden checks and representative TP/PP; save/resume must be explicitly onboarded |
| L3 | Stable combinations and regression | Explicitly onboarded parallel/optimizer/overlap combinations, compatibility, repeat/stress and calibrated performance checks |
| L4 | Model/scale/release acceptance | Explicitly onboarded long training, model/scale and candidate package/image acceptance |

L1 is not a CPU-only tier. No blanket CPU claim is made for `tests/unit_tests`.
L0 contract checks do not prove mounted datasets, tokenizer resources or runners
are available; existing hardware setup still validates actual resources.

Every existing unit group has `ci_level: L1`; every existing functional case has
`ci_level: L2` in `.github/configs/*.yml`. Classification granularity is a unit
group or functional case, per configured device. Existing pytest exclusions,
markers, experimental passes and transient retry policies remain in effect.
Upstream tests outside these platform configurations are not implicitly supported.

[project.yaml](../../../.flagcicd/project.yaml) declares all five levels and a frozen
`migration_baseline` of the original entry IDs per platform. Definitions remain
in platform YAML; the baseline stores identities only. New entries require an
explicit valid `ci_level`, and do not automatically expand the PR baseline.
Missing baseline entries/platforms, duplicate IDs, overlapping unit-group paths,
missing setup/training/config/golden files and invalid configuration fail L0.
Changing the baseline requires an explicit reviewed configuration change.

L3/L4 are defined, but **no platform currently has registered L3/L4 entries**.
Manual requests for those levels fail explicitly. Register real tests, resource
requirements and pass criteria before declaring capability; a lower-tier pass
must never be described as complete L3/L4 acceptance. Calibration and hardware
validation are required before introducing performance gates.

## Default PR diff routing and supplemental requests

Automatic PR CI routes platforms by changed paths by default, with no label
required. Each selected platform runs its full frozen baseline. Push CI retains
its existing event/path filters. The legacy `ci:diff` label is no longer required
and has no effect on selection:

- `paths_watch` supports exact paths and directory prefixes ending in `/`.
- Vendor-only changes select affected platforms and their baseline entries.
- Shared/unknown paths, missing watch rules, absent or failed diffs fall back to all platforms.
- Documentation-only PRs omit hardware tests, while L0 still runs.
- Renames include old and new paths. Platform labels add to the result.
- `platform:cuda` cannot reduce shared-code validation to CUDA; `platform:all` adds all platforms.
- Runtime merge-scope reduction labels are disabled, including `ci:scope*` and
  `ci:reduce*`. Ordinary labels never authorize narrowing. Maintainers can request
  separate supplemental runs without changing `CI Complete`.

The new entry supports manual `workflow_dispatch` and `repository_dispatch`
(`event_type: flagcicd`) with `checkout_ref`, `platforms` (JSON array or `["all"]`),
`mode`, `test_level`, `task_id`, and `parent_run_id`. Target refs must contain
the classified platform configuration; legacy refs without `ci_level` fail L0.

| Mode | Selection and meaning |
| --- | --- |
| baseline (default) | Frozen baseline; level input is ignored; all platforms by default |
| level | Cumulative registered entries through L0-L4; requested hardware tier must actually exist on every selected platform |
| functional | Only registered functional cases through the chosen level, after L0 and necessary TE/runtime setup; always supplemental |

A selected-platform manual run is supplemental; it cannot complete the automatic
merge check. Functional-only runs skip unit tests but retain TE preparation,
resource setup and functional execution. Existing `/ci run` and `/test` commands
remain separate `CI Requested Tests` checks, never replacements for `CI Complete`.
`ci-requested.yml` does not inherit upload credentials.

## Nightly and weekly

`flagcicd-entry.yml` owns both schedules; `ci.yml` has no duplicate cron.

- Nightly: 02:00 Asia/Shanghai daily, all configured platforms, declared supported
  entries through L3.
- Weekly: Sunday 02:00 Asia/Shanghai, all configured platforms, declared supported
  entries through L4. This overlaps the daily run by design; both represent their
  independent requested scopes.

Neither schedule uses PR diffs or platform labels. Until L3/L4 suites are onboarded,
both run existing L1/L2 capabilities and record requested level, actual validated
level and capability gaps per platform. A supported-scope pass is not full L3/L4
or release acceptance. Schedules activate on the default branch, not merely when
this branch is pushed. L4 release jobs must eventually validate the actual
candidate artifact, not substitute a source checkout for artifact acceptance.

## Plans, reports and final gates

The planner validates the entire registered inventory and freezes the actual
source SHA, mode, selection reasons, scope, config hashes, required task IDs,
existing exclusions and unsupported capability gaps. That plan selects the exact
unit/functional matrices consumed by the existing platform pipeline.

The unit runner already writes per-group `unit-results.json`; reports check those
records for each device. Functional jobs now upload individual execution receipts
bound to run ID, attempt, source SHA, platform and case. A receipt is produced even
when training fails or is skipped. Missing receipts, skipped/cancelled tasks,
nonzero group exits, duplicate/stale reports and omitted required tasks cannot pass.
Successful receipts/reports from an earlier attempt of the same run remain reusable
when rerunning failed jobs. Future-attempt results are rejected.

Automatic `CI Complete` checks both upstream job status and the frozen plan's
required reports/tasks. The manual/periodic entry checks the same contract, with
`validation_scope` distinguishing baseline, selected level, supported capability
and supplemental results. Summaries distinguish requested `test_level` from actual `validated_level`
(L0 when a documentation-only PR requires no hardware), and
include `merge_acceptance`; a supplemental
or periodic pass is never promoted to baseline merge acceptance. Functional-only
reports leave `validated_level` empty; their executed task IDs describe the partial
verification rather than claiming complete L2/L3 acceptance.

Artifacts retained for 14 days include `flagcicd-plan`,
`flagcicd-report-<platform>`, `flagcicd-run-summary`, and functional execution
receipts. Existing raw unit coverage artifacts and FlagOps uploads are preserved.
Coverage remains per platform/device; line and branch percentages are separate.
Report schema is `flagcicd.report/v2` and plan schema is `flagcicd.plan/v2`.
Structured reports are GitHub artifacts, not a new backend API.

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

Unit coverage (`megatron/training/*`, `megatron/plugin/*`) is combined into
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
python -m unittest discover -s .github/scripts/ci -p 'test_*.py'
python -m unittest discover -s .github/scripts/tests -p 'test_*.py'
```

Before requiring `CI Complete`, verify real Actions runs for docs-only, vendor-only,
mixed/shared changes, failure, cancellation, label changes and comment commands.
Use a test base already containing this CI; the PR introducing these CI files
selects all platforms. Include a fork PR, an unauthorized comment, a push during
requested tests, and a transient failure followed by success. Intentional failures
belong in disposable branches. Local tests do not prove hardware/Actions behavior.
