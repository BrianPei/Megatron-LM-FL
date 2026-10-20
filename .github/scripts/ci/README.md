# Multi-platform CI

`ci.yml` runs lint, selects platforms, executes their pipelines, and reports the
required `CI Complete` gate. PRs target `main` or `devops`. Common pre-commit and
helper checks run once before the platform matrix, including requested tests.
All manual platform entrypoints reuse `_platform_pipeline.yml`, preserving image
and test-scope inputs. Every entrypoint runs common checks once before invoking
the platform pipeline; platform workflows contain no duplicate lint job. The shared test gate reports the platform result.

## FlagCICD and nightly

Adapted from `BrianPei/TransformerEngine-FL:ci/flagcicd-unified`.
The platform YAML files remain the source of truth for runners, images, test
matrices and TE wheel preparation. [`project.yaml`](../../../.flagcicd/project.yaml) defines the supported levels
and ordered risk rules; the planner reads those rules directly.

| Level | Scope |
| --- | --- |
| L0 | Validate selected platform configurations and run common CI checks; no hardware or TE build |
| L1 | Common checks, TE wheel preparation and existing unit tests |
| L2 | L1 plus existing functional training and golden-value validation |

L3/L4 are rejected until extended regression/release validation is implemented.
No uncalibrated coverage thresholds or synthetic performance baselines are added.

Use **Actions → FlagCICD Entry → Run workflow**, or `repository_dispatch` with
`event_type: flagcicd`. Both accept `checkout_ref`, `platforms`, `test_level`,
`risk_level`, `changed_files`, `task_id` and `parent_run_id`. The platform names
match `.github/configs/*.yml` (including `kunlunxin`, not TE's `kunlun`).
`platforms` and `changed_files` accept JSON arrays; dispatch also accepts arrays
as native JSON values. For example, a dispatch payload can contain:

```json
{
  "event_type": "flagcicd",
  "client_payload": {
    "checkout_ref": "cicd-nightly",
    "platforms": ["cuda", "ppu"],
    "test_level": "L2",
    "task_id": "training-validation-42"
  }
}
```

`changed_files` is an optional caller-supplied risk hint, not a computed diff.
Empty input preserves the requested level. Known runtime/training changes raise
it to L2/R3; CI/build/functional and unknown paths raise it to at least L2/R2.
Explicit requests can raise but cannot lower inferred requirements.

Nightly runs at **02:00 Asia/Shanghai** on the default branch and selects all
configured platforms at L2. Its schedule lives only in `flagcicd-entry.yml`;
`ci.yml` continues to handle automatic PR/push routing. GitHub activates scheduled
and repository dispatch events after the workflow reaches the default branch.

The source ref is resolved once to an immutable SHA and passed through the
existing `_platform_pipeline.yml` TE preparation and test chain. L0 skips that
chain. Common checks run once before hardware jobs. Existing manual and comment
commands keep their scope controls, retries, wheel cache and coverage uploads.

Artifacts retained for 14 days:

- `flagcicd-plan`: resolved platforms, scope, risk and task/source identity.
- `flagcicd-report-<platform>`: configuration, unit and functional job conclusions,
  with combined unit coverage reported separately for each device. These reports
  are also produced by existing platform entrypoints.
- `flagcicd-run-summary`: entry/nightly results and validated child reports.

Summaries check source SHA, run ID, attempt, task, scope and risk. Successful
platform reports from earlier attempts of the same run can be reused when rerunning
failed jobs; future-attempt reports are rejected. Missing,
duplicate, stale or failed reports cannot pass. Functional failures and skipped
required jobs cannot be hidden by successful unit coverage. L0 reports preflight
results without pretending to have coverage. Line percentages use line counts;
branch percentages use branch counts. Platform/device denominators are never
merged. Existing raw coverage artifacts and FlagOps backend uploads are retained;
structured reports use GitHub artifacts and do not introduce another backend API.

Run the local CI helper suites in [Validation](#validation). Hardware execution
and backend ingestion must be verified on actual runners.

## Platform selection

- `paths_watch` accepts exact files and directory prefixes ending in `/`.
- Vendor-only changes run the affected platforms; shared/unknown paths run all.
- Documentation (`docs/`, `.md`, `.rst`) skips hardware tests but still reports
  the gate. Revisit this rule if builds/tests start using documentation as input.
- Missing diffs/watch rules select all platforms; invalid configs fail.
- Renames include both old and new paths. Push/schedule/manual CI runs select all.
- Labels `platform:<config-name>` override path routing, matching TE. Multiple
  labels take their union; `platform:all` selects every platform. Removing all
  platform labels restores automatic routing. Narrow shared-code changes after review.
- A functional case shared by several platforms must appear in all their watch lists.

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
