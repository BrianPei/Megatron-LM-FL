import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "run_unit_groups", ROOT / ".github/scripts/run_unit_groups.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class UnitGroupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        script = self.root / "tests/test_utils/runners/run_ci_unit_tests.sh"
        script.parent.mkdir(parents=True)
        script.write_text(
            'mkdir -p "$CI_COVERAGE_DIRECTORY"\n'
            'printf "%s:%s:%s\\n" "$CI_TEST_GROUP" "${GROUP_ONLY:-unset}" '
            '"$CI_TEST_PATH" >> "$GITHUB_WORKSPACE/calls"\n'
            'printf "{\\"group\\":\\"%s\\"}" "$CI_TEST_GROUP" > '
            '"$CI_COVERAGE_DIRECTORY/coverage-fake-device-$CI_TEST_GROUP.json"\n'
            'if [ "$CI_TEST_GROUP" = failed ]; then exit 7; fi\n'
        )
        (self.root / "setup.sh").write_text(
            'test "$CI_TEST_SUITE" = unit_group\n'
            'if [ "$CI_TEST_GROUP" = failed ]; then export GROUP_ONLY=local; fi\n'
            'echo "GROUP_ONLY=leaked" >> "$GITHUB_ENV"\n'
            'echo "/leaked/path" >> "$GITHUB_PATH"\n'
        )
        self.env = {
            **os.environ,
            "GITHUB_WORKSPACE": str(self.root),
            "GITHUB_ENV": str(self.root / "env"),
            "GITHUB_PATH": str(self.root / "path"),
            "GITHUB_STEP_SUMMARY": str(self.root / "summary"),
            "CI_PLATFORM": "fake",
            "CI_DEVICE": "device",
            "CI_SETUP_SCRIPT": "setup.sh",
        }

    def run_groups(self, groups):
        return subprocess.run(
            [sys.executable, str(ROOT / ".github/scripts/run_unit_groups.py")],
            env={**self.env, "CI_TEST_GROUPS": json.dumps(groups)},
            capture_output=True,
            text=True,
            timeout=10,
        )

    def test_failure_continues_without_environment_or_report_leaks(self):
        result = self.run_groups(
            [{"name": "failed", "path": "tests/a tests/b"}, {"name": "passed", "path": "tests/c"}]
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(
            (self.root / "calls").read_text().splitlines(),
            ["failed:local:tests/a tests/b", "passed:unset:tests/c"],
        )
        reports = self.root / "coverage-report"
        for name in ("failed", "passed"):
            self.assertEqual(
                json.loads((reports / f"coverage-fake-device-{name}.json").read_text()),
                {"group": name},
            )
        self.assertEqual(
            json.loads((reports / "unit-results.json").read_text()),
            [{"name": "failed", "exit_code": 7}, {"name": "passed", "exit_code": 0}],
        )
        self.assertIn("| failed | 7 |", (self.root / "summary").read_text())
        self.assertFalse((self.root / "env").exists())
        self.assertFalse((self.root / "path").exists())

    def test_success(self):
        result = self.run_groups([{"name": "passed", "path": "tests/a"}])
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_empty_invalid_and_duplicate_groups_fail_before_running(self):
        for groups in (
            [],
            [{"name": "../bad", "path": "tests/a"}],
            [{"name": "same", "path": "a"}, {"name": "same", "path": "b"}],
            [{"name": "empty", "path": ""}],
        ):
            with self.subTest(groups=groups):
                self.assertNotEqual(self.run_groups(groups).returncode, 0)
                self.assertFalse((self.root / "calls").exists())

    def test_hook_failure_is_a_group_failure_and_next_group_still_runs(self):
        (self.root / "setup.sh").write_text('if [ "$CI_TEST_GROUP" = failed ]; then exit 9; fi\n')
        result = self.run_groups([{"name": "failed", "path": "a"}, {"name": "passed", "path": "b"}])
        self.assertEqual(result.returncode, 1)
        self.assertEqual((self.root / "calls").read_text(), "passed:unset:b\n")
        self.assertIn("exit code 9", result.stdout)

    def test_real_platform_hooks_keep_group_settings_local(self):
        script = self.root / "tests/test_utils/runners/run_ci_unit_tests.sh"
        script.write_text(
            'printf "%s|%s|%s|%s\\n" "$CI_TEST_GROUP" "${MASTER_PORT:-}" '
            '"${MCCL_P2P_DISABLE:-}" "${PYTHONPATH:-}" >> "$GITHUB_WORKSPACE/hooks"\n'
        )
        for platform, groups in (
            ("musa", ["models", "core"]),
            ("ascend", ["models", "core"]),
            ("metax", ["data", "core"]),
        ):
            with self.subTest(platform=platform):
                setup = (ROOT / f".github/scripts/set_env_{platform}.sh").read_text()
                setup = setup.replace(
                    'source "$SCRIPT_DIR/set_env_common.sh"',
                    f'source "{ROOT}/.github/scripts/set_env_common.sh"',
                ).replace("/tmp/megatron-ci-stubs", str(self.root / "stubs"))
                (self.root / "setup.sh").write_text("python3() { :; }\n" + setup)
                result = self.run_groups([{"name": name, "path": "tests/a"} for name in groups])
                self.assertEqual(result.returncode, 0, result.stderr)
                rows = (self.root / "hooks").read_text().splitlines()[-2:]
                first, second = [row.split("|") for row in rows]
                if platform in ("musa", "ascend"):
                    self.assertIn(str(self.root / "stubs"), first[3])
                    self.assertNotIn(str(self.root / "stubs"), second[3])
                if platform == "musa":
                    self.assertEqual((first[1], second[1]), ("29601", "29600"))
                if platform == "metax":
                    self.assertEqual((first[2], second[2]), ("1", ""))

    def test_timeout_kills_the_group_and_returns_failure(self):
        (self.root / "setup.sh").write_text('echo $$ > "$GITHUB_WORKSPACE/pid"\nsleep 30\n')
        env = {**self.env, "CI_SETUP_SCRIPT": str(self.root / "setup.sh"), "CI_TEST_GROUP": "slow"}
        self.assertEqual(RUNNER.run_group(env, timeout=0.3), 124)
        pid = int((self.root / "pid").read_text())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    def test_process_group_cleanup_also_runs_after_success_and_interruption(self):
        for outcome in (0, KeyboardInterrupt()):
            process = mock.Mock(pid=12345)
            process.wait.side_effect = [outcome, 0]
            with (
                mock.patch.object(RUNNER.subprocess, "Popen", return_value=process),
                mock.patch.object(RUNNER.os, "killpg") as kill,
            ):
                if isinstance(outcome, BaseException):
                    with self.assertRaises(KeyboardInterrupt):
                        RUNNER.run_group({})
                else:
                    self.assertEqual(RUNNER.run_group({}), 0)
                kill.assert_called_once_with(12345, signal.SIGKILL)

    def test_combined_coverage_includes_normal_and_partial_group_data(self):
        groups = [self.root / "first", self.root / "second"]
        for directory in groups:
            directory.mkdir()
            (directory / ".coveragerc").write_text("[run]\nparallel = true\n")
        (groups[0] / ".coverage").touch()
        (groups[1] / ".coverage.rank1").touch()
        output = self.root / "coverage-all.json"
        with mock.patch.object(RUNNER.subprocess, "run") as run:
            RUNNER.combine_coverage(groups, output, {"CI_PYTHON_BIN": "/custom/python"})
        self.assertEqual(run.call_count, 2)
        combine = run.call_args_list[0].args[0]
        self.assertEqual(combine[0], "/custom/python")
        self.assertIn(str(groups[0] / ".coverage"), combine)
        self.assertIn(str(groups[1] / ".coverage.rank1"), combine)
        self.assertNotIn(str(groups[0] / ".coveragerc"), combine)
        self.assertIn(str(output), run.call_args_list[1].args[0])

    def test_reporting_failure_does_not_override_test_result(self):
        (self.root / ".coverage").touch()
        (self.root / ".coveragerc").touch()
        with mock.patch.object(RUNNER.subprocess, "run", side_effect=OSError("unavailable")):
            RUNNER.combine_coverage([self.root], self.root / "all.json", {})

    def test_real_coverage_combines_completed_and_interrupted_groups(self):
        source = self.root / "megatron/training/example.py"
        source.parent.mkdir(parents=True)
        source.write_text(
            "import sys\nif sys.argv[1] == 'first':\n    value = 1\nelse:\n    value = 2\n"
        )
        directories = [self.root / "first", self.root / "second"]
        for directory in directories:
            directory.mkdir()
            config = directory / ".coveragerc"
            config.write_text(f"[run]\nparallel = true\ndata_file = {directory}/.coverage\n")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "coverage",
                    "run",
                    f"--rcfile={config}",
                    str(source),
                    directory.name,
                ],
                check=True,
                capture_output=True,
                timeout=10,
            )
        # The first group finished reporting; the second left raw rank data.
        subprocess.run(
            [
                sys.executable,
                "-m",
                "coverage",
                "combine",
                f"--rcfile={directories[0] / '.coveragerc'}",
                str(directories[0]),
            ],
            check=True,
            capture_output=True,
            timeout=10,
        )
        output = self.root / "combined.json"
        RUNNER.combine_coverage(directories, output, {**self.env, "CI_PYTHON_BIN": sys.executable})
        report = json.loads(output.read_text())
        measured = next(iter(report["files"].values()))
        self.assertEqual(measured["executed_lines"], [1, 2, 3, 5])
        self.assertTrue((directories[0] / ".coverage").exists())
        self.assertTrue(list(directories[1].glob(".coverage.*")))


class UnitWorkflowTests(unittest.TestCase):
    def test_one_job_prepares_once_and_calls_existing_group_runner(self):
        workflow = yaml.safe_load((ROOT / ".github/workflows/unit_tests_common.yml").read_text())
        self.assertEqual(list(workflow["jobs"]), ["unit_test"])
        job = workflow["jobs"]["unit_test"]
        self.assertNotIn("strategy", job)
        steps = {step["name"]: step for step in job["steps"]}
        self.assertEqual(steps["Setup platform environment"]["env"]["CI_TEST_GROUP"], "__all__")
        runner = steps["Run unit test groups sequentially"]
        self.assertEqual(runner["env"]["CI_TEST_GROUPS"], "${{ inputs.test_groups }}")
        for name in ("Install configured runtime packages", "Install TE-FL wheel"):
            self.assertIn(name, steps)
        for step in job["steps"]:
            if step.get("uses") == "actions/checkout@v4":
                self.assertEqual(step["with"]["fetch-depth"], 1)
            self.assertNotEqual(step.get("uses"), "actions/download-artifact@v4")
        for name in ("Upload Coverage Report", "Upload Coverage Report to FlagCICD"):
            self.assertIn("!cancelled()", steps[name]["if"])
        self.assertIn(
            "CI_COVERAGE_DIRECTORY",
            (ROOT / "tests/test_utils/runners/run_ci_unit_tests.sh").read_text(),
        )

    def test_all_platforms_have_install_free_group_hooks(self):
        for name in ("cuda", "musa", "ascend", "enflame", "hygon", "kunlunxin", "metax"):
            script = (ROOT / f".github/scripts/set_env_{name}.sh").read_text()
            hook = script.split("  unit_group)\n", 1)[1].split("    ;;", 1)[0]
            self.assertNotIn("pip", hook)
            self.assertNotIn("setup_unit_environment", hook)
        musa = (ROOT / ".github/scripts/set_env_musa.sh").read_text()
        self.assertIn("${CI_TEST_GROUP:-}\" = __all__", musa)


if __name__ == "__main__":
    unittest.main()
