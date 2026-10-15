"""Hardware-independent checks for report generation and secret forwarding."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import performance_report
import yaml

ROOT = Path(__file__).resolve().parents[3]


class PerformanceReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.logs = self.root / ".retry-logs/functional-smoke"
        self.logs.mkdir(parents=True)
        self.environment = {
            "GITHUB_WORKSPACE": str(self.root),
            "CI_PLATFORM": "cuda",
            "CI_DEVICE": "a100",
            "CI_IMAGE": "test-image",
            "CI_SOURCE_REF": "tested-sha",
            "CI_TEST_CONFIG": json.dumps({"model": "gpt", "test_case": "smoke"}),
            "CI_TEST_EXIT_CODE": "0",
        }

    def report(self, exit_code="0"):
        return performance_report.build_report(
            {**self.environment, "CI_TEST_EXIT_CODE": exit_code}, self.root
        )

    def attempt(self, number, text, exit_code=0):
        (self.logs / "result.json").write_text(
            json.dumps({"attempts": [{"attempt": number, "exit_code": exit_code}]})
        )
        (self.logs / f"attempt-{number}.log").write_text(text)

    @staticmethod
    def line(iteration, ms, throughput=10):
        return (
            f"[rank0] iteration {iteration:8d}/      10 | consumed samples: 8 | "
            f"elapsed time per iteration (ms): {ms} | "
            f"throughput per GPU (TFLOP/s/GPU): {throughput} |\n"
        )

    def test_final_retry_only_and_warmup_excluded(self):
        self.attempt(1, self.line(2, 999), exit_code=1)
        self.attempt(
            2, "unrelated output\n" + self.line(1, 1000) + self.line(2, 12) + self.line(3, 14)
        )
        report = self.report()
        case = report["cases"][0]
        self.assertEqual(case["latency"]["samples"], [12, 14])
        self.assertEqual(case["latency"]["median"], 13)
        self.assertEqual(case["throughput"]["unit"], "TFLOP/s/GPU")
        self.assertEqual(report["summary"]["status"], "passed")
        self.assertEqual(report["environment"]["commit"], "tested-sha")
        json.dumps(report, allow_nan=False)

    def test_failure_keeps_partial_measurements(self):
        self.attempt(1, self.line(2, 20), exit_code=1)
        report = self.report("1")
        self.assertEqual(report["summary"]["failed"], 1)
        self.assertEqual(report["cases"][0]["latency"]["median"], 20)

    def test_missing_results_are_not_successful_measurements(self):
        self.assertEqual(self.report()["summary"]["status"], "unavailable")
        self.assertEqual(self.report("1")["summary"]["status"], "failed")

    def test_main_writes_failure_artifact(self):
        with patch.dict(
            performance_report.os.environ, {**self.environment, "CI_TEST_EXIT_CODE": "1"}
        ):
            performance_report.main()
        path = self.root / "performance-report/perf-smoke-cuda-a100-smoke.json"
        report = json.loads(path.read_text())
        self.assertEqual(report["summary"]["status"], "failed")
        self.assertEqual(report["environment"]["commit"], "tested-sha")

    def test_warmup_only_and_invalid_measurements(self):
        log = self.logs / "output.log"
        log.write_text(self.line(1, 100))
        self.assertEqual(performance_report.measurements(log), {})
        for value in ("nan", "inf", "-1", "0"):
            with self.subTest(value=value):
                log.write_text(self.line(2, value))
                with self.assertRaises(ValueError):
                    performance_report.measurements(log)


class ReportWorkflowTests(unittest.TestCase):
    def test_secret_reaches_both_uploaders(self):
        workflows = {
            path.name: yaml.safe_load(path.read_text())
            for path in (ROOT / ".github/workflows").glob("*.yml")
        }
        callers = {
            "_platform_pipeline.yml",
            "all_tests_common.yml",
            "unit_tests_common.yml",
            "functional_tests_common.yml",
        }
        for name in callers:
            self.assertIn("FLAGCICD_API_TOKEN", workflows[name][True]["workflow_call"]["secrets"])
        for name, workflow in workflows.items():
            for job in workflow.get("jobs", {}).values():
                target = job.get("uses", "").removeprefix("./.github/workflows/")
                if target in callers and name != "ci-requested.yml":
                    self.assertEqual(
                        job["secrets"], {"FLAGCICD_API_TOKEN": "${{ secrets.FLAGCICD_API_TOKEN }}"}
                    )
        for name in ("unit_tests_common.yml", "functional_tests_common.yml"):
            upload = next(
                step
                for job in workflows[name]["jobs"].values()
                for step in job.get("steps", [])
                if step.get("uses") == "flagos-ai/FlagOps/actions/post-pytest-report@v2"
            )
            self.assertTrue(upload["continue-on-error"])
            self.assertEqual(upload["with"]["api_token"], "${{ secrets.FLAGCICD_API_TOKEN }}")
            self.assertNotIn("user_id", upload["with"])
            job_name = upload["with"]["job_name"]
            if name == "functional_tests_common.yml":
                self.assertEqual(job_name, "${{ env.PERFORMANCE_REPORT_NAME }}")
                job_name = workflows[name]["jobs"]["functional_test_train"]["env"][
                    "PERFORMANCE_REPORT_NAME"
                ]
            self.assertIn("${{ inputs.platform }}", job_name)
            self.assertIn("${{ inputs.device }}", job_name)


if __name__ == "__main__":
    unittest.main()
