import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def workflow(name):
    return yaml.safe_load((ROOT / "workflows" / name).read_text())


class RuntimeSnapshotWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.document = workflow("unit_tests_common.yml")
        self.producer = self.document["jobs"]["unit_runtime_prepare"]
        self.consumer = self.document["jobs"]["unit_test"]

    def test_producer_matches_consumer_platform_and_has_no_test_group_matrix(self):
        self.assertEqual(self.producer["runs-on"], self.consumer["runs-on"])
        for field in ("image", "volumes", "options"):
            self.assertEqual(self.producer["container"][field], self.consumer["container"][field])
        self.assertNotIn("strategy", self.producer)
        self.assertEqual(self.producer["env"]["CI_TEST_GROUP"], "__all__")
        self.assertEqual(self.consumer["needs"], "unit_runtime_prepare")
        self.assertIn("needs.unit_runtime_prepare.result == 'success'", self.consumer["if"])

    def test_consumer_failure_and_disabled_producer_gates(self):
        for enabled in (True, False):
            for cancelled in (True, False):
                for result in ("success", "failure", "cancelled", "skipped"):
                    expression = self.consumer["if"]
                    for key, value in (
                        ("always()", True),
                        ("cancelled()", cancelled),
                        ("inputs.runtime_snapshot", enabled),
                        ("needs.unit_runtime_prepare.result", result),
                    ):
                        expression = expression.replace(key, repr(value))
                    expression = (
                        expression.replace("!", "not ").replace("&&", " and ").replace("||", " or ")
                    )
                    actual = eval(" ".join(expression.split()), {"__builtins__": {}}, {})
                    self.assertEqual(
                        actual,
                        not cancelled
                        and (result == "success" or (not enabled and result == "skipped")),
                    )

    def test_producer_bundles_dependencies_without_extra_environment_validation(self):
        names = [step["name"] for step in self.producer["steps"]]
        ordered = [
            "Setup platform environment",
            "Install configured runtime packages",
            "Bundle prepared unit dependencies",
            "Upload unit runtime snapshot",
        ]
        self.assertEqual(
            sorted(names.index(name) for name in ordered), [names.index(name) for name in ordered]
        )
        upload = self.producer["steps"][-1]
        self.assertEqual(upload["with"]["compression-level"], 0)
        self.assertEqual(upload["with"]["if-no-files-found"], "error")
        self.assertNotIn("Install TE-FL wheel", names)
        self.assertNotIn("Record unmodified image identity", names)

    def test_consumer_restores_before_local_setup_and_bypasses_online_paths(self):
        steps = self.consumer["steps"]
        names = [step["name"] for step in steps]
        self.assertLess(
            names.index("Install prepared unit runtime offline"),
            names.index("Setup platform environment"),
        )
        for step in steps:
            if "TE-FL" in step["name"] or step["name"] == "Install configured runtime packages":
                self.assertIn("!inputs.runtime_snapshot", step["if"])
        for job in (self.producer, self.consumer):
            for step in job["steps"]:
                if step.get("uses") == "actions/checkout@v4":
                    self.assertEqual(step["with"]["fetch-depth"], 1)

    def test_rerun_consumes_producer_outputs_not_current_attempt(self):
        steps = {step["name"]: step for step in self.consumer["steps"]}
        self.assertEqual(
            steps["Download prepared unit runtime"]["with"]["name"],
            "${{ needs.unit_runtime_prepare.outputs.artifact_name }}",
        )
        export = next(step for step in self.producer["steps"] if step.get("id") == "snapshot")
        self.assertIn("github.run_attempt", export["env"]["CI_UNIT_SNAPSHOT_NAME"])
        self.assertNotIn("github.run_attempt", str(self.consumer))

    def test_platforms_use_generic_snapshot_flag_and_ppu_fast_path(self):
        common = workflow("all_tests_common.yml")
        unit = common["jobs"]["unit_tests"]
        self.assertIn("unit_runtime_snapshot", unit["with"]["runtime_snapshot"])
        self.assertIn("unit_tests", common["jobs"]["all_tests_complete"]["needs"])
        self.assertNotIn("runtime_snapshot", common["jobs"]["functional_tests"]["with"])
        self.assertIn("!inputs.runtime_snapshot", self.consumer["if"])
        for name in ("cuda", "musa", "ascend", "enflame", "hygon", "kunlunxin", "metax", "ppu"):
            config = yaml.safe_load((ROOT / "configs" / f"{name}.yml").read_text())
            self.assertEqual(
                config["test_matrix"]["unit"].get("runtime_snapshot", True), name != "ppu"
            )
            script = (ROOT.parent / config["setup_script"]).read_text()
            self.assertIn("  activate)", script)

    def test_group_specific_dependencies_and_compatibility_are_retained(self):
        musa = (ROOT / "scripts/set_env_musa.sh").read_text()
        self.assertIn(
            'if [[ "${CI_TEST_GROUP:-}" = models || "${CI_TEST_GROUP:-}" = __all__ ]]; then', musa
        )
        for name, hook in (
            ("musa", "install_musa_compatibility_layer"),
            ("enflame", "patch_coverage_for_torch_gcu"),
            ("metax", "configure_metax_unit_runtime"),
        ):
            script = (ROOT / "scripts" / f"set_env_{name}.sh").read_text()
            self.assertIn(f"  {hook}\n", script)


if __name__ == "__main__":
    unittest.main()
