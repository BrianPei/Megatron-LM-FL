"""Hardware-independent tests for logical unit-group routing and shadow plans."""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import tempfile
import unittest

import generate_matrix
import generate_test_plan as planner
import unit_group_routing as routing

ROOT = pathlib.Path(__file__).resolve().parents[3]
CONFIGS = ROOT / ".github/configs"
CATALOG = CONFIGS / "unit_group_routes.yml"


def load_catalog() -> routing.Catalog:
    return routing.load_catalog(CATALOG)


def load_platforms() -> dict[str, routing.PlatformUnit]:
    rules = generate_matrix._load_watch_rules(CONFIGS)
    return {name: routing.load_platform(CONFIGS / f"{name}.yml") for name in rules}


def build_plan(
    changed: list[str],
    *,
    platforms: list[str] | None = None,
    selection: str = "affected",
    level: str = "L1",
    event: str = "pull_request",
    catalog: routing.Catalog | None = None,
    units: dict[str, routing.PlatformUnit] | None = None,
) -> dict:
    catalog = catalog or load_catalog()
    units = units or load_platforms()
    return planner.build_plan(
        event_name=event,
        stage="pr" if event == "pull_request" else "nightly",
        selection=selection,
        level=level,
        changed=changed,
        unknown_diff=False,
        platforms=sorted(units) if platforms is None else platforms,
        catalog=catalog,
        platform_units=units,
        source_sha="source",
        base_sha="base",
    )


def platform_plan(plan: dict, name: str) -> dict:
    return next(item for item in plan["platforms"] if item["name"] == name)


def selected_names(plan: dict, name: str) -> list[str]:
    return [group["name"] for group in platform_plan(plan, name)["selected_groups"]]


class CatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog()
        cls.platforms = load_platforms()

    def test_catalog_is_not_a_platform_config(self):
        self.assertIn("unit_group_routes", generate_matrix._NON_PLATFORM_STEMS)
        self.assertNotIn("unit_group_routes", self.platforms)

    def test_catalog_paths_and_dependencies_are_valid(self):
        self.assertEqual(self.catalog.validate(ROOT), [])

    def test_every_unit_test_directory_has_a_logical_group(self):
        for directory in sorted((ROOT / "tests/unit_tests").iterdir()):
            if not directory.is_dir() or directory.name == "__pycache__":
                continue
            probe = f"tests/unit_tests/{directory.name}/"
            with self.subTest(directory=probe):
                self.assertIsNotNone(self.catalog.matches(probe))

    def test_every_platform_group_exists_in_the_catalog(self):
        for name, platform in self.platforms.items():
            for group_name in platform.groups:
                with self.subTest(platform=name, group=group_name):
                    self.assertIn(group_name, self.catalog.groups)

    def test_required_groups_are_executable_everywhere(self):
        for group_name in self.catalog.required_groups("L1"):
            for name, platform in self.platforms.items():
                with self.subTest(platform=name, group=group_name):
                    self.assertIn(group_name, platform.groups)

    def test_dependency_graph_is_acyclic(self):
        errors = [error for error in self.catalog.validate(ROOT) if "cycle" in error]
        self.assertEqual(errors, [])

    def test_glob_rules_do_not_cross_directories(self):
        self.assertEqual(self.catalog.matches("tests/unit_tests/test_basic.py"), "core")
        self.assertEqual(
            self.catalog.matches("tests/unit_tests/models/test_gpt_model.py"), "models"
        )
        self.assertIsNone(self.catalog.matches("tests/unit_tests/test_data/fixture.py"))


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog()
        cls.platforms = load_platforms()

    def test_transformer_change_selects_transformer_models_and_core(self):
        plan = build_plan(["megatron/core/transformer/attention.py"], platforms=["cuda"])
        self.assertEqual(selected_names(plan, "cuda"), ["core", "models", "transformer"])
        groups = {group["name"]: group for group in platform_plan(plan, "cuda")["selected_groups"]}
        self.assertEqual(
            groups["transformer"]["reasons"],
            [{"type": "direct", "path": "megatron/core/transformer/attention.py"}],
        )
        self.assertIn(
            {"type": "dependency", "source_group": "transformer"}, groups["models"]["reasons"]
        )
        self.assertTrue(groups["core"]["required"])

    def test_distributed_change_selects_parallel_dependencies(self):
        plan = build_plan(
            ["megatron/core/distributed/param_and_grad_buffer.py"], platforms=["cuda"]
        )
        self.assertEqual(
            selected_names(plan, "cuda"),
            ["core", "distributed", "models", "pipeline_parallel", "tensor_parallel"],
        )

    def test_dist_checkpointing_change_does_not_select_models(self):
        plan = build_plan(["megatron/core/dist_checkpointing/serialization.py"], platforms=["cuda"])
        self.assertEqual(selected_names(plan, "cuda"), ["core", "dist_checkpointing"])

    def test_test_directory_maps_to_its_group(self):
        for path, expected in (
            ("tests/unit_tests/models/test_gpt_model.py", "models"),
            ("tests/unit_tests/distributed/test_data_parallel.py", "distributed"),
            ("tests/unit_tests/tensor_parallel/test_layers.py", "tensor_parallel"),
            ("tests/unit_tests/inference/test_engine.py", "inference"),
            ("tests/unit_tests/rl/test_agent.py", "rl"),
            ("tests/unit_tests/training/test_training.py", "training"),
        ):
            with self.subTest(path=path):
                self.assertEqual(self.catalog.matches(path), expected)

    def test_others_test_directories_map_to_others(self):
        for directory in ("post_training", "tokenizers", "utils"):
            with self.subTest(directory=directory):
                self.assertEqual(
                    self.catalog.matches(f"tests/unit_tests/{directory}/test_x.py"), "others"
                )

    def test_plugin_tests_execute_in_core(self):
        plan = build_plan(["megatron/plugin/tests/test_override.py"], platforms=["cuda"])
        self.assertEqual(selected_names(plan, "cuda"), ["core"])

    def test_shared_path_falls_back_to_every_group(self):
        plan = build_plan([".github/workflows/ci.yml"], platforms=["cuda"])
        self.assertTrue(plan["global_groups"]["fallback"])
        self.assertEqual(set(plan["global_groups"]["selected"]), set(self.catalog.groups))
        self.assertEqual(
            selected_names(plan, "cuda"), platform_plan(plan, "cuda")["configured_groups"]
        )

    def test_build_inputs_fall_back_to_every_group(self):
        for path in ("uv.lock", "docker/Dockerfile", "pyproject.toml", "setup.py"):
            with self.subTest(path=path):
                plan = build_plan([path], platforms=["cuda"])
                self.assertTrue(plan["global_groups"]["fallback"])

    def test_unknown_path_falls_back_and_is_reported(self):
        plan = build_plan(["megatron/core/new_shared_module.py"], platforms=["cuda"])
        self.assertTrue(plan["global_groups"]["fallback"])
        self.assertEqual(
            plan["global_groups"]["unmapped_paths"], ["megatron/core/new_shared_module.py"]
        )

    def test_mixed_pr_takes_the_union(self):
        plan = build_plan(
            [
                "megatron/core/transformer/attention.py",
                "megatron/core/dist_checkpointing/serialization.py",
            ],
            platforms=["cuda"],
        )
        selected = selected_names(plan, "cuda")
        for name in ("transformer", "models", "dist_checkpointing", "core"):
            self.assertIn(name, selected)

    def test_documentation_only_selects_nothing(self):
        for changed in (["docs/guide.md"], ["README.md"], ["docs/guide.rst", "notes.md"]):
            with self.subTest(changed=changed):
                plan = build_plan(changed, platforms=[])
                self.assertTrue(plan["global_groups"]["documentation_only"])
                self.assertEqual(plan["global_groups"]["selected"], [])
                self.assertEqual(plan["platforms"], [])

    def test_documentation_mixed_with_code_runs_code(self):
        plan = build_plan(
            ["docs/guide.md", "megatron/core/models/gpt/gpt_model.py"], platforms=["cuda"]
        )
        self.assertFalse(plan["global_groups"]["documentation_only"])
        self.assertEqual(selected_names(plan, "cuda"), ["core", "models"])

    def test_nightly_selects_all_logical_groups(self):
        plan = build_plan([], selection="all", event="schedule", platforms=["cuda"])
        self.assertEqual(set(plan["global_groups"]["selected"]), set(self.catalog.groups))
        self.assertEqual(
            selected_names(plan, "cuda"), platform_plan(plan, "cuda")["configured_groups"]
        )

    def test_rename_considers_old_and_new_paths(self):
        plan = build_plan(
            ["megatron/core/models/legacy.py", "megatron/core/transformer/renamed.py"],
            platforms=["cuda"],
        )
        self.assertIn("transformer", selected_names(plan, "cuda"))
        self.assertIn("models", selected_names(plan, "cuda"))

    def test_vendor_plugin_maps_to_plugin_and_core(self):
        plan = build_plan(["megatron/plugin/Ascend/ascend_plugin.py"], platforms=["ascend"])
        self.assertEqual(selected_names(plan, "ascend"), ["core"])
        self.assertEqual(
            [group["name"] for group in platform_plan(plan, "ascend")["not_configured"]], ["plugin"]
        )

    def test_level_two_adds_models(self):
        plan = build_plan(["megatron/core/config.py"], platforms=["cuda"], level="L2")
        self.assertEqual(selected_names(plan, "cuda"), ["core", "models"])


class MaterializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog()
        cls.platforms = load_platforms()

    def test_platform_without_transformer_reports_not_configured(self):
        plan = build_plan(["megatron/core/transformer/attention.py"], platforms=["ascend"])
        self.assertEqual(selected_names(plan, "ascend"), ["core", "models"])
        self.assertEqual(
            [group["name"] for group in platform_plan(plan, "ascend")["not_configured"]],
            ["transformer"],
        )

    def test_hygon_missing_groups_are_reported(self):
        plan = build_plan(
            ["megatron/core/distributed/param_and_grad_buffer.py"], platforms=["hygon"]
        )
        self.assertEqual(
            [group["name"] for group in platform_plan(plan, "hygon")["not_configured"]],
            ["distributed"],
        )
        self.assertIn("pipeline_parallel", selected_names(plan, "hygon"))

    def test_missing_required_group_is_blocked(self):
        unit = routing.PlatformUnit(
            name="temp",
            path=pathlib.Path("temp.yml"),
            data={},
            device_types=("dev",),
            nproc_per_node=8,
            groups={"models": {"name": "models", "path": "tests/unit_tests/models/"}},
        )
        plan = build_plan(
            ["megatron/core/models/gpt/gpt_model.py"], platforms=["temp"], units={"temp": unit}
        )
        blocked = [group["name"] for group in platform_plan(plan, "temp")["blocked"]]
        self.assertEqual(blocked, ["core"])
        self.assertEqual(selected_names(plan, "temp"), ["models"])

    def test_plan_contains_identity_and_hash_fields(self):
        plan = build_plan(["megatron/core/models/gpt/gpt_model.py"], platforms=["cuda"])
        for field in (
            "schema_version",
            "source_sha",
            "base_sha",
            "catalog_hash",
            "config_hash",
            "plan_hash",
        ):
            self.assertIn(field, plan)
        self.assertTrue(plan["plan_hash"].startswith("sha256:"))

    def test_plan_hash_is_stable_and_order_independent(self):
        changed = [
            "megatron/core/transformer/attention.py",
            "megatron/core/models/gpt/gpt_model.py",
        ]
        first = build_plan(changed, platforms=["cuda"])
        second = build_plan(list(reversed(changed)), platforms=["cuda"])
        self.assertEqual(first["plan_hash"], second["plan_hash"])
        third = build_plan(changed + ["megatron/core/config.py"], platforms=["cuda"])
        self.assertNotEqual(first["plan_hash"], third["plan_hash"])

    def test_selected_and_not_configured_are_sorted(self):
        plan = build_plan(
            ["megatron/core/models/gpt/gpt_model.py", "megatron/core/transformer/attention.py"],
            platforms=["cuda"],
        )
        for platform in plan["platforms"]:
            names = [group["name"] for group in platform["selected_groups"]]
            self.assertEqual(names, sorted(names))
            not_configured = [group["name"] for group in platform["not_configured"]]
            self.assertEqual(not_configured, sorted(not_configured))


class PathExistenceTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("git"), "git is required")
    def test_untracked_paths_do_not_count_as_repository_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            subprocess.run(["git", "-C", str(root), "init", "-q"], check=True)
            (root / "tracked.txt").write_text("tracked\n")
            (root / "untracked.txt").write_text("untracked\n")
            (root / "tests").mkdir()
            (root / "tests/test_plan.py").write_text("pass\n")
            subprocess.run(
                ["git", "-C", str(root), "add", "tracked.txt", "tests/test_plan.py"],
                check=True,
            )

            self.assertTrue(routing.path_exists(root, "tracked.txt"))
            self.assertTrue(routing.path_exists(root, "tests/test_*.py"))
            self.assertFalse(routing.path_exists(root, "untracked.txt"))
            self.assertFalse(routing.path_exists(root, "missing/"))


class PlatformSelectionTests(unittest.TestCase):
    def test_unknown_platform_is_rejected(self):
        with self.assertRaises(ValueError):
            planner._parse_platforms('["not-a-platform"]', {"cuda"})

    def test_platform_json_must_be_a_string_array(self):
        with self.assertRaises(ValueError):
            planner._parse_platforms('["cuda", 1]', {"cuda"})


if __name__ == "__main__":
    unittest.main()
