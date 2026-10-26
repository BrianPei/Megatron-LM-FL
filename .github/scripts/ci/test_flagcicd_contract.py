#!/usr/bin/env python3
"""Tests for the repository-facing FlagCICD contract."""

from __future__ import annotations

import pathlib
import tempfile
import unittest

import flagcicd_contract
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[3]


class ContractTests(unittest.TestCase):
    def test_project_matches_legacy_catalog(self) -> None:
        project, digest = flagcicd_contract.load_project(ROOT / ".flagcicd/project.yaml")
        legacy = yaml.safe_load((ROOT / ".github/configs/unit_group_routes.yml").read_text())
        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertEqual(flagcicd_contract.contract_catalog_errors(project, legacy), [])

    def test_project_has_explicit_stage_semantics(self) -> None:
        project, _ = flagcicd_contract.load_project(ROOT / ".flagcicd/project.yaml")
        self.assertEqual(project["stages"]["pr"]["selection"], "affected")
        self.assertEqual(project["stages"]["nightly"]["selection"], "configured")
        self.assertTrue(project["stages"]["nightly"]["fixed_level"])
        self.assertEqual(set(project["stages"]), {"pr", "branch", "nightly", "manual"})
        self.assertEqual(set(project["levels"]), {"L1", "L2"})
        self.assertEqual(project["gate_policy"]["pr"]["required_plan_status"], ["ready", "partial"])

    def test_removed_configuration_is_rejected(self) -> None:
        cases = [
            ("project", "resource_classes", {}),
            ("global", "default_timeout_minutes", 1),
            ("group", "resource_class", "single_device"),
            ("group", "parallel_safe", True),
            ("group", "timeout_minutes", 1),
            ("group", "kind", "unit"),
            ("group", "evidence", "junit"),
            ("stage", "full_configured_matrix", True),
            ("stage", "status", "planned"),
            ("level", "status", "planned"),
        ]
        for section, key, value in cases:
            with self.subTest(section=section, key=key), tempfile.TemporaryDirectory() as directory:
                project, _ = flagcicd_contract.load_project(ROOT / ".flagcicd/project.yaml")
                target = {
                    "project": project,
                    "global": project["global"],
                    "group": project["test_groups"][0],
                    "stage": project["stages"]["pr"],
                    "level": project["levels"]["L1"],
                }[section]
                target[key] = value
                path = pathlib.Path(directory) / "project.yaml"
                path.write_text(yaml.safe_dump(project, sort_keys=False))
                with self.assertRaisesRegex(flagcicd_contract.ContractError, "unsupported fields"):
                    flagcicd_contract.load_project(path)

    def test_project_and_legacy_catalog_must_have_the_same_levels(self) -> None:
        project, _ = flagcicd_contract.load_project(ROOT / ".flagcicd/project.yaml")
        legacy = flagcicd_contract.project_to_catalog(project)
        legacy["levels"].pop("L2")
        errors = flagcicd_contract.contract_catalog_errors(project, legacy)
        self.assertIn("missing active project levels", errors[0])
        self.assertIn("L2", project["levels"])

    def test_stage_cannot_reference_unknown_level(self) -> None:
        project, _ = flagcicd_contract.load_project(ROOT / ".flagcicd/project.yaml")
        project["stages"]["pr"]["default_level"] = "L4"
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "project.yaml"
            path.write_text(yaml.safe_dump(project, sort_keys=False))
            with self.assertRaisesRegex(flagcicd_contract.ContractError, "unknown level"):
                flagcicd_contract.load_project(path)


if __name__ == "__main__":
    unittest.main()
