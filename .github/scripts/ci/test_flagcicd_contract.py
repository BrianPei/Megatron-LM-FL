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
        self.assertTrue(project["stages"]["nightly"]["full_configured_matrix"])
        self.assertEqual(project["stages"]["release"]["status"], "planned")
        self.assertEqual(project["gate_policy"]["pr"]["required_plan_status"], ["ready", "partial"])

    def test_project_to_catalog_drops_contract_only_fields(self) -> None:
        project, _ = flagcicd_contract.load_project(ROOT / ".flagcicd/project.yaml")
        catalog = flagcicd_contract.project_to_catalog(project)
        self.assertEqual(set(catalog["groups"]), {item["name"] for item in project["test_groups"]})
        self.assertNotIn("kind", catalog["groups"]["core"])
        self.assertNotIn("evidence", catalog["groups"]["core"])

    def test_project_and_legacy_catalog_must_have_the_same_levels(self) -> None:
        project, _ = flagcicd_contract.load_project(ROOT / ".flagcicd/project.yaml")
        legacy = flagcicd_contract.project_to_catalog(project)
        legacy["levels"].pop("L4")
        self.assertEqual(flagcicd_contract.contract_catalog_errors(project, legacy), [])
        legacy["levels"].pop("L2")
        errors = flagcicd_contract.contract_catalog_errors(project, legacy)
        self.assertIn("missing active project levels", errors[0])
        self.assertIn("L2", project["levels"])

    def test_active_stage_cannot_reference_planned_level(self) -> None:
        project, _ = flagcicd_contract.load_project(ROOT / ".flagcicd/project.yaml")
        project["stages"]["pr"]["default_level"] = "L4"
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "project.yaml"
            path.write_text(yaml.safe_dump(project, sort_keys=False))
            with self.assertRaisesRegex(flagcicd_contract.ContractError, "active stage pr"):
                flagcicd_contract.load_project(path)


if __name__ == "__main__":
    unittest.main()
