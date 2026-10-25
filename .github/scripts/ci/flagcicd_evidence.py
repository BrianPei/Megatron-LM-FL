"""Validate successful unit execution from per-rank, per-phase JUnit evidence."""

from __future__ import annotations

import pathlib
import xml.etree.ElementTree as ET
from typing import Any

COUNTS = ("tests", "skipped", "failures", "errors", "passed")


def report_names(nproc: int, experimental: bool) -> list[str]:
    """List the reports produced by the configured distributed invocations."""
    phases = ("normal", "experimental") if experimental else ("normal",)
    return sorted(f"{phase}-{rank}.xml" for phase in phases for rank in range(nproc))


def validate(evidence: Any, expected: list[str] | None = None) -> list[str]:
    """Reject incomplete, inconsistent, failed, or entirely skipped executions."""
    if not isinstance(evidence, dict) or evidence.get("schema_version") != 1:
        return ["missing or unsupported unit evidence"]
    reports = evidence.get("reports")
    if not isinstance(reports, dict) or not reports:
        return ["missing JUnit reports"]
    errors = []
    if expected is not None and sorted(reports) != sorted(expected):
        errors.append("JUnit report set differs from configured ranks/phases")
    passed = 0
    for name, counts in reports.items():
        if not isinstance(counts, dict) or any(
            type(counts.get(key)) is not int or counts[key] < 0 for key in COUNTS
        ):
            errors.append(f"invalid JUnit counts: {name}")
            continue
        if counts["tests"] != sum(counts[key] for key in COUNTS[1:]):
            errors.append(f"inconsistent JUnit counts: {name}")
        if counts["failures"] or counts["errors"]:
            errors.append(f"JUnit contains failures/errors: {name}")
        passed += counts["passed"]
    if passed == 0:
        errors.append("unit group executed no passing tests (empty or all skipped)")
    return errors


def collect(directory: pathlib.Path, expected: list[str]) -> dict[str, Any]:
    """Parse actual test cases; no individual phase is required to be nonempty."""
    reports = {}
    for path in sorted(directory.glob("*.xml")):
        root = ET.parse(path).getroot()
        if root.tag not in {"testsuite", "testsuites"}:
            raise ValueError(f"invalid JUnit root: {path.name}")
        cases = list(root.iter("testcase"))
        counts = {
            "tests": len(cases),
            "skipped": len(list(root.iter("skipped"))),
            "failures": len(list(root.iter("failure"))),
            "errors": len(list(root.iter("error"))),
        }
        counts["passed"] = counts["tests"] - sum(counts[key] for key in COUNTS[1:4])
        # A truncated report must not conceal counts advertised by its suites.
        suites = [node for node in root.iter("testsuite") if not node.findall("testsuite")]
        for key in COUNTS[:4]:
            if suites and all(key in node.attrib for node in suites):
                if sum(int(node.attrib[key]) for node in suites) != counts[key]:
                    raise ValueError(f"JUnit {key} count mismatch: {path.name}")
        reports[path.name] = counts
    evidence = {"schema_version": 1, "reports": reports}
    errors = validate(evidence, expected)
    if errors:
        raise ValueError("; ".join(errors))
    return evidence
