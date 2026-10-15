#!/usr/bin/env python3
"""Report performance observations from the existing functional training smoke tests."""

import json
import math
import os
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

ITERATION = re.compile(r"\biteration\s+(\d+)\s*/\s*\d+\s*\|")
LATENCY = re.compile(r"elapsed time per iteration \(ms\):\s*(\S+)")
THROUGHPUT = re.compile(r"throughput per GPU \(TFLOP/s/GPU\):\s*(\S+)")


def summarize(samples: list[float], unit: str) -> dict:
    """Keep the samples as well as their descriptive statistics."""
    return {
        "unit": unit,
        "samples": samples,
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
    }


def measurements(log: Path) -> dict:
    """Parse rank-zero training output, excluding iteration one as warmup."""
    latency, throughput = [], []
    with log.open(errors="replace") as stream:
        for line in stream:
            iteration = ITERATION.search(line)
            if iteration is None or int(iteration[1]) <= 1:
                continue
            for pattern, samples in ((LATENCY, latency), (THROUGHPUT, throughput)):
                match = pattern.search(line)
                if match:
                    value = float(match[1])
                    if not math.isfinite(value) or value <= 0:
                        raise ValueError(f"Invalid performance measurement: {match[0]}")
                    samples.append(value)
    return {
        name: summarize(samples, unit)
        for name, samples, unit in (
            ("latency", latency, "ms"),
            ("throughput", throughput, "TFLOP/s/GPU"),
        )
        if samples
    }


def build_report(environment: dict, workspace: Path) -> dict:
    """Use only the final retry attempt and keep test failure distinct from missing metrics."""
    config = json.loads(environment["CI_TEST_CONFIG"])
    name = config["test_case"]
    logs = workspace / ".retry-logs" / f"functional-{name}"
    exit_code = int(environment["CI_TEST_EXIT_CODE"])
    case = {
        "name": name,
        "model": config["model"],
        "exit_code": exit_code,
        "status": "unavailable" if exit_code == 0 else "failed",
    }
    try:
        result = json.loads((logs / "result.json").read_text())
        attempt = result["attempts"][-1]
        log = logs / f"attempt-{int(attempt['attempt'])}.log"
        case["attempt"] = attempt["attempt"]
        case.update(measurements(log))
        if "latency" not in case:
            case["error"] = "No measured training iterations after warmup"
        if attempt["exit_code"] != 0:
            case["status"] = "failed"
        elif exit_code == 0 and "latency" in case:
            case["status"] = "passed"
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        case["error"] = f"{type(error).__name__}: {error}"
    return {
        "schema_version": 1,
        "report_type": "performance_smoke",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": environment["CI_PLATFORM"],
            "device": environment["CI_DEVICE"],
            "image": environment["CI_IMAGE"],
            "commit": environment["CI_SOURCE_REF"],
            "run_id": environment.get("GITHUB_RUN_ID", ""),
            "run_attempt": environment.get("GITHUB_RUN_ATTEMPT", ""),
        },
        "measurement": {
            "method": "megatron_training_log",
            "excluded_iterations": [1],
            "configuration": config,
        },
        "cases": [case],
        "summary": {
            "total": 1,
            "passed": int(case["status"] == "passed"),
            "failed": int(case["status"] == "failed"),
            "unavailable": int(case["status"] == "unavailable"),
            "status": case["status"],
        },
    }


def main() -> None:
    """Write a per-platform, per-device, per-case artifact for upload."""
    workspace = Path(os.environ.get("GITHUB_WORKSPACE", ".")).resolve()
    report = build_report(dict(os.environ), workspace)
    name = report["cases"][0]["name"]
    output = workspace / "performance-report"
    output.mkdir(exist_ok=True)
    path = output / f"perf-smoke-{os.environ['CI_PLATFORM']}-{os.environ['CI_DEVICE']}-{name}.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Performance smoke report: {path} ({report['summary']['status']})")


if __name__ == "__main__":
    main()
