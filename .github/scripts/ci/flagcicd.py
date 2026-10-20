#!/usr/bin/env python3
"""Plan FlagCICD requests and report existing Megatron CI results."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml
from generate_matrix import _load_watch_rules

SCHEMA = 'flagcicd.report/v1'


def string_list(value: object, name: str) -> list[str]:
    """Validate external arrays before using them in paths or job outputs."""
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f'{name} must be an array of nonempty strings')
    return value


def plan(root: Path, request: dict) -> dict:
    """Resolve supported levels and validate the selected platform capabilities."""
    contract = yaml.safe_load(Path('.flagcicd/project.yaml').read_text())
    platforms = sorted(set(string_list(request.get('platforms', ['cuda']), 'platforms')))
    available = _load_watch_rules(root / '.github/configs')
    if not platforms or set(platforms) - available.keys():
        raise ValueError(f'platforms must be selected from {sorted(available)}')
    level = request.get('test_level', 'L1')
    risk = request.get('risk_level', 'auto')
    if level not in contract['levels']:
        raise ValueError('Only L0, L1 and L2 are implemented; L3/L4 are unsupported')
    if risk not in ('auto', 'R1', 'R2', 'R3', 'R4'):
        raise ValueError('risk_level must be auto or R1-R4')
    risk = 'R1' if risk == 'auto' else risk
    changed = string_list(request.get('changed_files', []), 'changed_files')
    matched = []
    rules = contract['risk_rules']
    for path in changed:
        path = path.removeprefix('./')
        # Ordered rules allow plugin unit tests to stay L1 instead of matching runtime.
        rule = next(
            (
                rule
                for rule in rules['rules']
                if any(fnmatch.fnmatchcase(path, pattern) for pattern in rule['paths'])
            ),
            None,
        )
        minimum = rule['minimum_level'] if rule else rules['unknown_level']
        inferred_risk = rule['risk'] if rule else rules['unknown_risk']
        level = max(level, minimum)
        risk = max(risk, inferred_risk)
        matched.append({'path': path, 'rule': rule['id'] if rule else 'unknown'})
    for platform in platforms:
        config = yaml.safe_load((root / f'.github/configs/{platform}.yml').read_text())
        for field in ('ci_image', 'runner_labels', 'device_types', 'setup_script'):
            if not config.get(field):
                raise ValueError(f'{platform}: missing {field}')
        if not (root / config['setup_script']).is_file():
            raise ValueError(f'{platform}: setup_script does not exist')
        matrix = config.get('test_matrix', {})
        if level != 'L0' and not matrix.get('unit', {}).get('groups'):
            raise ValueError(f'{platform}: unit test groups are empty')
        if level == 'L2' and not matrix.get('functional', {}).get('train'):
            raise ValueError(f'{platform}: functional test matrix is empty')
    sha = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    metadata = {
        'source_sha': sha,
        'run_id': os.environ.get('GITHUB_RUN_ID', 'local'),
        'attempt': int(os.environ.get('GITHUB_RUN_ATTEMPT', '1')),
        'task_id': request.get('task_id') or f"megatron-{os.environ.get('GITHUB_RUN_ID', 'local')}",
        'parent_run_id': request.get('parent_run_id', ''),
        'test_level': level,
        'risk_level': risk,
        'risk_rule_version': rules['version'],
    }
    for key in ('task_id', 'parent_run_id'):
        if not isinstance(metadata[key], str):
            raise ValueError(f'{key} must be a string')
    return {
        'schema_version': 'flagcicd.plan/v1',
        'platforms': platforms,
        'metadata': metadata,
        'matched_rules': matched,
        'run_unit_tests': level != 'L0',
        'run_functional_tests': level == 'L2',
    }


def status_for(results: dict, required: list[str]) -> str:
    """Treat missing and skipped required jobs as failures."""
    values = [results.get(key) for key in required]
    if 'cancelled' in values:
        return 'cancelled'
    if results.get('configuration') != 'success':
        return 'infra_error'
    return 'passed' if all(value == 'success' for value in values) else 'test_failed'


def base_report(metadata: dict) -> dict:
    """Attach the same source and task identity to every report."""
    return {
        **metadata,
        'schema_version': SCHEMA,
        'repository': os.environ.get('GITHUB_REPOSITORY', 'Megatron-LM-FL'),
        'created_at': datetime.now(timezone.utc).isoformat(),
    }


def platform_report(
    root: Path, platform: str, metadata: dict, results: dict, coverage_dir: Path
) -> dict:
    """Report workflow conclusions and coverage separately for each device."""
    config = yaml.safe_load((root / f'.github/configs/{platform}.yml').read_text())
    sha = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    level = metadata.get('test_level') or os.environ['TEST_LEVEL']
    metadata = {
        'run_id': os.environ.get('GITHUB_RUN_ID', 'local'),
        'task_id': '',
        'parent_run_id': '',
        'risk_level': 'R1',
        'risk_rule_version': 'v1',
        **metadata,
        'attempt': int(os.environ.get('GITHUB_RUN_ATTEMPT', '1')),
        'test_level': level,
        'source_sha': sha,
    }
    required = ['configuration']
    if os.environ.get('RUN_UNIT_TESTS', str(level != 'L0').lower()) == 'true':
        required.append('unit')
    if os.environ.get('RUN_FUNCTIONAL_TESTS', str(level == 'L2').lower()) == 'true':
        required.append('functional')
    status = status_for(results, required)
    errors = []
    metrics = []
    for device in config['device_types']:
        coverage = {'available': False, 'scope': 'unit', 'device': device}
        paths = list(coverage_dir.rglob(f'coverage-{platform}-{device}-all.json'))
        if paths:
            try:
                if len(paths) != 1:
                    raise ValueError('duplicate combined coverage reports')
                payload = json.loads(paths[0].read_text())
                totals = payload['totals']
                covered, total = totals['covered_lines'], totals['num_statements']
                if not 0 <= covered <= total:
                    raise ValueError('invalid line counts')
                coverage.update(
                    available=True,
                    metric_type='line',
                    covered=covered,
                    total=total,
                    percent=100 * covered / total if total else 0,
                )
                if 'num_branches' in totals:
                    branches = totals['num_branches']
                    if not 0 <= totals['covered_branches'] <= branches:
                        raise ValueError('invalid branch counts')
                    coverage['branch'] = {
                        'covered': totals['covered_branches'],
                        'total': branches,
                        'percent': 100 * totals['covered_branches'] / branches if branches else 0,
                    }
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f'{device}: invalid coverage: {exc}')
        elif 'unit' in required and results.get('unit') == 'success':
            errors.append(f'{device}: missing combined coverage report')
        metrics.append(coverage)
    return {
        **base_report(metadata),
        'platform': {
            'name': platform,
            'image_ref': os.environ.get('CI_IMAGE', config['ci_image']),
            'config_hash': hashlib.sha256(
                (root / f'.github/configs/{platform}.yml').read_bytes()
            ).hexdigest(),
        },
        'status': 'report_error' if errors and status == 'passed' else status,
        'child_jobs': results,
        'metrics': {
            'coverage': {
                'available': any(m['available'] for m in metrics),
                'aggregation': 'per_device_only',
                'children': metrics,
            }
        },
        'errors': errors,
    }


def aggregate(plan_data: dict, reports_dir: Path, results: dict) -> dict:
    """Reject missing, duplicate or stale reports and include the execution gate."""
    metadata = {**plan_data['metadata'], 'attempt': int(os.environ.get('GITHUB_RUN_ATTEMPT', '1'))}
    children, errors = [], []
    expected = set(plan_data['platforms'])
    seen = set()
    for path in sorted(reports_dir.rglob('flagcicd-report-*.json')):
        try:
            report = json.loads(path.read_text())
            platform = report['platform']['name']
            if report['schema_version'] != SCHEMA or platform not in expected or platform in seen:
                raise ValueError('invalid schema, unexpected or duplicate platform')
            for key in (
                'source_sha',
                'run_id',
                'task_id',
                'parent_run_id',
                'test_level',
                'risk_level',
                'risk_rule_version',
            ):
                if report.get(key) != metadata[key]:
                    raise ValueError(f'mismatched {key}')
            # Successful platforms may be reused by GitHub's rerun-failed-jobs.
            if (
                not isinstance(report.get('attempt'), int)
                or not 1 <= report['attempt'] <= metadata['attempt']
            ):
                raise ValueError('invalid report attempt')
            if report.get('status') != 'passed':
                errors.append(f'{platform}: {report.get("status", "missing status")}')
            seen.add(platform)
            children.append(report)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f'{path}: {exc}')
    preflight = metadata['test_level'] == 'L0'
    if not preflight and expected - seen:
        errors.append(f'missing platforms: {sorted(expected - seen)}')
    required = ['configuration', 'lint'] + ([] if preflight else ['platforms'])
    status = status_for(results, required)
    if preflight and results.get('platforms') != 'skipped':
        errors.append('L0 must skip hardware jobs')
    return {
        **base_report(metadata),
        'report_type': 'run_summary',
        'platform': {'name': 'multi'},
        'status': 'report_error' if errors and status == 'passed' else status,
        'platforms': sorted(expected),
        'child_jobs': results,
        'children': children,
        'metrics': {'coverage': {'available': False, 'aggregation': 'per_platform_only'}},
        'errors': errors,
    }


def main() -> None:
    """Run the selected command using JSON inputs rather than shell interpolation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'report', 'aggregate'])
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--input', type=Path)
    parser.add_argument('--reports-dir', type=Path, default=Path('flagcicd-reports'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'plan':
        event = json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_text())
        request = (
            event.get('client_payload')
            if event.get('action') == 'flagcicd'
            else event.get('inputs', {})
        )
        request = dict(request or {})
        if os.environ.get('GITHUB_EVENT_NAME') == 'schedule':
            request = {
                'platforms': sorted(_load_watch_rules(args.root / '.github/configs')),
                'test_level': 'L2',
            }
        for key in ('platforms', 'changed_files'):
            if isinstance(request.get(key), str):
                request[key] = json.loads(request[key])
        payload = plan(args.root, request)
        with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
            for key in ('platforms', 'metadata', 'run_unit_tests', 'run_functional_tests'):
                output.write(f'{key}={json.dumps(payload[key], separators=(",", ":"))}\n')
            output.write(f'source_sha={payload["metadata"]["source_sha"]}\n')
    elif args.command == 'report':
        payload = platform_report(
            args.root,
            os.environ['PLATFORM'],
            json.loads(os.environ['FLAGCICD_METADATA']),
            json.loads(os.environ['JOB_RESULTS']),
            args.reports_dir,
        )
    else:
        results = json.loads(os.environ['JOB_RESULTS'])
        if args.input.is_file():
            payload = aggregate(json.loads(args.input.read_text()), args.reports_dir, results)
        else:
            payload = {
                **base_report({'run_id': os.environ.get('GITHUB_RUN_ID', 'local')}),
                'report_type': 'run_summary',
                'status': (
                    'cancelled' if results.get('configuration') == 'cancelled' else 'infra_error'
                ),
                'child_jobs': results,
                'errors': ['Resolved plan is unavailable; preflight did not complete'],
            }
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    if payload.get('status', 'passed') != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
