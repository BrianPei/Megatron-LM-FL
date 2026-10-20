#!/usr/bin/env python3
"""Plan registered Megatron tests and verify execution against that exact plan."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml
from generate_matrix import _changed_files, _load_watch_rules, select_platforms

SCHEMA = 'flagcicd.report/v2'
CONTRACT = Path(__file__).resolve().parents[3] / '.flagcicd/project.yaml'


def string_list(value: object, name: str) -> list[str]:
    """Require a list of nonempty strings at the external input boundary."""
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f'{name} must be an array of nonempty strings')
    return value


def source_sha(root: Path) -> str:
    """Read the source actually checked out, rather than a mutable branch name."""
    return subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()


def inventory(root: Path) -> dict:
    """Validate every registered entry and derive capabilities without duplicating matrices."""
    contract = yaml.safe_load(CONTRACT.read_text())
    if contract.get('schema_version') != 'flagcicd.project/v2' or set(contract['levels']) != {
        'L0',
        'L1',
        'L2',
        'L3',
        'L4',
    }:
        raise ValueError('Invalid project manifest levels/schema')
    if contract.get('report_schema') != SCHEMA:
        raise ValueError('Project and reporter schemas differ')
    platforms = {}
    known = _load_watch_rules(root / '.github/configs')
    if set(contract['migration_baseline']) != set(known):
        raise ValueError('Baseline platform inventory differs from registered platforms')
    for platform in known:
        path = root / f'.github/configs/{platform}.yml'
        config = yaml.safe_load(path.read_text())
        for field in ('ci_image', 'runner_labels', 'device_types', 'setup_script', 'te_fl'):
            if not config.get(field):
                raise ValueError(f'{platform}: missing {field}')
        devices = string_list(config['device_types'], 'device_types')
        if len(devices) != len(set(devices)):
            raise ValueError(f'{platform}: duplicate device')
        setup = root / config['setup_script']
        if not setup.is_file() or not setup.resolve().is_relative_to(root.resolve()):
            raise ValueError(f'{platform}: invalid setup_script')
        entries, seen, owners = [], set(), {}
        matrix = config.get('test_matrix', {})
        for kind, items in (
            ('unit', matrix.get('unit', {}).get('groups', [])),
            ('functional', matrix.get('functional', {}).get('train', [])),
        ):
            for item in items:
                name = item['name'] if kind == 'unit' else f"{item['model']}/{item['test_case']}"
                identity = f'{kind}:{name}'
                level = item.get('ci_level')
                if level not in contract['levels'] or kind not in contract['levels'][level].get(
                    'kinds', []
                ):
                    raise ValueError(f'{platform}/{identity}: missing or invalid ci_level')
                if identity in seen:
                    raise ValueError(f'{platform}/{identity}: duplicate entry')
                seen.add(identity)
                if kind == 'unit':
                    matches = set()
                    for pattern in shlex.split(item['path']):
                        found = glob.glob(str(root / pattern.split('::')[0]))
                        if not found:
                            raise ValueError(f'{platform}/{identity}: missing path {pattern}')
                        for match in found:
                            candidate = Path(match)
                            if not candidate.resolve().is_relative_to(root.resolve()):
                                raise ValueError('Test paths must stay inside the repository')
                            matches.update(
                                candidate.rglob('test_*.py') if candidate.is_dir() else [candidate]
                            )
                    for match in matches:
                        if match in owners:
                            raise ValueError(
                                f'{platform}: overlapping unit groups {owners[match]} / {name}: {match}'
                            )
                        owners[match] = name
                else:
                    case_root = root / 'tests/functional_tests/test_cases' / name
                    required = [case_root / 'model_config.yaml', root / item['training_script']]
                    if not item.get('enable_lightweight_mode', False):
                        required.append(
                            case_root
                            / f"golden_values_{item['golden_environment']}_{item['golden_platform']}.json"
                        )
                    if any(
                        not file.is_file() or not file.resolve().is_relative_to(root.resolve())
                        for file in required
                    ):
                        raise ValueError(
                            f'{platform}/{identity}: missing training/config/golden contract'
                        )
                    if type(item.get('gpus_per_node')) is not int or item['gpus_per_node'] < 1:
                        raise ValueError(f'{platform}/{identity}: invalid device count')
                entries.append(
                    {
                        'id': identity,
                        'kind': kind,
                        'level': level,
                        'baseline': identity in contract['migration_baseline'].get(platform, []),
                        'config': item,
                    }
                )
        baseline = string_list(
            contract['migration_baseline'].get(platform, []), 'migration_baseline'
        )
        if len(baseline) != len(set(baseline)) or set(baseline) - seen:
            raise ValueError(f'{platform}: baseline entries missing or duplicated')
        if not entries or not any(entry['baseline'] for entry in entries):
            raise ValueError(f'{platform}: empty registered baseline')
        platforms[platform] = {
            'entries': entries,
            'devices': devices,
            'config': config,
            'config_hash': hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return platforms


def task_ids(entries: list[dict], devices: list[str]) -> list[str]:
    """Use group/case and device as the execution granularity, preserving existing skips."""
    return sorted(
        f"{entry['kind']}:{device}:{entry['id'].split(':', 1)[1]}"
        for device in devices
        for entry in entries
    )


def plan(root: Path, request: dict, event_name: str = 'workflow_dispatch') -> dict:
    """Freeze supported scope, exclusions, baseline and source identity before testing."""
    catalog = inventory(root)
    mode = request.get('mode', 'baseline')
    if mode not in ('baseline', 'level', 'functional', 'nightly', 'weekly'):
        raise ValueError('Invalid execution mode')
    target = request.get('test_level', 'L2')
    contract = yaml.safe_load(CONTRACT.read_text())
    if mode in contract['schedules']:
        target = contract['schedules'][mode]['target_level']
    if target not in contract['levels']:
        raise ValueError('test_level must be L0-L4')
    if mode == 'baseline':
        target = 'L2'
    labels = string_list(request.get('labels', []), 'labels')
    selected = string_list(request.get('platforms', ['all']), 'platforms')
    reasons = ['Full registered baseline; no implicit diff reduction']
    if event_name == 'pull_request':
        selected, reasons = select_platforms(
            _load_watch_rules(root / '.github/configs'),
            request.get('changed_files', []),
            event_name,
            labels,
        )
        mode, target = 'baseline', 'L2'
    elif mode in ('nightly', 'weekly'):
        selected = sorted(catalog)
        reasons = [
            f'{mode}: all platforms, declared supported entries through {target}; no PR diff'
        ]
    elif 'all' in selected:
        selected = sorted(catalog)
    if set(selected) - catalog.keys() or (not selected and event_name != 'pull_request'):
        raise ValueError('Unknown or empty platform selection')
    selected = sorted(set(selected))
    plans = {}
    for platform in selected:
        info = catalog[platform]
        entries = info['entries']
        if mode == 'baseline':
            chosen = [entry for entry in entries if entry['baseline']]
        else:
            chosen = [
                entry
                for entry in entries
                if entry['level'] <= target
                and target != 'L0'
                and (mode != 'functional' or entry['kind'] == 'functional')
            ]
        capabilities = sorted({entry['level'] for entry in entries})
        if (
            mode in ('level', 'functional')
            and target != 'L0'
            and not any(entry['level'] == target for entry in chosen)
        ):
            raise ValueError(
                f'{platform}: no implemented {target} suite; supported levels: {capabilities}'
            )
        missing = [
            level
            for level in ('L1', 'L2', 'L3', 'L4')
            if level <= target and level not in capabilities
        ]
        config = info['config']
        plans[platform] = {
            'entry_ids': [entry['id'] for entry in chosen],
            'required_tasks': task_ids(chosen, info['devices']),
            'config_hash': info['config_hash'],
            'validated_level': (
                None
                if mode == 'functional'
                else max((entry['level'] for entry in chosen), default='L0')
            ),
            'highest_test_level': max((entry['level'] for entry in chosen), default='L0'),
            'run_unit_tests': any(entry['kind'] == 'unit' for entry in chosen),
            'run_functional_tests': any(entry['kind'] == 'functional' for entry in chosen),
            'capability_gaps': (
                missing if mode != 'functional' else [level for level in missing if level >= 'L2']
            ),
            'excluded_entries': [
                {'id': entry['id'], 'reason': 'outside requested scope'}
                for entry in entries
                if entry not in chosen
            ],
            'existing_unit_exclusions': config.get('test_matrix', {})
            .get('unit', {})
            .get('ignored_tests', []),
            'unit': [entry['config'] for entry in chosen if entry['kind'] == 'unit'],
            'functional': [entry['config'] for entry in chosen if entry['kind'] == 'functional'],
        }
    scope = (
        'supplemental'
        if mode == 'functional' or (event_name != 'pull_request' and selected != sorted(catalog))
        else (
            'supported_capabilities'
            if mode in ('nightly', 'weekly')
            else 'selected_level' if mode == 'level' else 'baseline'
        )
    )
    metadata = {
        'source_sha': source_sha(root),
        'run_id': os.environ.get('GITHUB_RUN_ID', 'local'),
        'attempt': int(os.environ.get('GITHUB_RUN_ATTEMPT', '1')),
        'task_id': request.get('task_id') or f"megatron-{os.environ.get('GITHUB_RUN_ID', 'local')}",
        'parent_run_id': request.get('parent_run_id', ''),
        'test_level': target,
        'mode': mode,
        'validation_scope': scope,
        'selection_reasons': reasons,
        'platform_plans': plans,
    }
    for key in ('task_id', 'parent_run_id'):
        if not isinstance(metadata[key], str):
            raise ValueError(f'{key} must be a string')
    unit = any(item['unit'] for item in plans.values())
    functional = any(item['functional'] for item in plans.values())
    return {
        'schema_version': 'flagcicd.plan/v2',
        'platforms': [platform for platform in selected if plans[platform]['required_tasks']],
        'metadata': metadata,
        'run_unit_tests': unit,
        'run_functional_tests': functional,
        'run_hardware': unit or functional,
    }


def select_tests(root: Path, platform: str, metadata: dict, unit: bool, functional: bool) -> dict:
    """Resolve the exact matrix from the frozen plan and reject changed configurations."""
    info = inventory(root)[platform]
    expected = metadata.get('platform_plans', {}).get(platform)
    if expected:
        if expected['config_hash'] != info['config_hash']:
            raise ValueError('Platform configuration changed after planning')
        ids = set(expected['entry_ids'])
        entries = [entry for entry in info['entries'] if entry['id'] in ids]
        if {entry['id'] for entry in entries} != ids or len(ids) != len(expected['entry_ids']):
            raise ValueError('Invalid planned entry identities')
    else:
        entries = [
            entry
            for entry in info['entries']
            if entry['baseline'] and (unit if entry['kind'] == 'unit' else functional)
        ]
    if any(
        (entry['kind'] == 'unit' and not unit) or (entry['kind'] == 'functional' and not functional)
        for entry in entries
    ):
        raise ValueError('Test switches would omit planned entries')
    return {
        'unit_test_groups': [entry['config'] for entry in entries if entry['kind'] == 'unit'],
        'train_test_matrix': [
            entry['config'] for entry in entries if entry['kind'] == 'functional'
        ],
        'required_tasks': task_ids(entries, info['devices']),
    }


def status_for(results: dict, required: list[str]) -> str:
    """Missing, skipped and cancelled required jobs cannot count as success."""
    values = [results.get(key) for key in required]
    if 'cancelled' in values:
        return 'cancelled'
    if results.get('configuration') != 'success':
        return 'infra_error'
    return 'passed' if all(value == 'success' for value in values) else 'test_failed'


def base_report(metadata: dict) -> dict:
    """Keep identity and selection context without repeating every platform matrix."""
    return {
        **{key: value for key, value in metadata.items() if key != 'platform_plans'},
        'schema_version': SCHEMA,
        'repository': os.environ.get('GITHUB_REPOSITORY', 'Megatron-LM-FL'),
        'created_at': datetime.now(timezone.utc).isoformat(),
    }


def platform_report(
    root: Path, platform: str, metadata: dict, results: dict, coverage_dir: Path
) -> dict:
    """Check actual group/case receipts and preserve per-device coverage."""
    unit = os.environ.get('RUN_UNIT_TESTS', 'true') == 'true'
    functional = os.environ.get('RUN_FUNCTIONAL_TESTS', 'true') == 'true'
    selection = select_tests(root, platform, metadata, unit, functional)
    info = inventory(root)[platform]
    metadata = {
        'run_id': os.environ.get('GITHUB_RUN_ID', 'local'),
        'task_id': '',
        'parent_run_id': '',
        'mode': 'legacy',
        'validation_scope': 'supplemental',
        **metadata,
        'attempt': int(os.environ.get('GITHUB_RUN_ATTEMPT', '1')),
        'source_sha': source_sha(root),
    }
    required = (
        ['configuration']
        + (['unit'] if selection['unit_test_groups'] else [])
        + (['functional'] if selection['train_test_matrix'] else [])
    )
    status = status_for(results, required)
    errors, metrics, executed = [], [], {}
    for device in info['devices']:
        coverage = {'available': False, 'scope': 'unit', 'device': device}
        paths = list(coverage_dir.rglob(f'coverage-{platform}-{device}-all.json'))
        if selection['unit_test_groups']:
            states = list(coverage_dir.glob(f'coverage-{platform}-{device}/unit-results.json'))
            if len(states) != 1:
                errors.append(f'{device}: missing or duplicate unit execution receipt')
            else:
                try:
                    receipts = json.loads(states[0].read_text())
                    names = set()
                    for record in receipts:
                        name = record['name']
                        key = f'unit:{device}:{name}'
                        if name in names or type(record['exit_code']) is not int:
                            raise ValueError('duplicate/invalid unit result')
                        names.add(name)
                        executed[key] = 'passed' if record['exit_code'] == 0 else 'test_failed'
                except (KeyError, TypeError, ValueError) as exc:
                    errors.append(f'{device}: {exc}')
        if paths:
            try:
                if len(paths) != 1:
                    raise ValueError('duplicate combined coverage')
                totals = json.loads(paths[0].read_text())['totals']
                covered, total = totals['covered_lines'], totals['num_statements']
                if not 0 <= covered <= total:
                    raise ValueError('invalid coverage counts')
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
                errors.append(f'{device}: {exc}')
        elif selection['unit_test_groups'] and results.get('unit') == 'success':
            errors.append(f'{device}: missing combined coverage')
        metrics.append(coverage)
    for path in coverage_dir.rglob('receipt.json'):
        try:
            receipt = json.loads(path.read_text())
            key = receipt['task_id']
            if (
                receipt['platform'] != platform
                or receipt['source_sha'] != metadata['source_sha']
                or (
                    receipt['run_id'] != metadata['run_id']
                    or type(receipt['attempt']) is not int
                    or not 1 <= receipt['attempt'] <= metadata['attempt']
                    or key in executed
                )
            ):
                raise ValueError('duplicate or stale execution receipt')
            executed[key] = 'passed' if receipt['outcome'] == 'success' else receipt['outcome']
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f'{path}: {exc}')
    expected = set(selection['required_tasks'])
    if set(executed) != expected or any(value != 'passed' for value in executed.values()):
        errors.append(
            f'Execution mismatch: missing={sorted(expected - executed.keys())}; '
            f'unexpected={sorted(executed.keys() - expected)}; all required tasks must pass'
        )
    planned = metadata.get('platform_plans', {}).get(platform, {})
    return {
        **base_report(metadata),
        'platform': {
            'name': platform,
            'config_hash': info['config_hash'],
            'image_ref': os.environ.get('CI_IMAGE', info['config']['ci_image']),
        },
        'status': 'report_error' if errors and status == 'passed' else status,
        'validated_level': planned.get('validated_level'),
        'capability_gaps': planned.get('capability_gaps', []),
        'required_tasks': sorted(expected),
        'executed_tasks': executed,
        'excluded_entries': planned.get('excluded_entries', []),
        'existing_unit_exclusions': info['config']
        .get('test_matrix', {})
        .get('unit', {})
        .get('ignored_tests', []),
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
    """Verify every required platform and task, never promote supplemental scope."""
    metadata = {**plan_data['metadata'], 'attempt': int(os.environ.get('GITHUB_RUN_ATTEMPT', '1'))}
    plans = metadata['platform_plans']
    expected = {platform for platform, item in plans.items() if item['required_tasks']}
    children, errors, seen = [], [], set()
    for path in sorted(reports_dir.rglob('flagcicd-report-*.json')):
        try:
            report = json.loads(path.read_text())
            platform = report['platform']['name']
            if report['schema_version'] != SCHEMA or platform not in expected or platform in seen:
                raise ValueError('Invalid, unexpected or duplicate platform')
            for key in (
                'source_sha',
                'run_id',
                'task_id',
                'parent_run_id',
                'test_level',
                'mode',
                'validation_scope',
            ):
                if report.get(key) != metadata[key]:
                    raise ValueError(f'mismatched {key}')
            if (
                type(report.get('attempt')) is not int
                or not 1 <= report['attempt'] <= metadata['attempt']
            ):
                raise ValueError('invalid report attempt')
            item = plans[platform]
            if report['platform']['config_hash'] != item['config_hash'] or (
                report.get('required_tasks') != sorted(item['required_tasks'])
                or set(report.get('executed_tasks', {})) != set(item['required_tasks'])
                or any(value != 'passed' for value in report['executed_tasks'].values())
                or report.get('validated_level') != item['validated_level']
                or report.get('capability_gaps') != item['capability_gaps']
                or report.get('errors')
                or report.get('status') != 'passed'
            ):
                raise ValueError('Planned task, capability or result mismatch')
            seen.add(platform)
            children.append(report)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f'{path}: {exc}')
    if expected - seen:
        errors.append(f'missing platforms: {sorted(expected - seen)}')
    status = status_for(results, ['configuration', 'lint'] + (['platforms'] if expected else []))
    if not expected and results.get('platforms') != 'skipped':
        errors.append('Hardware jobs should have been skipped')
    return {
        **base_report(metadata),
        'report_type': 'run_summary',
        'platform': {'name': 'multi'},
        'status': 'report_error' if errors and status == 'passed' else status,
        'required_tasks': {platform: item['required_tasks'] for platform, item in plans.items()},
        'capabilities': {
            platform: {'validated_level': item['validated_level'], 'gaps': item['capability_gaps']}
            for platform, item in plans.items()
        },
        'validated_level': (
            min((item['validated_level'] for item in plans.values()), default='L0')
            if status == 'passed' and not errors and metadata['mode'] != 'functional'
            else None
        ),
        'merge_acceptance': metadata['validation_scope'] == 'baseline'
        and status == 'passed'
        and not errors,
        'child_jobs': results,
        'children': children,
        'errors': errors,
    }


def main() -> None:
    """Normalize event JSON, publish a frozen plan, or validate actual execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'selection', 'report', 'aggregate', 'validate'])
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--input', type=Path)
    parser.add_argument('--reports-dir', type=Path, default=Path('flagcicd-reports'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command == 'validate':
        inventory(args.root)
        return
    if args.command == 'plan':
        event = json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_text())
        event_name = os.environ.get('GITHUB_EVENT_NAME', 'workflow_dispatch')
        request = dict(
            event.get('client_payload')
            if event.get('action') == 'flagcicd'
            else event.get('inputs', {}) or {}
        )
        if event_name == 'pull_request':
            request = {
                'labels': [label['name'] for label in event['pull_request'].get('labels', [])]
            }
            if 'ci:diff' in request['labels']:
                try:
                    request['changed_files'] = _changed_files(os.environ.get('BASE_REF', 'main'))
                except (OSError, subprocess.CalledProcessError):
                    request['changed_files'] = []
        if event_name == 'schedule':
            contract = yaml.safe_load(CONTRACT.read_text())
            request = {
                'mode': next(
                    name
                    for name, item in contract['schedules'].items()
                    if item['cron'] == event['schedule']
                )
            }
        for key in ('platforms',):
            if isinstance(request.get(key), str):
                request[key] = json.loads(request[key])
        payload = plan(args.root, request, event_name)
        with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
            for key in (
                'platforms',
                'metadata',
                'run_unit_tests',
                'run_functional_tests',
                'run_hardware',
            ):
                output.write(f'{key}={json.dumps(payload[key], separators=(",", ":"))}\n')
            output.write(f'source_sha={payload["metadata"]["source_sha"]}\n')
    elif args.command == 'selection':
        payload = select_tests(
            args.root,
            os.environ['PLATFORM'],
            json.loads(os.environ['FLAGCICD_METADATA']),
            os.environ['RUN_UNIT_TESTS'] == 'true',
            os.environ['RUN_FUNCTIONAL_TESTS'] == 'true',
        )
        with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
            for key in ('unit_test_groups', 'train_test_matrix'):
                output.write(f'{key}={json.dumps(payload[key], separators=(",", ":"))}\n')
        return
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
        payload = (
            aggregate(json.loads(args.input.read_text()), args.reports_dir, results)
            if args.input.is_file()
            else {
                **base_report({'run_id': os.environ.get('GITHUB_RUN_ID', 'local')}),
                'report_type': 'run_summary',
                'status': 'infra_error',
                'errors': ['Execution plan unavailable'],
                'child_jobs': results,
            }
        )
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    if summary := os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(summary, 'a') as stream:
            stream.write(
                '### Validation scope and execution plan\n\n```json\n'
                + json.dumps(payload, indent=2)
                + '\n```\n'
            )
    if payload.get('status', 'passed') != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
