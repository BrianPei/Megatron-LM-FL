"""Regression tests for level selection, source identity, and report gates."""

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import flagcicd
import yaml

ROOT = Path(__file__).resolve().parents[3]


class FlagCICDTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.reports = Path(self.directory.name)
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, {'GITHUB_RUN_ID': '42', 'GITHUB_RUN_ATTEMPT': '1'}).start()
        self.plan = flagcicd.plan(ROOT, {'platforms': ['cuda'], 'test_level': 'L2'})
        self.metadata = self.plan['metadata']
        self.results = {'configuration': 'success', 'unit': 'success', 'functional': 'success'}
        self.run_results = {'configuration': 'success', 'lint': 'success', 'platforms': 'success'}

    def report(self, results=None):
        return flagcicd.platform_report(
            ROOT, 'cuda', self.metadata, results or self.results, self.reports
        )

    def coverage(self):
        config = yaml.safe_load((ROOT / '.github/configs/cuda.yml').read_text())
        for device in config['device_types']:
            (self.reports / f'coverage-cuda-{device}-all.json').write_text(
                json.dumps(
                    {
                        'totals': {
                            'covered_lines': 8,
                            'num_statements': 10,
                            'covered_branches': 2,
                            'num_branches': 4,
                            'percent_covered': 71.4,
                        }
                    }
                )
            )

    def save_report(self, report):
        (self.reports / 'flagcicd-report-cuda.json').write_text(json.dumps(report))

    def test_risk_mapping_upgrades_without_downgrading(self):
        cases = [
            ('docs/guide.md', 'L0', 'R1'),
            ('tests/unit_tests/test_a.py', 'L1', 'R1'),
            ('megatron/plugin/tests/test_a.py', 'L1', 'R1'),
            ('megatron/core/transformer/attention.py', 'L2', 'R3'),
            ('.github/configs/ppu.yml', 'L2', 'R2'),
            ('new_runtime.py', 'L2', 'R2'),
        ]
        for path, level, risk in cases:
            with self.subTest(path=path):
                result = flagcicd.plan(ROOT, {'test_level': 'L0', 'changed_files': [path]})
                self.assertEqual(result['metadata']['test_level'], level)
                self.assertEqual(result['metadata']['risk_level'], risk)
        result = flagcicd.plan(
            ROOT, {'test_level': 'L2', 'risk_level': 'R4', 'changed_files': ['README.md']}
        )
        self.assertEqual(result['metadata']['test_level'], 'L2')
        self.assertEqual(result['metadata']['risk_level'], 'R4')

    def test_rejects_invalid_or_unimplemented_requests(self):
        for request in (
            {'platforms': ['../cuda']},
            {'platforms': []},
            {'platforms': 'cuda'},
            {'test_level': 'L3'},
            {'test_level': 'L4'},
            {'risk_level': 'R0'},
            {'changed_files': [1]},
            {'task_id': 123},
        ):
            with self.subTest(request=request), self.assertRaises(ValueError):
                flagcicd.plan(ROOT, request)

    def test_all_platform_capabilities_and_deduplication(self):
        platforms = sorted(flagcicd._load_watch_rules(ROOT / '.github/configs'))
        result = flagcicd.plan(ROOT, {'platforms': platforms + platforms, 'test_level': 'L2'})
        self.assertEqual(result['platforms'], platforms)
        self.assertTrue(result['run_unit_tests'])
        self.assertTrue(result['run_functional_tests'])

    def test_missing_coverage_cannot_pass(self):
        self.assertEqual(self.report()['status'], 'report_error')
        self.coverage()
        report = self.report()
        self.assertEqual(report['status'], 'passed')
        coverage = report['metrics']['coverage']['children'][0]
        # coverage.py percent_covered includes branches; line coverage uses line counts.
        self.assertEqual(coverage['percent'], 80)
        self.assertEqual(coverage['branch']['percent'], 50)
        self.save_report(report)
        self.assertEqual(
            flagcicd.aggregate(self.plan, self.reports, self.run_results)['status'], 'passed'
        )

    def test_functional_failure_and_cancellation_cannot_pass(self):
        self.coverage()
        for result, status in [
            ('failure', 'test_failed'),
            ('skipped', 'test_failed'),
            ('cancelled', 'cancelled'),
        ]:
            with self.subTest(result=result):
                report = self.report({**self.results, 'functional': result})
                self.assertEqual(report['status'], status)
                self.save_report(report)
                self.assertNotEqual(
                    flagcicd.aggregate(self.plan, self.reports, self.run_results)['status'],
                    'passed',
                )

    def test_summary_rejects_missing_duplicate_and_stale_reports(self):
        self.assertEqual(
            flagcicd.aggregate(self.plan, self.reports, self.run_results)['status'], 'report_error'
        )
        self.coverage()
        report = self.report()
        for key in ('source_sha', 'run_id', 'attempt', 'task_id', 'test_level'):
            with self.subTest(key=key):
                stale = copy.deepcopy(report)
                stale[key] = 'stale'
                self.save_report(stale)
                summary = flagcicd.aggregate(self.plan, self.reports, self.run_results)
                self.assertEqual(summary['status'], 'report_error')
        self.save_report(report)
        (self.reports / 'flagcicd-report-duplicate.json').write_text(json.dumps(report))
        self.assertEqual(
            flagcicd.aggregate(self.plan, self.reports, self.run_results)['status'], 'report_error'
        )

    def test_rerun_reuses_successful_platforms_from_same_run(self):
        self.coverage()
        self.save_report(self.report())
        with patch.dict(os.environ, {'GITHUB_RUN_ATTEMPT': '2'}):
            summary = flagcicd.aggregate(self.plan, self.reports, self.run_results)
            self.assertEqual(summary['status'], 'passed')
            self.assertEqual(summary['attempt'], 2)
            self.assertEqual(summary['children'][0]['attempt'], 1)
            self.assertEqual(self.report()['attempt'], 2)
        report = self.report()
        report['attempt'] = 2
        self.save_report(report)
        self.assertEqual(
            flagcicd.aggregate(self.plan, self.reports, self.run_results)['status'], 'report_error'
        )

    def test_workflow_gate_overrides_successful_reports(self):
        self.coverage()
        self.save_report(self.report())
        for key in self.run_results:
            with self.subTest(key=key):
                summary = flagcicd.aggregate(
                    self.plan, self.reports, {**self.run_results, key: 'skipped'}
                )
                self.assertNotEqual(summary['status'], 'passed')

    def test_l0_skips_hardware_but_requires_preflight_and_lint(self):
        data = flagcicd.plan(ROOT, {'test_level': 'L0'})
        results = {**self.run_results, 'platforms': 'skipped'}
        self.assertFalse(data['run_unit_tests'])
        self.assertFalse(data['run_functional_tests'])
        self.assertEqual(flagcicd.aggregate(data, self.reports, results)['status'], 'passed')
        results['lint'] = 'failure'
        self.assertNotEqual(flagcicd.aggregate(data, self.reports, results)['status'], 'passed')

    def test_cli_manual_dispatch_and_nightly(self):
        events = [
            (
                'workflow_dispatch',
                {
                    'inputs': {
                        'platforms': '["ppu","cuda"]',
                        'test_level': 'L0',
                        'task_id': 'task\nvalue',
                    }
                },
                'L0',
            ),
            (
                'repository_dispatch',
                {
                    'action': 'flagcicd',
                    'client_payload': {
                        'platforms': ['cuda'],
                        'changed_files': ['megatron/training/x.py'],
                    },
                },
                'L2',
            ),
            ('schedule', {}, 'L2'),
        ]
        for event_name, event, level in events:
            with self.subTest(event=event_name):
                event_path = self.reports / 'event.json'
                event_path.write_text(json.dumps(event))
                outputs = self.reports / 'outputs'
                outputs.write_text('')
                output = self.reports / 'plan.json'
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / '.github/scripts/ci/flagcicd.py'),
                        'plan',
                        '--root',
                        str(ROOT),
                        '--output',
                        str(output),
                    ],
                    cwd=ROOT,
                    env={
                        **os.environ,
                        'GITHUB_EVENT_PATH': str(event_path),
                        'GITHUB_EVENT_NAME': event_name,
                        'GITHUB_OUTPUT': str(outputs),
                    },
                    check=True,
                )
                data = json.loads(output.read_text())
                self.assertEqual(data['metadata']['test_level'], level)
                self.assertEqual(len(outputs.read_text().splitlines()), 5)
                if event_name == 'schedule':
                    self.assertEqual(
                        data['platforms'],
                        sorted(flagcicd._load_watch_rules(ROOT / '.github/configs')),
                    )

    def test_one_nightly_schedule_and_pinned_source(self):
        workflows = {
            path.name: yaml.safe_load(path.read_text())
            for path in (ROOT / '.github/workflows').glob('*.yml')
        }
        self.assertNotIn('schedule', workflows['ci.yml'][True])
        entry = workflows['flagcicd-entry.yml']
        self.assertEqual(entry[True]['schedule'], [{'cron': '0 18 * * *'}])
        run = entry['jobs']['run_tests']
        self.assertEqual(run['uses'], './.github/workflows/_platform_pipeline.yml')
        self.assertEqual(run['with']['source_ref'], '${{ needs.plan.outputs.source_sha }}')
        self.assertIn('run_unit_tests', run['if'])


if __name__ == '__main__':
    unittest.main()
