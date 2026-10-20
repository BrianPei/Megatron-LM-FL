"""Exercise real plans, matrix selection and per-task execution gates without hardware."""

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
SHA = 'a' * 40


class FlagCICDTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.reports = self.root / 'reports'
        self.reports.mkdir()
        self.config_path = self.root / '.github/configs/cuda.yml'
        self.config_path.parent.mkdir(parents=True)
        self.config = {
            'ci_image': 'test-image',
            'runner_labels': ['test'],
            'device_types': ['a100'],
            'setup_script': 'setup.sh',
            'te_fl': {'prepare': True},
            'paths_watch': ['vendor/'],
            'test_matrix': {
                'unit': {'groups': [{'name': 'core', 'path': 'tests/unit/', 'ci_level': 'L1'}]},
                'functional': {
                    'train': [
                        {
                            'model': 'gpt',
                            'test_case': 'short',
                            'ci_level': 'L2',
                            'training_script': 'pretrain.py',
                            'gpus_per_node': 1,
                            'golden_environment': 'dev',
                            'golden_platform': 'test',
                        }
                    ]
                },
            },
        }
        for name in (
            'setup.sh',
            'pretrain.py',
            'tests/unit/test_a.py',
            'tests/functional_tests/test_cases/gpt/short/model_config.yaml',
            'tests/functional_tests/test_cases/gpt/short/golden_values_dev_test.json',
        ):
            file = self.root / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text('{}')
        self.save_config()
        self.contract = yaml.safe_load((ROOT / '.flagcicd/project.yaml').read_text())
        self.contract['migration_baseline'] = {'cuda': ['unit:core', 'functional:gpt/short']}
        self.contract_path = self.root / 'project.yaml'
        self.contract_path.write_text(yaml.safe_dump(self.contract))
        self.addCleanup(patch.stopall)
        patch.object(flagcicd, 'CONTRACT', self.contract_path).start()
        patch.object(flagcicd, 'source_sha', return_value=SHA).start()
        patch.dict(
            os.environ,
            {
                'GITHUB_RUN_ID': '42',
                'GITHUB_RUN_ATTEMPT': '1',
                'RUN_UNIT_TESTS': 'true',
                'RUN_FUNCTIONAL_TESTS': 'true',
            },
        ).start()
        self.results = {'configuration': 'success', 'unit': 'success', 'functional': 'success'}
        self.run_results = {'configuration': 'success', 'lint': 'success', 'platforms': 'success'}

    def save_config(self):
        self.config_path.write_text(yaml.safe_dump(self.config))

    def plan(self, **request):
        return flagcicd.plan(self.root, request)

    def receipts(self, data):
        selected = data['metadata']['platform_plans']['cuda']
        if selected['unit']:
            directory = self.reports / 'coverage-cuda-a100'
            directory.mkdir(exist_ok=True)
            (directory / 'unit-results.json').write_text(
                json.dumps([{'name': item['name'], 'exit_code': 0} for item in selected['unit']])
            )
            (directory / 'coverage-cuda-a100-all.json').write_text(
                json.dumps(
                    {'totals': {'covered_lines': 8, 'num_statements': 10, 'percent_covered': 12}}
                )
            )
        for item in selected['functional']:
            directory = self.reports / item['test_case']
            directory.mkdir(exist_ok=True)
            (directory / 'receipt.json').write_text(
                json.dumps(
                    {
                        'platform': 'cuda',
                        'source_sha': SHA,
                        'run_id': '42',
                        'attempt': 1,
                        'task_id': f"functional:a100:{item['model']}/{item['test_case']}",
                        'outcome': 'success',
                    }
                )
            )

    def report(self, data, **results):
        return flagcicd.platform_report(
            self.root, 'cuda', data['metadata'], {**self.results, **results}, self.reports
        )

    def save_report(self, report):
        (self.reports / 'flagcicd-report-cuda.json').write_text(json.dumps(report))

    def test_baseline_keeps_all_original_entries_and_defaults_all_platforms(self):
        data = self.plan()
        selected = data['metadata']['platform_plans']['cuda']
        self.assertEqual(selected['entry_ids'], ['unit:core', 'functional:gpt/short'])
        self.assertEqual(data['metadata']['validation_scope'], 'baseline')
        self.assertTrue(data['run_unit_tests'] and data['run_functional_tests'])
        self.assertEqual(set(data['metadata']['platform_plans']), {'cuda'})

    def test_all_five_levels_defined_but_empty_advanced_suites_cannot_pass(self):
        self.assertEqual(set(self.contract['levels']), {f'L{i}' for i in range(5)})
        for level in ('L3', 'L4'):
            with self.subTest(level=level), self.assertRaisesRegex(ValueError, 'no implemented'):
                self.plan(mode='level', test_level=level)
        data = self.plan(mode='level', test_level='L1')
        self.assertTrue(data['run_unit_tests'])
        self.assertFalse(data['run_functional_tests'])

    def test_schedules_cover_declared_capabilities_and_report_gaps(self):
        for mode, target in (('nightly', 'L3'), ('weekly', 'L4')):
            data = self.plan(mode=mode, platforms=['nonexistent'])
            self.assertEqual(data['platforms'], ['cuda'])
            self.assertEqual(data['metadata']['test_level'], target)
            self.assertEqual(data['metadata']['platform_plans']['cuda']['validated_level'], 'L2')
            self.assertIn('L3', data['metadata']['platform_plans']['cuda']['capability_gaps'])
            self.receipts(data)
            self.save_report(self.report(data))
            summary = flagcicd.aggregate(data, self.reports, self.run_results)
            self.assertEqual(summary['status'], 'passed')
            self.assertFalse(summary['merge_acceptance'])

    def test_new_advanced_entry_does_not_expand_baseline(self):
        extra = dict(
            self.config['test_matrix']['functional']['train'][0], ci_level='L3', test_case='stress'
        )
        self.config['test_matrix']['functional']['train'].append(extra)
        case = self.root / 'tests/functional_tests/test_cases/gpt/stress'
        case.mkdir()
        (case / 'model_config.yaml').write_text('{}')
        (case / 'golden_values_dev_test.json').write_text('{}')
        self.save_config()
        baseline = self.plan()
        self.assertEqual(len(baseline['metadata']['platform_plans']['cuda']['entry_ids']), 2)
        level = self.plan(mode='level', test_level='L3')
        self.assertEqual(len(level['metadata']['platform_plans']['cuda']['entry_ids']), 3)
        self.assertEqual(level['metadata']['platform_plans']['cuda']['validated_level'], 'L3')

    def test_manifest_rejects_unclassified_duplicate_overlapping_and_missing_entries(self):
        original = copy.deepcopy(self.config)
        mutations = [
            lambda c: c['test_matrix']['unit']['groups'][0].pop('ci_level'),
            lambda c: c['test_matrix']['unit']['groups'].clear(),
            lambda c: c['test_matrix']['unit']['groups'].append(
                dict(c['test_matrix']['unit']['groups'][0])
            ),
            lambda c: c['test_matrix']['unit']['groups'].append(
                {'name': 'overlap', 'ci_level': 'L1', 'path': 'tests/unit/'}
            ),
            lambda c: c['test_matrix']['unit']['groups'][0].update(path='missing/*.py'),
            lambda c: c['test_matrix']['functional']['train'][0].update(
                training_script='missing.py'
            ),
        ]
        for mutate in mutations:
            self.config = copy.deepcopy(original)
            mutate(self.config)
            self.save_config()
            with self.subTest(config=self.config), self.assertRaises(ValueError):
                flagcicd.inventory(self.root)

    def test_selection_freezes_config_and_switches_cannot_omit_tasks(self):
        data = self.plan()
        with self.assertRaisesRegex(ValueError, 'omit'):
            flagcicd.select_tests(self.root, 'cuda', data['metadata'], False, True)
        self.config['ci_image'] = 'changed'
        self.save_config()
        with self.assertRaisesRegex(ValueError, 'changed'):
            flagcicd.select_tests(self.root, 'cuda', data['metadata'], True, True)

    def test_missing_receipts_cannot_pass_even_when_workflow_says_success(self):
        data = self.plan()
        self.assertEqual(self.report(data)['status'], 'report_error')
        self.receipts(data)
        report = self.report(data)
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(report['metrics']['coverage']['children'][0]['percent'], 80)
        self.save_report(report)
        self.assertTrue(
            flagcicd.aggregate(data, self.reports, self.run_results)['merge_acceptance']
        )
        report['executed_tasks'].clear()
        self.save_report(report)
        self.assertEqual(
            flagcicd.aggregate(data, self.reports, self.run_results)['status'], 'report_error'
        )

    def test_functional_mode_is_supplemental_and_runs_without_units(self):
        data = self.plan(mode='functional')
        self.assertTrue(data['run_hardware'])
        self.assertFalse(data['run_unit_tests'])
        self.receipts(data)
        with patch.dict(os.environ, {'RUN_UNIT_TESTS': 'false'}):
            report = self.report(data, unit='skipped')
        self.assertEqual(report['status'], 'passed')
        self.save_report(report)
        summary = flagcicd.aggregate(data, self.reports, self.run_results)
        self.assertEqual(summary['status'], 'passed')
        self.assertEqual(summary['validation_scope'], 'supplemental')
        self.assertFalse(summary['merge_acceptance'])

    def test_skipped_cancelled_and_failed_tasks_cannot_pass(self):
        data = self.plan()
        self.receipts(data)
        for outcome in ('failure', 'cancelled', 'skipped'):
            with self.subTest(outcome=outcome):
                report = self.report(data, functional=outcome)
                self.assertNotEqual(report['status'], 'passed')
                receipt = self.reports / 'short/receipt.json'
                value = json.loads(receipt.read_text())
                value['outcome'] = outcome
                receipt.write_text(json.dumps(value))
                self.assertNotEqual(self.report(data)['status'], 'passed')

    def test_summary_rejects_stale_duplicate_and_missing_reports(self):
        data = self.plan()
        self.assertNotEqual(
            flagcicd.aggregate(data, self.reports, self.run_results)['status'], 'passed'
        )
        self.receipts(data)
        report = self.report(data)
        for key in ('source_sha', 'run_id', 'task_id', 'test_level', 'validation_scope', 'attempt'):
            broken = copy.deepcopy(report)
            broken[key] = 'stale'
            self.save_report(broken)
            self.assertEqual(
                flagcicd.aggregate(data, self.reports, self.run_results)['status'], 'report_error'
            )
        self.save_report(report)
        (self.reports / 'flagcicd-report-copy.json').write_text(json.dumps(report))
        self.assertEqual(
            flagcicd.aggregate(data, self.reports, self.run_results)['status'], 'report_error'
        )

    def test_failed_job_rerun_can_reuse_same_run_success(self):
        data = self.plan()
        self.receipts(data)
        self.save_report(self.report(data))
        with patch.dict(os.environ, {'GITHUB_RUN_ATTEMPT': '2'}):
            summary = flagcicd.aggregate(data, self.reports, self.run_results)
            self.assertEqual(summary['status'], 'passed')
            self.assertEqual(summary['children'][0]['attempt'], 1)
        for key in self.run_results:
            summary = flagcicd.aggregate(data, self.reports, {**self.run_results, key: 'skipped'})
            self.assertNotEqual(summary['status'], 'passed')

    def test_l0_has_no_hardware_but_requires_static_checks(self):
        data = self.plan(mode='level', test_level='L0')
        self.assertFalse(data['run_hardware'])
        results = {**self.run_results, 'platforms': 'skipped'}
        self.assertEqual(flagcicd.aggregate(data, self.reports, results)['status'], 'passed')
        self.assertNotEqual(
            flagcicd.aggregate(data, self.reports, {**results, 'lint': 'failure'})['status'],
            'passed',
        )

    def test_real_registered_platforms_validate(self):
        with patch.object(flagcicd, 'CONTRACT', ROOT / '.flagcicd/project.yaml'):
            self.assertEqual(len(flagcicd.inventory(ROOT)), 8)

    def test_failed_diff_falls_back_to_all_platforms(self):
        other = self.config_path.with_name('ascend.yml')
        other.write_text(yaml.safe_dump(self.config))
        self.contract['migration_baseline']['ascend'] = self.contract['migration_baseline']['cuda']
        self.contract_path.write_text(yaml.safe_dump(self.contract))
        event = self.root / 'event.json'
        event.write_text(
            json.dumps(
                {'pull_request': {'labels': [{'name': 'ci:diff'}, {'name': 'platform:cuda'}]}}
            )
        )
        output, github_output = self.root / 'plan.json', self.root / 'outputs'
        with (
            patch.dict(
                os.environ,
                {
                    'GITHUB_EVENT_PATH': str(event),
                    'GITHUB_EVENT_NAME': 'pull_request',
                    'GITHUB_OUTPUT': str(github_output),
                },
            ),
            patch.object(
                sys,
                'argv',
                ['flagcicd.py', 'plan', '--root', str(self.root), '--output', str(output)],
            ),
            patch.object(
                flagcicd, '_changed_files', side_effect=subprocess.CalledProcessError(1, 'git')
            ),
        ):
            flagcicd.main()
        self.assertEqual(json.loads(output.read_text())['platforms'], ['ascend', 'cuda'])

    def test_functional_receipt_step_records_real_source_and_skips(self):
        workflow = yaml.safe_load(
            (ROOT / '.github/workflows/functional_tests_common.yml').read_text()
        )
        steps = next(iter(workflow['jobs'].values()))['steps']
        training = next(step for step in steps if step.get('id') == 'training')
        self.assertEqual(training['name'], 'Run functional tests')
        record = next(
            step for step in steps if step.get('name') == 'Record functional execution result'
        )
        self.assertEqual(record['if'], 'always()')
        actual_sha = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True
        ).strip()
        for outcome in ('success', 'skipped', 'failure', 'cancelled'):
            subprocess.run(
                ['bash', '-ec', record['run']],
                cwd=self.reports,
                check=True,
                env={
                    **os.environ,
                    'GIT_DIR': str(ROOT / '.git'),
                    'PLATFORM': 'cuda',
                    'DEVICE': 'a100',
                    'CASE_MODEL': 'gpt',
                    'CASE_NAME': 'short',
                    'TEST_OUTCOME': outcome,
                },
            )
            receipt = json.loads((self.reports / 'receipt.json').read_text())
            self.assertEqual(receipt['source_sha'], actual_sha)
            self.assertEqual(receipt['task_id'], 'functional:a100:gpt/short')
            self.assertEqual(receipt['outcome'], outcome)

    def test_workflows_use_declared_schedule_and_per_platform_switches(self):
        contract = yaml.safe_load((ROOT / '.flagcicd/project.yaml').read_text())
        entry = yaml.safe_load((ROOT / '.github/workflows/flagcicd-entry.yml').read_text())
        self.assertEqual(
            {item['cron'] for item in entry[True]['schedule']},
            {item['cron'] for item in contract['schedules'].values()},
        )
        for name, job, producer in (
            ('ci.yml', 'platform-pipeline', 'generate-matrix'),
            ('flagcicd-entry.yml', 'run_tests', 'plan'),
        ):
            workflow = yaml.safe_load((ROOT / '.github/workflows' / name).read_text())
            inputs = workflow['jobs'][job]['with']
            self.assertEqual(
                inputs['source_ref'], '${{ needs.' + producer + '.outputs.source_sha }}'
            )
            self.assertIn(
                'platform_plans[matrix.platform].run_unit_tests', inputs['run_unit_tests']
            )
            self.assertIn(
                'platform_plans[matrix.platform].run_functional_tests',
                inputs['run_functional_tests'],
            )
            gate = workflow['jobs']['ci-complete' if name == 'ci.yml' else 'complete']
            self.assertTrue(
                any('flagcicd.py aggregate' in step.get('run', '') for step in gate['steps'])
            )

    def test_event_cli_binds_nightly_weekly_and_manual_modes(self):
        for event_name, event, expected in (
            (
                'workflow_dispatch',
                {'inputs': {'mode': 'functional', 'platforms': '["cuda"]'}},
                'functional',
            ),
            ('schedule', {'schedule': '0 18 * * *'}, 'nightly'),
            ('schedule', {'schedule': '0 18 * * 6'}, 'weekly'),
        ):
            event_file, output, github_output = (
                self.root / name for name in ('event.json', 'plan.json', 'outputs')
            )
            event_file.write_text(json.dumps(event))
            github_output.write_text('')
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
                check=True,
                env={
                    **os.environ,
                    'GITHUB_EVENT_NAME': event_name,
                    'GITHUB_EVENT_PATH': str(event_file),
                    'GITHUB_OUTPUT': str(github_output),
                },
            )
            self.assertEqual(json.loads(output.read_text())['metadata']['mode'], expected)


if __name__ == '__main__':
    unittest.main()
