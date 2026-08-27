#!/usr/bin/env python3
"""
Inject TE-FL verification step into existing test workflows.

This script modifies the common test workflows to add TE-FL provenance
verification before running tests, while maintaining backward compatibility.
"""

import sys
import yaml
from pathlib import Path
from typing import Any, Dict


def inject_verification_step(workflow_file: Path) -> bool:
    """
    Inject TE-FL verification step into workflow.

    Returns True if modified, False if already present or not applicable.
    """

    with open(workflow_file) as f:
        workflow = yaml.safe_load(f)

    if not workflow or 'jobs' not in workflow:
        return False

    modified = False

    # Find jobs that run in containers (unit_test, functional_test_train)
    for job_name, job_spec in workflow.get('jobs', {}).items():
        if not isinstance(job_spec, dict):
            continue

        # Check if this job uses a container
        if 'container' not in job_spec:
            continue

        # Check if verification step already exists
        steps = job_spec.get('steps', [])
        has_verification = any(
            step.get('name', '').startswith('Verify TE-FL')
            for step in steps
        )

        if has_verification:
            continue

        # Find the checkout step
        checkout_idx = None
        for i, step in enumerate(steps):
            if 'uses' in step and 'checkout@' in step['uses']:
                checkout_idx = i
                break

        if checkout_idx is None:
            continue

        # Insert verification step after checkout
        verification_step = {
            'name': 'Verify TE-FL provenance',
            'run': 'bash .github/scripts/verify_te_provenance.sh'
        }

        steps.insert(checkout_idx + 1, verification_step)
        job_spec['steps'] = steps
        modified = True

        print(f"  ✓ Injected verification into job: {job_name}")

    if modified:
        # Write back
        with open(workflow_file, 'w') as f:
            yaml.dump(workflow, f, default_flow_style=False, sort_keys=False)

        print(f"✓ Updated: {workflow_file}")

    return modified


def main():
    """Inject verification into all relevant workflows"""

    workflows_dir = Path('.github/workflows')

    if not workflows_dir.exists():
        print("ERROR: .github/workflows not found", file=sys.stderr)
        sys.exit(1)

    # Target workflows that run tests
    target_workflows = [
        'all_tests_common.yml',
        'functional_tests_common.yml',
    ]

    print("Injecting TE-FL verification steps...")
    print()

    modified_count = 0

    for workflow_name in target_workflows:
        workflow_file = workflows_dir / workflow_name

        if not workflow_file.exists():
            print(f"  ⊘ Skipped (not found): {workflow_name}")
            continue

        try:
            if inject_verification_step(workflow_file):
                modified_count += 1
            else:
                print(f"  ⊘ No changes needed: {workflow_name}")
        except Exception as e:
            print(f"  ✗ Failed: {workflow_name}: {e}", file=sys.stderr)

    print()
    print(f"Modified {modified_count} workflow(s)")


if __name__ == '__main__':
    main()
