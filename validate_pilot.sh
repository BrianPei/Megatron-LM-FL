#!/bin/bash
# Quick validation script for the minimal TE-FL integration pilot

set -euo pipefail

cd "$(dirname "$0")"

echo "==================================="
echo "TE-FL Integration Pilot Validation"
echo "==================================="
echo ""

EXIT_CODE=0

# 1. Shell syntax
echo "[1/6] Checking shell script syntax..."
if bash -n .github/scripts/verify_te_runtime.sh; then
  echo "  ✓ verify_te_runtime.sh: valid"
else
  echo "  ✗ verify_te_runtime.sh: syntax error"
  EXIT_CODE=1
fi
echo ""

# 2. YAML syntax
echo "[2/6] Checking YAML syntax..."
for workflow in \
  .github/workflows/all_tests_common.yml \
  .github/workflows/unit_tests_common.yml \
  .github/workflows/functional_tests_common.yml \
  .github/workflows/all_tests_musa.yml; do

  if python3 -c "import yaml; yaml.safe_load(open('$workflow'))" 2>/dev/null; then
    echo "  ✓ $(basename "$workflow"): valid"
  else
    echo "  ✗ $(basename "$workflow"): invalid"
    EXIT_CODE=1
  fi
done
echo ""

# 3. Git whitespace check
echo "[3/6] Checking git whitespace errors..."
if git diff --check HEAD 2>&1 | grep -q .; then
  echo "  ✗ Whitespace errors found"
  git diff --check HEAD
  EXIT_CODE=1
else
  echo "  ✓ No whitespace errors"
fi
echo ""

# 4. Verify image_override propagation
echo "[4/6] Verifying image_override propagation..."
if grep -q "image_override" .github/workflows/all_tests_common.yml && \
   grep -q "image_override" .github/workflows/all_tests_musa.yml; then
  echo "  ✓ image_override present in common and pilot workflows"
else
  echo "  ✗ image_override missing"
  EXIT_CODE=1
fi
echo ""

# 5. Verify same image to unit and functional
echo "[5/6] Verifying same image to unit and functional..."
# Grep for the specific pattern in the workflow file
if grep -q "needs.checkout_and_config.outputs.ci_image" .github/workflows/all_tests_common.yml; then
  OCCURRENCES=$(grep -c "needs.checkout_and_config.outputs.ci_image" .github/workflows/all_tests_common.yml)
  if [[ $OCCURRENCES -ge 2 ]]; then
    echo "  ✓ Both workflows reference the same ci_image output ($OCCURRENCES occurrences)"
  else
    echo "  ✗ ci_image referenced only $OCCURRENCES time(s), expected at least 2"
    EXIT_CODE=1
  fi
else
  echo "  ✗ checkout_and_config.outputs.ci_image not found in workflow"
  EXIT_CODE=1
fi
echo ""

# 6. Verify verification script exists
echo "[6/6] Checking verification script..."
if [[ -f .github/scripts/verify_te_runtime.sh && -x .github/scripts/verify_te_runtime.sh ]]; then
  echo "  ✓ verify_te_runtime.sh exists and is executable"
else
  echo "  ✗ verify_te_runtime.sh missing or not executable"
  EXIT_CODE=1
fi
echo ""

# Summary
echo "==================================="
if [[ $EXIT_CODE -eq 0 ]]; then
  echo "✓ All validations passed"
  echo ""
  echo "Next steps:"
  echo "1. Review IMPLEMENTATION_REPORT.md"
  echo "2. Obtain FlagScale image with TE-FL"
  echo "3. Run pilot workflow:"
  echo ""
  echo "   gh workflow run all_tests_musa.yml \\"
  echo "     -f image_override='<image>@sha256:<digest>' \\"
  echo "     -f expected_te_fl_commit='<40-hex-sha>' \\"
  echo "     -f run_unit_tests=true \\"
  echo "     -f run_functional_tests=false"
else
  echo "✗ Some validations failed"
fi
echo "==================================="

exit $EXIT_CODE
