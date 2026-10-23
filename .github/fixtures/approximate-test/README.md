# Approximate comparison validation

These are deliberately constructed comparison inputs, not newly measured training results.
The unchanged baseline is `tests/functional_tests/test_cases/gpt/qwen3_0p6b_mcore_te_tp1_pp1_no_mmap_bin_files/golden_values_dev_dgx_a100.json`.

- `actual-within-tolerance.json`: every golden loss multiplied by 1.01 (1% deviation).
- `actual-outside-tolerance.json`: every golden loss multiplied by 1.50 (50% deviation).
- `model_config.yaml`: compare only `lm loss` using the existing regular pipeline.

The existing `ApproximateTest(atol=0, rtol=0.05)` is unchanged. With ten steps the comparator permits one failing step; all ten negative-case steps exceed the tolerance. The workflow runs with `--allow-nondeterministic-algo` so an exact comparison cannot obscure the approximate result.

Expected evidence: the positive step passes, then the negative step logs `Approximate comparison of lm loss: FAILED`, raises `AssertionError: The following metrics failed: lm loss`, and pytest exits 1. The negative failure must remain uncaught so GitHub marks the job failed. No model training, GPU execution, or end-to-end training regression is claimed.

The `np.isclose - per-step actual versus golden` step reads the existing `lm loss` approximate threshold and compares these same inputs with `np.isclose(actual, golden, rtol=check.rtol, atol=check.atol)`. It prints each step's actual value, golden value, absolute error, allowed error, and Boolean result, followed by the complete mask and passing count. Both expected masks are asserted. The uploaded `np-isclose-per-step.csv` preserves the full table. This diagnostic step succeeds when the controls match expectations; the original negative pipeline step still produces the intentional CI failure.
