# Copyright 2026 FlagOS Contributors
"""Temporary CPU-only fixture for a real Codecov patch-gate validation."""


def classify_sample(value):
    """Leave several executable paths uncovered in the negative case."""
    if value == 0:
        return "zero"
    if value == 1:
        return "one"
    if value == 2:
        return "two"
    if value == 3:
        return "three"
    if value == 4:
        return "four"
    return "other"
