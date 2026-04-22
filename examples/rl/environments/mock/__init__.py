# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Mock RL environments used by self-contained functional coverage.

These modules live under ``examples.rl.environments`` so the functional test
env_config can reference them through the same import path shape as the other
example RL environments, without adding test-only import hooks to the runtime.
"""
