from __future__ import annotations

import os
from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    regression = parser.getgroup("pear regressions")
    regression.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="Run tokenizer regressions that access pinned Hugging Face snapshots.",
    )
    regression.addoption(
        "--run-checkpoint-integration",
        action="store_true",
        default=False,
        help="Run multi-gigabyte PEAR checkpoint regressions.",
    )
    regression.addoption(
        "--tokenizer-baseline",
        type=Path,
        default=None,
        help="Compare tokenizer output with a JSON baseline recorded under 4.40.2.",
    )
    regression.addoption(
        "--write-tokenizer-baseline",
        type=Path,
        default=None,
        help="Write a tokenizer JSON baseline (intended for the 4.40.2 CI job).",
    )
    regression.addoption(
        "--checkpoint-baseline",
        type=Path,
        default=None,
        help="Compare checkpoint scores/key mismatches with a 4.40.2 JSON baseline.",
    )
    regression.addoption(
        "--write-checkpoint-baseline",
        type=Path,
        default=None,
        help="Write the heavyweight checkpoint baseline under Transformers 4.40.2.",
    )
    regression.addoption(
        "--hf-cache-dir",
        type=Path,
        default=None,
        help="Optional shared Hugging Face cache for opt-in regression tests.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "network: accesses immutable Hugging Face snapshots; skipped by default",
    )
    config.addinivalue_line(
        "markers",
        "checkpoint_integration: downloads and evaluates PEAR checkpoints; "
        "skipped by default",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    run_checkpoints = config.getoption("--run-checkpoint-integration") or _truthy_env(
        "PEAR_RUN_CHECKPOINT_INTEGRATION"
    )
    run_network = (
        config.getoption("--run-network")
        or run_checkpoints
        or _truthy_env("PEAR_RUN_NETWORK_TESTS")
    )
    skip_network = pytest.mark.skip(
        reason="pass --run-network to run Hugging Face tokenizer regressions"
    )
    skip_checkpoints = pytest.mark.skip(
        reason=(
            "pass --run-checkpoint-integration to download and test both "
            "multi-gigabyte PEAR checkpoints"
        )
    )
    for item in items:
        if "checkpoint_integration" in item.keywords and not run_checkpoints:
            item.add_marker(skip_checkpoints)
        elif "network" in item.keywords and not run_network:
            item.add_marker(skip_network)


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}
