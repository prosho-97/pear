from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
import pytest

import pear
from pear.inference import (
    build_pairwise_samples,
    build_reference_anchored_samples,
    score_pairwise,
    score_reference_anchored,
)
from pear.mbr import pear_utility_matrix, select_mbr_hypothesis


class RecordingMetric:
    def __init__(self, values: dict[str, float] | None = None) -> None:
        self.calls: list[tuple[list[dict[str, str]], dict[str, Any]]] = []
        self.values = values

    def predict(self, samples: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        self.calls.append((samples, kwargs))
        if self.values is None:
            scores = [float(index) + 0.25 for index in range(len(samples))]
        else:
            scores = [
                self.values[sample["mt_0"]] - self.values[sample["mt_1"]]
                for sample in samples
            ]
        return {"scores": scores}


def test_public_version_matches_installed_distribution_metadata() -> None:
    try:
        installed_version = version("pear-mt")
    except PackageNotFoundError:
        assert pear.__version__ == "0+unknown"
    else:
        assert installed_version == "1.0.0"
        assert pear.__version__ == installed_version


def test_sample_builders_preserve_order_and_reference_role() -> None:
    expected = [
        {"src": "s1", "mt_0": "a1", "mt_1": "b1"},
        {"src": "s2", "mt_0": "a2", "mt_1": "b2"},
    ]
    assert build_pairwise_samples(["s1", "s2"], ["a1", "a2"], ["b1", "b2"]) == expected
    assert (
        build_reference_anchored_samples(["s1", "s2"], ["a1", "a2"], ["b1", "b2"])
        == expected
    )


def test_sample_builder_rejects_unequal_lengths() -> None:
    with pytest.raises(ValueError, match=r"got \[2, 1, 2\]"):
        build_pairwise_samples(["s1", "s2"], ["a1"], ["b1", "b2"])


def test_score_pairwise_forwards_prediction_options() -> None:
    metric = RecordingMetric()

    scores = score_pairwise(
        metric,  # type: ignore[arg-type]
        ["s1", "s2"],
        ["a1", "a2"],
        ["b1", "b2"],
        batch_size=3,
        gpus=0,
        progress_bar=False,
    )

    assert scores == [0.25, 1.25]
    assert metric.calls == [
        (
            [
                {"src": "s1", "mt_0": "a1", "mt_1": "b1"},
                {"src": "s2", "mt_0": "a2", "mt_1": "b2"},
            ],
            {
                "batch_size": 3,
                "gpus": 0,
                "progress_bar": False,
                "output_field": "scores",
            },
        )
    ]


def test_score_both_evaluates_forward_and_reverse_orders() -> None:
    metric = RecordingMetric({"better": 2.0, "worse": -1.0})

    scores = score_reference_anchored(
        metric,  # type: ignore[arg-type]
        ["source"],
        ["better"],
        ["worse"],
        mode="both",
    )

    assert scores == {"forward": [3.0], "reverse": [-3.0]}
    assert metric.calls[0][0][0] == {
        "src": "source",
        "mt_0": "better",
        "mt_1": "worse",
    }
    assert metric.calls[1][0][0] == {
        "src": "source",
        "mt_0": "worse",
        "mt_1": "better",
    }


def test_score_pairwise_rejects_unknown_mode_after_no_reverse_call() -> None:
    metric = RecordingMetric()

    with pytest.raises(ValueError, match="mode must be"):
        score_pairwise(
            metric,  # type: ignore[arg-type]
            ["s"],
            ["a"],
            ["b"],
            mode="invalid",  # type: ignore[arg-type]
        )

    assert len(metric.calls) == 1


@pytest.mark.parametrize("mode", ["half", "full"])
def test_mbr_utility_matrix_and_selection(mode: str) -> None:
    metric = RecordingMetric({"best": 3.0, "middle": 2.0, "worst": 0.0})

    matrix = pear_utility_matrix(
        metric,  # type: ignore[arg-type]
        "source",
        ["best", "middle", "worst"],
        mode=mode,  # type: ignore[arg-type]
        batch_size=2,
        gpus=0,
        progress_bar=False,
    )

    np.testing.assert_array_equal(
        matrix,
        np.array(
            [
                [0.0, 1.0, 3.0],
                [-1.0, 0.0, 2.0],
                [-3.0, -2.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    assert select_mbr_hypothesis(matrix) == (0, 2.0)
    expected_pairs = 3 if mode == "half" else 6
    assert len(metric.calls[0][0]) == expected_pairs


def test_mbr_handles_single_hypothesis_and_rejects_unknown_mode() -> None:
    metric = RecordingMetric({"only": 1.0})
    matrix = pear_utility_matrix(metric, "source", ["only"])  # type: ignore[arg-type]
    np.testing.assert_array_equal(matrix, np.zeros((1, 1), dtype=np.float32))
    assert select_mbr_hypothesis(matrix) == (0, 0.0)
    assert metric.calls == []

    with pytest.raises(ValueError, match="mode must be"):
        pear_utility_matrix(
            metric,  # type: ignore[arg-type]
            "source",
            ["a", "b"],
            mode="invalid",  # type: ignore[arg-type]
        )
