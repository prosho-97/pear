"""Minimum Bayes risk decoding utilities for PEAR."""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

from pear.inference import score_pairwise
from pear.models.base import MetricModel

MbrMode = Literal["full", "half"]


def pear_utility_matrix(
    model: MetricModel,
    source: str,
    hypotheses: Sequence[str],
    *,
    mode: MbrMode = "half",
    batch_size: int = 8,
    gpus: int | str = "auto",
    progress_bar: bool = True,
) -> np.ndarray:
    """Compute a PEAR pairwise utility matrix for one source segment.

    ``full`` evaluates all off-diagonal ordered pairs. ``half`` evaluates only
    one triangular half and fills the opposite direction by antisymmetry, which
    is the efficient PEAR MBR mode described in the paper.
    """
    n = len(hypotheses)
    matrix = np.zeros((n, n), dtype=np.float32)
    pairs: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if mode == "half" and j < i:
                continue
            pairs.append((i, j))
    if mode not in {"full", "half"}:
        raise ValueError("mode must be 'full' or 'half'.")
    if not pairs:
        return matrix
    scores = score_pairwise(
        model,
        [source] * len(pairs),
        [hypotheses[i] for i, _ in pairs],
        [hypotheses[j] for _, j in pairs],
        batch_size=batch_size,
        gpus=gpus,
        progress_bar=progress_bar,
    )
    assert isinstance(scores, list)
    for (i, j), score in zip(pairs, scores):
        matrix[i, j] = score
        if mode == "half":
            matrix[j, i] = -score
    return matrix


def select_mbr_hypothesis(
    utility_matrix: np.ndarray,
) -> tuple[int, float]:
    """Return the index and expected utility of the best MBR hypothesis."""
    utilities = utility_matrix.sum(axis=1) / max(utility_matrix.shape[0] - 1, 1)
    index = int(np.argmax(utilities))
    return index, float(utilities[index])
