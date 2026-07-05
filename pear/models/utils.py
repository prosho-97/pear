"""Small inference-only helpers for PEAR models."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Iterator, Mapping, Sequence

from torch.utils.data import Sampler


class Prediction(OrderedDict):
    """Dictionary-like prediction container with attribute access.

    The original training code used a larger Hugging Face-style ``ModelOutput``
    wrapper. Inference only needs a lightweight object that supports both
    ``prediction["scores"]`` and ``prediction.scores`` access patterns.
    """

    def __init__(self, mapping: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__()
        if mapping is not None:
            kwargs = {**mapping, **kwargs}
        for key, value in kwargs.items():
            if value is not None:
                self[key] = value

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        if key.startswith("_"):
            super().__setattr__(key, value)
        else:
            self[key] = value

    def to_tuple(self) -> tuple[Any, ...]:
        """Return non-null values as a tuple for legacy compatibility."""
        return tuple(self.values())


class OrderedSampler(Sampler[int]):
    """Sampler that returns indices in a deterministic precomputed order."""

    def __init__(self, indices: Sequence[int]) -> None:
        self.indices = list(indices)

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def restore_list_order(
    sorted_list: Sequence[Any], sort_ids: Sequence[int]
) -> list[Any]:
    """Restore values sorted by ``sort_ids`` back to their original order."""
    restored = [None for _ in range(len(sorted_list))]
    for original_index, value in zip(sort_ids, sorted_list):
        restored[original_index] = value
    return restored
