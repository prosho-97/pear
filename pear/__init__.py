"""PEAR-MT: pairwise QE metrics for machine translation evaluation."""

from importlib.metadata import PackageNotFoundError, version

from pear.inference import (
    build_pairwise_samples,
    build_reference_anchored_samples,
    load_metric,
    score_pairwise,
    score_reference_anchored,
)
from pear.mbr import pear_utility_matrix, select_mbr_hypothesis
from pear.models import download_model, load_from_checkpoint

try:
    __version__ = version("pear-mt")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "build_pairwise_samples",
    "build_reference_anchored_samples",
    "download_model",
    "load_from_checkpoint",
    "load_metric",
    "pear_utility_matrix",
    "score_pairwise",
    "score_reference_anchored",
    "select_mbr_hypothesis",
]
