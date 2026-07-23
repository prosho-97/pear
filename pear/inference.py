"""Public PEAR inference API."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal, Sequence

from pear.models import (
    DEFAULT_HF_MODELS,
    DEFAULT_HF_REVISIONS,
    download_model,
    load_from_checkpoint,
)
from pear.models.base import MetricModel

PearMode = Literal["single", "both"]


def load_metric(
    model: str | Path = "pear",
    *,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    strict: bool = False,
    revision: str | None = None,
    encoder_revision: str | None = None,
) -> MetricModel:
    """Load a PEAR checkpoint from a local path or from Hugging Face.

    ``model`` may be a checkpoint path, a Hugging Face repo id, or one of the
    aliases ``pear``/``pear-xl``. The public checkpoints are expected under the
    ``Prosho`` Hugging Face namespace. Official repositories default to
    immutable revisions; pass ``revision`` and/or ``encoder_revision`` to
    override the checkpoint and base-encoder revisions.
    """
    model_name = str(model)
    model_path = Path(model).expanduser() if not isinstance(model, Path) else model
    trusted_checkpoint = False
    if model_path.is_file():
        if revision is not None:
            raise ValueError(
                "revision is only valid when loading a Hugging Face model, "
                "not a local checkpoint."
            )
        checkpoint_path = model_path
    else:
        repo_id = DEFAULT_HF_MODELS.get(model_name, model_name)
        official_revision = DEFAULT_HF_REVISIONS.get(repo_id)
        if revision is None:
            revision = DEFAULT_HF_REVISIONS.get(
                model_name, DEFAULT_HF_REVISIONS.get(repo_id)
            )
        trusted_checkpoint = (
            official_revision is not None and revision == official_revision
        )
        checkpoint_path = Path(
            download_model(
                repo_id,
                saving_directory=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
            )
        )
    return load_from_checkpoint(
        checkpoint_path,
        strict=strict,
        local_files_only=local_files_only,
        class_identifier="pairwise_metric",
        encoder_revision=encoder_revision,
        trusted_checkpoint=trusted_checkpoint,
    )


def build_pairwise_samples(
    sources: Sequence[str],
    translations_a: Sequence[str],
    translations_b: Sequence[str],
) -> list[dict[str, str]]:
    """Build PEAR pairwise-QE samples from parallel source/MT lists."""
    _validate_equal_lengths(sources, translations_a, translations_b)
    return [
        {"src": src, "mt_0": mt_0, "mt_1": mt_1}
        for src, mt_0, mt_1 in zip(sources, translations_a, translations_b)
    ]


def build_reference_anchored_samples(
    sources: Sequence[str], translations: Sequence[str], references: Sequence[str]
) -> list[dict[str, str]]:
    """Build PEAR reference-anchored samples by using references as ``mt_1``."""
    return build_pairwise_samples(sources, translations, references)


def score_pairwise(
    model: MetricModel,
    sources: Sequence[str],
    translations_a: Sequence[str],
    translations_b: Sequence[str],
    *,
    mode: PearMode = "single",
    batch_size: int = 8,
    gpus: int | str = "auto",
    progress_bar: bool = True,
) -> list[float] | dict[str, list[float]]:
    """Score translation pairs with PEAR.

    Positive scores mean ``translations_a`` is preferred over
    ``translations_b``. In ``both`` mode the reverse order is also scored.
    """
    samples = build_pairwise_samples(sources, translations_a, translations_b)
    forward = _predict_scores(model, samples, batch_size, gpus, progress_bar)
    if mode == "single":
        return forward
    if mode != "both":
        raise ValueError("mode must be 'single' or 'both'.")
    reverse_samples = build_pairwise_samples(sources, translations_b, translations_a)
    reverse = _predict_scores(model, reverse_samples, batch_size, gpus, progress_bar)
    return {"forward": forward, "reverse": reverse}


def score_reference_anchored(
    model: MetricModel,
    sources: Sequence[str],
    translations: Sequence[str],
    references: Sequence[str],
    *,
    mode: PearMode = "single",
    batch_size: int = 8,
    gpus: int | str = "auto",
    progress_bar: bool = True,
) -> list[float] | dict[str, list[float]]:
    """Score each translation against a reference anchor."""
    return score_pairwise(
        model,
        sources,
        translations,
        references,
        mode=mode,
        batch_size=batch_size,
        gpus=gpus,
        progress_bar=progress_bar,
    )


def _predict_scores(
    model: MetricModel,
    samples: list[dict[str, str]],
    batch_size: int,
    gpus: int | str,
    progress_bar: bool,
) -> list[float]:
    prediction = model.predict(
        samples,
        batch_size=batch_size,
        gpus=gpus,
        progress_bar=progress_bar,
        output_field="scores",
    )
    return [float(score) for score in prediction["scores"]]


def _validate_equal_lengths(*items: Iterable[object]) -> None:
    lengths = [len(item) for item in items]  # type: ignore[arg-type]
    if len(set(lengths)) != 1:
        raise ValueError(f"Input sequences must have equal lengths; got {lengths}.")
