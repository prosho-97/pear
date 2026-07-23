from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, sentinel

import pytest

import pear.inference as inference
import pear.models as models
from pear.models.base import DEFAULT_ENCODER_REVISIONS, resolve_encoder_revision

PEAR_REVISION = "3cc8b7894f6542fde430cc2a9f96a82fa6a679ff"
PEAR_XL_REVISION = "1f3b730fca2e223642270749310bb429b4ce5c0a"
INFOXLM_REVISION = "d616d637f0720deda963cebbfc630657d2b7d3ae"
XLMR_XL_REVISION = "aa5d120255845efeebc9b7f42822a1dd0f9ece9d"


@pytest.mark.parametrize(
    ("model", "repo_id", "revision"),
    [
        ("pear", "Prosho/pear", PEAR_REVISION),
        ("Prosho/pear", "Prosho/pear", PEAR_REVISION),
        ("pear-xl", "Prosho/pear-xl", PEAR_XL_REVISION),
        ("Prosho/pear-xl", "Prosho/pear-xl", PEAR_XL_REVISION),
        ("someone/custom-pear", "someone/custom-pear", None),
    ],
)
def test_download_model_resolves_revisions_and_forwards_hub_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model: str,
    repo_id: str,
    revision: str | None,
) -> None:
    snapshot_download = Mock(return_value="/cache/snapshots/commit")
    monkeypatch.setattr(models, "snapshot_download", snapshot_download)

    checkpoint = models.download_model(
        model,
        saving_directory=tmp_path,
        local_files_only=True,
    )

    assert checkpoint == "/cache/snapshots/commit/checkpoints/model.ckpt"
    snapshot_download.assert_called_once_with(
        repo_id=repo_id,
        cache_dir=tmp_path,
        local_files_only=True,
        revision=revision,
    )


def test_download_model_explicit_revision_overrides_official_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_download = Mock(return_value="/cache/custom")
    monkeypatch.setattr(models, "snapshot_download", snapshot_download)

    models.download_model("pear", revision="experiment")

    assert snapshot_download.call_args.kwargs["revision"] == "experiment"


def test_download_model_wraps_hub_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    hub_error = RuntimeError("offline")
    monkeypatch.setattr(models, "snapshot_download", Mock(side_effect=hub_error))

    with pytest.raises(KeyError, match="missing/model") as exc_info:
        models.download_model("missing/model")

    assert exc_info.value.__cause__ is hub_error


@pytest.mark.parametrize(
    ("model", "expected_repo", "expected_revision"),
    [
        ("pear", "Prosho/pear", PEAR_REVISION),
        ("Prosho/pear", "Prosho/pear", PEAR_REVISION),
        ("pear-xl", "Prosho/pear-xl", PEAR_XL_REVISION),
        ("Prosho/pear-xl", "Prosho/pear-xl", PEAR_XL_REVISION),
        ("someone/custom-pear", "someone/custom-pear", None),
    ],
)
def test_load_metric_resolves_hub_model_and_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model: str,
    expected_repo: str,
    expected_revision: str | None,
) -> None:
    checkpoint = tmp_path / "downloaded" / "checkpoints" / "model.ckpt"
    download_model = Mock(return_value=str(checkpoint))
    load_from_checkpoint = Mock(return_value=sentinel.metric)
    monkeypatch.setattr(inference, "download_model", download_model)
    monkeypatch.setattr(inference, "load_from_checkpoint", load_from_checkpoint)

    result = inference.load_metric(
        model,
        cache_dir=tmp_path / "cache",
        local_files_only=True,
        strict=True,
        encoder_revision="encoder-commit",
    )

    assert result is sentinel.metric
    download_model.assert_called_once_with(
        expected_repo,
        saving_directory=tmp_path / "cache",
        local_files_only=True,
        revision=expected_revision,
    )
    load_from_checkpoint.assert_called_once_with(
        checkpoint,
        strict=True,
        local_files_only=True,
        class_identifier="pairwise_metric",
        encoder_revision="encoder-commit",
    )


def test_load_metric_explicit_revision_overrides_pin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    download_model = Mock(return_value=str(tmp_path / "model.ckpt"))
    monkeypatch.setattr(inference, "download_model", download_model)
    monkeypatch.setattr(
        inference, "load_from_checkpoint", Mock(return_value=sentinel.metric)
    )

    inference.load_metric("pear", revision="checkpoint-commit")

    assert download_model.call_args.kwargs["revision"] == "checkpoint-commit"


def test_load_metric_rejects_hub_revision_for_local_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.touch()

    with pytest.raises(ValueError, match="only valid.*Hugging Face"):
        inference.load_metric(checkpoint, revision="checkpoint-commit")


def test_load_metric_allows_encoder_revision_for_local_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.touch()
    load_from_checkpoint = Mock(return_value=sentinel.metric)
    monkeypatch.setattr(inference, "load_from_checkpoint", load_from_checkpoint)

    result = inference.load_metric(
        checkpoint,
        encoder_revision="encoder-commit",
        local_files_only=True,
    )

    assert result is sentinel.metric
    assert load_from_checkpoint.call_args.kwargs["encoder_revision"] == "encoder-commit"


@pytest.mark.parametrize("encoder_revision", [None, "encoder-commit"])
def test_load_from_checkpoint_forwards_only_explicit_encoder_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    encoder_revision: str | None,
) -> None:
    checkpoint = tmp_path / "model" / "checkpoints" / "model.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    metric_class = Mock()
    metric_class.load_from_checkpoint.return_value = sentinel.metric
    monkeypatch.setitem(models.str2model, "pairwise_metric", metric_class)

    result = models.load_from_checkpoint(
        checkpoint,
        strict=True,
        local_files_only=True,
        encoder_revision=encoder_revision,
    )

    assert result is sentinel.metric
    args, kwargs = metric_class.load_from_checkpoint.call_args
    assert args == (checkpoint,)
    assert kwargs["load_pretrained_weights"] is False
    assert kwargs["strict"] is True
    assert kwargs["local_files_only"] is True
    if encoder_revision is None:
        assert "encoder_revision" not in kwargs
    else:
        assert kwargs["encoder_revision"] == encoder_revision


def test_load_from_checkpoint_validates_local_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Invalid checkpoint path"):
        models.load_from_checkpoint(tmp_path / "missing.ckpt")


def test_official_encoder_revisions_are_immutable_and_overridable() -> None:
    assert DEFAULT_ENCODER_REVISIONS == {
        "microsoft/infoxlm-large": INFOXLM_REVISION,
        "facebook/xlm-roberta-xl": XLMR_XL_REVISION,
    }
    assert resolve_encoder_revision("microsoft/infoxlm-large", None) == INFOXLM_REVISION
    assert resolve_encoder_revision("facebook/xlm-roberta-xl", None) == XLMR_XL_REVISION
    assert resolve_encoder_revision("someone/custom", None) is None
    assert resolve_encoder_revision("microsoft/infoxlm-large", "override") == "override"
