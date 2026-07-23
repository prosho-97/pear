from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from unittest.mock import Mock, sentinel

import numpy as np
import pytest

import pear
import pear.commands.main as cli


@pytest.mark.parametrize("program", ["pear", "pear-mt"])
def test_cli_version_uses_distribution_version(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    program: str,
) -> None:
    monkeypatch.setattr(sys, "argv", [program, "--version"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"{program} {pear.__version__}"


def test_score_cli_forwards_revision_options_and_writes_tsv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "segments.tsv"
    output_path = tmp_path / "scores.tsv"
    input_path.write_text(
        "src\tmt_0\tmt_1\nsource\tbetter\tworse\n",
        encoding="utf-8",
    )
    load_metric = Mock(return_value=sentinel.metric)
    score_pairwise = Mock(return_value={"forward": [1.5], "reverse": [-1.5]})
    monkeypatch.setattr(cli, "load_metric", load_metric)
    monkeypatch.setattr(cli, "score_pairwise", score_pairwise)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pear",
            "score",
            "--hf-model",
            "organization/model",
            "--revision",
            "checkpoint-commit",
            "--encoder-revision",
            "encoder-commit",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--local-files-only",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--both",
            "--batch-size",
            "4",
            "--gpus",
            "0",
        ],
    )

    cli.main()

    load_metric.assert_called_once_with(
        "organization/model",
        cache_dir=tmp_path / "cache",
        local_files_only=True,
        revision="checkpoint-commit",
        encoder_revision="encoder-commit",
    )
    score_pairwise.assert_called_once_with(
        sentinel.metric,
        ["source"],
        ["better"],
        ["worse"],
        mode="both",
        batch_size=4,
        gpus=0,
    )
    with output_path.open(newline="", encoding="utf-8") as output_file:
        rows = list(csv.DictReader(output_file, delimiter="\t"))
    assert rows == [
        {
            "src": "source",
            "mt_0": "better",
            "mt_1": "worse",
            "pear_forward": "1.5",
            "pear_reverse": "-1.5",
        }
    ]


def test_mbr_cli_selects_hypothesis_and_writes_jsonl(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "nbest.jsonl"
    output_path = tmp_path / "selected.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "src": "source",
                "hypotheses": ["best", "worse"],
                "metadata": "preserved",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    load_metric = Mock(return_value=sentinel.metric)
    matrix = np.array([[0.0, 2.0], [-2.0, 0.0]], dtype=np.float32)
    utility_matrix = Mock(return_value=matrix)
    select = Mock(return_value=(0, 2.0))
    monkeypatch.setattr(cli, "load_metric", load_metric)
    monkeypatch.setattr(cli, "pear_utility_matrix", utility_matrix)
    monkeypatch.setattr(cli, "select_mbr_hypothesis", select)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pear",
            "mbr",
            "--checkpoint",
            str(tmp_path / "local.ckpt"),
            "--encoder-revision",
            "encoder-commit",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--utility",
            "full",
            "--batch-size",
            "2",
            "--gpus",
            "0",
        ],
    )

    cli.main()

    load_metric.assert_called_once_with(
        tmp_path / "local.ckpt",
        cache_dir=None,
        local_files_only=False,
        revision=None,
        encoder_revision="encoder-commit",
    )
    utility_matrix.assert_called_once_with(
        sentinel.metric,
        "source",
        ["best", "worse"],
        mode="full",
        batch_size=2,
        gpus=0,
    )
    assert select.call_args.args[0] is matrix
    item = json.loads(output_path.read_text(encoding="utf-8"))
    assert item == {
        "src": "source",
        "hypotheses": ["best", "worse"],
        "metadata": "preserved",
        "selected_index": 0,
        "selected": "best",
        "utility": 2.0,
    }


def test_cli_rejects_conflicting_explicit_model_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pear",
            "score",
            "--checkpoint",
            str(tmp_path / "model.ckpt"),
            "--hf-model",
            "organization/model",
            "--input",
            str(tmp_path / "input.tsv"),
        ],
    )

    with pytest.raises(SystemExit, match="Use only one"):
        cli.main()
