"""Opt-in compatibility regressions.

Normal ``pytest`` runs are entirely offline. The intended cross-version flow is:

1. Under Transformers 4.40.2, record baselines with
   ``--run-network --write-tokenizer-baseline=/tmp/tokenizers.json`` and
   ``--run-checkpoint-integration --write-checkpoint-baseline=/tmp/pear.json``.
2. Under the latest allowed Transformers 5.x on the same hardware, pass those
   artifacts back through ``--tokenizer-baseline`` and
   ``--checkpoint-baseline``.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import transformers
from transformers import AutoTokenizer

from pear import load_metric, score_pairwise
from pear.models import download_model
from pear.models.base import DEFAULT_ENCODER_REVISIONS

MULTILINGUAL_SEGMENTS = [
    "Machine translation evaluation.",
    "Già, perché l’umanità è multilingue.",
    "机器翻译评估。",
    "تقييم الترجمة الآلية.",
]

BACKBONES = {
    "infoxlm-large": (
        "microsoft/infoxlm-large",
        "d616d637f0720deda963cebbfc630657d2b7d3ae",
    ),
    "xlm-roberta-xl": (
        "facebook/xlm-roberta-xl",
        "aa5d120255845efeebc9b7f42822a1dd0f9ece9d",
    ),
}

DEFAULT_TOKENIZER_BASELINE = (
    Path(__file__).parent / "fixtures" / "tokenizers-4.40.2.json"
)

EXPECTED_XLMR_SPECIAL_IDS = {
    "bos_token_id": 0,
    "pad_token_id": 1,
    "eos_token_id": 2,
    "unk_token_id": 3,
    "mask_token_id": 250001,
}


@pytest.mark.network
def test_pinned_multilingual_tokenizers_match_440_baseline(
    pytestconfig: pytest.Config,
) -> None:
    baseline_path = _option_path(pytestconfig, "--tokenizer-baseline")
    write_path = _option_path(pytestconfig, "--write-tokenizer-baseline")
    if baseline_path is None and write_path is None:
        baseline_path = DEFAULT_TOKENIZER_BASELINE
    _ensure_exclusive_paths(baseline_path, write_path, "tokenizer")
    _require_4402_for_baseline_write(write_path)
    cache_dir = _option_path(pytestconfig, "--hf-cache-dir")

    models: dict[str, Any] = {}
    for name, (repo_id, revision) in BACKBONES.items():
        assert DEFAULT_ENCODER_REVISIONS[repo_id] == revision
        tokenizer = AutoTokenizer.from_pretrained(
            repo_id,
            revision=revision,
            cache_dir=cache_dir,
        )
        encoded = tokenizer(
            MULTILINGUAL_SEGMENTS,
            padding=True,
            truncation=True,
        )
        special_ids = {
            key: getattr(tokenizer, key) for key in EXPECTED_XLMR_SPECIAL_IDS
        }
        assert special_ids == EXPECTED_XLMR_SPECIAL_IDS
        models[name] = {
            "repo_id": repo_id,
            "revision": revision,
            "tokenizer_class": type(tokenizer).__name__,
            "special_token_ids": special_ids,
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    payload = {
        "schema_version": 1,
        "transformers_version": transformers.__version__,
        "segments": MULTILINGUAL_SEGMENTS,
        "models": models,
    }
    if write_path is not None:
        _write_json(write_path, payload)
    elif baseline_path is not None:
        baseline = _read_json(baseline_path)
        assert baseline["schema_version"] == 1
        assert baseline["transformers_version"] == "4.40.2"
        assert baseline["segments"] == MULTILINGUAL_SEGMENTS
        # The concrete class name changed in Transformers 5 for some models;
        # token semantics, not implementation naming, define compatibility.
        for model_name in BACKBONES:
            expected = baseline["models"][model_name]
            actual = models[model_name]
            assert actual["repo_id"] == expected["repo_id"]
            assert actual["revision"] == expected["revision"]
            assert actual["special_token_ids"] == expected["special_token_ids"]
            assert actual["input_ids"] == expected["input_ids"]
            assert actual["attention_mask"] == expected["attention_mask"]


@pytest.mark.network
@pytest.mark.checkpoint_integration
def test_released_checkpoints_match_440_scores_and_key_mismatches(
    pytestconfig: pytest.Config,
) -> None:
    baseline_path = _option_path(pytestconfig, "--checkpoint-baseline")
    write_path = _option_path(pytestconfig, "--write-checkpoint-baseline")
    _ensure_exclusive_paths(baseline_path, write_path, "checkpoint")
    _require_4402_for_baseline_write(write_path)
    cache_dir = _option_path(pytestconfig, "--hf-cache-dir")

    sources = [
        "The cat is sleeping on the sofa.",
        "Scientific results should be reproducible.",
        "Machine translation connects people.",
    ]
    preferred = [
        "Il gatto dorme sul divano.",
        "I risultati scientifici dovrebbero essere riproducibili.",
        "La traduzione automatica mette in contatto le persone.",
    ]
    alternatives = [
        "Il divano dorme sul gatto.",
        "I risultati scientifici mangiano riproducibili.",
        "Le persone traducono una macchina sconnessa.",
    ]

    model_payloads: dict[str, Any] = {}
    for alias in ("pear", "pear-xl"):
        checkpoint_path = Path(download_model(alias, saving_directory=cache_dir))
        checkpoint_keys = _checkpoint_state_keys(checkpoint_path)
        metric = load_metric(alias, cache_dir=cache_dir)
        parameter = next(metric.parameters())
        model_dtype = str(parameter.dtype).removeprefix("torch.")
        model_device = str(parameter.device)
        del parameter
        scores = score_pairwise(
            metric,
            sources,
            preferred,
            alternatives,
            mode="both",
            batch_size=2,
            gpus=0,
            progress_bar=False,
        )
        assert isinstance(scores, dict)
        forward = np.asarray(scores["forward"], dtype=np.float64)
        reverse = np.asarray(scores["reverse"], dtype=np.float64)
        assert np.isfinite(forward).all()
        assert np.isfinite(reverse).all()
        np.testing.assert_array_equal(np.sign(forward), -np.sign(reverse))

        model_payloads[alias] = {
            "device": model_device,
            "dtype": model_dtype,
            "forward": forward.tolist(),
            "reverse": reverse.tolist(),
            "state_dict_mismatches": _state_dict_mismatches(checkpoint_keys, metric),
        }
        del metric
        gc.collect()

    devices = {model["device"] for model in model_payloads.values()}
    assert devices == {"cpu"}
    payload = {
        "schema_version": 1,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "device": devices.pop(),
        "sources": sources,
        "preferred": preferred,
        "alternatives": alternatives,
        "models": model_payloads,
    }
    if write_path is not None:
        _write_json(write_path, payload)
    elif baseline_path is not None:
        baseline = _read_json(baseline_path)
        assert baseline["schema_version"] == 1
        assert baseline["transformers_version"] == "4.40.2"
        assert baseline["device"] == "cpu"
        assert baseline["sources"] == sources
        assert baseline["preferred"] == preferred
        assert baseline["alternatives"] == alternatives
        for alias, actual in model_payloads.items():
            expected = baseline["models"][alias]
            assert actual["device"] == expected["device"]
            assert actual["dtype"] == expected["dtype"]
            assert actual["state_dict_mismatches"] == expected["state_dict_mismatches"]
            for order in ("forward", "reverse"):
                actual_scores = np.asarray(actual[order])
                expected_scores = np.asarray(expected[order])
                np.testing.assert_allclose(
                    actual_scores,
                    expected_scores,
                    rtol=1e-4,
                    atol=1e-4,
                )
                np.testing.assert_array_equal(
                    np.sign(actual_scores), np.sign(expected_scores)
                )
                np.testing.assert_array_equal(
                    np.argsort(actual_scores), np.argsort(expected_scores)
                )


def _checkpoint_state_keys(checkpoint_path: Path) -> set[str]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device("cpu"),
        weights_only=False,
        mmap=True,
    )
    checkpoint_state = checkpoint.get("state_dict", checkpoint)
    keys = set(checkpoint_state)
    del checkpoint_state
    del checkpoint
    gc.collect()
    return keys


def _state_dict_mismatches(
    checkpoint_keys: set[str], metric: Any
) -> dict[str, list[str]]:
    model_keys = set(metric.state_dict())
    return {
        "missing": sorted(model_keys - checkpoint_keys),
        "unexpected": sorted(checkpoint_keys - model_keys),
    }


def _option_path(config: pytest.Config, name: str) -> Path | None:
    value = config.getoption(name)
    return None if value is None else Path(value)


def _ensure_exclusive_paths(
    baseline_path: Path | None,
    write_path: Path | None,
    kind: str,
) -> None:
    if baseline_path is not None and write_path is not None:
        pytest.fail(
            f"Choose either the {kind} comparison baseline or write target, not both."
        )


def _require_4402_for_baseline_write(write_path: Path | None) -> None:
    if write_path is not None and transformers.__version__ != "4.40.2":
        pytest.fail(
            "Compatibility baselines must be recorded with Transformers 4.40.2; "
            f"found {transformers.__version__}."
        )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        pytest.fail(f"Regression baseline does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
