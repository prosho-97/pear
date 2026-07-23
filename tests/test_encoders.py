from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, sentinel

import pytest

import pear.encoders.xlmr as xlmr
import pear.encoders.xlmr_xl as xlmr_xl


@pytest.mark.parametrize(
    ("module", "encoder_class", "model_attribute"),
    [
        (xlmr, xlmr.XLMREncoder, "XLMRobertaModel"),
        (xlmr_xl, xlmr_xl.XLMRXLEncoder, "XLMRobertaXLModel"),
    ],
)
def test_encoder_uses_auto_tokenizer_and_forwards_pretrained_arguments(
    monkeypatch: pytest.MonkeyPatch,
    module: object,
    encoder_class: type,
    model_attribute: str,
) -> None:
    auto_tokenizer = SimpleNamespace(
        from_pretrained=Mock(return_value=sentinel.tokenizer)
    )
    model_factory = Mock()
    model_factory.from_pretrained.return_value = sentinel.model
    monkeypatch.setattr(module, "AutoTokenizer", auto_tokenizer)
    monkeypatch.setattr(module, model_attribute, model_factory)

    encoder = encoder_class(
        "organization/encoder",
        load_pretrained_weights=True,
        local_files_only=True,
        revision="immutable-commit",
    )

    assert encoder.tokenizer is sentinel.tokenizer
    assert encoder.model is sentinel.model
    auto_tokenizer.from_pretrained.assert_called_once_with(
        "organization/encoder",
        revision="immutable-commit",
        local_files_only=True,
    )
    model_factory.from_pretrained.assert_called_once_with(
        "organization/encoder",
        add_pooling_layer=False,
        revision="immutable-commit",
        local_files_only=True,
    )


@pytest.mark.parametrize(
    ("module", "encoder_class", "model_attribute", "config_attribute"),
    [
        (
            xlmr,
            xlmr.XLMREncoder,
            "XLMRobertaModel",
            "XLMRobertaConfig",
        ),
        (
            xlmr_xl,
            xlmr_xl.XLMRXLEncoder,
            "XLMRobertaXLModel",
            "XLMRobertaXLConfig",
        ),
    ],
)
def test_encoder_forwards_revision_when_loading_config_without_weights(
    monkeypatch: pytest.MonkeyPatch,
    module: object,
    encoder_class: type,
    model_attribute: str,
    config_attribute: str,
) -> None:
    auto_tokenizer = SimpleNamespace(
        from_pretrained=Mock(return_value=sentinel.tokenizer)
    )
    config_factory = SimpleNamespace(from_pretrained=Mock(return_value=sentinel.config))
    model = SimpleNamespace(config=sentinel.config)
    model_factory = Mock(return_value=model)
    monkeypatch.setattr(module, "AutoTokenizer", auto_tokenizer)
    monkeypatch.setattr(module, config_attribute, config_factory)
    monkeypatch.setattr(module, model_attribute, model_factory)

    encoder = encoder_class(
        "organization/encoder",
        load_pretrained_weights=False,
        local_files_only=True,
        revision="immutable-commit",
    )

    assert encoder.model is model
    config_factory.from_pretrained.assert_called_once_with(
        "organization/encoder",
        revision="immutable-commit",
        local_files_only=True,
    )
    model_factory.assert_called_once_with(
        sentinel.config,
        add_pooling_layer=False,
    )


@pytest.mark.parametrize(
    ("encoder_class", "module"),
    [
        (xlmr.XLMREncoder, xlmr),
        (xlmr_xl.XLMRXLEncoder, xlmr_xl),
    ],
)
def test_encoder_from_pretrained_forwards_all_arguments(
    monkeypatch: pytest.MonkeyPatch,
    encoder_class: type,
    module: object,
) -> None:
    init = Mock(return_value=None)
    monkeypatch.setattr(encoder_class, "__init__", init)

    encoder_class.from_pretrained(
        "organization/encoder",
        load_pretrained_weights=False,
        local_files_only=True,
        revision="immutable-commit",
    )

    init.assert_called_once_with(
        "organization/encoder",
        False,
        True,
        "immutable-commit",
    )
