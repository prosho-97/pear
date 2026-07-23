r"""Inference-only PEAR model base class."""

from __future__ import annotations

import abc
import logging
import os
import warnings
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Literal, Optional, Union

import numpy as np

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="pkg_resources is deprecated as an API.*",
)
warnings.filterwarnings("ignore", message=".*isinstance\\(treespec.*")

import pytorch_lightning as ptl
import torch
from torch.utils.data import DataLoader, SequentialSampler

from pear.encoders import str2encoder

from .utils import OrderedSampler, Prediction, restore_list_order

logger = logging.getLogger(__name__)

DEFAULT_ENCODER_REVISIONS = {
    "microsoft/infoxlm-large": "d616d637f0720deda963cebbfc630657d2b7d3ae",
    "facebook/xlm-roberta-xl": "aa5d120255845efeebc9b7f42822a1dd0f9ece9d",
}


def resolve_encoder_revision(
    pretrained_model: str, encoder_revision: str | None
) -> str | None:
    """Resolve immutable revisions for the encoders used by official models."""
    if encoder_revision is not None:
        return encoder_revision
    return DEFAULT_ENCODER_REVISIONS.get(pretrained_model)


class MetricModel(ptl.LightningModule, metaclass=abc.ABCMeta):
    """Shared inference functionality for released PEAR checkpoints.

    The published PEAR, PEAR_KD, and PEAR-XL checkpoints all use the encoder's
    last hidden layer. Extra legacy hparams are accepted via ``**unused_hparams``
    for Lightning checkpoint compatibility, but non-``last`` blends are
    intentionally unsupported in the lightweight inference package.
    """

    def __init__(
        self,
        keep_embeddings_frozen: bool = False,
        encoder_model: str = "InfoXLM",
        pretrained_model: str = "microsoft/infoxlm-large",
        blend: str = "last",
        load_pretrained_weights: bool = True,
        local_files_only: bool = False,
        encoder_revision: str | None = None,
        **unused_hparams: Any,
    ) -> None:
        super().__init__()
        encoder_revision = resolve_encoder_revision(pretrained_model, encoder_revision)
        self.save_hyperparameters()

        if self.hparams["blend"] != "last":
            raise ValueError(
                "pear-mt currently supports the released PEAR checkpoints, all of "
                "which use blend='last'."
            )

        self.encoder = str2encoder[self.hparams["encoder_model"]].from_pretrained(
            self.hparams["pretrained_model"],
            load_pretrained_weights=load_pretrained_weights,
            local_files_only=local_files_only,
            revision=encoder_revision,
        )

        if self.hparams["keep_embeddings_frozen"]:
            self.encoder.freeze_embeddings()

    @abc.abstractmethod
    def prepare_sample(
        self, sample: List[dict], stage: str = "predict"
    ) -> Dict[str, torch.Tensor]:
        """Tokenize one inference batch."""

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> Prediction:
        """Run a forward pass."""

    @abc.abstractmethod
    def requires_references(self) -> bool:
        """Return whether the metric requires references."""

    def configure_optimizers(self):
        """Disable optimizers in the inference-only package."""
        return None

    def freeze_encoder(self) -> None:
        """Freeze encoder parameters."""
        logger.info("Encoder model frozen.")
        self.encoder.freeze()

    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the CLS embedding from the encoder's last hidden layer."""
        layers = self.encoder(input_ids, attention_mask)
        return layers[-1][:, 0, :]

    def prepare_for_inference(
        self, sample: List[Dict[str, str]]
    ) -> Dict[str, torch.Tensor]:
        """Collate function used by the prediction dataloader."""
        return self.prepare_sample(sample, stage="predict")

    def predict(
        self,
        samples: List[Dict[str, str]],
        batch_size: int = 16,
        progress_bar: bool = True,
        gpus: int | Literal["auto"] = "auto",
        accelerator: str = "auto",
        length_batching: bool = True,
        output_field: Literal["scores"] = "scores",
        devices: Optional[Union[List[int], str, int]] = None,
        num_workers: Optional[int] = None,
    ) -> Prediction:
        """Run PEAR inference for a list of prepared text samples."""
        if output_field != "scores":
            raise ValueError("pear-mt inference only exposes segment scores.")
        accelerator, devices, resolved_gpus = _resolve_inference_device(
            gpus=gpus, accelerator=accelerator, devices=devices
        )

        sampler = SequentialSampler(samples)
        sort_ids = None
        if length_batching and resolved_gpus < 2:
            sort_ids = np.argsort(
                [len(sample.get("src", sample.get("ref", ""))) for sample in samples]
            )
            sampler = OrderedSampler(sort_ids.tolist())

        is_windows = os.name == "nt"
        if num_workers is None:
            num_workers = 0 if is_windows else 2 * resolved_gpus
        elif is_windows and num_workers != 0:
            logger.warning("Use num_workers=0 for PEAR inference on Windows.")

        self.eval()
        dataloader = DataLoader(
            dataset=samples,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=self.prepare_for_inference,
            num_workers=num_workers,
            multiprocessing_context=(
                "fork" if torch.backends.mps.is_available() else None
            ),
        )

        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*Consider increasing the value of the `num_workers` argument` .*",
        )
        with _quiet_lightning_logs():
            trainer = ptl.Trainer(
                devices=devices,
                logger=False,
                accelerator=accelerator,
                strategy="auto",
                enable_progress_bar=progress_bar,
                enable_checkpointing=False,
                enable_model_summary=False,
            )
            predictions = trainer.predict(
                self, dataloaders=dataloader, return_predictions=True
            )
        scores = torch.cat([pred.scores for pred in predictions], dim=0).tolist()

        if length_batching and sort_ids is not None:
            scores = restore_list_order(scores, sort_ids)

        return Prediction(scores=scores, system_score=sum(scores) / len(scores))


def _resolve_inference_device(
    *,
    gpus: int | Literal["auto"],
    accelerator: str,
    devices: Optional[Union[List[int], str, int]],
) -> tuple[str, Optional[Union[List[int], str, int]], int]:
    """Resolve PEAR's small public device API into Lightning arguments."""
    if gpus == "auto":
        use_accelerator = torch.cuda.is_available() or torch.backends.mps.is_available()
        if use_accelerator:
            return accelerator, 1 if devices is None else devices, 1
        return "cpu", "auto" if devices is None else devices, 0

    if not isinstance(gpus, int):
        raise ValueError("gpus must be 0, 1, or 'auto'.")
    if gpus > 1:
        raise ValueError("PEAR inference currently supports at most one GPU.")
    if gpus < 0:
        raise ValueError("gpus must be 0, 1, or 'auto'.")
    if gpus == 1 and devices is not None:
        if isinstance(devices, list) and len(devices) != gpus:
            raise ValueError("devices must match gpus when gpus=1.")
        return accelerator, devices, 1
    if gpus == 1:
        return accelerator, 1, 1
    return "cpu", "auto" if devices is None else devices, 0


@contextmanager
def _quiet_lightning_logs() -> Iterator[None]:
    """Suppress Lightning startup banners/tips during repeated inference calls."""
    lightning_loggers = [
        "lightning",
        "lightning.pytorch",
        "lightning_fabric",
        "pytorch_lightning",
    ]
    previous_levels = {
        name: logging.getLogger(name).level for name in lightning_loggers
    }
    try:
        for name in lightning_loggers:
            logging.getLogger(name).setLevel(logging.ERROR)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*GPU available but not used.*")
            warnings.filterwarnings("ignore", message=".*litlogger.*")
            warnings.filterwarnings("ignore", message=".*litmodels.*")
            warnings.filterwarnings("ignore", message=".*isinstance\\(treespec.*")
            yield
    finally:
        for name, level in previous_levels.items():
            logging.getLogger(name).setLevel(level)
