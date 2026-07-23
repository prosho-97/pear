import os
from pathlib import Path
from typing import Union

import torch
import yaml
from huggingface_hub import snapshot_download

from .base import MetricModel
from .pairwise.pairwise_metric import PairwiseMetric

str2model = {"pairwise_metric": PairwiseMetric}

DEFAULT_HF_MODELS = {
    "pear": "Prosho/pear",
    "pear-xl": "Prosho/pear-xl",
}

DEFAULT_HF_REVISIONS = {
    "pear": "3cc8b7894f6542fde430cc2a9f96a82fa6a679ff",
    "Prosho/pear": "3cc8b7894f6542fde430cc2a9f96a82fa6a679ff",
    "pear-xl": "1f3b730fca2e223642270749310bb429b4ce5c0a",
    "Prosho/pear-xl": "1f3b730fca2e223642270749310bb429b4ce5c0a",
}


def download_model(
    model: str,
    saving_directory: Union[str, Path, None] = None,
    local_files_only: bool = False,
    revision: str | None = None,
) -> str:
    """Download a PEAR model from Hugging Face Hub and return model.ckpt path."""
    repo_id = DEFAULT_HF_MODELS.get(model, model)
    if revision is None:
        revision = DEFAULT_HF_REVISIONS.get(model, DEFAULT_HF_REVISIONS.get(repo_id))
    try:
        model_path = snapshot_download(
            repo_id=repo_id,
            cache_dir=saving_directory,
            local_files_only=local_files_only,
            revision=revision,
        )
    except Exception as exc:
        raise KeyError(f"Model '{model}' not found.") from exc
    return os.path.join(model_path, "checkpoints", "model.ckpt")


def load_from_checkpoint(
    checkpoint_path: Union[str, os.PathLike[str], Path],
    reload_hparams: bool = False,
    strict: bool = False,
    local_files_only: bool = False,
    class_identifier: str = "pairwise_metric",
    encoder_revision: str | None = None,
) -> MetricModel:
    """Load a PEAR pairwise metric model from a Lightning checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Invalid checkpoint path: {checkpoint_path}")

    parent_folder = checkpoint_path.parents[1]
    hparams_file = parent_folder / "hparams.yaml"
    if hparams_file.is_file():
        with hparams_file.open() as yaml_file:
            hparams = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)
        model_class = str2model[hparams.get("class_identifier", class_identifier)]
    elif reload_hparams:
        raise FileNotFoundError(f"Missing hparams.yaml in {parent_folder}")
    else:
        model_class = str2model[class_identifier]

    checkpoint_kwargs = {}
    if encoder_revision is not None:
        checkpoint_kwargs["encoder_revision"] = encoder_revision

    return model_class.load_from_checkpoint(
        checkpoint_path,
        load_pretrained_weights=False,
        hparams_file=hparams_file if reload_hparams else None,
        map_location=torch.device("cpu"),
        strict=strict,
        local_files_only=local_files_only,
        **checkpoint_kwargs,
    )
