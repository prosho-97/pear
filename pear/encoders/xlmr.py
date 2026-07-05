r"""
XLM-RoBERTa Encoder
==============
    Pretrained XLM-RoBERTa  encoder from Hugging Face.
"""

from typing import List, Union

import torch
from transformers import (
    XLMRobertaConfig,
    XLMRobertaModel,
    XLMRobertaTokenizerFast,
    XLMRobertaXLModel,
)

from pear.encoders.base import Encoder


class XLMREncoder(Encoder):
    """
    XLM-RoBERTa encoder.

    Args:
        pretrained_model (str): Pretrained model from Hugging Face.
        load_pretrained_weights (bool): If set to `True` loads the pretrained weights from Hugging Face. Default: `True`.
        local_files_only (bool): Whether or not to only look at local files. Default: `False`.
    """

    def __init__(
        self,
        pretrained_model: str,
        load_pretrained_weights: bool = True,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(
            pretrained_model, local_files_only=local_files_only
        )
        self.model: Union[XLMRobertaModel, XLMRobertaXLModel]
        if load_pretrained_weights:
            self.model = XLMRobertaModel.from_pretrained(
                pretrained_model, add_pooling_layer=False
            )
        else:
            self.model = XLMRobertaModel(
                XLMRobertaConfig.from_pretrained(
                    pretrained_model, local_files_only=local_files_only
                ),
                add_pooling_layer=False,
            )

    @property
    def output_units(self) -> int:
        """
        Hidden size of the encoder model.

        Returns:
            int: Hidden size of the encoder model.
        """
        return self.model.config.hidden_size

    @property
    def max_positions(self) -> int:
        """
        Max number of tokens the encoder handles.

        Returns:
            int: Max number of tokens the encoder handles.
        """
        return self.model.config.max_position_embeddings - 2

    @property
    def num_layers(self) -> int:
        """
        Number of model layers available.

        Returns:
            int: Number of model layers available.
        """
        return self.model.config.num_hidden_layers + 1

    @property
    def size_separator(self):
        """Size of the seperator.
        E.g: For BERT is just 1 (`[SEP]`) but models such as XLM-R and InfoXLM use 2 (`</s></s>`).

        Returns:
            int: Number of tokens used between two segments.
        """
        return 2

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model: str,
        load_pretrained_weights: bool = True,
        local_files_only: bool = False,
    ) -> Encoder:
        """Function that loads a pretrained encoder from Hugging Face.

        Args:
            pretrained_model (str):Name of the pretrain model to be loaded.
            load_pretrained_weights (bool): If set to True loads the pretrained weights from Hugging Face. Default: `True`.
            local_files_only (bool): Whether or not to only look at local files. Default: `False`.

        Returns:
            Encoder: XLMREncoder object.
        """
        return XLMREncoder(pretrained_model, load_pretrained_weights, local_files_only)

    def layerwise_lr(self, lr: float, decay: float) -> List:
        """
        Calculates the learning rate for each encoder layer by applying a small decay.

        Args:
            lr (float): Learning rate for the highest encoder layer.
            decay (float): decay percentage for the lower layers.

        Returns:
            List: List of model parameters for all layers and the corresponding learning rate.
        """
        # Last layer keeps LR
        opt_parameters = [
            {
                "params": self.model.encoder.layer[-1].parameters(),
                "lr": lr,
            }
        ]
        # Decay at each layer.
        for i in range(2, self.num_layers):
            opt_parameters.append(
                {
                    "params": self.model.encoder.layer[-i].parameters(),
                    "lr": lr * decay ** (i - 1),
                }
            )
        # Embedding Layer
        opt_parameters.append(
            {
                "params": self.model.embeddings.parameters(),
                "lr": lr * decay ** (self.num_layers),
            }
        )
        return opt_parameters

    def freeze_embeddings(self) -> None:
        """Freezes the embedding layer."""
        for param in self.model.embeddings.parameters():
            param.requires_grad = False

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Encoder model forward.

        Args:
            input_ids (torch.Tensor): Tokenized batch.
            attention_mask (torch.Tensor): Batch attention mask.

        Returns:
            torch.Tensor: All layers embeddings.
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states  # tuple: (emb, layer1, ..., layerN)
