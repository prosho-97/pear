r"""
XLM-RoBERTa-XL Encoder
==============
    Pretrained XLM-RoBERTa-XL  encoder from Hugging Face.
"""

from transformers import AutoTokenizer, XLMRobertaXLConfig, XLMRobertaXLModel

from pear.encoders.base import Encoder
from pear.encoders.xlmr import XLMREncoder


class XLMRXLEncoder(XLMREncoder):
    """
    XLM-RoBERTA-XL encoder.

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
        revision: str | None = None,
    ) -> None:
        super(Encoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model,
            revision=revision,
            local_files_only=local_files_only,
        )
        if load_pretrained_weights:
            self.model = XLMRobertaXLModel.from_pretrained(
                pretrained_model,
                add_pooling_layer=False,
                revision=revision,
                local_files_only=local_files_only,
            )
        else:
            self.model = XLMRobertaXLModel(
                XLMRobertaXLConfig.from_pretrained(
                    pretrained_model,
                    revision=revision,
                    local_files_only=local_files_only,
                ),
                add_pooling_layer=False,
            )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model: str,
        load_pretrained_weights: bool = True,
        local_files_only: bool = False,
        revision: str | None = None,
    ) -> Encoder:
        """Function that loads a pretrained encoder from Hugging Face.

        Args:
            pretrained_model (str): Name of the pretrain model to be loaded.
            load_pretrained_weights (bool): If set to True loads the pretrained weights from Hugging Face. Default: `True`.
            local_files_only (bool): Whether or not to only look at local files. Default: `False`.
            revision (str | None): Hugging Face revision to load. Default: `None`.

        Returns:
            Encoder: XLMRXLEncoder object.
        """
        return XLMRXLEncoder(
            pretrained_model,
            load_pretrained_weights,
            local_files_only,
            revision,
        )
