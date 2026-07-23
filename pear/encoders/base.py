r"""
Encoder Model base
====================
    Module defining the common interface between all pretrained encoder models.
"""

import abc
from typing import Dict, List

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerBase


class Encoder(nn.Module, metaclass=abc.ABCMeta):
    """Base class for an encoder model."""

    tokenizer: PreTrainedTokenizerBase
    model: nn.Module

    @property
    @abc.abstractmethod
    def output_units(self) -> int:
        """
        Hidden size of the encoder model.

        Returns:
            int: Hidden size of the encoder model.
        """
        pass

    @property
    @abc.abstractmethod
    def max_positions(self) -> int:
        """
        Max number of tokens the encoder handles.

        Returns:
            int: Max number of tokens the encoder handles.
        """
        pass

    @property
    @abc.abstractmethod
    def num_layers(self) -> int:
        """
        Number of model layers available.

        Returns:
            int: Number of model layers available.
        """
        pass

    @property
    @abc.abstractmethod
    def size_separator(self) -> int:
        """Size of the seperator.
        E.g: For BERT is just 1 (`[SEP]`) but models such as XLM-R and InfoXLM use 2 (`</s></s>`).

        Returns:
            int: Number of tokens used between two segments.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def from_pretrained(
        cls,
        pretrained_model: str,
        load_pretrained_weights: bool = True,
        local_files_only: bool = False,
        revision: str | None = None,
    ) -> "Encoder":
        """
        Function that loads a pretrained encoder and the respective tokenizer.

        Args:
            pretrained_model (str): Name of the pretrained model.
            load_pretrained_weights (bool): Whether to load pretrained weights.
            local_files_only (bool): Whether to only use locally cached files.
            revision (str | None): Hugging Face revision to load.

        Returns:
            Encoder: Pretrained model from Hugging Face.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError

    def freeze(self) -> None:
        """
        Freezes the entire encoder.
        """
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """
        Unfreezes the entire encoder.
        """
        for param in self.parameters():
            param.requires_grad = True

    @abc.abstractmethod
    def freeze_embeddings(self) -> None:
        """
        Freezes the embedding layer.
        """
        pass

    @abc.abstractmethod
    def layerwise_lr(self, lr: float, decay: float) -> List:
        """
        Calculates the learning rate for each encoder layer by applying a small decay.

        Args:
            lr (float): Learning rate for the highest encoder layer.
            decay (float): decay percentage for the lower layers.

        Returns:
            List: List of model parameters for all layers and the corresponding learning rate.
        """
        pass

    @abc.abstractmethod
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
        pass

    def prepare_sample(
        self,
        sample: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Receives a list of strings and applies tokenization and vectorization.

        Args:
            sample (List[str]): List with text segments to be tokenized and padded.

        Returns:
            Dict[str, torch.Tensor]: dict with `'input_ids'` and `'attention_mask'`.
        """
        return {
            k: v
            for k, v in self.tokenizer(
                sample,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_positions,
            ).items()
        }

    def pad_tensor(
        self, tensor: torch.Tensor, length: int, padding_index: int
    ) -> torch.Tensor:
        """
        Pad a tensor to length with padding_index.

        Args:
            tensor (torch.Tensor): Tensor to pad.
            length (int): Sequence length after padding.
            padding_index (int): Index to pad tensor with.

        Returns:
            torch.Tensor: Input batch
        """
        n_padding = length - tensor.shape[0]
        assert n_padding >= 0
        if n_padding == 0:
            return tensor
        padding = tensor.new(n_padding, *tensor.shape[1:]).fill_(padding_index)
        return torch.cat((tensor, padding), dim=0)

    def concat_sequences(
        self,
        inputs: List[Dict[str, torch.Tensor]],
        return_span_masks: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Concatenate K tokenized sequences per example into a single encoder input, distributing the length budget across segments before concatenation.
        Optionally it returns span masks for SRC/MT0/MT1 content tokens (separators and pads are 0).
        Works for:
            - K=2 : SRC + TR        (Absolute QE)
            - K=3 : SRC + TR1 + TR2 (Pairwise)

        Args:
            inputs (List[Dict[str, torch.Tensor]]): List of model inputs.
            return_span_masks (bool): Whether to return span masks. Default: `False`.

        Returns:
            Dict[str, torch.Tensor]: Returns a single model input with all sentences concatenated into a single input.
        """
        if len(inputs) == 0:
            raise ValueError("concat_sequences expects at least one input dict.")

        # 1) Strip padding via attention masks → python lists of ids per segment/example
        segments: List[List[List[int]]] = []
        for encoder_input in inputs:
            ids = encoder_input["input_ids"]  # [B, T]
            am = encoder_input["attention_mask"].bool()
            seg_i = [ids[b][am[b]].tolist() for b in range(ids.size(0))]
            segments.append(seg_i)

        K = len(segments)
        B = len(segments[0])

        if K != 2 and K != 3:
            raise ValueError(
                f"concat_sequences expects K=2 or K=3 segments, got K={K}."
            )

        # 2) Specials budget (depends on family via size_separator)
        # BERT: [CLS] s1 [SEP] s2 [SEP] s3 [SEP] → specials = 1 + K
        # RoBERTa/XLM-R/InfoXLM:
        #   K>=2: <s> s1 </s></s> s2 </s></s> s3 </s> ... → K + 2 (two double boundaries)
        if self.size_separator == 1:
            specials = 1 + K
        else:
            specials = 2 if K == 1 else 2 + (K - 1) * self.size_separator

        total_content_budget = max(self.max_positions - specials, 0)

        # Ids
        sep_id = self.tokenizer.sep_token_id
        cls_id = self.tokenizer.cls_token_id
        if cls_id is None or sep_id is None:
            raise ValueError("Tokenizer must define cls_token_id and sep_token_id.")
        assert isinstance(cls_id, int) and isinstance(sep_id, int), (
            f"cls_token_id and sep_token_id must be integers, got "
            f"{type(cls_id)}: {cls_id!r}, {type(sep_id)}: {sep_id!r}"
        )

        batch_tokens: List[torch.Tensor] = []
        # When masks requested, we store per-example seg-ids (0=special/pad, 1=SRC, 2=MT0, 3=MT1)
        segid_lists: List[List[int]] = [] if return_span_masks else None  # type: ignore

        for b in range(B):
            # content tokens per segment (strip outer specials)
            contents = [segments[k][b][1:-1] for k in range(K)]
            raw_lens = torch.tensor([len(c) for c in contents], dtype=torch.long)

            # 3) Equal-split allocation with round-robin handout
            q = total_content_budget // K
            alloc = torch.full((K,), q, dtype=torch.long)
            alloc = torch.minimum(alloc, raw_lens)  # respect per-segment caps
            leftover = int(total_content_budget - int(alloc.sum().item()))
            while leftover > 0:
                elig = torch.nonzero(raw_lens > alloc, as_tuple=False).flatten()
                if elig.numel() == 0:
                    break
                for t in elig.tolist():
                    i = int(t)
                    alloc[i] += 1
                    leftover -= 1
                    if leftover == 0:
                        break

            # truncate contents to allocated budgets
            contents = [contents[k][: int(alloc[k].item())] for k in range(K)]

            merged: List[int] = []
            segids: List[int] = [] if return_span_masks else None  # type: ignore

            if self.size_separator == 1:
                # BERT: [CLS] s1 [SEP] s2 [SEP] (s3 [SEP])*
                merged.append(cls_id)
                if segids is not None:
                    segids.append(0)
                # s1 (SRC)
                for t in contents[0]:
                    merged.append(t)
                    if segids is not None:
                        segids.append(1)
                merged.append(sep_id)
                if segids is not None:
                    segids.append(0)
                # s2 (MT0)
                for t in contents[1]:
                    merged.append(t)
                    if segids is not None:
                        segids.append(2)
                merged.append(sep_id)
                if segids is not None:
                    segids.append(0)
                # s3 (MT1) if present
                if K == 3:
                    for t in contents[2]:
                        merged.append(t)
                        if segids is not None:
                            segids.append(3)
                    merged.append(sep_id)
                    if segids is not None:
                        segids.append(0)
            else:
                # RoBERTa/XLM-R/InfoXLM
                merged.append(cls_id)
                if segids is not None:
                    segids.append(0)
                # <s> s1 </s></s> s2 </s></s> [ + s3 </s>]
                # s1 (SRC)
                for t in contents[0]:
                    merged.append(t)
                    if segids is not None:
                        segids.append(1)
                # first double boundary
                merged.append(sep_id)
                merged.append(sep_id)
                if segids is not None:
                    segids.append(0)
                    segids.append(0)
                # s2 (MT0)
                for t in contents[1]:
                    merged.append(t)
                    if segids is not None:
                        segids.append(2)
                # second double boundary
                merged.append(sep_id)
                merged.append(sep_id)
                if segids is not None:
                    segids.append(0)
                    segids.append(0)
                # s3.. (MT1 if present)
                if K == 3:
                    for t in contents[2]:
                        merged.append(t)
                        if segids is not None:
                            segids.append(3)
                    merged.append(sep_id)
                    if segids is not None:
                        segids.append(0)

            assert (
                len(merged) <= self.max_positions
            ), f"Merged length {len(merged)} exceeds max_positions {self.max_positions}."

            batch_tokens.append(torch.tensor(merged, dtype=torch.long))
            if return_span_masks:
                segid_lists.append(segids)  # type: ignore

        # 4) Pad and build attention mask
        lengths = [t.size(0) for t in batch_tokens]
        max_len = max(lengths)
        pad_raw = self.tokenizer.pad_token_id
        if not isinstance(pad_raw, int):
            raise ValueError(
                f"Tokenizer must have integer pad_token_id; got {pad_raw!r}."
            )
        pad_id: int = pad_raw

        padded = [self.pad_tensor(t, max_len, pad_id) for t in batch_tokens]
        padded = torch.stack(padded, dim=0).contiguous()  # [B, L]
        lens_t = torch.tensor(lengths, dtype=torch.long)
        arange = torch.arange(max_len, device=padded.device)
        attention_mask = (arange[None, :] < lens_t[:, None]).long()  # [B, L]

        out = {"input_ids": padded, "attention_mask": attention_mask}

        if return_span_masks:
            # Build masks from segids (0=special/pad)
            src_mask = torch.zeros((B, max_len), dtype=torch.long)
            mt0_mask = torch.zeros((B, max_len), dtype=torch.long)
            mt1_mask = torch.zeros((B, max_len), dtype=torch.long) if K == 3 else None
            for b in range(B):
                Lb = lengths[b]
                seg = segid_lists[b]
                # copy segids into [B,L], pad tail stays 0
                for i in range(Lb):
                    sid = seg[i]
                    if sid == 1:
                        src_mask[b, i] = 1
                    elif sid == 2:
                        mt0_mask[b, i] = 1
                    elif sid == 3 and K == 3:
                        mt1_mask[b, i] = 1  # type: ignore
            out["src_mask"] = src_mask
            out["mt0_mask"] = mt0_mask
            if K == 3:
                out["mt1_mask"] = mt1_mask  # type: ignore

        return out
