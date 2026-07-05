r"""Inference-only PEAR pairwise metric model."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, cast

import torch
import torch.nn.functional as F
from torch import nn

from pear.models.base import MetricModel
from pear.models.utils import Prediction
from pear.modules import FeedForward


class PairwiseMetric(MetricModel):
    """Inference-only PEAR model for pairwise MT quality estimation.

    The released PEAR checkpoints are regression models with an antisymmetric
    source-aware head over last-layer encoder token states. Training-only
    Lightning hparams are accepted via ``**unused_hparams`` and ignored.
    """

    def __init__(
        self,
        keep_embeddings_frozen: bool = False,
        encoder_model: str = "InfoXLM",
        pretrained_model: str = "microsoft/infoxlm-large",
        blend: str = "last",
        dropout: float = 0.1,
        load_pretrained_weights: bool = True,
        local_files_only: bool = False,
        hidden_sizes: Optional[List[int]] = None,
        activations: str = "GELU",
        task: str = "regression",
        layer_norm_before_head: bool = False,
        use_antisymmetric_head: bool = True,
        learn_score_scale: bool = True,
        score_alpha: float = 1.0,
        final_activation: Optional[str] = None,
        final_init_std: Optional[float] = 0.02,
        final_init_bias: Optional[float] = 0.0,
        **unused_hparams: Any,
    ) -> None:
        super().__init__(
            keep_embeddings_frozen=keep_embeddings_frozen,
            encoder_model=encoder_model,
            pretrained_model=pretrained_model,
            blend=blend,
            load_pretrained_weights=load_pretrained_weights,
            local_files_only=local_files_only,
            **unused_hparams,
        )
        self.save_hyperparameters()

        if self.hparams["task"] != "regression":
            raise ValueError("pear-mt only supports released regression PEAR models.")
        if not self.hparams["use_antisymmetric_head"]:
            raise ValueError(
                "pear-mt only supports released PEAR models with use_antisymmetric_head=True."
            )

        if self.hparams["learn_score_scale"]:
            alpha = max(float(self.hparams.get("score_alpha", 1.0)), 1e-4)
            self._alpha_raw = nn.Parameter(
                torch.tensor(math.log(math.expm1(alpha)), dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "score_scale",
                torch.tensor(
                    float(self.hparams.get("score_alpha", 1.0)), dtype=torch.float32
                ),
                persistent=False,
            )

        self._feat_dim = self.encoder.output_units
        if self.hparams["layer_norm_before_head"]:
            self.pre_head_norm = nn.LayerNorm(self._feat_dim)
            self.span_proj = nn.Sequential(
                nn.Linear(3 * self._feat_dim, self._feat_dim),
                nn.GELU(),
                nn.Dropout(self.hparams["dropout"]),
            )
        else:
            # Matches the released PEAR checkpoints, where the span projection
            # owns the LayerNorm parameters loaded from ``span_proj.2``.
            self.pre_head_norm = nn.Identity()
            self.span_proj = nn.Sequential(
                nn.Linear(3 * self._feat_dim, self._feat_dim),
                nn.GELU(),
                nn.LayerNorm(self._feat_dim),
                nn.Dropout(self.hparams["dropout"]),
            )
        self._init_span_proj()

        self.pair_head = FeedForward(
            in_dim=self._feat_dim,
            hidden_sizes=hidden_sizes or [512, 256, 128],
            activations=activations,
            dropout=dropout,
            final_activation=final_activation,
            final_init_std=final_init_std,
            final_init_bias=final_init_bias,
        )

    def _init_span_proj(self) -> None:
        for module in self.span_proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(
                    module.weight, a=0.0, mode="fan_in", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @property
    def feature_dim(self) -> int:
        """Dimension of the pairwise head input features."""
        return self._feat_dim

    def requires_references(self) -> bool:
        """PEAR pairwise QE does not require references."""
        return False

    def prepare_sample(
        self, sample: List[Dict[str, str]], stage: str = "predict"
    ) -> Dict[str, torch.Tensor]:
        """Tokenize a PEAR inference batch with ``src``, ``mt_0``, and ``mt_1``."""
        if stage != "predict":
            raise ValueError("The pear-mt package only supports prediction batches.")
        inputs = {key: [row[key] for row in sample] for key in sample[0]}
        return self.encoder.concat_sequences(
            [
                self.encoder.prepare_sample([str(x) for x in inputs["src"]]),
                self.encoder.prepare_sample([str(x) for x in inputs["mt_0"]]),
                self.encoder.prepare_sample([str(x) for x in inputs["mt_1"]]),
            ],
            return_span_masks=True,
        )

    def _masked_mean(
        self, x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        if (
            mask.dim() != 2
            or x.dim() != 3
            or x.size(0) != mask.size(0)
            or x.size(1) != mask.size(1)
        ):
            raise ValueError(
                f"_masked_mean expects x:[B,T,D], mask:[B,T]; got x={tuple(x.shape)}, mask={tuple(mask.shape)}"
            )
        weights = mask.to(dtype=x.dtype)
        return (x * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(dim=1).clamp_min(
            eps
        ).unsqueeze(-1)

    def _current_score_scale(self) -> torch.Tensor:
        if self.hparams["learn_score_scale"]:
            return F.softplus(self._alpha_raw) + 1e-6
        return cast(torch.Tensor, self.score_scale)

    def score_from_z(self, z: torch.Tensor) -> torch.Tensor:
        """Apply the checkpoint's regression score scale to raw logits."""
        return self._current_score_scale().to(z.dtype) * z

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> Prediction:
        """Run the PEAR antisymmetric pairwise scoring head."""
        src_mask = kwargs.get("src_mask")
        mt0_mask = kwargs.get("mt0_mask")
        mt1_mask = kwargs.get("mt1_mask")
        if src_mask is None or mt0_mask is None or mt1_mask is None:
            raise ValueError(
                "PEAR inference requires src_mask, mt0_mask, and mt1_mask."
            )

        token_features = self.encoder(input_ids, attention_mask)[-1]
        h_src = self._masked_mean(token_features, src_mask)
        h0 = self._masked_mean(token_features, mt0_mask)
        h1 = self._masked_mean(token_features, mt1_mask)

        f0 = torch.cat([h0, h0 * h_src, (h0 - h_src).abs()], dim=-1)
        f1 = torch.cat([h1, h1 * h_src, (h1 - h_src).abs()], dim=-1)
        a0 = self.pre_head_norm(self.span_proj(f0))
        a1 = self.pre_head_norm(self.span_proj(f1))

        z = self.pair_head(a0).squeeze(-1) - self.pair_head(a1).squeeze(-1)
        return Prediction(score=self.score_from_z(z), z=z)

    def predict_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
    ) -> Prediction:
        """PyTorch Lightning prediction hook."""
        return Prediction(scores=self(**batch).score.detach())
