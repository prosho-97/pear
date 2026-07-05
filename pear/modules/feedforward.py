"""Small MLP head used by released PEAR checkpoints."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn


class FeedForward(nn.Module):
    """Feed-forward regression head compatible with PEAR checkpoints."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        hidden_sizes: Sequence[int] = (512, 256, 128),
        activations: str = "GELU",
        dropout: float = 0.1,
        final_activation: Optional[str] = None,
        final_init_std: Optional[float] = 0.02,
        final_init_bias: Optional[float] = 0.0,
    ) -> None:
        super().__init__()
        if not hidden_sizes:
            raise ValueError("FeedForward requires at least one hidden layer size.")

        self._act_name = activations.strip().lower()
        layers: list[nn.Module] = []
        previous_dim = in_dim
        for hidden_dim in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(previous_dim, int(hidden_dim)),
                    self._activation(),
                    nn.Dropout(dropout),
                ]
            )
            previous_dim = int(hidden_dim)
        layers.append(nn.Linear(previous_dim, int(out_dim)))
        if final_activation is not None:
            layers.append(self._activation(final_activation))

        self.ff = nn.Sequential(*layers)
        self._apply_initialization(final_init_std, final_init_bias)

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        """Run the MLP head, matching half-precision checkpoint dtypes if needed."""
        ff_dtypes = {param.dtype for param in self.ff.parameters()}
        if ff_dtypes == {torch.float16} and in_features.dtype != torch.float16:
            in_features = in_features.to(torch.float16)
        elif ff_dtypes == {torch.bfloat16} and in_features.dtype != torch.bfloat16:
            in_features = in_features.to(torch.bfloat16)
        return self.ff(in_features)

    def _activation(self, activation: Optional[str] = None) -> nn.Module:
        name = (activation or self._act_name).strip().lower()
        aliases = {
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "silu": nn.SiLU,
            "swish": nn.SiLU,
            "tanh": nn.Tanh,
            "identity": nn.Identity,
        }
        try:
            return aliases[name]()
        except KeyError as exc:
            raise ValueError(
                f"Unsupported activation: {activation or self._act_name!r}"
            ) from exc

    def _apply_initialization(
        self, final_init_std: Optional[float], final_init_bias: Optional[float]
    ) -> None:
        linear_layers = [m for m in self.ff if isinstance(m, nn.Linear)]
        for layer in linear_layers[:-1]:
            self._init_hidden(layer)

        final = linear_layers[-1]
        if final_init_std is None:
            self._init_hidden(final, init_bias=final_init_bias is None)
        else:
            nn.init.normal_(final.weight, mean=0.0, std=float(final_init_std))
        if final_init_bias is not None:
            nn.init.constant_(final.bias, float(final_init_bias))

    def _init_hidden(self, layer: nn.Linear, init_bias: bool = True) -> None:
        if self._act_name in {"gelu", "silu", "swish", "relu"}:
            nn.init.kaiming_uniform_(
                layer.weight, a=0.0, mode="fan_in", nonlinearity="relu"
            )
        else:
            gain = nn.init.calculate_gain("tanh") if self._act_name == "tanh" else 1.0
            nn.init.xavier_uniform_(layer.weight, gain=gain)
        if init_bias:
            nn.init.zeros_(layer.bias)
