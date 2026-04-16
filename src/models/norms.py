from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ChannelLayerNorm2d(nn.Module):
    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = F.layer_norm(
            x.permute(0, 2, 3, 1),
            normalized_shape=(self.num_channels,),
            weight=self.weight,
            bias=self.bias,
            eps=self.eps,
        )
        return normalized.permute(0, 3, 1, 2)
