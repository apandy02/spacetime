"""
This module contains the building blocks for a 2D attention mechanism.
Author: Aryaman Pandya
"""

import math
from typing import Optional

import torch
import torch.nn as nn


def scaled_dot_product_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        d_k: int,
        mask: torch.Tensor
) -> torch.Tensor:
    """
    Scaled dot product attention.

    Args:
        q: query tensor
        k: key tensor
        d_k: dimension of the key
        mask: whether to use a mask

    Returns:
        attention: attention tensor
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask:
        mask = torch.tril(torch.ones(scores.shape)).to(q.device)
        scores = scores.masked_fill(mask == 0, float('-inf'))
    return nn.Softmax(-1)(scores)


class Attention(nn.Module):
    """
    Multihead attention.
    """
    def __init__(
        self,
        dropout: float,
        num_heads: int,
        num_channels: int,
        num_groups: int = 8,
        d_k: Optional[int] = None,
        mask: bool = False
    ):
        """
        TODO: change to ViT style, expecting patches

        Args:
            d_k: dimension of the key
            dropout: dropout rate
            num_heads: number of heads
            num_channels: number of channels
            num_groups: number of groups for group normalization
            mask: whether to use a mask
        """
        super(Attention, self).__init__()
        self.d_k = d_k if d_k is not None else num_channels
        self.num_heads = num_heads

        self.query_projection = nn.Linear(num_channels, num_heads * self.d_k)
        self.key_projection = nn.Linear(num_channels, num_heads * self.d_k)
        self.value_projection = nn.Linear(num_channels, num_heads * self.d_k)

        self.group_norm = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=num_channels
        )
        self.output_layer = nn.Linear(num_heads * self.d_k, num_channels)
        self.dropout = nn.Dropout(dropout)
        self.mask = mask
        self.num_channels = num_channels

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        forward pass for 2D attention.

        Args:
            x: input tensor
            y: optional tensor for cross-attention
        """
        batch_size, _, _, _ = x.shape
        x = self.group_norm(x)
        residual = x

        if y is not None:
            k, q, v = y, x, y
        else:
            k, q, v = x, x, x

        k_len, q_len, v_len = k.size(1), q.size(1), v.size(1)

        k = self.key_projection(k).view(batch_size, k_len, self.num_heads, self.d_k)
        q = self.query_projection(q).view(batch_size, q_len, self.num_heads, self.d_k)
        v = self.value_projection(v).view(batch_size, v_len, self.num_heads, self.d_k)

        attention = scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            self.d_k,
            self.mask
        )
        output = torch.matmul(attention, v.transpose(1, 2))
        output = self.output_layer(output.transpose(1, 2).contiguous().view(batch_size, q_len, -1))

        return self.dropout(output) + residual


class STAttention(nn.Module):
    """
    Spatial-temporal attention block, as implemented in TimeSformer (Bertasius et al., 2020).

    TODO: currently just a copy of the 2D attention.
    """
    def __init__(
        self,
        dropout: float,
        num_heads: int,
        num_channels: int,
        num_groups: int = 8,
        d_k: Optional[int] = None,
    ):
        """
        Args:
            d_k: dimension of the key
            dropout: dropout rate
            num_heads: number of heads
            num_channels: number of channels
            num_groups: number of groups for group normalization
            mask: whether to use a mask
        """
        super(STAttention, self).__init__()
        self.spatial_attention = Attention(
            dropout=dropout,
            num_heads=num_heads,
            num_channels=num_channels,
            num_groups=num_groups,
            d_k=d_k,
            mask=False,
        )

        self.temporal_attention = Attention(
            dropout=dropout,
            num_heads=num_heads,
            num_channels=num_channels,
            num_groups=num_groups,
            d_k=d_k,
            mask=True,  # since temporal attention is causal
        )

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        forward pass for spatial-temporal attention.
        """
        return torch.Tensor(0)  # TODO: implement
