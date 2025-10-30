"""
This module contains the building blocks for a 2D attention mechanism.
Author: Aryaman Pandya
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Class implementation of the position wise MLP
    """

    def __init__(
        self, d_model: int, d_ff: int, dropout: float, num_layers: int = 2
    ) -> None:
        super(MLP, self).__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model if i == 0 else d_ff, d_ff, bias=True), nn.ReLU()
                )
                for i in range(num_layers - 1)
            ]
        )
        self.layers.append(nn.Linear(d_ff, d_model, bias=True))
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        for layer in self.layers:
            x = layer(x)
        return self.dropout(x) + residual


class Attention(nn.Module):
    """
    Multihead attention.
    """

    def __init__(
        self,
        dropout: float,
        num_heads: int,
        d_model: int,
        num_groups: int = 8,
        d_k: Optional[int] = None,
        is_masked: bool = False,
    ):
        """
        Args:
            d_k: dimension of the key
            dropout: dropout rate
            num_heads: number of heads
            d_model: number of channels
            num_groups: number of groups for group normalization
            mask: whether to use a mask
        """
        super(Attention, self).__init__()
        self.d_k = d_k if d_k is not None else d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.mask = is_masked
        self.d_model = d_model

        self.query_projection = nn.Linear(d_model, num_heads * self.d_k)
        self.key_projection = nn.Linear(d_model, num_heads * self.d_k)
        self.value_projection = nn.Linear(d_model, num_heads * self.d_k)

        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=d_model)
        self.output_layer = nn.Linear(num_heads * self.d_k, d_model)

    def attention_values(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes attention values.
        """
        batch_size = x.shape[0]
        residual = x

        if y is not None:
            k, q, v = y, x, y
        else:
            k, q, v = x, x, x

        k_len, q_len, v_len = k.size(1), q.size(1), v.size(1)

        k = (
            self.key_projection(k)
            .view(batch_size, k_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        q = (
            self.query_projection(q)
            .view(batch_size, q_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        v = (
            self.value_projection(v)
            .view(batch_size, v_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )

        attn_mask = None
        if self.mask:
            attn_mask = torch.tril(
                torch.ones(q_len, k_len, device=q.device, dtype=torch.bool)
            )

        output = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=self.mask and attn_mask is None,
        )
        output = self.output_layer(
            output.transpose(1, 2).contiguous().view(batch_size, q_len, -1)
        )

        return self.dropout(output) + residual

    def forward(self, x, y=None):
        """
        forward pass for the attention mechanism.

        Args:
            x: input tensor
            y: optional tensor for cross-attention
        """
        return self.attention_values(x, y)


class STTransformerLayer(nn.Module):
    """
    Spatial-temporal transformer layer block
    """

    def __init__(
        self,
        num_heads: int,
        d_model: int,
        d_linear: int,
        num_linear_layers: int = 2,
        num_groups: int = 8,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super(STTransformerLayer, self).__init__()
        self.norm1, self.norm2, self.norm3 = (
            nn.LayerNorm(d_model),
            nn.LayerNorm(d_model),
            nn.LayerNorm(d_model),
        )
        self.mha_space = Attention(dropout, num_heads, d_model, num_groups)
        self.mha_time = Attention(
            dropout, num_heads, d_model, num_groups, is_masked=causal
        )
        self.mlp = MLP(d_model, d_linear, dropout, num_linear_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass for spatial-temporal encoder layer.

        Args:
            x: input tensor
        """
        batch, time, patches, channels = x.shape
        h = x.permute(0, 2, 1, 3).reshape(
            batch * patches, time, channels
        )  # convert to [B, patches, time, channels] to compute attention across time dimension
        h = self.mha_time(self.norm1(h))
        h = h.reshape(batch, patches, time, channels).permute(0, 2, 1, 3) + x
        h2 = h.reshape(batch * time, patches, channels)
        h2 = self.mha_space(self.norm2(h2))
        h2 = h2.reshape(batch, time, patches, channels) + h
        return self.mlp(self.norm3(h2)) + h2
