"""
This module contains the building blocks for a 2D attention mechanism.
Author: Aryaman Pandya
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class MLP(nn.Module): 
    """
    Class implementation of the position wise MLP
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float, num_layers: int = 2) -> None:
        super(MLP, self).__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model if i == 0 else d_ff, d_ff, bias=True),
                nn.ReLU()
            ) for i in range(num_layers - 1)
        ])
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
        num_channels: int,
        num_groups: int = 8,
        d_k: Optional[int] = None,
        is_masked: bool = False
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
        super(Attention, self).__init__()
        self.d_k = d_k if d_k is not None else num_channels
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.mask = is_masked
        self.num_channels = num_channels

        self.query_projection = nn.Linear(num_channels, num_heads * self.d_k)
        self.key_projection = nn.Linear(num_channels, num_heads * self.d_k)
        self.value_projection = nn.Linear(num_channels, num_heads * self.d_k)

        self.group_norm = nn.GroupNorm(
            num_groups=num_groups, 
            num_channels=num_channels
        )
        self.output_layer = nn.Linear(num_heads * self.d_k, num_channels)
        

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

        k = self.key_projection(k).view(batch_size, k_len, self.num_heads, self.d_k).transpose(1, 2)
        q = self.query_projection(q).view(batch_size, q_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.value_projection(v).view(batch_size, v_len, self.num_heads, self.d_k).transpose(1, 2)

        attn_mask = None
        if self.mask:
            attn_mask = torch.tril(torch.ones(q_len, k_len, device=q.device, dtype=torch.bool))
        
        output = F.scaled_dot_product_attention(
            query=q,
            key=k, 
            value=v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=self.mask and attn_mask is None
        )
        output = self.output_layer(output.transpose(1, 2).contiguous().view(batch_size, q_len, -1))

        return self.dropout(output) + residual

    def forward(self, x, y=None):
        """
        forward pass for the attention mechanism.

        Args:
            x: input tensor
            y: optional tensor for cross-attention
        """
        return self.attention_values(x, y)



class STAttention(nn.Module):
    """
    Spatial-temporal attention block, as implemented in TimeSformer (Bertasius et al., 2020).

    TODO: currently just a copy of the 2D attention.
    """
    def __init__(
        self,
        dropout: float,
        num_heads: int,
        d_model: int,
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
            num_channels=d_model,
            num_groups=num_groups,
            d_k=d_k,
            is_masked=False,
        )

        self.temporal_attention = Attention(
            dropout=dropout,
            num_heads=num_heads,
            num_channels=d_model,
            num_groups=num_groups,
            d_k=d_k,
            is_masked=True,  # since temporal attention is causal
        )

        self.mlp = MLP(
            d_model=d_model,
            d_ff=4096,  # TODO: change to appropriate value
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass for spatial-temporal attention module.

        Args:
            x: input tensor
        """
        h1 = self.spatial_attention(x) + x
        h2 = self.temporal_attention(h1) + h1
        return self.mlp(h2) + h2


class STEncoderLayer(nn.Module):
    """
    Encoder layer block for ViT
    """
    def __init__(
        self, 
        num_heads: int,
        num_channels: int,
        d_linear: int,
        num_linear_layers: int = 2,
        num_groups: int = 8,
        dropout: float = 0.1,
    ):
        super(STEncoderLayer, self).__init__()
        self.norm1, self.norm2, self.norm3 = (
            nn.LayerNorm(num_channels),  nn.LayerNorm(num_channels), nn.LayerNorm(num_channels)
        )
        self.mha_space = Attention(dropout, num_heads, num_channels, num_groups)
        self.mha_time = Attention(dropout, num_heads, num_channels, num_groups)
        self.mlp = MLP(num_channels, d_linear, dropout, num_linear_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass for spatial-temporal encoder layer.

        Args:
            x: input tensor
        """
        batch, time, patches, channels = x.shape
        h = x.permute(0, 2, 1, 3).reshape(batch*patches, time, channels)  # convert to [B, patches, time, channels] to compute attention across time dimension
        h = self.mha_time(self.norm1(h))
        h = h.reshape(batch, patches, time, channels).permute(0, 2, 1, 3) + x
        h2 = h.reshape(batch*time, patches, channels)
        h2 = self.mha_space(self.norm2(h2))
        h2  = h2.reshape(batch, time, patches, channels) + h
        return self.mlp(self.norm3(h2)) + h2


class STEncoder(nn.Module):
    """
    Spatial-temporal encoder block, as implemented in TimeSformer (Bertasius et al., 2020).
    """
    def __init__(
        self, 
        num_heads: int,
        num_channels: int,
        num_layers: int,
        d_linear: int,
        num_linear_layers: int = 2,
        num_groups: int = 8,
        dropout: float = 0.1,
    ):
        super(STEncoder, self).__init__()
        self.layers = nn.ModuleList([
            STEncoderLayer(
                num_heads, num_channels, d_linear, num_linear_layers, num_groups, dropout
            ) for _ in range(num_layers)
        ])
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x 
