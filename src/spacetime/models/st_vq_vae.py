from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from spacetime.modules.transformer import STTransformerLayer
from spacetime.modules.utils import build_causal_mask, patchify, unpatchify


class STVQVae(nn.Module):
    """
    space time vq vae
    """

    def __init__(
        self,
        num_heads: int,
        d_model: int,
        num_layers: int,
        d_linear: int,
        codebook_size: int,
        codebook_dim: int,
        patch_size: int,
        frame_height: int,
        frame_width: int,
        num_frames: int,
        num_linear_layers: int = 2,
        num_groups: int = 8,
        dropout: float = 0.1,
    ):
        super(STVQVae, self).__init__()
        self.patch_size = patch_size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.num_frames = num_frames
        self.n_patch_h = frame_height // patch_size
        self.n_patch_w = frame_width // patch_size
        self.n_patches = self.n_patch_h * self.n_patch_w

        d_patches = 3 * patch_size * patch_size  # hardcoded for RGB

        self.d_model_projection = nn.Linear(d_patches, d_model)

        self.pos_embed_space = nn.Parameter(torch.zeros(1, 1, self.n_patches, d_model))
        self.pos_embed_time = nn.Parameter(torch.zeros(1, self.num_frames, 1, d_model))

        self.encoder = VQVAEVideoEncoder(
            num_heads,
            d_model,
            num_layers,
            d_linear,
            codebook_dim,
            num_linear_layers,
            num_groups,
            dropout=dropout,
            mask=build_causal_mask,
        )
        self.codebook = nn.Parameter(
            torch.randn(codebook_size, codebook_dim) * 0.02, requires_grad=True
        )
        self.decoder = VQVAEVideoDecoder(
            num_heads,
            d_model,
            num_layers,
            d_linear,
            codebook_dim,
            input_image_channels=3,
            patch_size=patch_size,
        )
        self.vector_quantizer = EMAVectorQuantizer(codebook_size, codebook_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the STVQVAE.
        """
        x = patchify(x, self.patch_size)  # [B, F, NUM_P, DIM_P]
        x = self.d_model_projection(x)  # [B, F, NUM_P, D_model]
        z_e = self.encoder(x + self.pos_embed_space + self.pos_embed_time)
        z_q, _ = self.vector_quantizer(z_e)
        return unpatchify(self.decoder(z_q), self.patch_size, self.n_patch_h, self.n_patch_w), z_e, z_q


class VQVAEVideoEncoder(nn.Module):
    """
    vq vae video encoder
    """

    def __init__(
        self,
        num_heads,
        d_model: int,
        num_layers: int,
        d_linear: int,
        codebook_dim: int,
        num_linear_layers: int = 2,
        num_groups: int = 8,
        dropout: float = 0.1,
        mask: Optional[Callable] = None,
    ):
        super(VQVAEVideoEncoder, self).__init__()
        self.causal_st_encoder = nn.ModuleList(
            [
                STTransformerLayer(
                    num_heads,
                    d_model,
                    d_linear,
                    num_linear_layers,
                    num_groups,
                    dropout,
                    mask=mask,
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.codebook_projector = nn.Linear(d_model, codebook_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the VQVAEVideoEncoder.
        inputs [B,T,N,D_model] -> causal encoder -> layer norm -> linear projection -> outputs [B,T,N,D_codebook]
        """
        for layer in self.causal_st_encoder:
            x = layer(x)

        x = self.layer_norm(x)
        x = self.codebook_projector(x)
        return x


class VQVAEVideoDecoder(nn.Module):
    """
    vq vae video decoder
    """

    def __init__(
        self,
        num_heads,
        d_model: int,
        num_layers: int,
        d_linear: int,
        codebook_dim: int,
        input_image_channels: int,
        patch_size: int,
        num_linear_layers: int = 2,
        num_groups: int = 8,
        dropout: float = 0.1,
    ):
        super(VQVAEVideoDecoder, self).__init__()
        self.d_model_projection = nn.Linear(codebook_dim, d_model)
        self.st_decoder = nn.ModuleList(
            [
                STTransformerLayer(
                    num_heads,
                    d_model,
                    d_linear,
                    num_linear_layers,
                    num_groups,
                    dropout,
                    mask=None,
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.reconstruction_projector = nn.Linear(
            d_model, input_image_channels * patch_size * patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the VQVAEVideoDecoder.
        inputs [B,T,N,D_codebook] -> linear projection [B,T,N,D_model] -> st decoder -> layer norm -> outputs
        """
        x = self.d_model_projection(x)
        for layer in self.st_decoder:
            x = layer(x)
        x = self.layer_norm(x)
        x = self.reconstruction_projector(x)
        return x

class EMAVectorQuantizer(nn.Module):
    """
    EMA vector quantizer
    """
    def __init__(self, codebook_size: int, codebook_dim: int, decay: float = 0.98, eps: float = 1e-5):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.decay = decay
        self.eps = eps

        embed = torch.randn(codebook_size, codebook_dim)
        self.register_buffer('codebook', embed)
        self.register_buffer('ema_cluster_size', torch.zeros(codebook_size))
        self.register_buffer('ema_codebook', embed.clone())

    def forward(self, z_e: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the EMAVectorQuantizer.

        Args:
            z_e: encoded representation of shape [batch_size, frames, n_patches, codebook_dim]

        Returns:
            z_q: quantized representation of shape [batch_size, frames, n_patches, codebook_dim]
        """
        z_e_flat = z_e.reshape(-1, self.codebook_dim)

        z_sq = (z_e_flat ** 2).sum(dim=1, keepdim=True)
        e_sq = (self.codebook ** 2).sum(dim=1)
        dist = z_sq + e_sq - 2 * z_e_flat @ self.codebook.t()

        indices = dist.argmin(dim=1)
        z_q_flat = self.codebook[indices]
        z_q = z_q_flat.view_as(z_e)

        if self.training:
            with torch.no_grad():
                encodings = F.one_hot(indices, self.codebook_size).type(z_e_flat.dtype)
                cluster_size = encodings.sum(dim=0)

                self.ema_cluster_size.mul_(self.decay).add_(cluster_size * (1 - self.decay))
                codebook_sum = encodings.t() @ z_e_flat
                self.ema_codebook.mul_(self.decay).add_(codebook_sum * (1 - self.decay))

                n = self.ema_cluster_size.sum()
                cluster_size = (
                    (self.ema_cluster_size + self.eps)
                    / (n + self.codebook_size * self.eps) * n
                )

                cluster_size = torch.clamp(cluster_size, min=self.eps)
                self.codebook.copy_(self.ema_codebook / cluster_size.unsqueeze(1))

        return z_q, indices


def vanilla_vector_quantizer(z_e: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """
    Quantize encoded tensor by snapping its elements to the closest codebook entry.
    Quantizes each patch independently.

    Args:
        z_e: encoded representation of shape [batch_size, frames, n_patches, codebook_dim]
        codebook: codebook tensor of shape [codebook_size, codebook_dim]

    Returns:
        z_q: quantized representation of shape [batch_size, frames, n_patches, codebook_dim]
    """
    batch_size, frames, n_patches, codebook_dim = z_e.shape

    encoded = z_e.reshape(-1, codebook_dim)

    z_sq = (encoded ** 2).sum(dim=1, keepdim=True)
    e_sq = (codebook ** 2).sum(dim=1)     
    dist = z_sq + e_sq - 2 * encoded @ codebook.t()
    indices = dist.argmin(dim=1)
    return codebook[indices].reshape(batch_size, frames, n_patches, codebook_dim)