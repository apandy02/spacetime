from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from spacetime.modules.transformer import STTransformerLayer
from spacetime.modules.utils import build_causal_mask, patchify, unpatchify
from spacetime.modules.resnet import ResBlock

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
        conv_out_channels: Optional[list[int]] = None,
    ):
        super(STVQVae, self).__init__()
        self.patch_size = patch_size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.num_frames = num_frames

        conv_channels = conv_out_channels or [64, 64, 128, 128]
        strides = [2 if i % 2 == 0 else 1 for i in range(len(conv_channels))]

        div = 2 ** strides.count(2)
        h_s, w_s = frame_height // div, frame_width // div
        self.n_patch_h, self.n_patch_w = h_s // patch_size, w_s // patch_size
        self.n_patches = self.n_patch_h * self.n_patch_w

        self.encoder = VQVAEVideoEncoder(
            num_heads=num_heads,
            d_model=d_model,
            num_layers=num_layers,
            d_linear=d_linear,
            codebook_dim=codebook_dim,
            num_frames=num_frames,
            patch_size=patch_size,
            num_linear_layers=num_linear_layers,
            num_groups=num_groups,
            dropout=dropout,
            mask=build_causal_mask,
            conv_out_channels=conv_channels,
            conv_strides=strides,
            n_patches_h=self.n_patch_h,
            n_patches_w=self.n_patch_w,
        )
        self.codebook = nn.Parameter(
            torch.randn(codebook_size, codebook_dim) * 0.02, requires_grad=True
        )
        self.decoder = VQVAEVideoDecoder(
            num_heads=num_heads,
            d_model=d_model,
            num_layers=num_layers,
            d_linear=d_linear,
            codebook_dim=codebook_dim,
            input_image_channels=3,
            patch_size=patch_size,
            n_patches_h=self.n_patch_h,
            n_patches_w=self.n_patch_w,
            num_linear_layers=num_linear_layers,
            num_groups=num_groups,
            dropout=dropout,
            conv_out_channels=conv_channels,
            conv_stride=strides
        )
        self.vector_quantizer = EMAVectorQuantizer(codebook_size, codebook_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the STVQVAE.
        """
        z_e = self.encoder(x)
        z_q, _ = self.vector_quantizer(z_e)
        return self.decoder(z_q), z_e, z_q


class VQVAEVideoEncoder(nn.Module):
    """
    vq vae video encoder

    New structure:

    pixels → conv stem → patchify → latent grid → d_model projection → pos embeddings → ST attention → codebook projection → quantization
    """

    def __init__(
        self,
        num_heads,
        d_model: int,
        num_layers: int,
        d_linear: int,
        codebook_dim: int,
        num_frames: int,
        n_patches_h: int,
        n_patches_w: int,
        conv_out_channels: list[int],
        conv_strides: list[int],
        patch_size: int = 4,
        num_linear_layers: int = 2,
        num_groups: int = 8,
        dropout: float = 0.1,
        mask: Optional[Callable] = None,
    ):
        super(VQVAEVideoEncoder, self).__init__()

        self.patch_size, self.num_frames = patch_size, num_frames
        self.n_patches_h, self.n_patches_w = n_patches_h, n_patches_w
        self.n_patches = self.n_patches_h * self.n_patches_w

        self.conv_stem = nn.ModuleList([
            ResBlock(
                in_channels=3 if i == 0 else conv_out_channels[i-1],
                out_channels=conv_out_channels[i],
                activation=nn.ReLU,
                stride=conv_strides[i],
            )
            for i in range(len(conv_out_channels))
        ])

        d_patches = conv_out_channels[-1] * patch_size * patch_size

        self.d_model_projection = nn.Linear(d_patches, d_model)

        self.pos_embed_space = nn.Parameter(torch.zeros(1, 1, self.n_patches, d_model))
        self.pos_embed_time = nn.Parameter(torch.zeros(1, self.num_frames, 1, d_model))

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
        
        """
        b, c, f, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.reshape(b * f, c, h, w)
        for block in self.conv_stem:
            x = block(x)

        _, c_new, h_new, w_new = x.shape
        x = x.reshape(b, f, c_new, h_new, w_new)
        x = patchify(x, self.patch_size)

        x = self.d_model_projection(x) + self.pos_embed_space + self.pos_embed_time

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
        n_patches_h: int,
        n_patches_w: int,
        conv_strides: list[int],
        conv_out_channels: Optional[list[int]],
        num_linear_layers: int = 2,
        num_groups: int = 8,
        dropout: float = 0.1,
    ):
        super(VQVAEVideoDecoder, self).__init__()

        self.input_image_channels = input_image_channels
        self.patch_size = patch_size
        self.n_patches_h, self.n_patches_w = n_patches_h, n_patches_w

        self.c_latent = conv_out_channels[-1]

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
                for _ in range(num_lay  §ers)
            ]
        )

        self.layer_norm = nn.LayerNorm(d_model)

        d_patches = self.c_latent * patch_size * patch_size
        self.reconstruction_projector = nn.Linear(d_model, d_patches)

        downsample_indices = [i for i, s in enumerate(conv_strides) if s == 2]
        encoder_stage_channels = [conv_out_channels[i] for i in downsample_indices]
        decoder_stage_channels = list(reversed(encoder_stage_channels))  # e.g. [128, 64]

        conv_layers: list[nn.Module] = []
        in_channels = self.c_latent
        for stage_out_ch in decoder_stage_channels:
            conv_layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
            conv_layers.append(
                nn.Conv2d(in_channels, stage_out_ch, kernel_size=3, padding=1)
            )
            conv_layers.append(
                ResBlock(
                    in_channels=stage_out_ch,
                    out_channels=stage_out_ch,
                    activation=nn.ReLU,
                    stride=1,
                )
            )
            in_channels = stage_out_ch

        self.conv_stem = nn.Sequential(*conv_layers)
        self.final_conv = nn.Conv2d(in_channels, self.input_image_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the VQVAEVideoDecoder.
        inputs [B,T,N,D_codebook] -> linear projection [B,T,N,D_model] -> st decoder
        -> layer norm -> reconstruction_projector -> unpatchify -> conv up-stem -> pixels
        """
        x = self.d_model_projection(x)      # [B, T, N, d_model]
        for layer in self.st_decoder:
            x = layer(x)
        x = self.layer_norm(x)
        x = self.reconstruction_projector(x)  # [B, T, N, C_latent * p * p]

        x = unpatchify(x, self.patch_size, self.n_patches_h, self.n_patches_w)  # [B, T, C_latent, H', W']

        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)      # [B·T, C_latent, H', W']

        x = self.conv_stem(x)              # [B·T, C_dec, H, W]
        x = self.final_conv(x)             # [B·T, 3, H, W]
        x = x.reshape(b, t, self.input_image_channels, x.shape[-2], x.shape[-1])
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