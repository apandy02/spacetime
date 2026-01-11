from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from spacetime.modules.quantizers import (
    EMAVectorQuantizer,
    QuantizerType,
    VanillaVectorQuantizer,
)
from spacetime.modules.transformer import STTransformerLayer
from spacetime.modules.utils import build_causal_mask, patchify, unpatchify


class VQVAEVideoEncoder(nn.Module):
    """
    VQ-VAE video encoder.
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
        gradient_checkpointing: bool = False,
    ):
        super(VQVAEVideoEncoder, self).__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.layers = nn.ModuleList(
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
        [B,T,N,D_model] -> encoder -> layer norm -> projection -> [B,T,N,D_codebook]
        """
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

        return self.codebook_projector(self.layer_norm(x))


class VQVAEVideoDecoder(nn.Module):
    """
    VQ-VAE video decoder.
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
        gradient_checkpointing: bool = False,
    ):
        super(VQVAEVideoDecoder, self).__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.codebook_projection = nn.Linear(codebook_dim, d_model)
        self.layers = nn.ModuleList(
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
        [B,T,N,D_codebook] -> projection -> decoder -> layer norm -> [B,T,N,patch_pixels]
        """
        x = self.codebook_projection(x)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

        return self.reconstruction_projector(self.layer_norm(x))


class STVQVae(nn.Module):
    """
    Spatial-temporal VQ-VAE.
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
        quantizer_type: QuantizerType = QuantizerType.EMA,
        quantizer_decay: float = 0.95,
        quantizer_eps: float = 1e-5,
        dead_code_threshold: float = 0.01,
        dead_code_noise: float = 1e-4,
        gradient_checkpointing: bool = False,
    ):
        super(STVQVae, self).__init__()
        self.patch_size = patch_size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.num_frames = num_frames
        self.n_patch_h = frame_height // patch_size
        self.n_patch_w = frame_width // patch_size
        self.n_patches = self.n_patch_h * self.n_patch_w

        d_patches = 3 * patch_size * patch_size

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
            gradient_checkpointing=gradient_checkpointing,
        )
        self.decoder = VQVAEVideoDecoder(
            num_heads,
            d_model,
            num_layers,
            d_linear,
            codebook_dim,
            input_image_channels=3,
            patch_size=patch_size,
            gradient_checkpointing=gradient_checkpointing,
        )
        if quantizer_type == QuantizerType.EMA:
            self.vector_quantizer = EMAVectorQuantizer(
                codebook_size,
                codebook_dim,
                decay=quantizer_decay,
                eps=quantizer_eps,
                dead_code_threshold=dead_code_threshold,
                dead_code_noise=dead_code_noise,
            )
        elif quantizer_type == QuantizerType.VANILLA:
            self.vector_quantizer = VanillaVectorQuantizer(codebook_size, codebook_dim)
        else:
            raise ValueError(f"Invalid quantizer type: {quantizer_type}")

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the STVQVAE.

        Returns:
            x_pred: reconstructed input
            z_e: encoder output (normalized)
            z_q: quantized representation
            indices: codebook indices
        """
        x = patchify(x, self.patch_size)
        x = self.d_model_projection(x)
        z_e_raw = self.encoder(x + self.pos_embed_space + self.pos_embed_time)
        z_q, indices, z_e = self.vector_quantizer(z_e_raw)

        z_q_st = z_e + (z_q - z_e).detach()
        x_pred = unpatchify(
            self.decoder(z_q_st), self.patch_size, self.n_patch_h, self.n_patch_w
        )
        return x_pred, z_e, z_q, indices

    def quantizer_factory(self, quantizer_type: QuantizerType) -> nn.Module:
        if quantizer_type == QuantizerType.VANILLA:
            return VanillaVectorQuantizer
        if quantizer_type == QuantizerType.EMA:
            return EMAVectorQuantizer
        raise ValueError(f"Invalid quantizer type: {quantizer_type}")
