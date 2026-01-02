from typing import Callable, Optional

import torch
import torch.nn as nn

from spacetime.modules.quantizers import (
    EMAVectorQuantizer,
    QuantizerType,
    VanillaVectorQuantizer,
)
from spacetime.modules.transformer import STTransformerLayer
from spacetime.modules.utils import build_causal_mask, patchify, unpatchify


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
        quantizer_type: QuantizerType = QuantizerType.EMA,
        quantizer_decay: float = 0.95,
        quantizer_eps: float = 1e-5,
        dead_code_threshold: float = 0.01,
        dead_code_noise: float = 1e-4,
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
        self.decoder = VQVAEVideoDecoder(
            num_heads,
            d_model,
            num_layers,
            d_linear,
            codebook_dim,
            input_image_channels=3,
            patch_size=patch_size,
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
        self, x: torch.Tensor, return_indices: bool = False
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """
        Forward pass for the STVQVAE.
        """
        x = patchify(x, self.patch_size)  # [B, F, NUM_P, DIM_P]
        x = self.d_model_projection(x)  # [B, F, NUM_P, D_model]
        z_e = self.encoder(x + self.pos_embed_space + self.pos_embed_time)
        z_q, indices = self.vector_quantizer(z_e)
        z_q_st = z_e + (z_q - z_e).detach()
        outputs = (
            unpatchify(
                self.decoder(z_q_st), self.patch_size, self.n_patch_h, self.n_patch_w
            ),
            z_e,
            z_q,
        )
        if return_indices:
            return outputs + (indices,)
        return outputs

    def quantizer_factory(self, quantizer_type: QuantizerType) -> nn.Module:
        if quantizer_type == QuantizerType.VANILLA:
            return VanillaVectorQuantizer
        if quantizer_type == QuantizerType.EMA:
            return EMAVectorQuantizer
        raise ValueError(f"Invalid quantizer type: {quantizer_type}")
