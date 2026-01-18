from typing import Callable

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from spacetime.models.tokenizer.config import TokenizerConfig
from spacetime.modules.quantizers import (EMAVectorQuantizer, QuantizerType,
                                          VanillaVectorQuantizer)
from spacetime.modules.transformer import STTransformerLayer
from spacetime.modules.utils import patchify, unpatchify


class VQTokenizer(nn.Module):
    """
    Spatio-temporal VQ-VAE tokenizer.
    """

    def __init__(self, cfg: TokenizerConfig):
        """
        Initialize the ST Transformer-based VQ-VAE tokenizer.

        Args:
            cfg (TokenizerConfig): Configuration for the VQTokenizer.
        """
        super().__init__()
        model_cfg = cfg.model
        quant_cfg = cfg.quantizer

        self.patch_size = model_cfg.patch_size
        self.frame_height = model_cfg.frame_height
        self.frame_width = model_cfg.frame_width
        self.num_frames = model_cfg.num_frames
        self.n_patch_h = model_cfg.frame_height // model_cfg.patch_size
        self.n_patch_w = model_cfg.frame_width // model_cfg.patch_size
        self.n_patches = self.n_patch_h * self.n_patch_w

        d_patches = 3 * model_cfg.patch_size * model_cfg.patch_size

        self.d_model_projection = nn.Linear(d_patches, model_cfg.d_model)

        self.pos_embed_space = nn.Parameter(
            torch.zeros(1, 1, self.n_patches, model_cfg.d_model)
        )
        self.pos_embed_time = nn.Parameter(
            torch.zeros(1, self.num_frames, 1, model_cfg.d_model)
        )

        self.encoder = VQVAEVideoEncoder(
            model_cfg.num_heads,
            model_cfg.d_model,
            model_cfg.num_layers,
            model_cfg.d_linear,
            quant_cfg.codebook_dim,
            model_cfg.num_linear_layers,
            model_cfg.num_groups,
            dropout=model_cfg.dropout,
            is_causal=True,
            gradient_checkpointing=model_cfg.gradient_checkpointing,
        )
        self.decoder = VQVAEVideoDecoder(
            model_cfg.num_heads,
            model_cfg.d_model,
            model_cfg.num_layers,
            model_cfg.d_linear,
            quant_cfg.codebook_dim,
            input_image_channels=3,
            patch_size=model_cfg.patch_size,
            gradient_checkpointing=model_cfg.gradient_checkpointing,
        )
        if quant_cfg.type == QuantizerType.EMA:
            self.vector_quantizer = EMAVectorQuantizer(
                quant_cfg.codebook_size,
                quant_cfg.codebook_dim,
                decay=quant_cfg.decay,
                eps=quant_cfg.eps,
                dead_code_threshold=quant_cfg.dead_code_threshold,
                dead_code_noise=quant_cfg.dead_code_noise,
            )
        elif quant_cfg.type == QuantizerType.VANILLA:
            self.vector_quantizer = VanillaVectorQuantizer(
                quant_cfg.codebook_size, quant_cfg.codebook_dim
            )
        else:
            raise ValueError(f"Invalid quantizer type: {quant_cfg.type}")

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
        z_q, indices, z_e = self.tokenize(x)
        z_q_st = z_e + (z_q - z_e).detach()
        
        x_pred = unpatchify(
            self.decoder(z_q_st), self.patch_size, self.n_patch_h, self.n_patch_w
        )
        return x_pred, z_e, z_q, indices

    def tokenize(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode inputs into quantized codebook vectors and indices without decoding.

        Args:
            x: [B, C, F, H, W] (the input video)
        Returns:
            z_q: [B, F, N, D_codebook] (the quantized video)
            indices: [B, F, N] (the codebook indices)
            z_e: [B, F, N, D_model] (the encoder outputs)
        """
        x = patchify(x, self.patch_size)
        x = self.d_model_projection(x)
        z_e_raw = self.encoder(x + self.pos_embed_space + self.pos_embed_time)
        return self.vector_quantizer(z_e_raw)


    def quantizer_factory(self, quantizer_type: QuantizerType) -> nn.Module:
        if quantizer_type == QuantizerType.VANILLA:
            return VanillaVectorQuantizer
        if quantizer_type == QuantizerType.EMA:
            return EMAVectorQuantizer
        raise ValueError(f"Invalid quantizer type: {quantizer_type}")


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
        is_causal: bool = True,
        mask: Callable | None = None,
        gradient_checkpointing: bool = False,
    ):
        super(VQVAEVideoEncoder, self).__init__()
        self.gradient_checkpointing = gradient_checkpointing
        use_causal = is_causal if mask is None else False
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
                    is_causal=use_causal,
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
