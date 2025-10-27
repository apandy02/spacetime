import torch
import torch.nn as nn

from spacetime.modules.transformer import STTransformerLayer


class STVQVae(nn.Module):
    """
    space time vq vae tokenizer
    """

    def __init__(
        self,
        num_heads: int,
        d_model: int,
        num_layers: int,
        d_linear: int,
        codebook_size: int,
        latent_dim: int,
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

        self.pos_embed_space = nn.Parameter(torch.zeros(1, 1, self.n_patches, d_model))
        self.pos_embed_time = nn.Parameter(torch.zeros(1, self.num_frames, 1, d_model))

        self.encoder = VQVAEVideoEncoder(
            num_heads,
            d_model,
            num_layers,
            d_linear,
            latent_dim,
            num_linear_layers,
            num_groups,
            dropout=dropout,
        )
        self.codebook = nn.Parameter(
            torch.randn(codebook_size, latent_dim) * 0.02, requires_grad=True
        )
        self.decoder = VQVAEVideoDecoder(
            num_heads,
            d_model,
            num_layers,
            d_linear,
            latent_dim,
            input_image_channels=3,
            patch_size=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the STVQVAE.
        """
        x = self._patchify(x)
        z_e = self.encoder(x + self.pos_embed_space + self.pos_embed_time)
        z_q = self._quantize_encoder_output(z_e)
        z_q_st = z_e + (z_q - z_e).detach()
        return self._unpatchify(self.decoder(z_q_st)), z_e, z_q

    def _quantize_encoder_output(self, z_e: torch.Tensor) -> torch.Tensor:
        """
        Quantizing the encoded tensor by snapping its elements to the closest codebook
        entry. Quantizes each patch independently.

        Args:
            z_e: encoded representation of shape [batch_size, frames, n_patches, latent_dim]

        Returns:
            z_q: quantized representation of shape [batch_size, frames, n_patches, latent_dim]
        """
        batch_size, frames, n_patches, latent_dim = z_e.shape

        encoded = z_e.reshape(-1, latent_dim)
        quantized = self.codebook[
            torch.argmin(torch.cdist(encoded, self.codebook), dim=1)
        ]
        z_q = quantized.reshape(batch_size, frames, n_patches, latent_dim)
        return z_q

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Splits a batch of videos into non-overlapping patches.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, F, H, W]

        Returns:
            torch.Tensor: Patchified tensor of shape [B, F, NUM_P, DIM_P],
                        where DIM_P = channels * patch_size * patch_size.
        """
        batch_size, channels, frames, height, width = x.shape
        n_patch_h = height // self.patch_size
        n_patch_w = width // self.patch_size
        x = x.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
        x = x.reshape(
            batch_size,
            frames,
            channels,
            n_patch_h,
            self.patch_size,
            n_patch_w,
            self.patch_size,
        )
        x = x.permute(0, 1, 3, 5, 2, 4, 6)  # [B, F, n_patch_h, n_patch_w, C, p_h, p_w]
        return x.reshape(
            batch_size,
            frames,
            n_patch_h * n_patch_w,
            channels * self.patch_size * self.patch_size,
        )

    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Unpatches a batch of videos from patches back to full frames.
        Reverses the patchification done by _patchify.

        Args:
            x (torch.Tensor): Input tensor of shape [B, F, NUM_P, DIM_P]

        Returns:
            torch.Tensor: Unpatchified tensor of shape [B, C, F, H, W]
        """
        batch_size, frames, _, patch_dim = x.shape
        n_patch_h = self.n_patch_h
        n_patch_w = self.n_patch_w
        channels = patch_dim // (self.patch_size * self.patch_size)

        # Reshape to separate patch dimensions
        x = x.reshape(
            batch_size,
            frames,
            n_patch_h,
            n_patch_w,
            channels,
            self.patch_size,
            self.patch_size,
        )
        x = x.permute(0, 1, 4, 2, 5, 3, 6)  # [B, F, C, n_patch_h, p_h, n_patch_w, p_w]
        # Merge patches
        x = x.reshape(
            batch_size,
            frames,
            channels,
            n_patch_h * self.patch_size,
            n_patch_w * self.patch_size,
        )
        # Final permute to get [B, C, F, H, W]
        return x.permute(0, 2, 1, 3, 4)


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
                    causal=True,
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
                    causal=False,
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
        x = self._unpatchify(x)
        return x
