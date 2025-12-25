import torch
from torch import nn

from spacetime.models.tokenizer.model import VQVAEVideoEncoder
from spacetime.modules.transformer import STTransformerLayer
from spacetime.modules.utils import (build_anti_causal_mask, build_causal_mask,
                                     patchify, unpatchify)


def quantize(z_e: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    codebook_dim = codebook.shape[1]
    encoded = z_e.reshape(-1, codebook_dim)
    z_sq = (encoded**2).sum(dim=1, keepdim=True)
    e_sq = (codebook**2).sum(dim=1)
    dist = z_sq + e_sq - 2 * encoded @ codebook.t()
    indices = dist.argmin(dim=1)
    return codebook[indices].reshape_as(z_e)


class LatentActionModel(nn.Module):
    """
    Latent action model (VAE with quantized latent actions).
    """

    def __init__(
        self,
        num_heads: int,
        d_model: int,
        num_layers: int,
        d_linear: int,
        num_discrete_actions: int,
        codebook_dim: int,
        num_linear_layers: int = 2,
        num_groups: int = 8,
        dropout: float = 0.1,
        patch_size: int = 16,
        frame_height: int = 128,
        frame_width: int = 128,
        num_frames: int = 16,
    ):
        super(LatentActionModel, self).__init__()
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

        self.anti_causal_encoder = VQVAEVideoEncoder(
            num_heads=num_heads,
            d_model=d_model,
            num_layers=num_layers,
            d_linear=d_linear,
            codebook_dim=codebook_dim,
            num_linear_layers=num_linear_layers,
            num_groups=num_groups,
            dropout=dropout,
            mask=build_anti_causal_mask,
        )

        self.codebook = nn.Parameter(
            torch.randn(num_discrete_actions, codebook_dim) * 0.02, requires_grad=True
        )
        self.decoder = LatentActionDecoder(
            num_heads=num_heads,
            d_model=d_model,
            num_layers=num_layers,
            d_linear=d_linear,
            codebook_dim=codebook_dim,
            input_image_channels=3,
            patch_size=patch_size,
            num_linear_layers=num_linear_layers,
            num_groups=num_groups,
            dropout=dropout,
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the LatentActionModel (VAE with quantized latent actions).
        Takes as input:
            - x: [B, C, F, H, W] (the input video)
        Returns:
            - frame_embeddings: [B, F, NUM_P, patch_dim], corresponding to the reconstructed frame patches
            - encoder_output: [B, F, NUM_P, D_model], corresponding to the encoded frame embeddings
            - action_codes: [B, F, NUM_P, D_codebook], corresponding to the quantized action codes
        """
        x = patchify(x, self.patch_size)  # [B, F, NUM_P, DIM_P]
        frame_embeddings = self.d_model_projection(x)
        encoder_input = frame_embeddings + self.pos_embed_space + self.pos_embed_time
        encoder_output = self.anti_causal_encoder(encoder_input)
        action_codes = quantize(encoder_output, self.codebook)

        st_estimator_action_codes = (
            encoder_output + (action_codes - encoder_output).detach()
        )
        frame_reconstructions = self.decoder(
            st_estimator_action_codes,
            frame_embeddings,
            self.pos_embed_space,
            self.pos_embed_time,
        )
        return (
            unpatchify(
                frame_reconstructions, self.patch_size, self.n_patch_h, self.n_patch_w
            ),
            encoder_output,
            action_codes,
        )


class LatentActionDecoder(nn.Module):
    """
    Latent action decoder block.
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
        super(LatentActionDecoder, self).__init__()
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
                    mask=build_causal_mask,
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.token_type_embed = nn.Parameter(torch.zeros(2, d_model))
        self.reconstruction_projector = nn.Linear(
            d_model, input_image_channels * patch_size * patch_size
        )

    def forward(
        self,
        action_embeddings: torch.Tensor,
        frame_embeddings: torch.Tensor,
        pos_embed_space: torch.Tensor,
        pos_embed_time: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the LatentActionDecoder.
        Takes as input:
            - action_embeddings: [B, T, N, D_codebook] (the latent action tokens)
            - frame_embeddings: [B, T, N, D_model] (embedded frames)

        Does:
            - Projects action embeddings to d_model: [B, T, N, D_model]
            - Interleaves action and frame tokens: [a_1, f_1, a_2, f_2, ..., a_T, f_T]
              This creates a sequence of length 2T where each frame f_t can attend to
              its corresponding action a_t and all prior tokens via causal masking.
            - Passes the resulting tensor through a stack of causal ST Transformer blocks
            - Applies layer normalization
            - Extracts only the frame positions (odd indices) for reconstruction
        Outputs:
            - [B, T, N, patch_dim], corresponding to the reconstructed frame patches
        """
        action_embeddings = self.d_model_projection(action_embeddings)

        frame_embeddings = (
            frame_embeddings
            + pos_embed_space
            + pos_embed_time
            + self.token_type_embed[0].view(1, 1, 1, -1)
        )
        action_embeddings = (
            action_embeddings
            + pos_embed_space
            + pos_embed_time
            + self.token_type_embed[1].view(1, 1, 1, -1)
        )

        # Interleave action and frame tokens along time dimension
        batch_size, T, N, D = frame_embeddings.shape
        x = torch.stack([action_embeddings, frame_embeddings], dim=2)  # [B, T, 2, N, D]
        x = x.reshape(
            batch_size, 2 * T, N, D
        )  # [a_1, f_1, a_2, f_2, ..., a_T, f_T] -> [B, 2T, N, D]

        for layer in self.st_decoder:
            x = layer(x)
        x = self.layer_norm(x)

        # Extract only frame positions (indices 1, 3, 5, ... i.e., odd indices)
        x = x[:, 1::2, :, :]  # [B, T, N, D]
        x = self.reconstruction_projector(x)
        return x
