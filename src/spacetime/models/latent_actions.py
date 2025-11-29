import torch
import torch.nn as nn

from spacetime.models.st_vq_vae import VQVAEVideoEncoder
from spacetime.modules.transformer import STTransformerLayer
from spacetime.modules.utils import build_anti_causal_mask, build_causal_mask


class LatentActionModel(nn.Module):
    """
    Latent action model as per Genie.
    """
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        num_layers: int,
        d_linear: int,
        codebook_dim: int,
        num_linear_layers: int = 2,
        num_groups: int = 8,
        dropout: float = 0.1,
        patch_size: int = 16,
    ):
        super(LatentActionModel, self).__init__()
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        action_embeddings = self.anti_causal_encoder(x)
        frame_embeddings = self.decoder(action_embeddings, x)
        return frame_embeddings

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
        self.reconstruction_projector = nn.Linear(
            d_model, input_image_channels * patch_size * patch_size
        )

    def forward(self, action_embeddings: torch.Tensor, frame_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the LatentActionDecoder.
        Takes as input:
            - action_embeddings: [B, T, N, D_codebook] (the latent action tokens)
            - frame_embeddings: [B, T, N, D_model] (embedded frames)

        The method:
            - Projects action embeddings to d_model: [B, T, N, D_model]
            - Prepends action embeddings to frame embeddings along the time dimension
            - Passes the resulting tensor through a stack of causal ST Transformer blocks
            - Applies layer normalization
            - Produces reconstructed frames using a projector, using only the frame embeddings (excluding action embeddings) for reconstruction
        Outputs:
            - [B, T, N, patch_dim], corresponding to the reconstructed frame patches
        """
        action_embeddings = self.d_model_projection(action_embeddings)
        # Prepend action_embedding along time (dim=1)
        x = torch.cat([action_embeddings, frame_embeddings], dim=1)
        for layer in self.st_decoder:
            x = layer(x)
        x = self.layer_norm(x)
        x = self.reconstruction_projector(x[:,1:,:,:])  # don't pass in the prepended action tokens for reconstruction
        return x
