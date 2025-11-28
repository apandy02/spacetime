import torch
import torch.nn as nn
from spacetime.models.st_vq_vae import VQVAEVideoEncoder

def build_anti_causal_mask(seq_len: int, device=None, dtype=None) -> torch.Tensor:
    """
    Anti causal mask for the transformer. that allows attention to one token in the future.
    """
    tril = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=dtype))
    superdiag = torch.diag(torch.ones(seq_len - 1, device=device, dtype=dtype), diagonal=1)
    mask = tril + superdiag
    return mask.clamp(max=1)

class LatentActionModel(nn.Module):
    """
    Latent action model as per Genie.
    """
    def __init__(
        self,
        num_heads:int,
        d_model: int,
        num_layers: int,
        d_linear: int,
        codebook_dim: int,
        num_linear_layers: int = 2,
        num_groups: int = 8,
        dropout: float = 0.1,
    ):
        super(LatentActionModel, self).__init__()
        self.anti_causal_encoder = VQVAEVideoEncoder(
            num_heads,
            d_model,
            num_layers,
            d_linear,
            codebook_dim,
            num_linear_layers,
            num_groups,
            dropout=dropout,
            mask=build_anti_causal_mask,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the latent action model.
        """
        return self.anti_causal_encoder(x)