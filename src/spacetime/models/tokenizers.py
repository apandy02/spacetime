import torch 
import torch.nn as nn
from spacetime.modules.transformer import STEncoder 


class STVQVae(nn.Module):
    """
    space time vq vae tokenizer
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
        super(STVQVae, self).__init__()



class VQVAEVideoEncoder(nn.Module):
    """
    vq vae video encoder
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
        super(VQVAEVideoEncoder, self).__init__()
        self.encoder = STEncoder(
            num_heads, num_channels, num_layers, d_linear, num_linear_layers, num_groups, dropout
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)