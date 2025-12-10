import torch
from torch import nn

from spacetime.modules.transformer import STTransformerLayer
from spacetime.modules.utils import build_causal_mask


class DynamicsModel(nn.Module):
    """
    MaskGIT based Video Dynamics model 
    """
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        num_layers: int,
        d_linear: int,
        action_codebook_size: int,
        action_dim: int,
        token_codebook_size: int,
        token_dim: int,
        num_tokens: int,
        num_linear_layers: int = 2,
        num_groups: int = 8,
        dropout: float = 0.1,
        num_frames: int = 16,
    ):
        super(DynamicsModel, self).__init__()
        self.n_tokens, self.num_frames = num_tokens, num_frames
        self.action_codebook_size, self.token_codebook_size = action_codebook_size, token_codebook_size
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

        self.pos_embed_space = nn.Parameter(torch.zeros(1, 1, self.n_tokens, d_model))
        self.pos_embed_time = nn.Parameter(torch.zeros(1, self.num_frames, 1, d_model))

        self.action_projector = nn.Linear(action_dim, d_model)
        self.token_projector = nn.Linear(token_dim, d_model)
    

    def forward(self, tokens: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        forward pass for maskgit based dynamics model 
        """
        tokens = self.token_projector(tokens)
        actions = self.action_projector(actions)

        tokens += self.pos_embed_space + self.pos_embed_time + actions

        tokens = self._mask(tokens)

         for layer in self.st_decoder:
            x = layer(x)
        
        x = self.layer_norm(x)
        return x


    

    def _mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Bernoulli masking 
        """
        return tokens
