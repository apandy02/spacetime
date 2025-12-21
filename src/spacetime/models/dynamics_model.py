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
        n_heads: int,
        d_model: int,
        n_layers: int,
        d_linear: int,
        action_codebook_size: int,
        action_dim: int,
        token_codebook_size: int,
        token_dim: int,
        n_tokens: int,
        p_sample: float = 0.2,
        n_linear_layers: int = 2,
        n_groups: int = 8,
        dropout: float = 0.1,
        n_frames: int = 16,
    ):
        super(DynamicsModel, self).__init__()
        self.n_tokens, self.n_frames = n_tokens, n_frames
        self.d_model = d_model
        self.p_sample = p_sample
        self.action_codebook_size, self.token_codebook_size = (
            action_codebook_size,
            token_codebook_size,
        )
        self.st_decoder = nn.ModuleList(
            [
                STTransformerLayer(
                    n_heads,
                    d_model,
                    d_linear,
                    n_linear_layers,
                    n_groups,
                    dropout,
                    mask=build_causal_mask,
                )
                for _ in range(n_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(d_model)

        self.pos_embed_space = nn.Parameter(torch.zeros(1, 1, self.n_tokens, d_model))
        self.pos_embed_time = nn.Parameter(torch.zeros(1, self.n_frames, 1, d_model))

        self.mask_embed = nn.Parameter(torch.zeros(d_model))

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
            tokens = layer(tokens)

        tokens = self.layer_norm(tokens)
        return tokens

    def _mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Bernoulli masking
        """
        batch_size, n_frames, n_tokens, _ = tokens.shape[0]
        a = torch.empty((batch_size, n_frames, n_tokens)).uniform_(0, 1)
        mask = a < self.p_sample
        tokens = torch.where(mask[..., None], self.mask_embed, tokens)
        return tokens
