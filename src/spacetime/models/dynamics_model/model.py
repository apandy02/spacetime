import torch
from torch import nn

from spacetime.models.genie.config import DynamicsConfig, LamConfig
from spacetime.models.tokenizer.config import TokenizerConfig
from spacetime.modules.transformer import STTransformerLayer
from spacetime.modules.utils import build_causal_mask


class DynamicsModel(nn.Module):
    """
    MaskGIT based Video Dynamics model
    """

    def __init__(
        self,
        dynamics_cfg: DynamicsConfig,
        lam_cfg: LamConfig,
        tokenizer_cfg: TokenizerConfig,
    ):
        """
        Initialize the MaskGIT based Video Dynamics model.
        Args:
            dynamics_cfg (DynamicsConfig): Configuration for the DynamicsModel.
            lam_cfg (LamConfig): Configuration for the LatentActionModel.
            tokenizer_cfg (TokenizerConfig): Configuration for the VQTokenizer.
        """
        super(DynamicsModel, self).__init__()
        
        n_heads = dynamics_cfg.n_heads
        d_model = dynamics_cfg.d_model
        n_layers = dynamics_cfg.n_layers
        d_linear = dynamics_cfg.d_linear
        n_linear_layers = dynamics_cfg.n_linear_layers
        n_groups = dynamics_cfg.n_groups
        dropout = dynamics_cfg.dropout
        p_sample = dynamics_cfg.p_sample
        
        action_codebook_size = lam_cfg.num_discrete_actions
        action_dim = lam_cfg.codebook_dim
        n_frames = lam_cfg.num_frames
        
        token_codebook_size = tokenizer_cfg.quantizer.codebook_size
        token_dim = tokenizer_cfg.quantizer.codebook_dim
        
        n_tokens = (tokenizer_cfg.model.frame_height // tokenizer_cfg.model.patch_size) * \
                   (tokenizer_cfg.model.frame_width // tokenizer_cfg.model.patch_size)
        
        self.n_tokens = n_tokens
        self.n_frames = n_frames
        self.d_model = d_model
        self.p_sample = p_sample
        self.action_codebook_size = action_codebook_size
        self.action_dim = action_dim
        self.token_codebook_size = token_codebook_size
        self.token_dim = token_dim
        
        self.st_decoder = nn.ModuleList(
            [
                STTransformerLayer(
                    n_heads,
                    self.d_model,
                    d_linear,
                    n_linear_layers,
                    n_groups,
                    dropout,
                    mask=build_causal_mask,
                )
                for _ in range(n_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(self.d_model)

        self.pos_embed_space = nn.Parameter(torch.zeros(1, 1, self.n_tokens, self.d_model))
        self.pos_embed_time = nn.Parameter(torch.zeros(1, self.n_frames, 1, self.d_model))

        self.mask_embed = nn.Parameter(torch.zeros(self.d_model))

        self.action_projector = nn.Linear(self.action_dim, self.d_model)
        self.token_projector = nn.Linear(self.token_dim, self.d_model)

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
        batch_size, n_frames, n_tokens, _ = tokens.shape
        a = torch.empty((batch_size, n_frames, n_tokens)).uniform_(0, 1)
        mask = a < self.p_sample
        tokens = torch.where(mask[..., None], self.mask_embed, tokens)
        return tokens
