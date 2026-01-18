import torch
from torch import nn

from spacetime.models.dynamics_model.dynamics_model import DynamicsModel
from spacetime.models.genie.config import LamConfig, DynamicsConfig
from spacetime.models.tokenizer.config import TokenizerConfig
from spacetime.models.latent_actions.model import LatentActionModel
from spacetime.models.tokenizer.model import VQTokenizer


class GenieModel(nn.Module):
    """
    Genie
    """
    def __init__(
        self,
        lam_cfg: LamConfig,
        dyn_cfg: DynamicsConfig,
        tokenizer_cfg: TokenizerConfig,
        pretrained_tokenizer: VQTokenizer,
    ) -> None:
        super().__init__()

        self.lam = LatentActionModel(
            num_heads=lam_cfg.num_heads,
            d_model=lam_cfg.d_model,
            num_layers=lam_cfg.num_layers,
            d_linear=lam_cfg.d_linear,
            num_discrete_actions=lam_cfg.num_discrete_actions,
            codebook_dim=lam_cfg.codebook_dim,
            patch_size=lam_cfg.patch_size,
            frame_height=lam_cfg.frame_height,
            frame_width=lam_cfg.frame_width,
            num_frames=lam_cfg.num_frames,
            num_linear_layers=lam_cfg.num_linear_layers,
            num_groups=lam_cfg.num_groups,
            dropout=lam_cfg.dropout,
        )
        self.dynamics = DynamicsModel(
            dynamics_cfg=dyn_cfg,
            lam_cfg=lam_cfg,
            tokenizer_cfg=tokenizer_cfg,
        )
        self.tokenizer = pretrained_tokenizer

    def forward(
        self,
        lam_inputs: torch.Tensor | None = None,
        dynamics_tokens: torch.Tensor | None = None,
        dynamics_actions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Run the requested submodules and return their outputs.

        TODO: implement exact flow from paper diagram. 
        """
        outputs: dict[str, torch.Tensor] = {}
        if lam_inputs is not None:
            outputs["lam"] = self.lam(lam_inputs)
        if dynamics_tokens is not None and dynamics_actions is not None:
            outputs["dynamics"] = self.dynamics(dynamics_tokens, dynamics_actions)
        return outputs
