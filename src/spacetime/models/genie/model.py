from dataclasses import dataclass

import torch
from torch import nn

from spacetime.models.genie.config import DynamicsConfig, LamConfig
from spacetime.models.genie.dynamics import DynamicsModel
from spacetime.models.genie.latent_actions import LatentActionModel
from spacetime.models.tokenizer.config import TokenizerConfig
from spacetime.models.tokenizer.model import VQTokenizer


@dataclass
class GenieOutput:
    lam_reconstruction: torch.Tensor
    z_e: torch.Tensor
    actions: torch.Tensor
    action_indices: torch.Tensor
    output_tokens: torch.Tensor
    token_indices: torch.Tensor
    token_embeddings: torch.Tensor


class GenieModel(nn.Module):
    """
    Genie: Generative Interactive Environments, Bruce et. al. 2024

    Model wrapper: pretrained tokenizer, and learnable latent action + dynamics models.
    """

    def __init__(
        self,
        lam_cfg: LamConfig,
        dyn_cfg: DynamicsConfig,
        tokenizer_cfg: TokenizerConfig,
        pretrained_tokenizer: VQTokenizer,
    ) -> None:
        super().__init__()

        self.lam = LatentActionModel(lam_cfg)
        self.dynamics = DynamicsModel(
            dynamics_cfg=dyn_cfg,
            lam_cfg=lam_cfg,
            tokenizer_cfg=tokenizer_cfg,
        )
        self.tokenizer = pretrained_tokenizer

    def forward(self, x: torch.Tensor) -> GenieOutput:
        """
        Video [B, C, F, H, W] → tokenizer (frozen) for tokens, LAM for actions + reconstruction,
        dynamics for predicted tokens. Returns GenieOutput for reconstruction, VQ, and dynamics losses.
        """
        with torch.no_grad():
            token_embeddings, token_indices, _ = self.tokenizer.tokenize(x)
            token_indices = token_indices.view(token_embeddings.shape[:-1])

        lam_reconstruction, z_e, actions, action_indices = self.lam(x)
        output_tokens = self.dynamics(token_embeddings[:, :-1], actions[:, :-1].detach())

        return GenieOutput(
            lam_reconstruction=lam_reconstruction,
            z_e=z_e,
            actions=actions,
            action_indices=action_indices,
            output_tokens=output_tokens,
            token_indices=token_indices,
            token_embeddings=token_embeddings,
        )
