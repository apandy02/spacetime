import math

import torch
import torch.nn.functional as F

from spacetime.modules.quantizers import QuantizerType


def compute_entropy_loss(indices: torch.Tensor, codebook_size: int) -> torch.Tensor:
    """
    Compute normalized entropy loss to encourage diverse codebook usage.

    Returns a value in [0, 1] where 0 means perfect uniform usage
    and 1 means complete collapse to a single code.
    """
    flat_indices = indices.reshape(-1)
    counts = torch.bincount(flat_indices, minlength=codebook_size).float()
    probs = counts / (counts.sum() + 1e-8)
    entropy = -(probs * torch.log(probs + 1e-8)).sum()
    max_entropy = math.log(codebook_size)
    return (max_entropy - entropy) / max_entropy


def compute_lpips_loss(lpips_metric, x_pred: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Compute LPIPS perceptual loss between prediction and target.
    """
    B, C, F, H, W = x.shape
    to_lpips = lambda t: ((t * 2.0) - 1.0).reshape(B * F, C, H, W)
    return lpips_metric(to_lpips(x_pred), to_lpips(x)).mean()


def compute_vq_losses(
    x_pred: torch.Tensor,
    x: torch.Tensor,
    z_e: torch.Tensor,
    z_quantized: torch.Tensor,
    indices: torch.Tensor,
    *,
    beta: float,
    quantizer_type: QuantizerType,
    codebook_size: int,
    entropy_weight: float,
    lpips_weight: float,
    lpips_metric,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute reconstruction, commitment, codebook, entropy, and LPIPS losses.
    """
    recon_loss = F.mse_loss(x_pred, x)
    commit_loss = F.mse_loss(z_e, z_quantized.detach())
    is_vanilla = quantizer_type == QuantizerType.VANILLA
    codebook_loss = F.mse_loss(z_quantized, z_e.detach()) if is_vanilla else 0.0
    entropy_loss = compute_entropy_loss(indices, codebook_size) if entropy_weight else None
    lpips_loss = compute_lpips_loss(lpips_metric, x_pred, x) if lpips_weight else None

    loss = recon_loss + (beta * commit_loss) + codebook_loss
    if entropy_loss is not None:
        loss = loss + entropy_weight * entropy_loss
    if lpips_loss is not None:
        loss = loss + lpips_weight * lpips_loss

    losses = {
        "loss": loss,
        "recon_loss": recon_loss,
        "commit_loss": commit_loss,
        "codebook_loss": codebook_loss if is_vanilla else None,
        "entropy_loss": entropy_loss,
        "lpips_loss": lpips_loss,
    }
    return loss, losses
