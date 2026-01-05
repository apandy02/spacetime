from enum import Enum

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class QuantizerType(Enum):
    """Quantizer type enum"""

    VANILLA = "vanilla"
    EMA = "ema"


class EMAVectorQuantizer(nn.Module):
    """
    EMA vector quantizer
    """

    def __init__(
        self,
        codebook_size: int,
        codebook_dim: int,
        decay: float = 0.95,
        eps: float = 1e-5,
        dead_code_threshold: float = 0.01,
        dead_code_noise: float = 1e-4,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.decay = decay
        self.eps = eps
        self.dead_code_threshold = dead_code_threshold
        self.dead_code_noise = dead_code_noise

        embed = torch.randn(codebook_size, codebook_dim)
        self.register_buffer("codebook", embed)
        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_codebook", embed.clone())

    def forward(
        self, z_e: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the EMAVectorQuantizer.

        Args:
            z_e: encoded representation of shape [batch_size, frames, n_patches, codebook_dim]

        Returns:
            z_q: quantized representation of shape [batch_size, frames, n_patches, codebook_dim]
            indices: codebook indices of shape [batch_size, frames, n_patches]
            z_e_normalized: L2-normalized encoder output (same shape as z_e)
        """
        z_e_flat = z_e.reshape(-1, self.codebook_dim)

        z_e_normalized = F.normalize(z_e_flat, dim=-1)
        codebook_normalized = F.normalize(self.codebook.to(z_e.dtype), dim=-1)

        dist = 2 - 2 * (z_e_normalized @ codebook_normalized.t())

        indices = dist.argmin(dim=1)
        z_q_flat = codebook_normalized[indices]
        z_q = z_q_flat.view_as(z_e)
        z_e_norm_out = z_e_normalized.view_as(z_e)

        if self.training:
            with torch.no_grad():
                cluster_size = torch.bincount(
                    indices, minlength=self.codebook_size
                ).to(z_e_normalized.dtype)

                codebook_sum = torch.zeros(
                    self.codebook_size,
                    self.codebook_dim,
                    device=z_e_normalized.device,
                    dtype=z_e_normalized.dtype,
                )
                codebook_sum.index_add_(0, indices, z_e_normalized)
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(cluster_size)
                    dist.all_reduce(codebook_sum)
                self.ema_cluster_size.mul_(self.decay).add_(
                    cluster_size * (1 - self.decay)
                )
                self.ema_codebook.mul_(self.decay).add_(codebook_sum * (1 - self.decay))

                n = self.ema_cluster_size.sum()
                cluster_size = (
                    (self.ema_cluster_size + self.eps)
                    / (n + self.codebook_size * self.eps)
                    * n
                )

                cluster_size = torch.clamp(cluster_size, min=self.eps)
                self.codebook.copy_(self.ema_codebook / cluster_size.unsqueeze(1))
                if self.dead_code_threshold > 0:
                    self._refresh_dead_codes(cluster_size, n, z_e_normalized)

        return z_q, indices, z_e_norm_out

    def _refresh_dead_codes(
        self,
        cluster_size: torch.Tensor,
        total: torch.Tensor,
        z_e_flat: torch.Tensor,
    ) -> None:
        """Reinitialize dead codes from active ones to prevent codebook collapse."""
        if total.item() <= 0:
            return
        avg_cluster = total / self.codebook_size
        dead = cluster_size < (self.dead_code_threshold * avg_cluster)
        if not dead.any():
            return
        if z_e_flat.numel() == 0:
            return
        num_dead = int(dead.sum().item())
        z_e_flat = z_e_flat.reshape(-1, self.codebook_dim)
        if z_e_flat.shape[0] < num_dead:
            rand_idx = torch.randint(
                0, z_e_flat.shape[0], (num_dead,), device=z_e_flat.device
            )
        else:
            rand_idx = torch.randperm(z_e_flat.shape[0], device=z_e_flat.device)[
                :num_dead
            ]
        new_codes = z_e_flat[rand_idx].clone()
        if self.dead_code_noise > 0:
            new_codes.add_(torch.randn_like(new_codes) * self.dead_code_noise)
        self.ema_cluster_size[dead] = avg_cluster
        self.ema_codebook[dead] = new_codes * avg_cluster
        self.codebook[dead] = new_codes


class VanillaVectorQuantizer(nn.Module):
    """
    Quantize encoded tensor by snapping its elements to the closest codebook entry.
    Quantizes each patch independently. Uses L2 normalization for stable training.

    Args:
        codebook_size: number of codebook entries
        codebook_dim: dimensionality of codebook vectors

    Usage:
        quantizer = VanillaVectorQuantizer(codebook_size, codebook_dim)
        z_q, indices, z_e_normalized = quantizer(z_e)
    """

    def __init__(self, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.codebook = nn.Parameter(
            torch.randn(codebook_size, codebook_dim) * 0.02, requires_grad=True
        )

    def forward(
        self, z_e: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z_e: encoded representation of shape [batch_size, frames, n_patches, codebook_dim]
        Returns:
            z_q: quantized representation [batch_size, frames, n_patches, codebook_dim]
            indices: indices of closest codebook entries [batch_size, frames, n_patches]
            z_e_normalized: L2-normalized encoder output (same shape as z_e)
        """
        batch_size, frames, n_patches, codebook_dim = z_e.shape
        assert codebook_dim == self.codebook_dim

        encoded = z_e.reshape(-1, codebook_dim)

        # L2 normalize encoder outputs and codebook for stable quantization
        z_e_normalized = F.normalize(encoded, dim=-1)
        codebook_normalized = F.normalize(self.codebook, dim=-1)

        # Distance calculation using normalized vectors (equivalent to cosine distance)
        dist = 2 - 2 * (z_e_normalized @ codebook_normalized.t())
        indices = dist.argmin(dim=1)
        z_q = codebook_normalized[indices].reshape(
            batch_size, frames, n_patches, codebook_dim
        )
        z_e_norm_out = z_e_normalized.reshape(
            batch_size, frames, n_patches, codebook_dim
        )
        indices = indices.view(batch_size, frames, n_patches)
        return z_q, indices, z_e_norm_out
