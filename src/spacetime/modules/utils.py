"""
Utility functions for transformer and video processing operations.
"""

import torch


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Splits a batch of videos into non-overlapping patches.

    Args:
        x (torch.Tensor): Input tensor of shape [B, C, F, H, W]
        patch_size (int): Size of each patch (height and width)

    Returns:
        torch.Tensor: Patchified tensor of shape [B, F, NUM_P, DIM_P],
                    where DIM_P = channels * patch_size * patch_size.
    """
    batch_size, channels, frames, height, width = x.shape
    n_patch_h = height // patch_size
    n_patch_w = width // patch_size
    x = x.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
    x = x.reshape(
        batch_size,
        frames,
        channels,
        n_patch_h,
        patch_size,
        n_patch_w,
        patch_size,
    )
    x = x.permute(0, 1, 3, 5, 2, 4, 6)  # [B, F, n_patch_h, n_patch_w, C, p_h, p_w]
    return x.reshape(
        batch_size,
        frames,
        n_patch_h * n_patch_w,
        channels * patch_size * patch_size,
    )


def unpatchify(x: torch.Tensor, patch_size: int, n_patch_h: int, n_patch_w: int) -> torch.Tensor:
    """
    Unpatches a batch of videos from patches back to full frames.
    Reverses the patchification done by patchify.

    Args:
        x (torch.Tensor): Input tensor of shape [B, F, NUM_P, DIM_P]
        patch_size (int): Size of each patch (height and width)
        n_patch_h (int): Number of patches along height
        n_patch_w (int): Number of patches along width

    Returns:
        torch.Tensor: Unpatchified tensor of shape [B, C, F, H, W]
    """
    batch_size, frames, _, patch_dim = x.shape
    channels = patch_dim // (patch_size * patch_size)

    # Reshape to separate patch dimensions
    x = x.reshape(
        batch_size,
        frames,
        n_patch_h,
        n_patch_w,
        channels,
        patch_size,
        patch_size,
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6)  # [B, F, C, n_patch_h, p_h, n_patch_w, p_w]
    # Merge patches
    x = x.reshape(
        batch_size,
        frames,
        channels,
        n_patch_h * patch_size,
        n_patch_w * patch_size,
    )
    # Final permute to get [B, C, F, H, W]
    return x.permute(0, 2, 1, 3, 4)


def build_causal_mask(seq_len: int, device=None, dtype=None) -> torch.Tensor:
    """
    Causal mask: each position can only attend to itself and earlier positions.
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    return ~mask


def build_anti_causal_mask(seq_len: int, device=None, dtype=None) -> torch.Tensor:
    """
    Anti causal mask for the transformer that allows attention to one token in the future.
    """
    tril = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    superdiag = torch.diag(torch.ones(seq_len - 1, device=device, dtype=torch.bool), diagonal=1)
    mask = tril | superdiag
    return ~mask
