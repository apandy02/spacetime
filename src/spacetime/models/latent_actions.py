import torch

def build_anti_causal_mask(seq_len: int, device=None, dtype=None) -> torch.Tensor:
    tril = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=dtype))
    superdiag = torch.diag(torch.ones(seq_len - 1, device=device, dtype=dtype), diagonal=1)
    mask = tril + superdiag
    return mask.clamp(max=1)

