import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch


@dataclass
class ParamGroupConfig:
    """
    Configuration for a single parameter group in the optimizer.
    """

    params: Iterable[torch.nn.Parameter]
    lr: float
    warmup_steps: int = 0
    min_lr_ratio: float = 0.0


def build_optimizer_with_schedule(
    param_groups: list[ParamGroupConfig],
    total_steps: int | None,
    betas: tuple[float, float] = (0.9, 0.999),
    weight_decay: float = 0.0,
) -> dict | torch.optim.AdamW:
    """
    Build AdamW optimizer with per-group warmup + cosine decay schedules.

    Args:
        param_groups: List of parameter group configurations.
        total_steps: Total training steps for scheduler. If None, returns optimizer only.
        betas: AdamW beta coefficients.
        weight_decay: AdamW weight decay.

    Returns:
        Lightning-compatible dict with optimizer + scheduler, or just optimizer if total_steps is None.
    """
    if not param_groups:
        raise ValueError("Must provide at least one parameter group")

    optimizer_param_groups = [{"params": list(cfg.params), "lr": cfg.lr} for cfg in param_groups]

    optimizer = torch.optim.AdamW(
        optimizer_param_groups,
        betas=betas,
        weight_decay=weight_decay,
    )

    if total_steps is None or total_steps <= 0:
        return optimizer

    lr_lambdas = [_make_lr_lambda(cfg, total_steps) for cfg in param_groups]

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambdas)

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        },
    }


def _make_lr_lambda(cfg: ParamGroupConfig, total_steps: int):
    """
    Create learning rate schedule function for a single parameter group.

    Schedule: warmup + cosine decay

    Args:
        cfg: Parameter group configuration.
        total_steps: Total training steps.

    Returns:
        Lambda function that maps step -> lr_multiplier.
    """

    def lr_lambda(step: int) -> float:
        if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps

        decay_steps = total_steps - cfg.warmup_steps
        if decay_steps <= 0:
            return 1.0

        progress = (step - cfg.warmup_steps) / decay_steps
        progress = min(1.0, progress)  # Clamp to [0, 1]
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

        return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine

    return lr_lambda
