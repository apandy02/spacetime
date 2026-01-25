import logging
import sys
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from lightning.pytorch.loggers import CSVLogger, WandbLogger

from spacetime.utils.wandb import is_rank_zero


def get_logger(name: str = "spacetime", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def setup_wandb_csv_loggers(
    *,
    project: str,
    config_parts: Iterable[Any],
    effective_batch_size: int | None = None,
    extra_config: dict[str, Any] | None = None,
    csv_flush_logs_every_n_steps: int = 1,
) -> list | bool:
    """
    Configure WandB and CSV loggers for rank-zero process.
    """
    if not is_rank_zero():
        return False

    merged_config: dict[str, Any] = {}
    for part in config_parts:
        merged_config.update(_serialize_wandb_config(_as_dict(part)))
    if extra_config:
        merged_config.update(_serialize_wandb_config(extra_config))
    if effective_batch_size is not None:
        merged_config["effective_batch_size"] = effective_batch_size

    wandb_logger = WandbLogger(project=project, config=merged_config)
    run_id = wandb_logger.experiment.id
    csv_logger = CSVLogger(
        save_dir="lightning_logs",
        name=run_id,
        flush_logs_every_n_steps=csv_flush_logs_every_n_steps,
    )
    return [csv_logger, wandb_logger]


def _as_dict(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return value
    raise TypeError(f"Unsupported config type for wandb logging: {type(value)}")


def _serialize_wandb_config(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _serialize_wandb_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_wandb_config(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_serialize_wandb_config(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    return value
