from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from spacetime.models.tokenizer.config import Hyperparameters
from spacetime.models.tokenizer.model import VQTokenizer


def load_pretrained_tokenizer_from_checkpoint(
    checkpoint_path: str | Path,
    wandb_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[VQTokenizer, Hyperparameters]:
    cfg = load_pretrained_tokenizer_config(wandb_path)
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_state = {
        key.replace("model.", "", 1): value
        for key, value in state_dict.items()
        if key.startswith("model.")
    }
    model = VQTokenizer(cfg)
    model.load_state_dict(model_state, strict=True)
    model.eval()
    return model, cfg


def load_pretrained_tokenizer_config(wandb_path: str | Path) -> Hyperparameters:
    config_path = _resolve_wandb_config_path(wandb_path)
    cfg_dict = _load_wandb_config(config_path)
    model_cfg = cfg_dict.get("model", {})
    quant_cfg = cfg_dict.get("quantizer", {})
    beta_cfg = cfg_dict.get("beta", {})
    optimizer_cfg = cfg_dict.get("optimizer", {})
    loss_cfg = cfg_dict.get("loss", {})

    cfg = Hyperparameters()
    for key, value in model_cfg.items():
        setattr(cfg.model, key, value)
    for key, value in quant_cfg.items():
        setattr(cfg.quantizer, key, value)
    for key, value in beta_cfg.items():
        setattr(cfg.beta, key, value)
    for key, value in optimizer_cfg.items():
        setattr(cfg.optimizer, key, value)
    for key, value in loss_cfg.items():
        setattr(cfg.loss, key, value)
    return cfg


def _unwrap_wandb_values(data: Any) -> Any:
    if isinstance(data, dict):
        if set(data.keys()) == {"value"}:
            return _unwrap_wandb_values(data["value"])
        return {key: _unwrap_wandb_values(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_unwrap_wandb_values(item) for item in data]
    return data


def _load_wandb_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Unexpected wandb config format in {config_path}")
    raw.pop("_wandb", None)
    return _unwrap_wandb_values(raw)


def _resolve_wandb_config_path(wandb_path: str | Path) -> Path:
    path = Path(wandb_path)
    if path.is_dir():
        candidate = path / "files" / "config.yaml"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"No config.yaml found under {path}/files")
    if path.is_file():
        return path
    raise FileNotFoundError(f"Wandb path not found: {path}")
