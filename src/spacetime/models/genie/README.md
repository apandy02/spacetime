# Genie: Generative Interactive Environments

Reproduction of [Genie (Bruce et al., 2024)](https://arxiv.org/abs/2402.15391) — a world model that learns latent actions from unlabelled video and uses them to drive autoregressive frame generation.

## Architecture

The model has three components, combined in `GenieModel` (`model.py`):

| Component | File | Description |
|---|---|---|
| **VQ-VAE Tokenizer** | `spacetime.models.tokenizer` | Pretrained; encodes frames into discrete token grids. Frozen during Genie training. |
| **Latent Action Model (LAM)** | `latent_actions.py` | VAE with a VQ bottleneck. Anti-causal ST-transformer encodes consecutive frames into per-timestep discrete action codes. |
| **Dynamics Model** | `dynamics.py` | MaskGIT-style causal ST-transformer. Predicts next-frame token logits conditioned on past tokens + latent actions. |

## Training

`train.py` / `training_module.py` — joint training of LAM + dynamics with a frozen tokenizer. Losses: LAM reconstruction, VQ commitment, dynamics cross-entropy, and LPIPS.

```
uv run python -m spacetime.models.genie.train --help
```

## Inference

`inference.py` — evaluation script that runs two modes:

- **Single-step reconstruction**: one-step dynamics prediction vs ground truth.
- **Multi-step rollout**: autoregressive generation from context frames using LAM-inferred actions.

```
uv run python -m spacetime.models.genie.inference \
    --checkpoint lightning_logs/<run_id>/version_0/checkpoints/<ckpt>.ckpt \
    --tokenizer_checkpoint lightning_logs/<tok_id>/version_0/checkpoints/<ckpt>.ckpt \
    --tokenizer_wandb_path wandb/run-<date>-<tok_id>/files/config.yaml \
    --output_dir outputs/genie_eval/<run_id>
```

## Config

`config.py` — dataclass hierarchy (`LamConfig`, `DynamicsConfig`, `TrainingConfig`, `Config`) configuring architecture dimensions, loss weights, and training hyperparameters.
