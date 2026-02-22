"""Genie inference: single-step reconstruction and multi-step autoregressive rollout."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.utils import make_grid

from spacetime.models.genie.config import Config
from spacetime.models.genie.training_module import GenieTrainingModule
from spacetime.utils import get_logger
from spacetime.utils.data import ProcgenShardDataset

logger = get_logger("spacetime.genie.inference")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model from %s", args.checkpoint)
    module = load_model(
        args.checkpoint,
        args.tokenizer_checkpoint,
        args.tokenizer_wandb_path,
        args.device,
    )

    logger.info("Loading validation data from %s", args.shard_dir)
    val_loader = get_val_dataloader(Path(args.shard_dir), batch_size=args.batch_size)
    batch = _load_eval_batch(val_loader, args.num_samples)

    if args.mode == "single-step":
        _run_single_step(module, batch, args, output_dir)
    else:
        _run_multi_step(module, batch, args, output_dir)

    logger.info("Results saved to %s", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genie inference and evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Genie checkpoint")
    parser.add_argument(
        "--tokenizer_checkpoint", type=str, required=True, help="Path to tokenizer checkpoint"
    )
    parser.add_argument(
        "--tokenizer_wandb_path",
        type=str,
        required=True,
        help="Path to tokenizer wandb config.yaml",
    )
    parser.add_argument(
        "--shard_dir", type=str, default="data/procgen_heist/shards", help="Path to data shards"
    )
    parser.add_argument("--mode", type=str, choices=["single-step", "multi-step"], required=True)
    parser.add_argument(
        "--output_dir", type=str, default="outputs/genie_eval", help="Output directory"
    )
    parser.add_argument("--batch_size", type=int, default=48, help="Batch size for data loading")
    parser.add_argument(
        "--num_samples", type=int, default=8, help="Number of samples to evaluate and save"
    )
    parser.add_argument(
        "--num_context_frames",
        type=int,
        default=1,
        help="Context frames for rollout (multi-step only)",
    )
    parser.add_argument(
        "--num_rollout_frames", type=int, default=15, help="Rollout length (multi-step only)"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    return parser.parse_args()


def load_model(
    checkpoint_path: str,
    tokenizer_checkpoint: str,
    tokenizer_wandb_path: str,
    device: str = "cuda",
) -> GenieTrainingModule:
    """Load trained Genie model from checkpoint."""
    cfg = Config(
        tokenizer_checkpoint=Path(tokenizer_checkpoint),
        tokenizer_wandb_path=Path(tokenizer_wandb_path),
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"]

    fixed_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("._orig_mod", "")
        fixed_state_dict[new_key] = value

    module = GenieTrainingModule(cfg)
    load_result = module.load_state_dict(fixed_state_dict, strict=False)
    if load_result.missing_keys:
        logger.warning("Missing checkpoint keys: %s", load_result.missing_keys)
    if load_result.unexpected_keys:
        logger.warning("Unexpected checkpoint keys: %s", load_result.unexpected_keys)
    module.eval()
    module.to(device)
    return module


def get_val_dataloader(
    shard_dir: Path,
    batch_size: int = 48,
    train_ratio: float = 0.8,
    num_workers: int = 4,
) -> DataLoader:
    dataset = ProcgenShardDataset(shard_dir, normalize=True)
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    return DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def _load_eval_batch(val_loader: DataLoader, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch = next(iter(val_loader))
    available = min(num_samples, batch[0].shape[0])
    if available < num_samples:
        logger.warning(
            "Requested %d samples but only %d are available in the first validation batch",
            num_samples,
            available,
        )
    return batch[0][:available], batch[1][:available]


def _run_single_step(module, batch, args, output_dir: Path):
    logger.info("Running single-step reconstruction (%d samples)", batch[0].shape[0])
    results = single_step_reconstruction(module, batch, args.device)

    avg_lpips = sum(results["lpips"]) / len(results["lpips"])
    avg_ce = sum(results["cross_entropy"]) / len(results["cross_entropy"])
    logger.info("Average LPIPS: %.4f", avg_lpips)
    logger.info("Average dynamics CE: %.4f", avg_ce)

    save_comparison_video(
        results["ground_truth"],
        results["predicted"],
        output_dir / "single_step_comparison.mp4",
        num_examples=args.num_samples,
    )


@torch.no_grad()
def single_step_reconstruction(
    module: GenieTrainingModule,
    batch: tuple[torch.Tensor, torch.Tensor],
    device: str = "cuda",
) -> dict:
    """
    Run single-step dynamics prediction.

    Returns dict with:
        - ground_truth: [B, C, F-1, H, W] frames 2..T
        - predicted: [B, C, F-1, H, W] dynamics predictions
        - lam_recon: [B, C, F-1, H, W] LAM reconstructions
        - lpips: per-example LPIPS scores
    """
    x = batch[0].to(device)

    genie_output = module.genie_model(x)

    ground_truth = x[:, :, 1:]
    pred_indices = genie_output.output_tokens.argmax(dim=-1)
    predicted = module.tokenizer.decode_indices(pred_indices)
    lam_recon = genie_output.lam_reconstruction
    cross_entropy = _compute_dynamics_cross_entropy(
        genie_output.output_tokens,
        genie_output.token_indices[:, 1:],
    )

    lpips_scores = []
    for i in range(x.shape[0]):
        lpips_val = 0
        for f in range(ground_truth.shape[2]):
            lpips_val += module.lpips_metric(
                ground_truth[i : i + 1, :, f], predicted[i : i + 1, :, f]
            ).item()
        lpips_scores.append(lpips_val / ground_truth.shape[2])

    return {
        "ground_truth": ground_truth,
        "predicted": predicted,
        "lam_recon": lam_recon,
        "lpips": lpips_scores,
        "cross_entropy": cross_entropy,
    }


def save_comparison_video(
    ground_truth: torch.Tensor,
    predicted: torch.Tensor,
    output_path: Path,
    num_examples: int = 48,
    fps: int = 4,
    max_columns: int = 4,
):
    """Save side-by-side GT|prediction comparison as an MP4 grid over time."""
    num_examples = min(num_examples, ground_truth.shape[0])
    if num_examples == 0:
        raise ValueError("No samples available to save comparison video")

    n_frames = min(ground_truth.shape[2], predicted.shape[2])
    comparisons = []
    for i in range(num_examples):
        gt_video = ground_truth[i, :, :n_frames].permute(1, 0, 2, 3)
        pred_video = predicted[i, :, :n_frames].permute(1, 0, 2, 3)
        comparisons.append(torch.cat([gt_video, pred_video], dim=3))

    tiled = torch.stack(comparisons, dim=1)
    nrow = min(max_columns, num_examples)
    grid_frames = []
    for t in range(n_frames):
        frame_grid = make_grid(tiled[t], nrow=nrow, padding=2, normalize=False)
        grid_frames.append(frame_grid)

    video = torch.stack(grid_frames, dim=0)
    _write_mp4(video, output_path, fps=fps)
    logger.info("Saved comparison video to %s", output_path)


def _run_multi_step(module, batch, args, output_dir: Path):
    logger.info("Running multi-step rollout (%d samples)", batch[0].shape[0])
    results = multi_step_rollout(
        module,
        batch,
        num_context_frames=args.num_context_frames,
        num_rollout_frames=args.num_rollout_frames,
        device=args.device,
    )

    logger.info("LPIPS per step: %s", [f"{v:.4f}" for v in results["lpips_per_step"]])

    for i in range(results["rollout"].shape[0]):
        save_rollout_video(
            results["ground_truth"][i : i + 1],
            results["rollout"][i : i + 1],
            output_dir / f"rollout_{i}.mp4",
        )

    save_comparison_video(
        results["ground_truth"],
        results["rollout"],
        output_dir / "rollout_comparison.mp4",
        num_examples=args.num_samples,
    )


@torch.no_grad()
def multi_step_rollout(
    module: GenieTrainingModule,
    batch: tuple[torch.Tensor, torch.Tensor],
    num_context_frames: int = 1,
    num_rollout_frames: int = 15,
    device: str = "cuda",
) -> dict:
    """
    Autoregressive multi-step rollout.

    Given first `num_context_frames`, predict next `num_rollout_frames` autoregressively.
    Uses LAM-inferred actions from ground truth video.

    Returns dict with:
        - ground_truth: [B, C, F, H, W] full ground truth
        - rollout: [B, C, F, H, W] context frames + predicted frames
        - lpips_per_step: LPIPS at each predicted timestep
    """
    x = batch[0].to(device)
    num_frames = x.shape[2]
    if num_context_frames < 1 or num_context_frames >= num_frames:
        raise ValueError(
            f"num_context_frames must be in [1, {num_frames - 1}] but got {num_context_frames}"
        )

    _, _, actions, _ = module.genie_model.lam(x)
    token_embeddings, _, _ = module.genie_model.tokenizer.tokenize(x)

    rollout_frames = [x[:, :, i] for i in range(num_context_frames)]
    current_token_embs = token_embeddings[:, :num_context_frames]
    lpips_per_step = []

    for step in range(num_rollout_frames):
        frame_idx = num_context_frames + step
        if frame_idx >= num_frames:
            break

        current_token_embs, current_actions = _apply_sliding_window(
            current_token_embs, actions, max_frames=actions.shape[1]
        )

        next_frame = _predict_next_frame(module.genie_model, current_token_embs, current_actions)
        rollout_frames.append(next_frame)

        lpips_val = module.lpips_metric(x[:, :, frame_idx], next_frame).mean().item()
        lpips_per_step.append(lpips_val)

        has_more_steps = step < num_rollout_frames - 1 and frame_idx + 1 < num_frames
        if has_more_steps:
            current_token_embs = _append_reencoded_frame(
                module.genie_model.tokenizer, current_token_embs, next_frame
            )

    rollout = torch.stack(rollout_frames, dim=2)

    return {
        "ground_truth": x[:, :, : rollout.shape[2]],
        "rollout": rollout,
        "lpips_per_step": lpips_per_step,
    }


def _apply_sliding_window(
    token_embs: torch.Tensor, actions: torch.Tensor, max_frames: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Trim token embeddings and align action codes for the same temporal prefix."""
    if token_embs.shape[1] > max_frames:
        token_embs = token_embs[:, -max_frames:]
    num_frames = token_embs.shape[1]
    if num_frames > actions.shape[1]:
        raise ValueError(
            f"Need {num_frames} action frames but only {actions.shape[1]} are available"
        )
    return token_embs, actions[:, :num_frames]


def _compute_dynamics_cross_entropy(
    output_tokens: torch.Tensor, target_indices: torch.Tensor
) -> list[float]:
    batch_size = output_tokens.shape[0]
    vocab_size = output_tokens.shape[-1]
    flat_logits = output_tokens.reshape(batch_size, -1, vocab_size)
    flat_targets = target_indices.reshape(batch_size, -1)
    per_token_ce = F.cross_entropy(flat_logits.transpose(1, 2), flat_targets, reduction="none")
    return per_token_ce.mean(dim=1).detach().cpu().tolist()


def _predict_next_frame(genie_model, token_embs, actions) -> torch.Tensor:
    """Run dynamics on the token sequence and decode the predicted next frame."""
    output_logits = genie_model.dynamics(token_embs, actions.detach())
    next_token_indices = output_logits[:, -1].argmax(dim=-1)
    return genie_model.tokenizer.decode_indices(next_token_indices.unsqueeze(1)).squeeze(2)


def _append_reencoded_frame(tokenizer, token_embs, frame) -> torch.Tensor:
    """Re-encode a predicted frame and append its tokens to the sequence."""
    next_embs, _, _ = tokenizer.tokenize(frame.unsqueeze(2))
    return torch.cat([token_embs, next_embs], dim=1)


def save_rollout_video(
    ground_truth: torch.Tensor,
    rollout: torch.Tensor,
    output_path: Path,
    fps: int = 4,
):
    """Save ground truth and rollout side-by-side as a video."""
    combined = torch.cat([ground_truth, rollout], dim=4)
    video = combined[0].permute(1, 0, 2, 3)
    _write_mp4(video, output_path, fps=fps)
    logger.info("Saved rollout video to %s", output_path)


def _write_mp4(video_tchw: torch.Tensor, output_path: Path, fps: int = 4):
    """Write [T, C, H, W] float video tensor to mp4 using imageio-ffmpeg directly."""
    import imageio_ffmpeg
    import numpy as np

    frames = (
        (video_tchw.clamp(0, 1) * 255).to(torch.uint8).cpu().permute(0, 2, 3, 1).contiguous().numpy()
    )
    height, width = frames.shape[1], frames.shape[2]
    writer = imageio_ffmpeg.write_frames(
        str(output_path),
        size=(width, height),
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        fps=fps,
        codec="libx264",
        macro_block_size=16,
    )
    writer.send(None)
    try:
        for frame in frames:
            writer.send(np.ascontiguousarray(frame))
    finally:
        writer.close()


if __name__ == "__main__":
    main()
