import argparse
import json
import os
from datetime import datetime

import numpy as np

try:
    from procgen import ProcgenEnv
except ImportError as exc:
    raise SystemExit(
        "procgen is required. Install with `pip install procgen`."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Procgen Heist clips with action labels."
    )
    parser.add_argument("--env", default="heist", help="Procgen env name.")
    parser.add_argument("--clip-len", type=int, default=16, help="Frames per clip.")
    parser.add_argument(
        "--num-clips", type=int, default=10000, help="Total clips to generate."
    )
    parser.add_argument(
        "--num-envs", type=int, default=8, help="Number of parallel envs."
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=256,
        help="Clips per shard file.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/procgen_heist",
        help="Output directory for shards and metadata.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for Procgen."
    )
    parser.add_argument(
        "--num-levels",
        type=int,
        default=0,
        help="Number of levels to sample (0 = unlimited).",
    )
    parser.add_argument(
        "--start-level",
        type=int,
        default=0,
        help="Start level offset (use different ranges for train/eval).",
    )
    parser.add_argument(
        "--distribution-mode",
        default="easy",
        choices=["easy", "hard", "exploration", "memory", "extreme"],
        help="Procgen distribution mode.",
    )
    parser.add_argument(
        "--repeat-action-prob",
        type=float,
        default=0.0,
        help="Probability of repeating the previous action (0 = random).",
    )
    return parser.parse_args()


def sample_actions(action_n: int, last_actions: np.ndarray, p_repeat: float) -> np.ndarray:
    if p_repeat <= 0.0:
        return np.random.randint(0, action_n, size=last_actions.shape[0])
    random_actions = np.random.randint(0, action_n, size=last_actions.shape[0])
    repeat_mask = np.random.rand(last_actions.shape[0]) < p_repeat
    actions = np.where(repeat_mask, last_actions, random_actions)
    return actions


def write_shard(
    shard_dir: str,
    shard_idx: int,
    frames: list[np.ndarray],
    actions: list[np.ndarray],
) -> None:
    shard_path = os.path.join(shard_dir, f"shard_{shard_idx:05d}.npz")
    frames_arr = np.stack(frames, axis=0).astype(np.uint8)
    actions_arr = np.stack(actions, axis=0).astype(np.int64)
    np.savez_compressed(shard_path, frames=frames_arr, actions=actions_arr)


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    shard_dir = os.path.join(args.out_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)

    env = ProcgenEnv(
        num_envs=args.num_envs,
        env_name=args.env,
        num_levels=args.num_levels,
        start_level=args.start_level,
        distribution_mode=args.distribution_mode,
        rand_seed=args.seed,
    )
    obs = env.reset()
    action_n = env.action_space.n

    buffers_frames = [[] for _ in range(args.num_envs)]
    buffers_actions = [[] for _ in range(args.num_envs)]
    last_actions = np.zeros(args.num_envs, dtype=np.int64)

    shard_frames: list[np.ndarray] = []
    shard_actions: list[np.ndarray] = []
    shard_idx = 0
    total_clips = 0

    while total_clips < args.num_clips:
        actions = sample_actions(action_n, last_actions, args.repeat_action_prob)
        last_actions = actions
        obs, _, dones, _ = env.step(actions)

        for env_idx in range(args.num_envs):
            if dones[env_idx]:
                buffers_frames[env_idx].clear()
                buffers_actions[env_idx].clear()
                continue

            frame = obs[env_idx]
            buffers_frames[env_idx].append(frame)
            buffers_actions[env_idx].append(actions[env_idx])

            if len(buffers_frames[env_idx]) == args.clip_len:
                clip = np.stack(buffers_frames[env_idx], axis=0)  # [F, H, W, C]
                clip = clip.transpose(3, 0, 1, 2)  # [C, F, H, W]
                shard_frames.append(clip)
                shard_actions.append(np.array(buffers_actions[env_idx], dtype=np.int64))
                buffers_frames[env_idx].clear()
                buffers_actions[env_idx].clear()
                total_clips += 1

                if len(shard_frames) >= args.shard_size or total_clips == args.num_clips:
                    write_shard(shard_dir, shard_idx, shard_frames, shard_actions)
                    shard_idx += 1
                    shard_frames.clear()
                    shard_actions.clear()

        if total_clips and total_clips % 1000 == 0:
            print(f"Generated {total_clips}/{args.num_clips} clips")

    meta = {
        "env": args.env,
        "clip_len": args.clip_len,
        "resolution": list(obs.shape[1:3]),
        "channels": int(obs.shape[-1]),
        "num_clips": args.num_clips,
        "num_envs": args.num_envs,
        "shard_size": args.shard_size,
        "num_levels": args.num_levels,
        "start_level": args.start_level,
        "distribution_mode": args.distribution_mode,
        "repeat_action_prob": args.repeat_action_prob,
        "seed": args.seed,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
