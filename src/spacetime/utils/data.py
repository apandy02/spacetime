import bisect
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class ProcgenShardDataset(Dataset):
    """
    Dataset for procgen shard .npz files with frames/actions.
    """

    def __init__(
        self,
        shard_dir: str | Path,
        return_actions: bool = True,
        normalize: bool = False,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.shard_dir = Path(shard_dir)
        self.return_actions = return_actions
        self.normalize = normalize
        self.transform = transform

        self.shards = sorted(self.shard_dir.glob("shard_*.npz"))
        if not self.shards:
            raise FileNotFoundError(f"No shard_*.npz files found in {self.shard_dir}")

        self._counts = []
        self._offsets = [0]
        for shard in self.shards:
            with np.load(shard) as data:
                n = data["frames"].shape[0]
            self._counts.append(n)
            self._offsets.append(self._offsets[-1] + n)

        # very basic last opened shard caching -- TODO: improve by using LRU cache or switching to a datapipe
        self._cached_shard_idx = None
        self._cached_frames = None
        self._cached_actions = None

    def __len__(self) -> int:
        return self._offsets[-1]

    def __getitem__(self, idx: int):
        if idx < 0:
            idx = self.__len__() + idx
        shard_idx = bisect.bisect_right(self._offsets, idx) - 1
        if shard_idx < 0 or shard_idx >= len(self.shards):
            raise IndexError("Index out of range")

        local_idx = idx - self._offsets[shard_idx]
        shard_path = self.shards[shard_idx]
        if self._cached_shard_idx != shard_idx:
            with np.load(shard_path) as data:
                self._cached_frames = data["frames"]
                self._cached_actions = data["actions"] if "actions" in data else None
            self._cached_shard_idx = shard_idx

        frames = self._cached_frames[local_idx]
        actions = (
            self._cached_actions[local_idx]
            if self._cached_actions is not None
            else None
        )

        frames = torch.from_numpy(frames)
        if self.normalize:
            frames = frames.float() / 255.0
        if self.transform is not None:
            frames = self.transform(frames)

        if self.return_actions:
            if actions is None:
                raise KeyError(f"No actions found in {shard_path}")
            actions = torch.from_numpy(actions)
            return frames, actions
        return frames
