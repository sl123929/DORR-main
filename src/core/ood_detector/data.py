"""Fixed state/item detector data built from ROLeR's replay trajectory format."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from tqdm.auto import tqdm


def representation_id(item_embeddings: torch.Tensor) -> str:
    values = item_embeddings.detach().cpu().contiguous().float().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


@dataclass
class DetectorDataset:
    states: torch.Tensor
    item_ids: torch.Tensor
    item_embeddings: torch.Tensor
    representation_version: str
    # Optional for compatibility with detector artifacts exported before DORR.
    # DORR uses this only to calibrate kNN-uncertainty on a held-out ID split.
    user_ids: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        self.states = self.states.detach().cpu().float().contiguous()
        self.item_ids = self.item_ids.detach().cpu().long().reshape(-1).contiguous()
        self.item_embeddings = self.item_embeddings.detach().cpu().float().contiguous()
        if self.states.ndim != 2 or self.item_embeddings.ndim != 2 or len(self.states) != len(self.item_ids):
            raise ValueError("states, item IDs, and item embedding table have incompatible shapes")
        if len(self.states) == 0:
            raise ValueError("detector dataset is empty")
        if self.item_ids.min() < 0 or self.item_ids.max() >= len(self.item_embeddings):
            raise ValueError("item IDs must index the frozen item embedding table")
        if not torch.isfinite(self.states).all() or not torch.isfinite(self.item_embeddings).all():
            raise ValueError("detector representations contain NaN or Inf")
        if self.user_ids is not None:
            self.user_ids = self.user_ids.detach().cpu().long().reshape(-1).contiguous()
            if len(self.user_ids) != len(self.states) or (self.user_ids < 0).any():
                raise ValueError("user IDs must be non-negative and aligned with states")

    @property
    def state_dim(self) -> int:
        return int(self.states.shape[1])

    @property
    def item_dim(self) -> int:
        return int(self.item_embeddings.shape[1])

    def split(self, calibration_fraction: float = 0.2, seed: int = 0):
        if not 0 < calibration_fraction < 1 or len(self.states) < 3:
            raise ValueError("need at least three samples and a calibration fraction in (0, 1)")
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(self.states), generator=generator)
        calibration_count = max(1, min(len(order) - 1, round(len(order) * calibration_fraction)))
        return self.select(order[calibration_count:]), self.select(order[:calibration_count])

    def select(self, indices: torch.Tensor) -> "DetectorDataset":
        return DetectorDataset(self.states[indices], self.item_ids[indices], self.item_embeddings,
                               self.representation_version,
                               None if self.user_ids is None else self.user_ids[indices])

    def save(self, path: Union[str, Path]) -> None:
        payload = {
            "states": self.states.numpy(),
            "item_ids": self.item_ids.numpy(),
            "item_embeddings": self.item_embeddings.numpy(),
            "representation_version": np.asarray(self.representation_version),
        }
        if self.user_ids is not None:
            payload["user_ids"] = self.user_ids.numpy()
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DetectorDataset":
        with np.load(path, allow_pickle=False) as data:
            version = str(data["representation_version"].item())
            return cls(torch.from_numpy(data["states"]), torch.from_numpy(data["item_ids"]),
                       torch.from_numpy(data["item_embeddings"]), version,
                       torch.from_numpy(data["user_ids"]) if "user_ids" in data.files else None)


@torch.no_grad()
def build_from_roler_buffer(buffer, state_tracker, batch_size: int = 1024,
                            representation_version: Optional[str] = None,
                            show_progress: bool = False) -> DetectorDataset:
    """Extract frozen state/action pairs from a ROLeR `VectorReplayBuffer`.

    The tracker is evaluated exactly as for replay observations.  This function does
    not mutate the buffer, the tracker, or the RL policy.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not hasattr(state_tracker, "embedding_dict") or "feat_item" not in state_tracker.embedding_dict:
        raise ValueError("state tracker does not expose a feat_item embedding table")
    item_table = state_tracker.embedding_dict.feat_item.weight.detach().cpu()
    catalog_size = int(getattr(state_tracker, "num_item", len(item_table) - 1))
    if catalog_size <= 0 or catalog_size > len(item_table):
        raise ValueError("invalid ROLeR item catalog size")
    item_table = item_table[:catalog_size].clone()
    indices = np.asarray(buffer.sample_indices(0), dtype=np.int64)
    if not len(indices):
        raise ValueError("ROLeR replay buffer is empty")
    was_training = state_tracker.training
    state_tracker.eval()
    states, action_ids, user_ids = [], [], []
    try:
        starts = range(0, len(indices), batch_size)
        if show_progress:
            starts = tqdm(starts, total=(len(indices) + batch_size - 1) // batch_size,
                          desc="Exporting tracker states", unit="batch")
        for start in starts:
            chunk = indices[start:start + batch_size]
            # is_obs=True constructs the state aligned with each recorded observation.
            extracted = state_tracker.forward(buffer=buffer, indices=chunk, reset=False, is_obs=True, is_train=True)
            extracted = extracted.detach().cpu().float()
            actions = torch.as_tensor(np.asarray(buffer[chunk].act), dtype=torch.long).reshape(-1)
            valid = (actions >= 0) & (actions < catalog_size)
            if valid.any():
                states.append(extracted[valid])
                action_ids.append(actions[valid])
                users = torch.as_tensor(np.asarray(buffer[chunk].obs)[:, 0], dtype=torch.long).reshape(-1)
                user_ids.append(users[valid])
    finally:
        state_tracker.train(was_training)
    if not states:
        raise ValueError("no valid catalog item IDs were found in the replay buffer")
    version = representation_version or representation_id(item_table)
    return DetectorDataset(torch.cat(states), torch.cat(action_ids), item_table, version,
                           torch.cat(user_ids))
