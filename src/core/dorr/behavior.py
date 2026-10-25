"""Frozen behavior-cloning policy used only for DORR reference candidates."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

import torch
from torch import nn
import torch.nn.functional as F


class BehaviorCloningPolicy(nn.Module):
    """Independent categorical policy trained on frozen ``(state, item_id)`` data."""

    format_version = 1

    def __init__(self, state_dim: int, catalog_size: int, hidden_sizes: Iterable[int] = (256, 256)) -> None:
        super().__init__()
        if state_dim <= 0 or catalog_size <= 1:
            raise ValueError("state_dim must be positive and catalog_size must exceed one")
        self.state_dim = int(state_dim)
        self.catalog_size = int(catalog_size)
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        layers: list[nn.Module] = []
        input_dim = self.state_dim
        for hidden_dim in self.hidden_sizes:
            if hidden_dim <= 0:
                raise ValueError("behavior hidden sizes must be positive")
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, self.catalog_size))
        self.network = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        states = torch.as_tensor(states, device=next(self.parameters()).device, dtype=torch.float32)
        if states.ndim != 2 or states.shape[1] != self.state_dim:
            raise ValueError(f"expected states [B, {self.state_dim}], got {tuple(states.shape)}")
        return self.network(states)

    def loss(self, states: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        item_ids = torch.as_tensor(item_ids, device=next(self.parameters()).device, dtype=torch.long).reshape(-1)
        if len(item_ids) != len(states) or (item_ids < 0).any() or (item_ids >= self.catalog_size).any():
            raise ValueError("behavior labels must be one legal item ID per state")
        return F.cross_entropy(self(states), item_ids)

    @torch.no_grad()
    def topk(self, states: torch.Tensor, count: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
        if count <= 0:
            raise ValueError("behavior candidate count must be positive")
        logits = self(states)
        count = min(int(count), self.catalog_size)
        values, item_ids = torch.topk(logits, count, dim=-1)
        return item_ids, values

    def checkpoint_payload(self, metadata: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "model": {
                "state_dim": self.state_dim,
                "catalog_size": self.catalog_size,
                "hidden_sizes": self.hidden_sizes,
            },
            "metadata": dict(metadata or {}),
            "state_dict": self.state_dict(),
        }

    def save(self, path: str | Path, metadata: Optional[dict[str, Any]] = None) -> None:
        torch.save(self.checkpoint_payload(metadata), Path(path))

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu",
             expected_metadata: Optional[dict[str, Any]] = None) -> "BehaviorCloningPolicy":
        payload = torch.load(Path(path), map_location=map_location)
        if payload.get("format_version") != cls.format_version:
            raise ValueError("unsupported behavior policy checkpoint format")
        metadata = payload.get("metadata", {})
        if expected_metadata:
            mismatch = {key: (expected_metadata[key], metadata.get(key)) for key in expected_metadata
                        if expected_metadata[key] != metadata.get(key)}
            if mismatch:
                raise ValueError(f"behavior policy metadata mismatch: {mismatch}")
        policy = cls(**payload["model"])
        policy.load_state_dict(payload["state_dict"])
        return policy


def train_behavior_policy(states: torch.Tensor, item_ids: torch.Tensor, state_dim: int, catalog_size: int,
                          config: Optional[dict[str, Any]] = None) -> tuple[BehaviorCloningPolicy, list[float]]:
    """Train behavior cloning from training-only frozen detector representations."""
    config = dict(config or {})
    device = torch.device(config.get("device", "cpu"))
    torch.manual_seed(int(config.get("seed", 0)))
    states = torch.as_tensor(states, dtype=torch.float32)
    item_ids = torch.as_tensor(item_ids, dtype=torch.long).reshape(-1)
    if len(states) == 0 or len(states) != len(item_ids):
        raise ValueError("behavior data requires non-empty aligned states and item IDs")
    model = BehaviorCloningPolicy(state_dim, catalog_size, config.get("hidden_sizes", (256, 256))).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 3e-4)))
    epochs = int(config.get("epochs", 20))
    batch_size = min(max(1, int(config.get("batch_size", 1024))), len(states))
    if epochs <= 0:
        raise ValueError("behavior policy epochs must be positive")
    losses: list[float] = []
    for epoch in range(epochs):
        for order in torch.randperm(len(states)).split(batch_size):
            optimizer.zero_grad()
            loss = model.loss(states[order].to(device), item_ids[order].to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        callback = config.get("epoch_callback")
        if callback is not None:
            callback(epoch + 1, epochs, losses[-1])
    return model, losses
