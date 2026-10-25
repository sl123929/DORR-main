"""Action-conditioned Q critic and target-network helpers for DORR."""

from __future__ import annotations

import copy
from typing import Iterable

import numpy as np
import torch
from torch import nn


class QCritic(nn.Module):
    """``Q(s, item_id)`` with an independent trainable action embedding table."""

    def __init__(self, state_dim: int, catalog_size: int, action_embed_dim: int = 64,
                 hidden_sizes: Iterable[int] = (256, 256)) -> None:
        super().__init__()
        if state_dim <= 0 or catalog_size <= 1 or action_embed_dim <= 0:
            raise ValueError("invalid Q critic dimensions")
        self.state_dim = int(state_dim)
        self.catalog_size = int(catalog_size)
        self.action_embed_dim = int(action_embed_dim)
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        self.action_embedding = nn.Embedding(self.catalog_size, self.action_embed_dim)
        layers: list[nn.Module] = []
        input_dim = self.state_dim + self.action_embed_dim
        for hidden_dim in self.hidden_sizes:
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        device = self.action_embedding.weight.device
        states = torch.as_tensor(states, device=device, dtype=torch.float32)
        item_ids = torch.as_tensor(item_ids, device=device, dtype=torch.long).reshape(-1)
        if states.ndim != 2 or states.shape[1] != self.state_dim or len(states) != len(item_ids):
            raise ValueError(f"expected aligned states [B, {self.state_dim}] and item IDs")
        if item_ids.numel() and ((item_ids < 0).any() or (item_ids >= self.catalog_size).any()):
            raise ValueError("Q critic item IDs are outside the catalog")
        return self.network(torch.cat([states, self.action_embedding(item_ids)], dim=-1)).squeeze(-1)

    def config(self) -> dict[str, object]:
        return {
            "state_dim": self.state_dim,
            "catalog_size": self.catalog_size,
            "action_embed_dim": self.action_embed_dim,
            "hidden_sizes": self.hidden_sizes,
        }


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    if not 0 < float(tau) <= 1:
        raise ValueError("target soft-update tau must be in (0, 1]")
    for target_parameter, source_parameter in zip(target.parameters(), source.parameters()):
        target_parameter.mul_(1.0 - tau).add_(source_parameter, alpha=tau)


def frozen_copy(module: nn.Module) -> nn.Module:
    target = copy.deepcopy(module)
    target.eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    return target


def discounted_returns(rewards, dones, gamma: float) -> np.ndarray:
    """Return every within-trajectory discounted return without crossing terminals."""
    if not 0 < gamma <= 1:
        raise ValueError("gamma must be in (0, 1]")
    rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
    dones = np.asarray(dones, dtype=bool).reshape(-1)
    if len(rewards) == 0 or len(rewards) != len(dones) or not np.isfinite(rewards).all():
        raise ValueError("finite aligned training rewards and dones are required")
    values = np.empty_like(rewards)
    accumulated = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        if dones[index]:
            accumulated = 0.0
        accumulated = rewards[index] + gamma * accumulated
        values[index] = accumulated
    return values


def discounted_return_quantile(rewards, dones, gamma: float, percentile: float = 1.0) -> float:
    """Deterministic discounted-return quantile from training trajectories only."""
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be in [0, 100]")
    values = discounted_returns(rewards, dones, gamma)
    return float(np.percentile(values, percentile))
