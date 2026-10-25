"""Feature-wise normalizers kept inside detector checkpoints."""

import torch
from torch import nn


class FeatureNormalizer(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("normalizer dimension must be positive")
        self.dim = int(dim)
        self.eps = float(eps)
        self.register_buffer("mean", torch.zeros(self.dim))
        self.register_buffer("std", torch.ones(self.dim))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, values: torch.Tensor) -> None:
        self._check(values)
        values = values.detach().float()
        if not torch.isfinite(values).all():
            raise ValueError("cannot fit a normalizer on NaN or Inf values")
        self.mean.copy_(values.mean(dim=0).to(self.mean.device))
        self.std.copy_(values.std(dim=0, unbiased=False).clamp_min(self.eps).to(self.std.device))
        self.fitted.fill_(True)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        self._check(values)
        if not bool(self.fitted.item()):
            raise RuntimeError("normalizer has not been fitted")
        return (values.float() - self.mean) / self.std

    def inverse(self, values: torch.Tensor) -> torch.Tensor:
        self._check(values)
        if not bool(self.fitted.item()):
            raise RuntimeError("normalizer has not been fitted")
        return values.float() * self.std + self.mean

    def _check(self, values: torch.Tensor) -> None:
        if values.ndim != 2 or values.shape[1] != self.dim:
            raise ValueError(f"expected [B, {self.dim}] features, got {tuple(values.shape)}")
