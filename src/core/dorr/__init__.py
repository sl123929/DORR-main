"""DORR components for discrete ROLeR item recommendation.

The package is intentionally independent of ``DOSER-main``.  The only detector
dependency is ROLeR's self-contained ``core.ood_detector`` checkpoint API.
"""

from .behavior import BehaviorCloningPolicy, train_behavior_policy
from .engine import DORRConfig, DORREngine, classify_ood, schedule_coefficients
from .qcritic import QCritic, discounted_return_quantile, discounted_returns, soft_update

__all__ = [
    "BehaviorCloningPolicy",
    "DORRConfig",
    "DORREngine",
    "QCritic",
    "classify_ood",
    "discounted_return_quantile",
    "discounted_returns",
    "schedule_coefficients",
    "soft_update",
    "train_behavior_policy",
]


def __getattr__(name):
    """Delay Tianshou's optional h5py import until transition code is needed."""
    if name == "DORRTransitionAdapter":
        from .transition import DORRTransitionAdapter
        return DORRTransitionAdapter
    raise AttributeError(name)
