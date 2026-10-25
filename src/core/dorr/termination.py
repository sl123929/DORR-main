"""Online leave prediction from observed training transitions, not a leave oracle."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def history_features(categories, actions, valid, candidates, max_turn, window=10):
    """Newest-first category overlap and masks; never include the current action."""
    n = len(candidates)
    features = np.zeros((n, 2 * window + 1), dtype=np.float32)
    for row, candidate in enumerate(candidates):
        target = set(categories[int(candidate)])
        for step in range(min(window, actions.shape[1])):
            if valid[row, step]:
                features[row, step] = len(target.intersection(categories[int(actions[row, step])])) / max(1, len(target))
                features[row, window + step] = 1
    features[:, -1] = valid.sum(axis=1) / max(1, max_turn)
    return features


def survival_gain(reward, ref_reward, value, ref_value, risk, ref_risk, gamma):
    return reward - ref_reward + gamma * ((1 - risk) * value - (1 - ref_risk) * ref_value)


class LeaveRisk(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(21, 32), nn.ReLU(), nn.Linear(32, 1))
        self.register_buffer('positives', torch.tensor(0))
        self.register_buffer('negatives', torch.tensor(0))

    def forward(self, features):
        return self.net(features).reshape(-1)

    def update(self, features, labels, optimizer):
        logits = self(features)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.positives.add_(labels.sum().long())
        self.negatives.add_((1 - labels).sum().long())
        return float(loss.detach()), float(((logits.detach().sigmoid() - labels) ** 2).mean())

    @property
    def ready(self):
        return bool(self.positives >= 32 and self.negatives >= 32)
