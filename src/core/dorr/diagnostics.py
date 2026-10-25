"""Persist unsmoothed DORR update metrics independently of Tianshou LazyLogger."""
import json
import math
from pathlib import Path


class TrainingDiagnosticsWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.update = 0
        self.module_step = 0

    def modules(self, epoch, tensors, scalars, every=20):
        """Sample every N minibatches, preserving masks/counts and full precision.

        Quantiles are within this minibatch, not global epoch quantiles.
        No raw user histories, model inputs or random sampling are persisted.
        """
        self.module_step += 1
        if (self.module_step - 1) % every:
            return
        import torch
        stats = {}
        for name, tensor in tensors.items():
            x = torch.as_tensor(tensor).detach().float().reshape(-1)
            finite = torch.isfinite(x)
            item = {"count": x.numel(), "nonfinite": int((~finite).sum())}
            x = x[finite]
            if x.numel():
                q = torch.quantile(x, x.new_tensor([0., .05, .5, .95, 1.])).cpu().tolist()
                item.update(dict(zip(("min", "p05", "p50", "p95", "max"), q)))
                item.update(mean=float(x.mean()), std=float(x.std(unbiased=False)))
            stats[name] = item
        record = {"epoch": epoch, "minibatch_step": self.module_step,
                  "sample_every_minibatches": every, "scalars": scalars, "distributions": stats}
        path = self.path.with_name("dorr_modules.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")

    def __call__(self, epoch, logs):
        self.update += 1
        metrics = {}
        for key, values in logs.items():
            if values:
                mean = sum(float(v) for v in values) / len(values)
                if not math.isfinite(mean):
                    raise FloatingPointError(f"non-finite training metric: {key}")
                metrics[key] = mean
        record = {"epoch": epoch, "update": self.update,
                  "aggregation": "unweighted minibatch mean within learn call", "metrics": metrics}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
