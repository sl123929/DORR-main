"""Structured DORR evaluation metrics written alongside the text training log."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


class DORRMetricsReporter:
    """Persist per-epoch and final NX_0 metrics without parsing console output."""

    def __init__(self, output_dir: str | Path, scenario: str = "NX_0", on_new_best=None) -> None:
        self.output_dir = Path(output_dir)
        self.scenario = scenario
        self.prefix = f"{scenario}_"
        self.history_path = self.output_dir / "dorr_metrics.jsonl"
        self.summary_path = self.output_dir / "dorr_metrics_summary.json"
        self.last: dict[str, Any] | None = None
        self.best_by_rtra: dict[str, Any] | None = None
        # Optional hook invoked with (epoch, record) whenever NX_0 Rtra improves,
        # so callers can persist the best policy weights at that moment.
        self.on_new_best = on_new_best

    def on_epoch_begin(self, epoch: int, **kwargs: Any) -> None:
        return None

    def on_train_begin(self, **kwargs: Any) -> None:
        return None

    def on_train_end(self, **kwargs: Any) -> None:
        return None

    def _record(self, epoch: int, results: Mapping[str, Any]) -> dict[str, Any]:
        required = ("n/ep", f"{self.prefix}n/st", f"{self.prefix}rew")
        missing = [key for key in required if key not in results]
        if missing:
            raise KeyError(f"cannot write DORR metrics; missing evaluation keys: {missing}")
        episodes = int(results["n/ep"])
        if episodes <= 0:
            raise ValueError("DORR evaluation must contain at least one episode")
        rtra = float(results[f"{self.prefix}rew"])
        length = float(results[f"{self.prefix}n/st"]) / episodes
        if length <= 0:
            raise ValueError("DORR evaluation trajectory length must be positive")
        domination = {key[len(self.prefix):]: float(value) for key, value in results.items()
                      if key.startswith(f"{self.prefix}ifeat_")}
        return {
            "epoch": int(epoch),
            "scenario": self.scenario,
            "num_trajectories": episodes,
            "Rtra": rtra,
            # ROLeR's logger calls Rtra / len_tra "ctr".  The experiment's
            # requested Reach metric uses that same NX_0 scalar.
            "Reach": rtra / length,
            "Length": length,
            # KuaiRec/KuaiRand expose ifeat_feat, Coat exposes per-feature rates,
            # and Yahoo has no item-feature domination metric.
            "MCD": domination.get("ifeat_feat"),
            "item_feature_domination": domination,
        }

    def _write_summary(self) -> None:
        if self.last is None or self.best_by_rtra is None:
            return
        payload = {
            "format_version": 1,
            "scenario": self.scenario,
            "definitions": {
                "Rtra": f"{self.prefix}rew",
                "Reach": f"{self.prefix}rew / ({self.prefix}n/st / n/ep)",
                "Length": f"{self.prefix}n/st / n/ep",
                "MCD": f"{self.prefix}ifeat_feat when available, otherwise null",
                "item_feature_domination": f"{self.prefix}ifeat_*",
            },
            "final": self.last,
            "best_by_Rtra": self.best_by_rtra,
        }
        temporary = self.summary_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.summary_path)

    def on_epoch_end(self, epoch: int, results=None, **kwargs: Any) -> None:
        if results is None:
            raise ValueError("DORR metrics reporter requires evaluation results")
        record = self._record(epoch, results)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self.last = record
        if self.best_by_rtra is None or record["Rtra"] > self.best_by_rtra["Rtra"]:
            self.best_by_rtra = record
            if self.on_new_best is not None:
                self.on_new_best(epoch, record)
        self._write_summary()
        return record
