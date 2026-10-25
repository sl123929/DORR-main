"""Standalone train/calibrate/score entry point for ROLeR diffusion OOD detectors."""

import argparse
import json
import logging
from typing import Dict, List

import torch
from tqdm.auto import tqdm

import os

_cpu_threads = int(os.environ.get("ROLeR_OOD_CPU_THREADS", "3"))
if not 1 <= _cpu_threads <= 3:
    raise ValueError("ROLeR_OOD_CPU_THREADS must be between 1 and 3")
torch.set_num_threads(_cpu_threads)
torch.set_num_interop_threads(_cpu_threads)

import sys
sys.path.extend(["./src"])

from core.ood_detector import DetectorDataset, OODDetectorBundle, calibrate_bundle, train_detector_bundle
from core.ood_detector.train import popularity_false_positive_rates


def main() -> None:
    parser = argparse.ArgumentParser(description="ROLeR standalone diffusion OOD detector")
    subparsers = parser.add_subparsers(dest="command", required=True)
    split = subparsers.add_parser("split", help="make reproducible training and calibration ID splits")
    split.add_argument("--dataset", required=True)
    split.add_argument("--train-output", required=True)
    split.add_argument("--calibration-output", required=True)
    split.add_argument("--calibration-fraction", type=float, default=0.2)
    split.add_argument("--seed", type=int, default=0)
    train = subparsers.add_parser("train", help="train only from an ID training dataset")
    train.add_argument("--dataset", required=True, help="DetectorDataset .npz produced from ROLeR offline data")
    train.add_argument("--checkpoint", required=True)
    train.add_argument("--architecture", choices=["separate", "shared"], default="separate")
    train.add_argument("--task-sampling", choices=["both", "random"], default="both")
    train.add_argument("--score-levels", type=int, default=1)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--hidden-dim", type=int, default=256)
    train.add_argument("--time-embed-dim", type=int, default=16)
    train.add_argument("--num-hidden-layers", type=int, default=4)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device", default="cuda")
    calibrate = subparsers.add_parser("calibrate", help="calibrate an existing detector on a held-out ID split")
    calibrate.add_argument("--dataset", required=True, help="held-out ID calibration DetectorDataset .npz")
    calibrate.add_argument("--reference-dataset", required=True, help="ID training dataset, used only for popularity diagnostics")
    calibrate.add_argument("--checkpoint", required=True, help="un-calibrated checkpoint written by train")
    calibrate.add_argument("--output-checkpoint", required=True)
    calibrate.add_argument("--percentiles", nargs="+", type=int, default=[90, 95, 99])
    calibrate.add_argument("--device", default="cuda")
    calibrate.add_argument("--stable-score-noise", action="store_true",
                           help="use batch-independent common probes and recalibrate both thresholds")
    score = subparsers.add_parser("score", help="score states and actual catalog item IDs")
    score.add_argument("--checkpoint", required=True)
    score.add_argument("--input", required=True,
                       help=".pt with states, item_ids, and representation metadata from the frozen tracker")
    score.add_argument("--percentile", type=int, default=95)
    score.add_argument("--device", default="cuda")
    score.add_argument("--batch-size", type=int, default=4096,
                       help="number of samples scored per progress-bar batch")
    score.add_argument("--output", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.command == "split":
        with tqdm(total=3, desc="Splitting ID data", unit="step") as progress:
            dataset = DetectorDataset.load(args.dataset)
            progress.update(1)
            train_data, calibration_data = dataset.split(args.calibration_fraction, args.seed)
            progress.update(1)
            train_data.save(args.train_output)
            calibration_data.save(args.calibration_output)
            progress.update(1)
        logging.info("wrote %s and %s", args.train_output, args.calibration_output)
    elif args.command == "train":
        dataset = DetectorDataset.load(args.dataset)
        config = vars(args)
        config["show_progress"] = True
        bundle = train_detector_bundle(dataset, config)
        bundle.save(args.checkpoint)
        logging.info("saved un-calibrated detector checkpoint to %s", args.checkpoint)
    elif args.command == "calibrate":
        dataset = DetectorDataset.load(args.dataset)
        reference = DetectorDataset.load(args.reference_dataset)
        bundle = OODDetectorBundle.load(args.checkpoint, map_location=args.device)
        if args.stable_score_noise:
            bundle.action.stable_score_noise = bundle.state.stable_score_noise = True
        bundle.action.to(args.device)
        bundle.state.to(args.device)
        thresholds = calibrate_bundle(bundle, dataset, args.percentiles, show_progress=True)
        with tqdm(total=1, desc="Checking popularity groups", unit="step") as progress:
            diagnostics = popularity_false_positive_rates(bundle, dataset, reference.item_ids, max(args.percentiles))
            progress.update(1)
        bundle.save(args.output_checkpoint)
        logging.info("saved calibrated detector checkpoint to %s", args.output_checkpoint)
        print(json.dumps({"thresholds": thresholds, "calibration_popularity_ood_rates": diagnostics}, indent=2))
    else:  # score
        payload = torch.load(args.input, map_location="cpu")
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("score input must include a metadata dict for the frozen representation")
        bundle = OODDetectorBundle.load(args.checkpoint, map_location=args.device, expected_metadata=metadata)
        bundle.action.to(args.device)
        bundle.state.to(args.device)
        states, ids = payload["states"], payload["item_ids"]
        output = _score_in_batches(bundle, states, ids, args.percentile, args.batch_size)
        torch.save(output, args.output)
        logging.info("saved scores to %s", args.output)


@torch.no_grad()
def _score_in_batches(bundle: OODDetectorBundle, states: torch.Tensor, item_ids: torch.Tensor,
                      percentile: int, batch_size: int) -> Dict[str, Dict[str, torch.Tensor]]:
    states = torch.as_tensor(states)
    item_ids = torch.as_tensor(item_ids).reshape(-1)
    if states.ndim != 2 or len(states) != len(item_ids):
        raise ValueError("score input requires states [B, D] and one item ID per state")
    if len(states) == 0 or batch_size <= 0:
        raise ValueError("score input and score batch size must be non-empty and positive")
    action_output: Dict[str, List[torch.Tensor]] = {"reconstruction_error": [], "normalized_score": [], "is_ood": []}
    state_output: Dict[str, List[torch.Tensor]] = {"reconstruction_error": [], "normalized_score": [], "is_ood": []}
    starts = range(0, len(states), batch_size)
    with tqdm(starts, total=(len(states) + batch_size - 1) // batch_size,
              desc="Scoring state/item batches", unit="batch") as progress:
        for start in progress:
            stop = min(start + batch_size, len(states))
            action_scores = bundle.score_action(states[start:stop], item_ids[start:stop], percentile)
            state_scores = bundle.score_state(states[start:stop], percentile)
            for name, values in action_scores.items():
                action_output[name].append(values.detach().cpu())
            for name, values in state_scores.items():
                state_output[name].append(values.detach().cpu())
    return {
        "action": {name: torch.cat(values) for name, values in action_output.items()},
        "state": {name: torch.cat(values) for name, values in state_output.items()},
    }


if __name__ == "__main__":
    main()
