"""Export frozen ROLeR ID representations for standalone OOD detector training.

The policy checkpoint is used only to recover the final state tracker.  This
script never updates a world model, policy, replay buffer, or detector.
"""

import argparse
import os
import sys
from pathlib import Path

import torch

_cpu_threads = int(os.environ.get("ROLeR_OOD_CPU_THREADS", "3"))
if not 1 <= _cpu_threads <= 3:
    raise ValueError("ROLeR_OOD_CPU_THREADS must be between 1 and 3")
torch.set_num_threads(_cpu_threads)
torch.set_num_interop_threads(_cpu_threads)

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--policy-checkpoint", required=True,
                        help="ROLeR checkpoint written with --is_save at the final policy epoch")
    parser.add_argument("--output", required=True, help="output DetectorDataset .npz path")
    parser.add_argument("--export-batch-size", type=int, default=1024)
    parser.add_argument("--source-root", required=True,
                        help="read-only ROLeR root containing environments/ and saved_models/")
    export_args, _ = parser.parse_known_args()

    source_root = Path(export_args.source_root).expanduser().resolve()
    if not (source_root / "environments").is_dir() or not (source_root / "saved_models").is_dir():
        raise ValueError("source root must contain environments/ and saved_models/")
    output_path = Path(export_args.output).expanduser()
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()
    if source_root == output_path or source_root in output_path.parents:
        raise ValueError("detector output must not be written inside the read-only source root")
    policy_checkpoint = Path(export_args.policy_checkpoint).expanduser().resolve()
    if source_root not in policy_checkpoint.parents:
        raise ValueError("policy checkpoint must be loaded from the read-only source root")

    # ROLeR data loaders use relative paths. Keep all reads rooted in the old
    # project, while output_path remains an absolute path under ROLeR-new.
    # PYTHONDONTWRITEBYTECODE is set by the pipeline to prevent __pycache__ writes.
    os.chdir(source_root)
    os.environ["ROLER_ENV_ROOT"] = str(source_root / "environments")
    os.environ["ROLER_CACHE_ROOT"] = str(PROJECT_ROOT / "environment_cache")

    # Import after chdir: read datasets/models via ROLeR-oral relative paths, but
    # execute the new self-contained detector implementation from PROJECT_ROOT.
    sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"),
                    str(PROJECT_ROOT / "src" / "DeepCTR-Torch"),
                    str(PROJECT_ROOT / "src" / "tianshou")]
    from policy_utils import prepare_buffer_via_offline_data, prepare_user_model_and_env
    from run_Policy_Main import get_args_all, prepare_envs, setup_state_tracker
    from core.configs import get_common_args
    from core.ood_detector import build_from_roler_buffer

    # Reuse policy parsing and construction so state semantics exactly match the
    # checkpoint. get_args_all intentionally ignores this exporter's arguments.
    args = get_args_all()
    args = get_common_args(args)
    from core.environment_paths import bind_environment_paths
    bind_environment_paths(args.env, source_root)
    ensemble_models = prepare_user_model_and_env(args)
    env, train_envs, test_envs_dict = prepare_envs(args, ensemble_models)
    state_tracker = setup_state_tracker(args, ensemble_models, env, train_envs, test_envs_dict)
    if state_tracker is None:
        raise ValueError(f"unsupported state tracker: {args.which_tracker}")

    checkpoint = torch.load(policy_checkpoint, map_location=args.device)
    if "state_tracker" not in checkpoint:
        raise ValueError("policy checkpoint does not contain a state_tracker state dict")
    state_tracker.load_state_dict(checkpoint["state_tracker"])
    _, buffer, _ = prepare_buffer_via_offline_data(args)
    dataset = build_from_roler_buffer(buffer, state_tracker, export_args.export_batch_size,
                                      show_progress=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save(output_path)
    print({
        "output": str(output_path),
        "samples": len(dataset.states),
        "state_dim": dataset.state_dim,
        "item_dim": dataset.item_dim,
        "catalog_size": len(dataset.item_embeddings),
        "representation_version": dataset.representation_version,
    })


if __name__ == "__main__":
    main()
