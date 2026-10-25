import os
import importlib
from pathlib import Path


def bind_environment_paths(env, source_root):
    mappings = {
        "KuaiRand-v0": ("KuaiRand_Pure", "KuaiRand", "data", "MF_results_GT/kuairand_is_click.csv"),
        "CoatEnv-v0": ("coat", "Coat", "", "../RL4Rec/data/coat_pseudoGT_ratingM.ascii"),
        "YahooEnv-v0": ("YahooR3", "Yahoo", "", "../RL4Rec/data/yahoo_pseudoGT_ratingM.ascii"),
    }
    if env not in mappings:
        return
    folder, name, data_folder, required = mappings[env]
    root = Path(source_root).resolve() / "environments" / folder
    if not root.is_dir():
        raise FileNotFoundError(f"Source environment directory missing: {root}")
    required_path = (root / required).resolve()
    if not required_path.is_file():
        raise FileNotFoundError(f"Source evaluation matrix missing: {required_path}")
    module = importlib.import_module(f"environments.{folder}.env.{name}")
    module.ROOTPATH = str(root)
    module.DATAPATH = str(root / data_folder)
    print(f"[DORR] {env} read-only environment root: {root}", flush=True)


def cache_path(read_path, environment):
    source = Path(read_path)
    if source.is_file():
        return str(source)
    root = Path(os.environ.get("ROLER_CACHE_ROOT", Path(__file__).resolve().parents[2] / "environment_cache"))
    target = root / environment / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    return str(target)
