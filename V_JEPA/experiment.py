import yaml
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from utils.logging import get_logger
log = get_logger()

def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_nested(cfg: dict, path: str, value) -> None:
    node = cfg
    keys = path.split(".")

    for key in keys[:-1]:
        node = node.setdefault(key, {})

    node[keys[-1]] = value

def flatten(d: dict, prefix: str = "") -> dict[str, object]:
    out = {}

    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k

        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v

    return out

@dataclass(slots=True)
class Experiment:
    name: str
    group: str
    config: dict

def build_experiments(
    config_path: str | Path = "config.yaml",
    experiments_path: str | Path = "experiments.yaml",
) -> list[Experiment]:

    base_cfg = load_yaml(config_path)
    spec = load_yaml(experiments_path)

    if not spec:
        raise ValueError(f"experiments.yaml at '{experiments_path}' is empty or invalid.")

    experiments: list[Experiment] = []

    for group_name, group in spec.items():
        keys = list(group.keys())
        values_list = [group[k] for k in keys]

        for combo in product(*values_list):
            cfg = deepcopy(base_cfg)
            parts = []

            for k, v in zip(keys, combo):
                set_nested(cfg, k, v)
                short = k.split(".")[-1]
                parts.append(f"{short}={v}")

            experiments.append(
                Experiment(
                    name=f"{group_name}__{'__'.join(parts)}",
                    group=group_name,
                    config=cfg,
                )
            )

    return experiments