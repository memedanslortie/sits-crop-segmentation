import yaml
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path

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
    config: dict[str, Any]

def build_experiments(
    config_path: str = "config.yaml",
    experiments_path: str = "experiments.yaml",
) -> list[Experiment]:

    base_cfg = load_yaml(config_path)
    spec = load_yaml(experiments_path)

    defaults = flatten(spec["defaults"])

    experiments: list[Experiment] = []

    for group_name, group in spec["groups"].items():

        paths = list(group.keys())
        value_lists = [group[p] for p in paths]

        for values in product(*value_lists):

            cfg = deepcopy(base_cfg)

            for path, value in defaults.items():
                set_nested(cfg, path, value)

            parts = []

            for path, value in zip(paths, values):
                set_nested(cfg, path, value)

                short_name = path.split(".")[-1]
                parts.append(f"{short_name}={value}")

            experiments.append(
                Experiment(
                    name=f"{group_name}__{'__'.join(parts)}",
                    group=group_name,
                    config=cfg,
                )
            )

    return experiments