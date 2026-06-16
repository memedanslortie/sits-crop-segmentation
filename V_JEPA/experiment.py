from copy import deepcopy
from dataclasses import dataclass
from typing import Any


def set_nested(cfg: dict, path: str, value):
    keys = path.split(".")
    node = cfg

    for key in keys[:-1]:
        node = node.setdefault(key, {})

    node[keys[-1]] = value

def flatten(d, prefix=""):
    result = {}

    for k, v in d.items():
        path = f"{prefix}.{k}" if prefix else k

        if isinstance(v, dict):
            result.update(flatten(v, path))
        else:
            result[path] = v

    return result


@dataclass(slots=True)
class Experiment:
    name: str
    group: str
    config: dict[str, Any]

import yaml
from copy import deepcopy


def build_experiments(
    base_config: dict,
    experiment_file: str,
) -> list[Experiment]:

    spec = yaml.safe_load(open(experiment_file))

    defaults = spec["defaults"]
    groups = spec["groups"]

    experiments = []

    for group_name, variations in groups.items():

        for path, values in variations.items():

            if not isinstance(values, list):
                values = [values]

            for value in values:

                cfg = deepcopy(base_config)

                for default_path, default_value in flatten(defaults).items():
                    set_nested(cfg, default_path, default_value)

                set_nested(cfg, path, value)

                experiments.append(
                    Experiment(
                        name=f"{group_name}_{value}",
                        group=group_name,
                        config=cfg,
                    )
                )

    return experiments