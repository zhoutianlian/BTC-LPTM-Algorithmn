from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def deep_update(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = deep_update(dict(out[k]), v)
        else:
            out[k] = v
    return out


def load_project_config(config_dir: str | Path) -> dict[str, Any]:
    config_dir = Path(config_dir)
    return {
        "feature_contract": load_yaml(config_dir / "feature_contract.yaml"),
        "model": load_yaml(config_dir / "model_config.yaml"),
        "state_definitions": load_yaml(config_dir / "state_definitions.yaml"),
    }


def load_run_config(path: str | Path) -> dict[str, Any]:
    return load_yaml(path)
