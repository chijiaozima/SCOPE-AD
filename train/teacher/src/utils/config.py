from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input."""
    result = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Configuration root must be a mapping: {config_path}")
    includes = raw.pop("include", [])
    if isinstance(includes, (str, Path)):
        includes = [includes]
    merged: dict[str, Any] = {}
    for include in includes:
        include_path = (config_path.parent / str(include)).resolve()
        merged = deep_merge(merged, load_config(include_path))
    merged = deep_merge(merged, raw)
    merged["_config_path"] = str(config_path)
    merged["_config_dir"] = str(config_path.parent)
    return merged


def project_root(config: Mapping[str, Any] | None = None) -> Path:
    if config and config.get("_config_path"):
        # All shipped configs live directly below PROJECT/configs.
        return Path(str(config["_config_path"])).resolve().parent.parent
    return Path(__file__).resolve().parents[2]


def resolve_path(config: Mapping[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root(config) / path).resolve()


def require_keys(mapping: Mapping[str, Any], keys: list[str], *, context: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise KeyError(f"Missing required keys in {context}: {missing}")
