"""Seeding, device selection, config loading and small I/O helpers."""
from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
COMMON_CONFIG = ROOT / "configs" / "common.yaml"


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU, CUDA and MPS generators)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(pref)


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config(path: str | Path) -> dict:
    """Load a run config. `_base: other.yaml` chains are resolved relative to the
    config's directory; configs/common.yaml is always the outermost base."""
    path = Path(path)
    cfg = _load_yaml(path)
    base_name = cfg.pop("_base", None)
    if base_name is not None:
        base = load_config(path.parent / base_name)
    elif path.resolve() != COMMON_CONFIG.resolve():
        base = _load_yaml(COMMON_CONFIG)
    else:
        base = {}
    merged = deep_merge(base, cfg)
    merged.setdefault("run_name", path.stem)
    return merged


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def dump_json(obj: Any, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=False, default=_json_default)


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")
