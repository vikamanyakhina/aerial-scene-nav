"""Загрузка YAML-конфигов с наследованием (_base)."""
import copy
from pathlib import Path

import yaml

from scenenav.features.textures import TEXTURE_FUNCS


def _merge(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = cfg.pop("_base", None)
    if base:
        cfg = _merge(load_config(path.parent / base), cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    feats = cfg["data"]["features"]
    if not feats or feats[0] != "rgb":
        raise ValueError(f"features должен начинаться с 'rgb', получено {feats}")
    unknown = [f for f in feats[1:] if f not in TEXTURE_FUNCS]
    if unknown:
        raise ValueError(f"Неизвестные признаки {unknown}; доступны {list(TEXTURE_FUNCS)}")
    if len(feats) > 2:
        raise ValueError("В прототипе допускается не более одного текстурного канала")
    if cfg["model"]["arch"] not in ("unet", "smp_unet"):
        raise ValueError("model.arch должен быть 'unet' или 'smp_unet'")


def in_channels_from_features(features) -> int:
    """Число входных каналов ВСЕГДА считается из списка признаков — рассинхрон невозможен."""
    return 3 + (len(features) - 1)
