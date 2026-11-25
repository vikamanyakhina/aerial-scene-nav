"""Датасет окон для обучения/оценки сегментации.

Число каналов определяется ТОЛЬКО списком features (rgb + не более одного текстурного канала).
Аугментации — собственные, на numpy: фотометрические искажения применяются только к RGB,
а текстурные каналы подвергаются лишь геометрическим преобразованиям (в прототипе Normalize
был рассчитан на 3 канала и ломался на 4).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from scenenav.classes import IGNORE_INDEX
from scenenav.config import in_channels_from_features

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def grid_starts(length: int, window: int, stride: int):
    """Начала окон вдоль оси; последнее окно прижато к краю, чтобы покрыть всё изображение."""
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


def normalize_stack(stack01: np.ndarray) -> np.ndarray:
    """(H,W,C) в [0,1] -> (C,H,W) float32: RGB — по ImageNet, текстуры — (x-0.5)/0.5."""
    out = stack01.astype(np.float32).copy()
    out[..., :3] = (out[..., :3] - IMAGENET_MEAN) / IMAGENET_STD
    if out.shape[2] > 3:
        out[..., 3:] = (out[..., 3:] - 0.5) / 0.5
    return np.ascontiguousarray(out.transpose(2, 0, 1))


def _read(path: Path) -> np.ndarray:
    return np.array(Image.open(path))


def _crop(arr: np.ndarray, y: int, x: int, size: int, pad_value) -> np.ndarray:
    crop = arr[y:y + size, x:x + size]
    ph, pw = size - crop.shape[0], size - crop.shape[1]
    if ph > 0 or pw > 0:
        pads = [(0, ph), (0, pw)] + [(0, 0)] * (arr.ndim - 2)
        crop = np.pad(crop, pads, constant_values=pad_value)
    return crop


class WindowDataset(Dataset):
    def __init__(self, root, ids, features=("rgb",), window=256, train=False,
                 samples_per_epoch=1000, seed=0):
        self.root = Path(root)
        self.ids = list(ids)
        self.features = list(features)
        self.window = window
        self.train = train
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.epoch = 0
        if not self.ids:
            raise ValueError("Пустой список изображений для датасета")

        self.rgb, self.masks, self.tex = [], [], []
        for uid in self.ids:
            self.rgb.append(_read(self.root / "images" / f"{uid}.png"))
            self.masks.append(_read(self.root / "masks" / f"{uid}.png"))
            self.tex.append([_read(self.root / "features" / f / f"{uid}.png") for f in self.features[1:]])

        areas = np.array([m.size for m in self.masks], dtype=np.float64)
        self._probs = areas / areas.sum()
        self.windows = []
        if not train:
            for k, m in enumerate(self.masks):
                for y in grid_starts(m.shape[0], window, window):
                    for x in grid_starts(m.shape[1], window, window):
                        self.windows.append((k, y, x))

    @property
    def in_channels(self) -> int:
        return in_channels_from_features(self.features)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self):
        return self.samples_per_epoch if self.train else len(self.windows)

    def _stack01(self, k, y=None, x=None, size=None):
        """Стек каналов в [0,1] для окна (или всего изображения при y is None)."""
        if y is None:
            rgb, tex = self.rgb[k], self.tex[k]
            parts = [rgb.astype(np.float32) / 255.0] + [t[..., None].astype(np.float32) / 255.0 for t in tex]
        else:
            rgb = _crop(self.rgb[k], y, x, size, 0)
            tex = [_crop(t, y, x, size, 0) for t in self.tex[k]]
            parts = [rgb.astype(np.float32) / 255.0] + [t[..., None].astype(np.float32) / 255.0 for t in tex]
        return np.concatenate(parts, axis=2)

    def full_input(self, k):
        """Всё изображение целиком: (C,H,W) тензор и маска (H,W) — для скользящего окна при финальной оценке."""
        return torch.from_numpy(normalize_stack(self._stack01(k))), self.masks[k]

    def __getitem__(self, idx):
        w = self.window
        if self.train:
            rng = np.random.default_rng([self.seed, self.epoch, idx])
            k = int(rng.choice(len(self.ids), p=self._probs))
            h_k, w_k = self.masks[k].shape
            y = int(rng.integers(0, max(h_k - w, 0) + 1))
            x = int(rng.integers(0, max(w_k - w, 0) + 1))
        else:
            k, y, x = self.windows[idx]
        stack = self._stack01(k, y, x, w)
        mask = _crop(self.masks[k], y, x, w, IGNORE_INDEX)
        if self.train:
            stack, mask = self._augment(stack, mask, rng)
        return torch.from_numpy(normalize_stack(stack)), torch.from_numpy(mask.astype(np.int64))

    @staticmethod
    def _augment(stack, mask, rng):
        if rng.random() < 0.5:
            stack, mask = stack[:, ::-1], mask[:, ::-1]
        if rng.random() < 0.5:
            stack, mask = stack[::-1], mask[::-1]
        r = int(rng.integers(0, 4))
        stack, mask = np.rot90(stack, r), np.rot90(mask, r)
        stack = np.ascontiguousarray(stack).copy()
        mask = np.ascontiguousarray(mask)
        rgb = stack[..., :3]
        if rng.random() < 0.5:  # яркость/контраст только для RGB
            a, b = rng.uniform(0.8, 1.2), rng.uniform(-0.1, 0.1)
            rgb = (rgb - 0.5) * a + 0.5 + b
        if rng.random() < 0.2:  # шум только для RGB
            rgb = rgb + rng.normal(0, rng.uniform(0.01, 0.03), rgb.shape)
        stack[..., :3] = np.clip(rgb, 0.0, 1.0)
        return stack, mask
