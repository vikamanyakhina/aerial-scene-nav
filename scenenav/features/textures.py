"""Текстурные признаки с ФИКСИРОВАННЫМ масштабированием в uint8 0..255.

Отличие от прототипа: раньше каждая карта нормировалась min-max по своему изображению,
из-за чего один и тот же признак имел разный масштаб на разных снимках (особенно дисперсия).
Здесь масштаб одинаков для всех изображений, поэтому карты сопоставимы.
Признаки считаются по ВСЕМУ изображению сразу (до нарезки на окна), без краевых артефактов.
"""
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.filters.rank import entropy as rank_entropy
from skimage.morphology import disk


def _gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _to_u8(x01: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x01 * 255.0), 0, 255).astype(np.uint8)


def compute_lbp(rgb: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
    """Равномерный LBP: коды 0..P+1 -> делим на P+1."""
    lbp = local_binary_pattern(_gray(rgb), P=n_points, R=radius, method="uniform")
    return _to_u8(lbp / float(n_points + 1))


def compute_entropy(rgb: np.ndarray, radius: int = 5) -> np.ndarray:
    """Локальная энтропия в круге радиуса radius; максимум для 8 бит = 8 бит -> делим на 8."""
    ent = rank_entropy(_gray(rgb), disk(radius))
    return _to_u8(ent / 8.0)


def compute_variance(rgb: np.ndarray, kernel: int = 9) -> np.ndarray:
    """Локальная дисперсия D = E[I^2] - E[I]^2; лог-масштаб, максимум дисперсии 255^2/4."""
    g = _gray(rgb).astype(np.float32)
    mean = cv2.blur(g, (kernel, kernel))
    mean_sq = cv2.blur(g * g, (kernel, kernel))
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return _to_u8(np.log1p(var) / np.log1p(255.0 ** 2 / 4.0))


def compute_gabor(rgb: np.ndarray) -> np.ndarray:
    """Максимум модуля отклика банка Габора (4 ориентации, lambda=10); ядра L1-нормированы."""
    g = _gray(rgb).astype(np.float32) / 255.0
    best = np.zeros_like(g)
    for theta in (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4):
        k = cv2.getGaborKernel((21, 21), 5.0, theta, 10.0, 0.5, 0.0, ktype=cv2.CV_32F)
        k = k / np.abs(k).sum()
        best = np.maximum(best, np.abs(cv2.filter2D(g, cv2.CV_32F, k)))
    return _to_u8(np.sqrt(np.clip(best, 0.0, 1.0)))


TEXTURE_FUNCS = {
    "lbp": compute_lbp,
    "entropy": compute_entropy,
    "variance": compute_variance,
    "gabor": compute_gabor,
}
