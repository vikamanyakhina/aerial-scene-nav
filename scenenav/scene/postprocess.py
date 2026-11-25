"""Морфологическая постобработка карты классов (в прототипе описана в отчёте, но в коде отсутствовала)."""
import cv2
import numpy as np
from scipy import ndimage as ndi


def clean_mask(mask: np.ndarray, n_classes: int, radius: int = 2, min_area: int = 100) -> np.ndarray:
    """Открытие + закрытие каждого класса, удаление мелких компонент, заполнение дыр ближайшими метками.

    Классы обрабатываются по убыванию площади, поэтому мелкие классы при конфликте имеют приоритет.
    Пиксели, ставшие «ничьими», получают метку ближайшего размеченного пикселя.
    """
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    areas = [(int((mask == c).sum()), c) for c in range(n_classes)]
    out = np.full(mask.shape, 255, dtype=np.uint8)
    for _, c in sorted(areas, reverse=True):
        b = (mask == c).astype(np.uint8)
        if b.sum() == 0:
            continue
        b = cv2.morphologyEx(b, cv2.MORPH_OPEN, se)
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, se)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
        keep = np.zeros(n, dtype=bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
        out[keep[labels]] = c
    holes = out == 255
    if holes.any() and not holes.all():
        _, (iy, ix) = ndi.distance_transform_edt(holes, return_indices=True)
        out = out[iy, ix]
    return out
