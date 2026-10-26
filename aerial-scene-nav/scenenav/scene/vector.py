"""Вектор описания сцены S = [p_1..p_K, x_1,y_1,...,x_K,y_K] и детерминированное текстовое/JSON-описание."""
from dataclasses import dataclass

import cv2
import numpy as np

from scenenav.classes import CLASS_NAMES, CLASS_NAMES_RU, IGNORE_INDEX

ZONES = {
    (0, 0): "в левой верхней части", (0, 1): "в верхней центральной части", (0, 2): "в правой верхней части",
    (1, 0): "в левой части", (1, 1): "в центре", (1, 2): "в правой части",
    (2, 0): "в левой нижней части", (2, 1): "в нижней центральной части", (2, 2): "в правой нижней части",
}


@dataclass
class SceneVector:
    p: np.ndarray        # (K,) доли классов, сумма = 1
    xy: np.ndarray       # (K,2) центроиды в [0,1]; для отсутствующего класса 0.5 (см. present)
    present: np.ndarray  # (K,) bool: доля >= min_share
    grid_p: np.ndarray   # (g,g,K) доли классов в ячейках сетки

    def flat(self, mode: str = "p"):
        """Возвращает (вектор, веса). Веса = 0 для координат отсутствующего класса — их нельзя сравнивать.

        mode: 'p' | 'p_xy' | 'grid'. Для расстояний используйте взвешенные варианты (см. retrieval).
        """
        k = len(self.p)
        if mode == "p":
            return self.p.copy(), np.ones(k)
        if mode == "p_xy":
            vec = np.concatenate([self.p, self.xy.reshape(-1)])
            w = np.concatenate([np.ones(k), np.repeat(self.present.astype(float), 2)])
            return vec, w
        if mode == "grid":
            v = self.grid_p.reshape(-1)
            return v, np.ones_like(v)
        raise ValueError(mode)


def scene_vector(mask: np.ndarray, n_classes: int, ignore: int = IGNORE_INDEX,
                 min_share: float = 0.005, grid: int = 3) -> SceneVector:
    h, w = mask.shape
    valid = mask != ignore
    total = max(int(valid.sum()), 1)
    p = np.array([(mask == c).sum() / total for c in range(n_classes)], dtype=np.float64)
    present = p >= min_share
    xy = np.full((n_classes, 2), 0.5)
    for c in range(n_classes):
        if present[c]:
            ys, xs = np.nonzero(mask == c)
            xy[c] = (xs.mean() / max(w - 1, 1), ys.mean() / max(h - 1, 1))
    grid_p = np.zeros((grid, grid, n_classes))
    ys_e = np.linspace(0, h, grid + 1).astype(int)
    xs_e = np.linspace(0, w, grid + 1).astype(int)
    for gy in range(grid):
        for gx in range(grid):
            cell = mask[ys_e[gy]:ys_e[gy + 1], xs_e[gx]:xs_e[gx + 1]]
            cv = cell != ignore
            if cv.any():
                grid_p[gy, gx] = np.bincount(cell[cv].astype(np.int64), minlength=n_classes)[:n_classes] / cv.sum()
    return SceneVector(p, xy, present, grid_p)


def _zone(cx: float, cy: float, w: int, h: int) -> str:
    col = min(int(cx / (w / 3)), 2)
    row = min(int(cy / (h / 3)), 2)
    return ZONES[(row, col)]


def describe_scene(mask: np.ndarray, n_classes: int, min_share: float = 0.01, min_area: int = 100) -> dict:
    """Детерминированное описание: по каждому классу — доля, число объектов, крупнейший объект и его положение."""
    h, w = mask.shape
    total = max(int((mask != IGNORE_INDEX).sum()), 1)
    classes = []
    for c in range(n_classes):
        share = float((mask == c).sum() / total)
        if share < min_share:
            continue
        n, labels, stats, cents = cv2.connectedComponentsWithStats((mask == c).astype(np.uint8), connectivity=8)
        objs = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
        if not objs:
            continue
        big = max(objs, key=lambda i: stats[i, cv2.CC_STAT_AREA])
        cx, cy = cents[big]
        classes.append({
            "class": CLASS_NAMES[c],
            "share_percent": round(share * 100, 1),
            "n_objects": len(objs),
            "largest_bbox_xywh": [int(stats[big, cv2.CC_STAT_LEFT]), int(stats[big, cv2.CC_STAT_TOP]),
                                  int(stats[big, cv2.CC_STAT_WIDTH]), int(stats[big, cv2.CC_STAT_HEIGHT])],
            "largest_centroid_xy": [round(float(cx), 1), round(float(cy), 1)],
            "largest_zone": _zone(cx, cy, w, h),
        })
    classes.sort(key=lambda d: -d["share_percent"])
    lines = ["На изображении обнаружены следующие области:"]
    for d in classes:
        ru = CLASS_NAMES_RU[d["class"]].capitalize()
        lines.append(f"- {ru}: {d['share_percent']}% площади, объектов: {d['n_objects']}, "
                     f"крупнейший расположен {d['largest_zone']} изображения.")
    return {"image_height": h, "image_width": w, "classes": classes, "text": "\n".join(lines)}
