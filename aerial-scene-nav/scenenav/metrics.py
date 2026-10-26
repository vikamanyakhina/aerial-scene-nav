"""Метрики сегментации (numpy, без torch).

Отличия от прототипа:
 * пиксели с меткой ignore_index (255) не участвуют в подсчёте;
 * IoU класса, которого нет ни в разметке, ни в предсказании, равен NaN и не тянет среднее вниз;
 * добавлена метрика ошибки вектора долей классов — именно она важна для поиска по эталонной базе.
"""
import numpy as np

from scenenav.classes import IGNORE_INDEX


class ConfusionMeter:
    def __init__(self, n_classes: int, ignore_index: int = IGNORE_INDEX):
        self.n = n_classes
        self.ignore = ignore_index
        self.cm = np.zeros((n_classes, n_classes), dtype=np.int64)

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        pred = np.asarray(pred).reshape(-1).astype(np.int64)
        target = np.asarray(target).reshape(-1).astype(np.int64)
        valid = (target != self.ignore) & (target >= 0) & (target < self.n)
        idx = self.n * target[valid] + pred[valid]
        self.cm += np.bincount(idx, minlength=self.n ** 2).reshape(self.n, self.n)

    def iou(self) -> np.ndarray:
        tp = np.diag(self.cm).astype(np.float64)
        denom = self.cm.sum(0) + self.cm.sum(1) - tp
        out = np.full(self.n, np.nan)
        ok = denom > 0
        out[ok] = tp[ok] / denom[ok]
        return out

    def dice(self) -> np.ndarray:
        tp = np.diag(self.cm).astype(np.float64)
        denom = self.cm.sum(0) + self.cm.sum(1)
        out = np.full(self.n, np.nan)
        ok = denom > 0
        out[ok] = 2 * tp[ok] / denom[ok]
        return out

    def pixel_accuracy(self) -> float:
        total = self.cm.sum()
        return float(np.diag(self.cm).sum() / total) if total else float("nan")

    def summary(self) -> dict:
        iou, dice = self.iou(), self.dice()
        return {
            "miou": float(np.nanmean(iou)) if np.any(~np.isnan(iou)) else float("nan"),
            "mdice": float(np.nanmean(dice)) if np.any(~np.isnan(dice)) else float("nan"),
            "pixel_acc": self.pixel_accuracy(),
            "class_iou": [None if np.isnan(v) else float(v) for v in iou],
            "confusion_matrix": self.cm.tolist(),
        }


def class_shares(mask: np.ndarray, n_classes: int, ignore_index: int = IGNORE_INDEX) -> np.ndarray:
    """Доли классов по валидным пикселям (сумма = 1)."""
    valid = mask != ignore_index
    total = valid.sum()
    if total == 0:
        return np.zeros(n_classes)
    counts = np.bincount(mask[valid].astype(np.int64), minlength=n_classes)[:n_classes]
    return counts / total


def share_mae(pred: np.ndarray, gt: np.ndarray, n_classes: int,
              ignore_index: int = IGNORE_INDEX) -> float:
    """Средняя абсолютная ошибка долей классов (в долях, 0.05 = 5 п.п.); пиксели ignore в gt исключаются."""
    valid = gt != ignore_index
    return float(np.abs(class_shares(pred[valid], n_classes, ignore_index)
                        - class_shares(gt[valid], n_classes, ignore_index)).mean())
