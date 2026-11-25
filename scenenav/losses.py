"""CE + Dice с поддержкой ignore_index и весов классов."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from scenenav.classes import IGNORE_INDEX


class DiceLoss(nn.Module):
    def __init__(self, ignore_index: int = IGNORE_INDEX, smooth: float = 1e-6):
        super().__init__()
        self.ignore, self.smooth = ignore_index, smooth

    def forward(self, logits, target):
        n = logits.shape[1]
        valid = (target != self.ignore).unsqueeze(1).float()
        probs = torch.softmax(logits, dim=1) * valid
        onehot = F.one_hot(target.clamp(max=n - 1) * (target != self.ignore), n).permute(0, 3, 1, 2).float() * valid
        inter = (probs * onehot).sum(dim=(0, 2, 3))
        card = probs.sum(dim=(0, 2, 3)) + onehot.sum(dim=(0, 2, 3))
        present = onehot.sum(dim=(0, 2, 3)) > 0  # классы, отсутствующие в батче, не учитываем
        dice = (2 * inter + self.smooth) / (card + self.smooth)
        return 1.0 - dice[present].mean() if present.any() else logits.sum() * 0.0


class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=1.0, dice_weight=1.0, class_weights=None, ignore_index=IGNORE_INDEX):
        super().__init__()
        w = None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32)
        self.ce = nn.CrossEntropyLoss(weight=w, ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index)
        self.cw, self.dw = ce_weight, dice_weight

    def forward(self, logits, target):
        return self.cw * self.ce(logits, target) + self.dw * self.dice(logits, target)


def auto_class_weights(counts) -> list:
    """Веса ~ 1/sqrt(частоты), нормированные к среднему 1 (схема 'Sqrt' из литературы по дрон-датасетам)."""
    import numpy as np
    freq = np.asarray(counts, dtype=np.float64)
    freq = np.maximum(freq / freq.sum(), 1e-6)
    w = 1.0 / np.sqrt(freq)
    return (w / w.mean()).tolist()
