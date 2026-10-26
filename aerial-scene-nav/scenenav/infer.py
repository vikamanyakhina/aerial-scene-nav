"""Предсказание на целом изображении скользящим окном с усреднением по перекрытию."""
import numpy as np
import torch

from scenenav.data.dataset import grid_starts


@torch.no_grad()
def predict_probs(model, x: torch.Tensor, window: int, device, stride: int = None, batch: int = 8) -> np.ndarray:
    """x: (C,H,W) -> вероятности (K,H,W) numpy."""
    model.eval()
    stride = stride or window // 2
    c, h, w = x.shape
    ph, pw = max(window - h, 0), max(window - w, 0)
    if ph or pw:
        x = torch.nn.functional.pad(x, (0, pw, 0, ph))
    _, H, W = x.shape
    coords = [(y, xx) for y in grid_starts(H, window, stride) for xx in grid_starts(W, window, stride)]
    acc, cnt = None, torch.zeros(H, W)
    for i in range(0, len(coords), batch):
        chunk = coords[i:i + batch]
        inp = torch.stack([x[:, y:y + window, xx:xx + window] for y, xx in chunk]).to(device)
        prob = torch.softmax(model(inp), dim=1).cpu()
        if acc is None:
            acc = torch.zeros(prob.shape[1], H, W)
        for (y, xx), p in zip(chunk, prob):
            acc[:, y:y + window, xx:xx + window] += p
            cnt[y:y + window, xx:xx + window] += 1
    return (acc / cnt)[:, :h, :w].numpy()


def predict_mask(model, x, window, device, stride=None) -> np.ndarray:
    return predict_probs(model, x, window, device, stride).argmax(0).astype(np.uint8)
