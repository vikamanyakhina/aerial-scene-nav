"""Обучение одного эксперимента.

    python -m scenenav.train --config configs/dubai_rgb_lbp.yaml --seed 42

Ключевые гарантии (в отличие от прототипа):
 * число входных каналов выводится из data.features и проверяется на реальном батче;
 * seed выставляется в начале КАЖДОГО запуска;
 * шаг планировщика вызывается без аргументов;
 * метрики считаются с ignore_index и NaN для отсутствующих классов;
 * в чекпоинт сохраняется полный конфиг эксперимента.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from scenenav.classes import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES
from scenenav.config import in_channels_from_features, load_config, validate_config
from scenenav.data.dataset import WindowDataset
from scenenav.data.prepare_dubai import check_split
from scenenav.infer import predict_mask
from scenenav.losses import CombinedLoss, auto_class_weights
from scenenav.metrics import ConfusionMeter, share_mae
from scenenav.models.factory import build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ids_for_tiles(manifest: pd.DataFrame, tiles) -> list:
    return manifest[manifest["tile"].isin(tiles)]["id"].tolist()


@torch.no_grad()
def eval_windows(model, loader, criterion, device) -> dict:
    model.eval()
    meter, loss_sum, n = ConfusionMeter(NUM_CLASSES), 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += criterion(out, y).item() * len(x)
        n += len(x)
        meter.update(out.argmax(1).cpu().numpy(), y.cpu().numpy())
    s = meter.summary()
    s["loss"] = loss_sum / max(n, 1)
    return s


@torch.no_grad()
def eval_full(model, ds: WindowDataset, window: int, device) -> dict:
    """Оценка на целых изображениях (скользящее окно) + ошибка вектора долей классов."""
    meter, maes = ConfusionMeter(NUM_CLASSES), []
    for k in range(len(ds.ids)):
        x, gt = ds.full_input(k)
        pred = predict_mask(model, x, window, device)
        meter.update(pred, gt)
        maes.append(share_mae(pred, gt, NUM_CLASSES))
    s = meter.summary()
    s["share_mae"] = float(np.mean(maes))
    return s


def main(cfg_path: str, seed: int = None, epochs: int = None) -> Path:
    cfg = load_config(cfg_path)
    if seed is not None:
        cfg["seed"] = seed
    if epochs is not None:
        cfg["train"]["epochs"] = epochs
    validate_config(cfg)
    seed_everything(cfg["seed"])

    d, t = cfg["data"], cfg["train"]
    root = Path(d["processed_dir"])
    manifest = pd.read_csv(root / "manifest.csv")
    for w in check_split(manifest, d["train_tiles"], d["val_tiles"], d["test_tiles"]):
        print("ПРЕДУПРЕЖДЕНИЕ:", w)

    features = d["features"]
    train_ds = WindowDataset(root, ids_for_tiles(manifest, d["train_tiles"]), features, d["window"],
                             train=True, samples_per_epoch=d["samples_per_epoch"], seed=cfg["seed"])
    val_ds = WindowDataset(root, ids_for_tiles(manifest, d["val_tiles"]), features, d["window"], train=False)
    test_ids = ids_for_tiles(manifest, d["test_tiles"])
    test_ds = WindowDataset(root, test_ids, features, d["window"], train=False) if test_ids else None

    device = get_device()
    in_ch = in_channels_from_features(features)
    model = build_model(cfg, in_ch, NUM_CLASSES).to(device)

    run_dir = Path(cfg["output_dir"]) / cfg["experiment"] / f"seed{cfg['seed']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    # --- проверка согласованности каналов на реальном батче ---
    x0, y0 = train_ds[0]
    assert x0.shape[0] == in_ch == train_ds.in_channels, f"Каналы: батч {x0.shape[0]}, модель {in_ch}"
    print(f"Эксперимент: {cfg['experiment']} | features={features} | входных каналов: {in_ch} | устройство: {device}")

    cw = cfg["loss"].get("class_weights")
    if cw == "auto":
        cols = [f"px_{n}" for n in CLASS_NAMES]
        cw = auto_class_weights(manifest[manifest["tile"].isin(d["train_tiles"])][cols].sum().values)
        print("Веса классов:", [round(v, 2) for v in cw])
    criterion = CombinedLoss(cfg["loss"]["ce"], cfg["loss"]["dice"], cw).to(device)

    tl = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=False, num_workers=t["num_workers"],
                    drop_last=True)
    vl = DataLoader(val_ds, batch_size=t["batch_size"], shuffle=False, num_workers=t["num_workers"])
    opt = AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = CosineAnnealingLR(opt, T_max=t["epochs"], eta_min=1e-6)

    best, bad, history = -1.0, 0, []
    for epoch in range(1, t["epochs"] + 1):
        train_ds.set_epoch(epoch)
        model.train()
        t0, run_loss = time.time(), 0.0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            run_loss += loss.item()
        sched.step()  # без аргументов
        val = eval_windows(model, vl, criterion, device)
        row = {"epoch": epoch, "train_loss": run_loss / max(len(tl), 1), "val_loss": val["loss"],
               "val_miou": val["miou"], "val_pixel_acc": val["pixel_acc"],
               "lr": opt.param_groups[0]["lr"], "sec": time.time() - t0}
        history.append(row)
        print(" | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        score = val["miou"] if not np.isnan(val["miou"]) else -1.0
        if score > best:
            best, bad = score, 0
            torch.save({"model": model.state_dict(), "config": cfg, "in_channels": in_ch,
                        "class_names": CLASS_NAMES, "epoch": epoch}, run_dir / "best.pt")
        else:
            bad += 1
            if bad >= t["patience"]:
                print("Ранняя остановка")
                break
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    result = {"experiment": cfg["experiment"], "seed": cfg["seed"], "features": features,
              "in_channels": in_ch, "best_epoch": ckpt["epoch"],
              "val": eval_full(model, val_ds, d["window"], device)}
    if test_ds is not None:
        result["test"] = eval_full(model, test_ds, d["window"], device)
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"val mIoU={result['val']['miou']:.4f} share_mae={result['val']['share_mae']:.4f}")
    if "test" in result:
        print(f"test mIoU={result['test']['miou']:.4f} share_mae={result['test']['share_mae']:.4f}")
    return run_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    main(a.config, a.seed, a.epochs)
