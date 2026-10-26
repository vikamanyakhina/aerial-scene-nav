"""Подготовка датасета Dubai: RGB-маски -> индексы классов, текстурные карты, manifest.csv.

Ожидаемая структура сырых данных (как в архиве Kaggle):
    <raw>/Tile 1/images/*.jpg   и   <raw>/Tile 1/masks/*.png   (имена файлов совпадают по stem)

Результат:
    <out>/images/<id>.png            RGB
    <out>/masks/<id>.png             индексы классов 0..4, 255 = Unlabeled
    <out>/features/<имя>/<id>.png    текстурные карты (uint8, фиксированный масштаб)
    <out>/manifest.csv               id, tile, height, width, число пикселей по классам

Запуск:  python -m scenenav.data.prepare_dubai --raw "<путь>" --out data/processed/dubai
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from scenenav.classes import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES, rgb_mask_to_ids
from scenenav.features.textures import TEXTURE_FUNCS

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _safe(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_-]+", "_", name).strip("_")


def find_pairs(raw: Path):
    """Пары (tile, stem, путь_к_изображению, путь_к_маске)."""
    pairs = []
    for img_dir in sorted(raw.rglob("images")):
        if not img_dir.is_dir():
            continue
        mask_dir = img_dir.parent / "masks"
        if not mask_dir.is_dir():
            continue
        masks = {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() in IMG_EXT}
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() in IMG_EXT and p.stem in masks:
                pairs.append((_safe(img_dir.parent.name), _safe(p.stem), p, masks[p.stem]))
    return pairs


def prepare(raw: Path, out: Path, features=tuple(TEXTURE_FUNCS), mask_tol: float = 30.0) -> pd.DataFrame:
    pairs = find_pairs(Path(raw))
    if not pairs:
        raise FileNotFoundError(f"В {raw} не найдено пар images/masks")
    out = Path(out)
    for sub in ["images", "masks"] + [f"features/{f}" for f in features]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    rows = []
    for tile, stem, img_path, mask_path in pairs:
        uid = f"{tile}__{stem}"
        rgb = np.array(Image.open(img_path).convert("RGB"))
        mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
        if rgb.shape[:2] != mask_rgb.shape[:2]:
            raise ValueError(f"{uid}: размеры изображения {rgb.shape[:2]} и маски {mask_rgb.shape[:2]} не совпадают")
        ids = rgb_mask_to_ids(mask_rgb, tol=mask_tol)

        Image.fromarray(rgb).save(out / "images" / f"{uid}.png")
        Image.fromarray(ids).save(out / "masks" / f"{uid}.png")
        for f in features:
            Image.fromarray(TEXTURE_FUNCS[f](rgb)).save(out / "features" / f / f"{uid}.png")

        counts = np.bincount(ids.reshape(-1), minlength=256)
        row = {"id": uid, "tile": tile, "height": rgb.shape[0], "width": rgb.shape[1]}
        row.update({f"px_{name}": int(counts[i]) for i, name in enumerate(CLASS_NAMES)})
        row["px_ignore"] = int(counts[IGNORE_INDEX])
        rows.append(row)
        print(f"{uid}: {rgb.shape[1]}x{rgb.shape[0]}")

    manifest = pd.DataFrame(rows)
    manifest.to_csv(out / "manifest.csv", index=False)
    return manifest


def tile_class_table(manifest: pd.DataFrame) -> pd.DataFrame:
    """Доли классов (в %) по тайлам — по ней выбирается разбиение train/val/test."""
    cols = [f"px_{n}" for n in CLASS_NAMES]
    g = manifest.groupby("tile")[cols].sum()
    return (g.div(g.sum(axis=1), axis=0) * 100).round(1).rename(columns=lambda c: c[3:])


def check_split(manifest: pd.DataFrame, train, val, test, min_pct: float = 1.0) -> list:
    """Предупреждения: пересечения тайлов между частями и классы с долей < min_pct% в какой-либо части."""
    warns = []
    sets = {"train": set(train), "val": set(val), "test": set(test)}
    for a in sets:
        for b in sets:
            if a < b and sets[a] & sets[b]:
                warns.append(f"Тайлы пересекаются между {a} и {b}: {sorted(sets[a] & sets[b])}")
    known = set(manifest["tile"])
    for name, s in sets.items():
        if s - known:
            warns.append(f"{name}: неизвестные тайлы {sorted(s - known)}; есть {sorted(known)}")
    cols = [f"px_{n}" for n in CLASS_NAMES]
    for name, s in sets.items():
        sub = manifest[manifest["tile"].isin(s)][cols].sum()
        if sub.sum() == 0:
            continue
        for c, v in (sub / sub.sum() * 100).items():
            if v < min_pct:
                warns.append(f"{name}: класс '{c[3:]}' занимает {v:.2f}% — метрика по нему будет шумной")
    return warns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", default="data/processed/dubai")
    ap.add_argument("--features", nargs="*", default=list(TEXTURE_FUNCS))
    args = ap.parse_args()
    m = prepare(Path(args.raw), Path(args.out), tuple(args.features))
    print(f"\nГотово: {len(m)} изображений, тайлов: {m['tile'].nunique()}")
    print("\nДоли классов по тайлам, %:")
    print(tile_class_table(m).to_string())
    print("\nИспользуйте эту таблицу для выбора data.train_tiles / val_tiles / test_tiles в конфигах.")


if __name__ == "__main__":
    main()
