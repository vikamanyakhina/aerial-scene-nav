"""Схема классов датасета Dubai (Humans in the Loop / MBRSC) и перевод RGB-масок в индексы.

Маски в датасете цветные (RGB), поэтому их нужно перевести в карту индексов классов.
HEX-значения в разных источниках отличаются на единицу (например, #8429F6 и #8529F6),
поэтому сопоставление идёт по ближайшему цвету с допуском, а не по точному равенству.
"""
import numpy as np

IGNORE_INDEX = 255

# Порядок задаёт индекс класса 0..4: building, land, road, vegetation, water.
# В описании датасета (Humans in the Loop) и в classes.json из архива указаны РАЗНЫЕ цвета:
# classes.json — цвета инструмента разметки, цвета самих масок могут быть другими.
# Поэтому поддерживаются обе палитры, а prepare_dubai выбирает подходящую автоматически.
PALETTES = {
    "hil": [  # цвета из описания датасета
        ("building", (60, 16, 152)),
        ("land", (132, 41, 246)),
        ("road", (110, 193, 228)),
        ("vegetation", (254, 221, 58)),
        ("water", (226, 169, 41)),
    ],
    "classes_json": [  # цвета из classes.json в архиве датасета
        ("building", (208, 2, 27)),
        ("land", (245, 166, 35)),
        ("road", (222, 89, 127)),
        ("vegetation", (65, 117, 5)),
        ("water", (80, 227, 194)),
    ],
}
DUBAI_PALETTE = PALETTES["hil"]
UNLABELED_RGB = (155, 155, 155)

CLASS_NAMES = [name for name, _ in DUBAI_PALETTE]
NUM_CLASSES = len(CLASS_NAMES)

# Названия для текстового описания сцены.
CLASS_NAMES_RU = {
    "building": "здания",
    "land": "открытая местность",
    "road": "дороги",
    "vegetation": "растительность",
    "water": "водные объекты",
}


def _distances(rgb: np.ndarray, palette_name: str) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"Ожидалась RGB-маска (H,W,3), получено {rgb.shape}")
    flat = rgb[..., :3].reshape(-1, 3).astype(np.float32)
    palette = [c for _, c in PALETTES[palette_name]] + [UNLABELED_RGB]
    return np.stack([np.sqrt(((flat - np.array(c, dtype=np.float32)) ** 2).sum(axis=1)) for c in palette])


def unknown_color_fraction(rgb: np.ndarray, palette_name: str = "hil", tol: float = 30.0) -> float:
    """Доля пикселей, цвет которых не совпадает ни с одним классом палитры (и не Unlabeled)."""
    return float((_distances(rgb, palette_name).min(axis=0) > tol).mean())


def rgb_mask_to_ids(rgb: np.ndarray, tol: float = 30.0, palette_name: str = "hil") -> np.ndarray:
    """RGB-маска (H,W,3) -> индексы классов (H,W) uint8; Unlabeled и неизвестные цвета -> 255."""
    dist = _distances(rgb, palette_name)
    nearest = dist.argmin(axis=0)
    ids = np.where(nearest < NUM_CLASSES, nearest, IGNORE_INDEX).astype(np.uint8)
    ids[dist.min(axis=0) > tol] = IGNORE_INDEX
    return ids.reshape(rgb.shape[:2])


def ids_to_rgb(ids: np.ndarray) -> np.ndarray:
    """Индексы классов -> цветная картинка для визуализации."""
    out = np.zeros(ids.shape + (3,), dtype=np.uint8)
    out[:] = UNLABELED_RGB
    for i, (_, color) in enumerate(DUBAI_PALETTE):
        out[ids == i] = color
    return out
