"""Схема классов датасета Dubai (Humans in the Loop / MBRSC) и перевод RGB-масок в индексы.

Маски в датасете цветные (RGB), поэтому их нужно перевести в карту индексов классов.
HEX-значения в разных источниках отличаются на единицу (например, #8429F6 и #8529F6),
поэтому сопоставление идёт по ближайшему цвету с допуском, а не по точному равенству.
"""
import numpy as np

IGNORE_INDEX = 255

# (имя, RGB). Порядок задаёт индекс класса 0..4.
DUBAI_PALETTE = [
    ("building", (60, 16, 152)),
    ("land", (132, 41, 246)),
    ("road", (110, 193, 228)),
    ("vegetation", (254, 221, 58)),
    ("water", (226, 169, 41)),
]
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


def rgb_mask_to_ids(rgb: np.ndarray, tol: float = 30.0) -> np.ndarray:
    """RGB-маска (H,W,3) -> индексы классов (H,W) uint8; Unlabeled и неизвестные цвета -> 255."""
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"Ожидалась RGB-маска (H,W,3), получено {rgb.shape}")
    flat = rgb[..., :3].reshape(-1, 3).astype(np.float32)
    palette = [c for _, c in DUBAI_PALETTE] + [UNLABELED_RGB]
    dist = np.stack(
        [np.sqrt(((flat - np.array(c, dtype=np.float32)) ** 2).sum(axis=1)) for c in palette]
    )
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
