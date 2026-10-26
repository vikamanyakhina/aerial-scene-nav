"""Сводная таблица экспериментов: среднее ± std по сидам.   python -m scenenav.compare runs"""
import json
import sys
from pathlib import Path

import numpy as np

from scenenav.classes import CLASS_NAMES


def _fmt(vals):
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    if not vals:
        return "—"
    return f"{np.mean(vals):.3f} ± {np.std(vals):.3f}" if len(vals) > 1 else f"{vals[0]:.3f}"


def main(runs_dir="runs", part="test"):
    groups = {}
    for f in sorted(Path(runs_dir).glob("*/seed*/metrics.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        if part in r:
            groups.setdefault(r["experiment"], []).append(r[part])
    if not groups:
        print("Нет метрик в", runs_dir)
        return
    head = ["experiment", "n_seeds", "mIoU", "pixel_acc", "share_MAE"] + [f"IoU_{c}" for c in CLASS_NAMES]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for name, rs in groups.items():
        cls = [_fmt([(r["class_iou"][i]) for r in rs]) for i in range(len(CLASS_NAMES))]
        row = [name, str(len(rs)), _fmt([r["miou"] for r in rs]), _fmt([r["pixel_acc"] for r in rs]),
               _fmt([r["share_mae"] for r in rs])] + cls
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main(*(sys.argv[1:3]))
