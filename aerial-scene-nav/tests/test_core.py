import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scenenav.classes import DUBAI_PALETTE, IGNORE_INDEX, NUM_CLASSES, UNLABELED_RGB, ids_to_rgb, rgb_mask_to_ids
from scenenav.config import in_channels_from_features, load_config, validate_config
from scenenav.data.prepare_dubai import check_split, prepare, tile_class_table
from scenenav.features.textures import TEXTURE_FUNCS
from scenenav.metrics import ConfusionMeter, class_shares, share_mae
from scenenav.scene.postprocess import clean_mask
from scenenav.scene.vector import describe_scene, scene_vector

ROOT = Path(__file__).resolve().parent.parent


class TestClasses(unittest.TestCase):
    def test_roundtrip(self):
        ids = np.random.default_rng(0).integers(0, NUM_CLASSES, (20, 30)).astype(np.uint8)
        self.assertTrue(np.array_equal(rgb_mask_to_ids(ids_to_rgb(ids)), ids))

    def test_off_by_one_hex_and_unlabeled(self):
        rgb = np.zeros((1, 3, 3), dtype=np.uint8)
        rgb[0, 0] = (133, 41, 246)         # land из другого источника (#8529F6)
        rgb[0, 1] = UNLABELED_RGB
        rgb[0, 2] = (0, 0, 0)              # неизвестный цвет
        ids = rgb_mask_to_ids(rgb)
        self.assertEqual(list(ids[0]), [1, IGNORE_INDEX, IGNORE_INDEX])


class TestTextures(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.img = rng.integers(0, 256, (64, 64, 3)).astype(np.uint8)

    def test_shape_dtype_determinism(self):
        for name, f in TEXTURE_FUNCS.items():
            a, b = f(self.img), f(self.img)
            self.assertEqual(a.shape, (64, 64), name)
            self.assertEqual(a.dtype, np.uint8, name)
            self.assertTrue(np.array_equal(a, b), name)

    def test_fixed_scale_not_per_image(self):
        flat = np.full((64, 64, 3), 128, dtype=np.uint8)
        flat[::2] += 4                      # очень слабая текстура
        for name in ("variance", "gabor"):
            self.assertLess(TEXTURE_FUNCS[name](flat).max(), 200, f"{name}: масштаб зависит от изображения")


class TestMetrics(unittest.TestCase):
    def test_absent_class_is_nan_and_ignore(self):
        m = ConfusionMeter(3)
        gt = np.array([[0, 0, 1, 255]])
        pred = np.array([[0, 1, 1, 2]])     # класс 2 есть только в pred на ignore-пикселе
        m.update(pred, gt)
        iou = m.iou()
        self.assertAlmostEqual(iou[0], 0.5)
        self.assertAlmostEqual(iou[1], 0.5)
        self.assertTrue(np.isnan(iou[2]))
        self.assertAlmostEqual(m.summary()["miou"], 0.5)

    def test_share_mae(self):
        gt = np.array([[0, 0, 1, 1]])
        self.assertEqual(share_mae(gt, gt, 2), 0.0)
        self.assertAlmostEqual(share_mae(np.array([[0, 0, 0, 1]]), gt, 2), 0.25)
        self.assertAlmostEqual(class_shares(np.array([[0, 255, 1, 1]]), 2).sum(), 1.0)


class TestScene(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((90, 90), dtype=np.uint8)
        self.mask[:, 45:] = 1               # правая половина — класс 1
        self.mask[10:12, 10:12] = 1         # шумовая точка

    def test_vector(self):
        sv = scene_vector(self.mask, 3, min_share=0.01)
        self.assertAlmostEqual(sv.p.sum(), 1.0)
        self.assertTrue(sv.present[0] and sv.present[1] and not sv.present[2])
        vec, w = sv.flat("p_xy")
        self.assertEqual(len(vec), 3 + 6)
        self.assertEqual(w[3 + 4], 0.0)     # координаты отсутствующего класса 2 не сравниваются
        self.assertEqual(sv.flat("grid")[0].shape, (27,))

    def test_clean_removes_speck(self):
        out = clean_mask(self.mask, 3, radius=1, min_area=20)
        self.assertEqual(out[10, 10], 0)
        self.assertEqual(out[50, 80], 1)

    def test_describe_is_deterministic(self):
        a, b = describe_scene(self.mask, 3), describe_scene(self.mask, 3)
        self.assertEqual(a["text"], b["text"])


class TestPrepareAndConfig(unittest.TestCase):
    def test_prepare_and_split_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw, out = Path(tmp) / "raw", Path(tmp) / "out"
            rng = np.random.default_rng(2)
            for tile in ("Tile 1", "Tile 2"):
                (raw / tile / "images").mkdir(parents=True)
                (raw / tile / "masks").mkdir(parents=True)
                ids = rng.integers(0, NUM_CLASSES, (40, 50)).astype(np.uint8)
                Image.fromarray(rng.integers(0, 256, (40, 50, 3)).astype(np.uint8)).save(raw / tile / "images" / "a.jpg")
                Image.fromarray(ids_to_rgb(ids)).save(raw / tile / "masks" / "a.png")
            m = prepare(raw, out, features=("lbp",))
            self.assertEqual(sorted(m["tile"]), ["Tile_1", "Tile_2"])
            self.assertTrue((out / "features" / "lbp" / "Tile_1__a.png").exists())
            saved = np.array(Image.open(out / "masks" / "Tile_1__a.png"))
            self.assertLessEqual(saved.max(), NUM_CLASSES - 1)
            self.assertEqual(tile_class_table(m).shape[0], 2)
            self.assertTrue(any("пересекаются" in w for w in check_split(m, ["Tile_1"], ["Tile_1"], ["Tile_2"])))
            self.assertTrue(any("неизвестные" in w for w in check_split(m, ["Tile_9"], ["Tile_1"], ["Tile_2"])))

    def test_all_configs_valid_and_channels(self):
        expected = {"rgb": 3, "rgb_lbp": 4, "rgb_entropy": 4, "rgb_variance": 4, "rgb_gabor": 4}
        for name, ch in expected.items():
            cfg = load_config(ROOT / "configs" / f"dubai_{name}.yaml")
            validate_config(cfg)
            self.assertEqual(in_channels_from_features(cfg["data"]["features"]), ch, name)
            self.assertEqual(cfg["train"]["epochs"], 40)   # наследование от base.yaml


if __name__ == "__main__":
    unittest.main()
