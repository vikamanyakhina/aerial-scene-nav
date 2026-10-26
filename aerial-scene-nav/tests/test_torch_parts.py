"""Тесты частей, требующих torch. Пропускаются, если torch не установлен."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

if HAVE_TORCH:
    from scenenav.data.dataset import WindowDataset, grid_starts
    from scenenav.losses import CombinedLoss
    from scenenav.models.unet import UNet


@unittest.skipUnless(HAVE_TORCH, "torch не установлен")
class TestTorchParts(unittest.TestCase):
    def _fake_root(self, tmp):
        root = Path(tmp)
        for sub in ("images", "masks", "features/lbp"):
            (root / sub).mkdir(parents=True)
        rng = np.random.default_rng(0)
        Image.fromarray(rng.integers(0, 256, (100, 120, 3)).astype(np.uint8)).save(root / "images" / "a.png")
        Image.fromarray(rng.integers(0, 5, (100, 120)).astype(np.uint8)).save(root / "masks" / "a.png")
        Image.fromarray(rng.integers(0, 256, (100, 120)).astype(np.uint8)).save(root / "features" / "lbp" / "a.png")
        return root

    def test_channels_match_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._fake_root(tmp)
            for feats, ch in ((["rgb"], 3), (["rgb", "lbp"], 4)):
                ds = WindowDataset(root, ["a"], feats, window=64, train=True, samples_per_epoch=4)
                x, y = ds[0]
                self.assertEqual(x.shape, (ch, 64, 64))
                self.assertEqual(ds.in_channels, ch)
                self.assertEqual(y.shape, (64, 64))
                ev = WindowDataset(root, ["a"], feats, window=64, train=False)
                self.assertEqual(ev[0][0].shape[0], ch)

    def test_grid_covers_edges(self):
        self.assertEqual(grid_starts(100, 64, 64), [0, 36])

    def test_unet_and_loss(self):
        for ch in (3, 4):
            net = UNet(ch, 5, base_channels=8)
            x = torch.randn(2, ch, 64, 64)
            out = net(x)
            self.assertEqual(out.shape, (2, 5, 64, 64))
        y = torch.randint(0, 5, (2, 64, 64))
        y[:, :8] = 255
        loss = CombinedLoss(class_weights=[1, 1, 1, 1, 1])(out, y)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
