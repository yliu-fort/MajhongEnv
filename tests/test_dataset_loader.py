import os
import sys
import numpy as np
import zarr

base = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(base)
sys.path.append(os.path.join(base, "src"))

from dataset_loader import RiichiDatasetZarr
from mahjong_features import NUM_FEATURES, NUM_TILES

import unittest

class TestRiichiDatasetZarrBasic(unittest.TestCase):
    def test_riichi_dataset_zarr_basic(self):
        tmp_path = 'tmp'
        root = os.path.join(tmp_path, "zarr_ds1")
        if os.path.exists(root):
            import shutil; shutil.rmtree(root)
        g = zarr.open_group(str(root), mode="w")
        imgs = np.zeros((2, NUM_FEATURES, NUM_TILES), dtype=np.uint8)
        lbls = np.array([1, 2], dtype=np.uint8)
        msks = np.ones((2, NUM_TILES), dtype=np.uint8)
        fns = np.array(["a", "b"], dtype="U1")

        g.create_array("images", data=imgs)
        g.create_array("labels", data=lbls)
        g.create_array("masks", data=msks)
        g.create_array("filenames", data=fns)

        ds = RiichiDatasetZarr(str(root))
        assert len(ds) == 2

        x, y, m = ds[0]
        self.assertEqual(x.shape, (NUM_FEATURES, NUM_TILES, NUM_TILES))
        self.assertEqual(y, 1)
        self.assertEqual(m.shape, (NUM_TILES,))
