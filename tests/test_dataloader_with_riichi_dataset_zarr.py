import os
import sys
import numpy as np
import zarr
import torch
from torch.utils.data import DataLoader

base = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(base)
sys.path.append(os.path.join(base, "src"))

from dataset_loader import RiichiDatasetZarr
from mahjong_features import NUM_FEATURES, NUM_TILES


def test_dataloader_with_riichi_dataset_zarr(tmp_path):
    root = tmp_path / "zarr_ds"
    g = zarr.open_group(str(root), mode="w")
    imgs = np.zeros((3, NUM_FEATURES, NUM_TILES), dtype=np.uint8)
    lbls = np.array([1, 2, 3], dtype=np.uint8)
    msks = np.ones((3, NUM_TILES), dtype=np.uint8)
    fns = np.array(["a", "b", "c"], dtype="U1")

    g.create_array("images", data=imgs)
    g.create_array("labels", data=lbls)
    g.create_array("masks", data=msks)
    g.create_array("filenames", data=fns)

    ds = RiichiDatasetZarr(str(root))
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    batches = list(loader)
    assert len(batches) == 2

    x0, y0, m0 = batches[0]
    assert x0.shape == (2, NUM_FEATURES, NUM_TILES, NUM_TILES)
    assert torch.equal(y0, torch.tensor([1, 2]))
    assert m0.shape == (2, NUM_TILES)

    x1, y1, m1 = batches[1]
    assert x1.shape == (1, NUM_FEATURES, NUM_TILES, NUM_TILES)
    assert y1.item() == 3
    assert m1.shape == (1, NUM_TILES)
