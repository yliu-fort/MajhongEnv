import os
import sys
import numpy as np
import zarr

base = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(base)
sys.path.append(os.path.join(base, "src"))

from dataset_builder_zarr import RiichiZarrDatasetBuilder
from mahjong_features import NUM_FEATURES, NUM_TILES


def test_write_data_to_zarr(tmp_path):
    out_dir = tmp_path / "zarr_ds"
    builder = RiichiZarrDatasetBuilder(shape=(NUM_FEATURES, NUM_TILES), out_dir=str(out_dir))

    imgs = np.zeros((3, NUM_FEATURES, NUM_TILES), dtype=np.uint8)
    lbls = np.array([0, 1, 2], dtype=np.uint8)
    msks = np.ones((3, NUM_TILES), dtype=np.uint8)

    n = builder.write_data(imgs, lbls, msks)
    assert n == 3

    g = zarr.open_group(str(out_dir), mode="r")
    assert g["images"].shape[0] == 3
    np.testing.assert_array_equal(g["labels"][:], lbls)
    g.store.close() if hasattr(g.store, "close") else None
