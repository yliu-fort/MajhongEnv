# build_zarr_dataset.py
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".", "src"))

import pickle, numpy as np
import torch
from typing import List, Optional, Sequence, Dict, Tuple, Iterable, Generator, Union, IO, Any
import xml.etree.ElementTree as ET
import sqlite3
import json, math, numpy as np
import zarr
from numcodecs import Blosc
from mahjong_features import RiichiResNetFeatures
from tenhou_to_mahjong import iter_discard_states

from tqdm import tqdm

TagLike = Union[str, bytes, os.PathLike, IO[bytes], IO[str]]




class RiichiZarrDatasetBuilder:
    def __init__(
        self, shape, squared=True,  # images:(N,C,H)  labels:(N,)  masks:(N,1,H)或(N,H), W=H
        out_dir="output/dataset_zarr",
        chunk_n=2048, chunk_hw=256,            # 分块参数：每块多少样本、空间块大小
        compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE),
        meta=None, splits=None, overwrite=True, dtype=torch.uint8
    ):
        N, C, H = shape

        if meta is None:
            meta = {"channel_names": [f"ch{i}" for i in range(C)],
                    "classes": None, "norm": "zscore"}

        if overwrite and os.path.exists(out_dir):
            import shutil; shutil.rmtree(out_dir)

        # 选一组适合的块尺度（按需要微调）
        chunks_img = (min(chunk_n, N), C, min(chunk_hw, H))
        chunks_msk = (min(chunk_n, N), min(chunk_hw, H))

        root = zarr.open_group(out_dir, mode="w")
        self.z_imgs = root.create_dataset(
            "images", shape=(N, C, H), chunks=chunks_img, dtype=dtype, compressor=compressor
        )
        self.z_lbls = root.create_dataset(
            "labels", shape=(N,), chunks=(min(chunk_n, N),), dtype=dtype, compressor=None
        )
        self.z_msks = root.create_dataset(
            "masks", shape=(N, H), chunks=chunks_msk, dtype=dtype, compressor=compressor
        )

        # 元信息 + 切分（可选）
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        if splits:
            with open(os.path.join(out_dir, "splits.json"), "w", encoding="utf-8") as f:
                json.dump({k: list(map(int, v)) for k, v in splits.items()}, f, indent=2)

        print(f"[OK] Zarr dataset built at: {out_dir}")
    
    def write_data(self, images, labels, masks, start=0, length=100):
        pass




def collect_discard_samples(xml: TagLike, extractor: RiichiResNetFeatures):
    """Return a list of **lightweight dicts** for easy JSON/pickle dumping.
    (If you want full dataclasses, iterate `iter_discard_states` directly.)
    """
    images = []
    labels = []
    masks = []

    for st, who, disc_t34, meta in iter_discard_states(xml):
        with torch.no_grad():
            out = extractor(st)
        legal_mask = out["legal_mask"].cpu().numpy().astype(np.uint8)
        x = out["x"][...,0].cpu().numpy().astype(np.uint8) # remove the broadcasted rows
        y = disc_t34
        # Save x, y and legal_mask like a image dataset like CIFAR-10
        images.append(x)
        labels.append(y)
        masks.append(legal_mask)

    return images, labels, masks

def read_xmls(xmls: List[TagLike]):
    """Read a list of xml strings/paths and return stacked numpy arrays."""
    imageset: List[np.ndarray] = []
    labelset: List[int] = []
    maskset: List[np.ndarray] = []

    extractor = RiichiResNetFeatures()
    
    for xml in xmls:
        images, labels, masks = collect_discard_samples(xml, extractor)
        imageset.extend(images)
        labelset.extend(labels)
        maskset.extend(masks)

    if imageset:
        images_arr = np.stack(imageset)
        labels_arr = np.asarray(labelset)
        masks_arr = np.stack(maskset)
    else:
        images_arr = np.empty((0,), dtype=np.float32)
        labels_arr = np.empty((0,), dtype=np.int64)
        masks_arr = np.empty((0,), dtype=np.uint8)

    return images_arr, labels_arr, masks_arr

# Count logs in sqlite database /logs
# with entries: log_id, date, is_tonpusen, is_sanma, is_speed, is_processed, was_error, log_content
def count_xmls_in_database(db, is_tonpusen=False, is_sanma=False, is_speed=False, is_processed=True, was_error=False):
    """Return the number of logs in the sqlite database matching filters."""
    conn = sqlite3.connect(db)
    try:
        cur = conn.cursor()
        query = "SELECT COUNT(*) FROM logs WHERE 1=1"
        params: List[int] = []

        def add_filter(column: str, value: Optional[bool]):
            nonlocal query
            if value is not None:
                query += f" AND {column} = ?"
                params.append(int(value))

        add_filter("is_tonpusen", is_tonpusen)
        add_filter("is_sanma", is_sanma)
        add_filter("is_speed", is_speed)
        add_filter("is_processed", is_processed)
        add_filter("was_error", was_error)

        cur.execute(query, params)
        (count,) = cur.fetchone()
        return int(count)
    finally:
        conn.close()

# Return logs (as xmls) from sqlite database /logs
def fetch_xmls_from_database(db, num_examples=100, start=0, is_tonpusen=False, is_sanma=False, is_speed=False, is_processed=True, was_error=False):
    """Fetch log XML strings from the database according to filters.

    Args:
        db: Path to the sqlite database.
        num_examples: Number of logs to fetch.
        start: Offset from which to start returning logs.
        is_tonpusen/is_sanma/is_speed/is_processed/was_error: Filtering flags
            corresponding to columns in the ``logs`` table.

    Returns:
        A list of XML strings from the ``log_content`` column.
    """
    conn = sqlite3.connect(db)
    try:
        cur = conn.cursor()
        query = "SELECT log_content FROM logs WHERE 1=1"
        params: List[int] = []

        def add_filter(column: str, value: Optional[bool]):
            nonlocal query
            if value is not None:
                query += f" AND {column} = ?"
                params.append(int(value))

        add_filter("is_tonpusen", is_tonpusen)
        add_filter("is_sanma", is_sanma)
        add_filter("is_speed", is_speed)
        add_filter("is_processed", is_processed)
        add_filter("was_error", was_error)

        query += " ORDER BY log_id LIMIT ? OFFSET ?"
        params.extend([num_examples, start])
        print("Fetch logs from database...")
        cur.execute(query, params)
        rows = cur.fetchall()
        print("OK")
        return [row[0] for row in rows]
    finally:
        conn.close()

def main():
    db = 'data/2016.db'
    total_logs = count_xmls_in_database(db)
    total_logs = 1000
    print(total_logs)
    dataset = RiichiZarrDatasetBuilder((1000000,29,34))
    sql_batch_size = 100
    for i in tqdm(np.arange(total_logs//sql_batch_size+1)*sql_batch_size):
        xmls = fetch_xmls_from_database(db, start=i, num_examples=sql_batch_size)
        print(len(xmls))
        images, labels, masks = read_xmls(xmls)
        dataset.write_data(images, labels, masks, start=i, length=sql_batch_size)
    

if __name__ == "__main__":
    main()