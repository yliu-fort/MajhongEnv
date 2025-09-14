# build_zarr_dataset.py
import os, sys, psutil
sys.path.append(os.path.join(os.path.dirname(__file__), ".", "src"))

import numpy as np
import torch
from typing import List, Optional, Sequence, Dict, Tuple, Iterable, Generator, Union, IO, Any
import xml.etree.ElementTree as ET
import sqlite3
import json, math
import zarr
from zarr.codecs import BloscCodec, BloscCname, BloscShuffle  # v3 codecs
from mahjong_features import RiichiResNetFeatures
from tenhou_to_mahjong import (
    iter_discard_states,
    iter_chi_states,
    iter_pon_states,
    iter_kan_states,
    iter_riichi_states,
)

from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

TagLike = Union[str, bytes, os.PathLike, IO[bytes], IO[str]]

# make_splits.py
from sklearn.model_selection import train_test_split

def make_holdout_split(labels, test_size=0.2, val_size=0.1, seed=42):
    N = labels.shape[0]; idx = np.arange(N)
    idx_trv, idx_te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=labels)
    val_size_adj = val_size / (1 - test_size)
    idx_tr, idx_va = train_test_split(idx_trv, test_size=val_size_adj, random_state=seed, stratify=labels[idx_trv])
    return {"train": idx_tr, "val": idx_va, "test": idx_te}

# 示例：
# splits = make_holdout_split(labels)
# json.dump({k: list(map(int,v)) for k,v in splits.items()}, open("splits.json","w"))


class RiichiZarrDatasetBuilder:
    def __init__(
        self, shape, squared=True,  # images:(N,C,H)  labels:(N,)  masks:(N,1,H)或(N,H), W=H
        out_dir="output/dataset_zarr",
        chunk_n=2048, chunk_hw=256,            # 分块参数：每块多少样本、空间块大小
        compressors=BloscCodec(cname=BloscCname.zstd, clevel=3, shuffle=BloscShuffle.bitshuffle),
        meta=None, splits=None, overwrite=True, dtype="uint8"
    ):
        self.shape = shape
        C, H = shape
        self.out_dir=out_dir

        if meta is None:
            meta = {"channel_names": [f"ch{i}" for i in range(C)],
                    "classes": None, "norm": "zscore"}

        if overwrite and os.path.exists(out_dir):
            import shutil; shutil.rmtree(out_dir)

        # 选一组适合的块尺度（按需要微调）
        chunks_img = (chunk_n, C, min(chunk_hw, H))
        chunks_msk = (chunk_n, min(chunk_hw, H))

        root = zarr.open_group(out_dir, mode="a")
        self.z_imgs = root.require_array(
            "images", shape=(0, C, H), chunks=chunks_img, dtype=dtype, compressors=None
        )
        self.z_lbls = root.require_array(
            "labels", shape=(0,), chunks=(chunk_n,), dtype=dtype, compressors=None
        )
        self.z_msks = root.require_array(
            "masks", shape=(0, H), chunks=chunks_msk, dtype=dtype, compressors=None
        )
        self.root = root

        # 元信息 + 切分（可选）
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"[OK] Zarr dataset built at: {out_dir}")
    
    def write_splits(self):
        splits = make_holdout_split(self.z_lbls, test_size=0.2, val_size=0.1, seed=42)
        with open(os.path.join(self.out_dir, "splits.json"), "w", encoding="utf-8") as f:
            json.dump({k: list(map(int, v)) for k, v in splits.items()}, f, indent=2)

    def write_data(self, images, labels, masks, start: int = 0, length: Optional[int] = None) -> int:
        """Write a batch of samples into the zarr datasets.

        Parameters
        ----------
        images, labels, masks: array-like
            Data to be written.  ``images`` is expected to have shape
            ``(N, C, H)``; ``labels`` has shape ``(N,)`` and ``masks``
            has shape ``(N, H)``.  ``start`` specifies the starting
            index in the underlying datasets.  ``length`` optionally
            limits how many samples from the inputs should be written.

        Returns
        -------
        int
            The number of samples that were written.
        """

        # Convert inputs to numpy arrays of the correct dtype.  Using
        # ``np.asarray`` instead of ``np.array`` allows already-numpy
        # inputs to be used without copying when possible.
        imgs = np.asarray(images)
        lbls = np.asarray(labels)
        msks = np.asarray(masks)

        n = lbls.shape[0]
        if length is not None:
            n = min(n, int(length))

        end = start + n
        if end > self.z_lbls.shape[0]:
            C, H = self.shape
            self.z_imgs.resize((end, C, H))
            self.z_lbls.resize((end, ))
            self.z_msks.resize((end, H))

        # Ensure dtype compatibility before assignment
        self.z_imgs[start:end, ...] = imgs[:n].astype(self.z_imgs.dtype, copy=False)
        self.z_lbls[start:end] = lbls[:n].astype(self.z_lbls.dtype, copy=False)
        self.z_msks[start:end, ...] = msks[:n].astype(self.z_msks.dtype, copy=False)

        return n
    
    def close(self):
        # Close self.root Zarr connection
        pass




def collect_discard_samples(xml: TagLike, extractor: RiichiResNetFeatures):
    """Return a list of **lightweight dicts** for easy JSON/pickle dumping.
    (If you want full dataclasses, iterate `iter_discard_states` directly.)
    """
    images = []
    labels = []
    masks = []

    for st, who, disc_t34, meta in iter_discard_states(xml):
        if st.riichi:
            continue

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

# Implementation of read_xmls for multi-CPU cores execution.
def _worker_fn(xml):
    return collect_discard_samples(xml, RiichiResNetFeatures())
    
def read_xmls_for_discards(xmls: List[TagLike]):
    """并行读取一组 xml 并返回堆叠后的 numpy 数组."""
    imageset, labelset, maskset = [], [], []

    # 自动获取 CPU 核心数
    cpu_count = psutil.cpu_count(logical=False) or 1
    # 保留一部分核心给系统，不要全用
    workers = max(1, cpu_count - 1)
    # 如果任务较少，不必开太多进程
    workers = min(workers, len(xmls))

    # 根据 CPU 核数调整 workers 数量（不要直接 72，内存可能吃不消）
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_worker_fn, xmls))

    for images, labels, masks in results:
        imageset.extend(images)
        labelset.extend(labels)
        maskset.extend(masks)

    if imageset:
        return (np.stack(imageset),
                np.asarray(labelset),
                np.stack(maskset))
    else:
        return (np.empty((0,), dtype=np.uint8),
                np.empty((0,), dtype=np.uint8),
                np.empty((0,), dtype=np.uint8))

def collect_chi_samples(xml: TagLike, extractor: RiichiResNetFeatures):
    images = []
    labels = []
    masks = []

    for st, who, called_t34, meta in iter_chi_states(xml):
        with torch.no_grad():
            out = extractor(st)
        x = out["x"][..., 0].cpu().numpy().astype(np.uint8)
        y = called_t34
        m = np.zeros((x.shape[-1],), dtype=np.uint8)
        if 0 <= y < m.shape[0]:
            m[y] = 1
        images.append(x)
        labels.append(y)
        masks.append(m)

    return images, labels, masks

def collect_pon_samples(xml: TagLike, extractor: RiichiResNetFeatures):
    images = []
    labels = []
    masks = []

    for st, who, called_t34, meta in iter_pon_states(xml):
        with torch.no_grad():
            out = extractor(st)
        x = out["x"][..., 0].cpu().numpy().astype(np.uint8)
        y = called_t34
        m = np.zeros((x.shape[-1],), dtype=np.uint8)
        if 0 <= y < m.shape[0]:
            m[y] = 1
        images.append(x)
        labels.append(y)
        masks.append(m)

    return images, labels, masks

def collect_kan_samples(xml: TagLike, extractor: RiichiResNetFeatures):
    images = []
    labels = []
    masks = []

    for st, who, base_t34, meta in iter_kan_states(xml):
        with torch.no_grad():
            out = extractor(st)
        x = out["x"][..., 0].cpu().numpy().astype(np.uint8)
        y = base_t34
        m = np.zeros((x.shape[-1],), dtype=np.uint8)
        if 0 <= y < m.shape[0]:
            m[y] = 1
        images.append(x)
        labels.append(y)
        masks.append(m)

    return images, labels, masks

def collect_riichi_samples(xml: TagLike, extractor: RiichiResNetFeatures):
    images = []
    labels = []
    masks = []

    for st, who, discard_t34, meta in iter_riichi_states(xml):
        with torch.no_grad():
            out = extractor(st)
        x = out["x"][..., 0].cpu().numpy().astype(np.uint8)
        y = discard_t34
        legal_mask = out["legal_mask"].cpu().numpy().astype(np.uint8)
        images.append(x)
        labels.append(y)
        masks.append(legal_mask)

    return images, labels, masks

def _worker_fn_chi(xml):
    return collect_chi_samples(xml, RiichiResNetFeatures())

def _worker_fn_pon(xml):
    return collect_pon_samples(xml, RiichiResNetFeatures())

def _worker_fn_kan(xml):
    return collect_kan_samples(xml, RiichiResNetFeatures())

def _worker_fn_riichi(xml):
    return collect_riichi_samples(xml, RiichiResNetFeatures())

def _parallel_collect(xmls: List[TagLike], worker):
    imageset, labelset, maskset = [], [], []

    cpu_count = psutil.cpu_count(logical=False) or 1
    workers = max(1, cpu_count - 1)
    workers = min(workers, len(xmls))

    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(worker, xmls))

    for images, labels, masks in results:
        imageset.extend(images)
        labelset.extend(labels)
        maskset.extend(masks)

    if imageset:
        return (np.stack(imageset), np.asarray(labelset), np.stack(maskset))
    else:
        return (
            np.empty((0,), dtype=np.uint8),
            np.empty((0,), dtype=np.uint8),
            np.empty((0,), dtype=np.uint8),
        )

def read_xmls_for_chi(xmls: List[TagLike]):
    return _parallel_collect(xmls, _worker_fn_chi)

def read_xmls_for_pon(xmls: List[TagLike]):
    return _parallel_collect(xmls, _worker_fn_pon)

def read_xmls_for_kan(xmls: List[TagLike]):
    return _parallel_collect(xmls, _worker_fn_kan)

def read_xmls_for_riichi(xmls: List[TagLike]):
    return _parallel_collect(xmls, _worker_fn_riichi)

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
        #print("Fetch logs from database...")
        cur.execute(query, params)
        rows = cur.fetchall()
        #print("OK")
        return [row[0] for row in rows]
    finally:
        conn.close()

def main() -> None:
    """Simple CLI entry point used for manual dataset building.

    The function expects a SQLite database in ``data/2016.db`` with a
    ``logs`` table containing Tenhou log XML strings.  When the database
    is not present the function will simply exit without doing any work
    so that importing this module during tests has no side effects.
    """

    print(f"物理核心数: {psutil.cpu_count(logical=False)}, 逻辑核心数: {psutil.cpu_count(logical=True)}")

    db = "data/2018.db"
    if not os.path.exists(db):
        print(f"Database '{db}' not found; skipping dataset build")
        return

    total_logs = count_xmls_in_database(db)
    
    # Gen Training dataset
    n_training = int(total_logs*0.99)
    dataset = RiichiZarrDatasetBuilder((29, 34), out_dir="output/training")
    dataset2 = RiichiZarrDatasetBuilder((29, 34), out_dir="output/training_chi")
    dataset3 = RiichiZarrDatasetBuilder((29, 34), out_dir="output/training_pon")
    dataset4 = RiichiZarrDatasetBuilder((29, 34), out_dir="output/training_kan")
    dataset5 = RiichiZarrDatasetBuilder((29, 34), out_dir="output/training_riichi")

    sql_batch_size = 256
    cursor = {'discard':0,'chi':0,'pon':0,'kan':0,'riichi':0}
    for start in tqdm(range(0, n_training, sql_batch_size)):
        xmls = fetch_xmls_from_database(db, start=start, num_examples=sql_batch_size)
        if not xmls:
            break
        # Discard samples
        images, labels, masks = read_xmls_for_discards(xmls)
        written = dataset.write_data(images, labels, masks, start=cursor['discard'])
        cursor['discard'] += written

        # Chi samples
        images, labels, masks = read_xmls_for_chi(xmls)
        if images.size:
            written = dataset2.write_data(images, labels, masks, start=cursor['chi'])
            cursor['chi'] += written

        # Pon samples
        images, labels, masks = read_xmls_for_pon(xmls)
        if images.size:
            written = dataset3.write_data(images, labels, masks, start=cursor['pon'])
            cursor['pon'] += written

        # Kan samples
        images, labels, masks = read_xmls_for_kan(xmls)
        if images.size:
            written = dataset4.write_data(images, labels, masks, start=cursor['kan'])
            cursor['kan'] += written

        # Riichi samples
        images, labels, masks = read_xmls_for_riichi(xmls)
        if images.size:
            written = dataset5.write_data(images, labels, masks, start=cursor['riichi'])
            cursor['riichi'] += written

    dataset.close()
    dataset2.close()
    dataset3.close()
    dataset4.close()
    dataset5.close()

    # Gen Test dataset (mirror training: discard/chi/pon/kan/riichi)
    dataset = RiichiZarrDatasetBuilder((29, 34), out_dir="output/test")
    dataset2 = RiichiZarrDatasetBuilder((29, 34), out_dir="output/test_chi")
    dataset3 = RiichiZarrDatasetBuilder((29, 34), out_dir="output/test_pon")
    dataset4 = RiichiZarrDatasetBuilder((29, 34), out_dir="output/test_kan")
    dataset5 = RiichiZarrDatasetBuilder((29, 34), out_dir="output/test_riichi")

    sql_batch_size = 256
    cursor = {'discard':0,'chi':0,'pon':0,'kan':0,'riichi':0}
    for start in tqdm(range(n_training, total_logs, sql_batch_size)):
        xmls = fetch_xmls_from_database(db, start=start, num_examples=sql_batch_size)
        if not xmls:
            break
        # Discard samples
        images, labels, masks = read_xmls_for_discards(xmls)
        written = dataset.write_data(images, labels, masks, start=cursor['discard'])
        cursor['discard'] += written

        # Chi samples
        images, labels, masks = read_xmls_for_chi(xmls)
        if images.size:
            written = dataset2.write_data(images, labels, masks, start=cursor['chi'])
            cursor['chi'] += written

        # Pon samples
        images, labels, masks = read_xmls_for_pon(xmls)
        if images.size:
            written = dataset3.write_data(images, labels, masks, start=cursor['pon'])
            cursor['pon'] += written

        # Kan samples
        images, labels, masks = read_xmls_for_kan(xmls)
        if images.size:
            written = dataset4.write_data(images, labels, masks, start=cursor['kan'])
            cursor['kan'] += written

        # Riichi samples
        images, labels, masks = read_xmls_for_riichi(xmls)
        if images.size:
            written = dataset5.write_data(images, labels, masks, start=cursor['riichi'])
            cursor['riichi'] += written

    dataset.close()
    dataset2.close()
    dataset3.close()
    dataset4.close()
    dataset5.close()


if __name__ == "__main__":
    main()
