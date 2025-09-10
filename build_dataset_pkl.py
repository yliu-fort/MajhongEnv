# save_as_pkl.py
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".", "src"))

import pickle, numpy as np
from typing import List, Optional, Sequence, Dict, Tuple, Iterable, Generator, Union, IO, Any
import xml.etree.ElementTree as ET
import sqlite3
from mahjong_features import RiichiResNetFeatures
from tenhou_to_mahjong import iter_discard_states

from tqdm import tqdm

TagLike = Union[str, bytes, os.PathLike, IO[bytes], IO[str]]

def save_as_pkl(images, labels, masks, filenames=None, out_file="dataset.pkl"):
    """
    把整个数据集打包进一个 .pkl 文件
    - images: (N,C,H,1) -> broadcast to (N,C,H,W).
    - labels: (N,)
    - masks:  (N,H)
    - filenames: (N,) 字符串，可选
    """
    N = images.shape[0]
    #if filenames is None:
    #    filenames = np.array([f"sample_{i}" for i in range(N)], dtype="U")

    dataset = {
        "images": images.astype(np.uint8),  # 用 uint8/float16 节省空间
        "labels": labels.astype(np.uint8),
        "masks": masks.astype(np.uint8),      # 0/1 掩码存 uint8 就够了
        "filenames": filenames,
        "meta": {
            "num_samples": N,
            "num_channels": images.shape[1],
            "image_shape": images.shape[2:],
            "classes": None,   # 可以填类别名列表
        }
    }

    print("Dumping to pickle file...")
    with open("output/" + out_file, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("OK")

    print(f"Saved dataset with {N} samples into {out_file}")

def collect_discard_samples(xml: TagLike, extractor: RiichiResNetFeatures):
    """Return a list of **lightweight dicts** for easy JSON/pickle dumping.
    (If you want full dataclasses, iterate `iter_discard_states` directly.)
    """
    images = []
    labels = []
    masks = []

    for st, who, disc_t34, meta in iter_discard_states(xml):
        out = extractor(st)
        legal_mask = out["legal_mask"]
        x = out["x"][...,0] # remove the broadcasted rows
        y = disc_t34
        # Save x, y and legal_mask like a image dataset like CIFAR-10
        images.append(x)
        labels.append(y)
        masks.append(legal_mask)

    return images, labels, masks

def read_xmls(xmls: Iterable[TagLike], total: Optional[int] = None):
    """Read XML strings/paths and return stacked numpy arrays.

    Parameters
    ----------
    xmls:
        Iterable of XML sources. Using an iterable instead of a materialised
        list prevents loading all logs into memory at once.
    total:
        Optional total count for progress reporting.
    """

    imageset: List[np.ndarray] = []
    labelset: List[int] = []
    maskset: List[np.ndarray] = []

    extractor = RiichiResNetFeatures()

    for xml in tqdm(xmls, total=total):
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
def iter_xmls_from_database(db, num_examples=10000, start=0, batch_size=1000,
                            is_tonpusen=False, is_sanma=False,
                            is_speed=False, is_processed=True,
                            was_error=False) -> Iterable[str]:
    """Yield log XML strings from the database according to filters.

    This generator streams results using ``fetchmany`` so the full result set
    isn't loaded into memory at once. This avoids huge memory spikes when
    requesting many logs.
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

        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break
            for row in rows:
                yield row[0]
        print("OK")
    finally:
        conn.close()


def main():
    db = 'data/2016.db'
    total = count_xmls_in_database(db)
    print(total)
    xmls_iter = iter_xmls_from_database(db, num_examples=total)
    images, labels, masks = read_xmls(xmls_iter, total=total)
    save_as_pkl(images, labels, masks)


if __name__ == "__main__":
    main()
