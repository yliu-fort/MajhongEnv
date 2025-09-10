# save_as_pkl.py
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".", "src"))

import pickle, numpy as np
from typing import List, Optional, Sequence, Dict, Tuple, Iterable, Generator, Union, IO, Any
import xml.etree.ElementTree as ET
from mahjong_features import RiichiResNetFeatures
from tenhou_to_mahjong import iter_discard_states

TagLike = Union[str, bytes, os.PathLike, IO[bytes], IO[str]]

def save_as_pkl(images, labels, masks, filenames=None, out_file="dataset.pkl"):
    """
    把整个数据集打包进一个 .pkl 文件
    - images: (N,C,H,W)
    - labels: (N,)
    - masks:  (N,1,H,W) 或 (N,H,W)
    - filenames: (N,) 字符串，可选
    """
    N = images.shape[0]
    if filenames is None:
        filenames = np.array([f"sample_{i}" for i in range(N)], dtype="U")

    dataset = {
        "images": images.astype(np.float32),  # 你也可以用 uint8/float16 节省空间
        "labels": labels.astype(np.int64),
        "masks": masks.astype(np.uint8),      # 0/1 掩码存 uint8 就够了
        "filenames": filenames,
        "meta": {
            "num_samples": N,
            "num_channels": images.shape[1],
            "image_shape": images.shape[2:],
            "classes": None,   # 你可以填类别名列表
        }
    }

    with open(out_file, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved dataset with {N} samples into {out_file}")

def collect_discard_samples(xml: TagLike):
    """Return a list of **lightweight dicts** for easy JSON/pickle dumping.
    (If you want full dataclasses, iterate `iter_discard_states` directly.)
    """
    images = []
    labels = []
    masks = []
    extractor = RiichiResNetFeatures()

    for st, who, disc_t34, meta in iter_discard_states(xml):
        out = extractor(st)
        legal_mask = out["legal_mask"]
        x = out["x"]
        y = disc_t34
        # Save x, y and legal_mask like a image dataset like CIFAR-10
        images.append(x)
        labels.append(y)
        masks.append(legal_mask)

    return images, labels, masks

def read_xmls(xmls: List[TagLike]):
    imageset = []
    labelset = []
    maskset = []
    for xml in xmls:
        images, labels, masks = collect_discard_samples(xml)
        imageset += images
        labelset += labels
        maskset += masks
    return imageset, labelset, maskset

# Count logs in sqlite database /logs
# with entries: log_id, date, is_tonpusen, is_sanma, is_speed, is_processed, was_error, log_content
def count_xmls_in_database(db, is_tonpusen=False, is_sanma=False, is_speed=False, is_processed=True, was_error=False):
    pass

# Return logs (as xmls) from sqlite database /logs
def fetch_xmls_from_database(db, num_examples=100, start=0, is_tonpusen=False, is_sanma=False, is_speed=False, is_processed=True, was_error=False):
    pass

def main():
    db = '2018.db'
    print(count_xmls_in_database(db))
    xmls = fetch_xmls_from_database(db)
    images, labels, masks = read_xmls(xmls)
    save_as_pkl(images, labels, masks)