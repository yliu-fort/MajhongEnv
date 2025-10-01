import json
import os, struct, numpy as np
import psutil
import sys
from concurrent.futures import ProcessPoolExecutor
sys.path.append(os.path.join(os.path.dirname(__file__), ".", "src"))

import torch
from torchvision import transforms
import webdataset as wds
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from tqdm import tqdm
import sqlite3

from mahjong_features import RiichiResNetFeatures, RiichiState, PlayerPublic, NUM_TILES, RIVER_LEN, HAND_LEN, DORA_MAX
#from dataset_builder_zarr import count_xmls_in_database, fetch_xmls_from_database
from tenhou_to_mahjong import iter_discard_states

# 固定长度常量
RECORD_SIZE = 400  # 近似值，不需要精确；只用于预分配/统计（实际解码按切片）

# bitpack / unpack 小工具
def pack_bits_bool(bools):  # e.g. [True, False, True] -> one byte 0b00000101
    v = 0
    for i, b in enumerate(bools):
        if b: v |= (1 << i)
    return np.uint8(v)

def pack_34bits(mask_iterable):  # 34个0/1 -> 5字节
    bits = 0
    for i, m in enumerate(mask_iterable):
        bits |= (int(m) & 1) << i
    return (bits & ((1<<40)-1)).to_bytes(5, "little")

def pack_253bits(mask_iterable):  # 253个0/1 -> 32字节
    out = bytearray((253 + 7) // 8)
    for i, m in enumerate(mask_iterable):
        if i >= 253:
            break
        if int(m) & 1:
            out[i >> 3] |= 1 << (i & 7)
    return bytes(out)

def pad_arr_u8(arr, length, pad=255):
    out = np.full((length,), pad, dtype=np.uint8)
    a = np.asarray(arr, dtype=np.int32)
    n = min(len(a), length)
    if n > 0:
        out[:n] = a[:n].astype(np.uint8)
    return out

def pack_uint16_offset(v):  # -1 or 0..135 -> uint16 (0..136)
    return np.uint16(0 if v < 0 else (v + 1))
    
def encode_record(s: "RiichiState-like-dict", label)->bytes:
    # 将你的 dict/对象转成定长块
    b = bytearray()

    def u8arr(x, n):
        a = np.asarray(x if x is not None else [255]*n, dtype=np.int32)
        a = np.clip(a, -1, 255).astype(np.uint8)
        b.extend(a.tobytes())

    # ego
    u8arr(s["hand_counts"], NUM_TILES)
    u8arr(s.get("meld_counts_self"), NUM_TILES)

    riichi_self = 1 if s.get("riichi", False) else 0
    dealer = 1 if s.get("dealer_self", False) else 0
    b.append(np.uint8(riichi_self))        # 1
    b.append(np.uint8(s.get("turn_number", 0)))  # 1
    b.append(np.uint8(s.get("honba", 0)))        # 1
    b.append(np.uint8(s.get("riichi_sticks", 0)))# 1
    b.append(np.uint8(dealer))                   # 1

    b.append(np.uint8(s.get("round_wind", 0)))
    b.append(np.uint8(s.get("seat_wind_self", 0)))

    b.extend(pad_arr_u8(s.get("river_self", []), RIVER_LEN).tobytes())

    # opponents
    for name in ["left", "across", "right"]:
        pp = s[name]
        b.extend(pad_arr_u8(pp.get("river", []), RIVER_LEN).tobytes())
        b.append(np.uint8(1 if pp.get("riichi", False) else 0))
        b.append(np.uint8(pp.get("riichi_turn", 0)))
        u8arr(pp.get("meld_counts"), NUM_TILES)
        b.extend(pack_uint16_offset(pp.get("score", -1)).tobytes())
        b.append(np.uint8(pp.get("rank", 0)))

    # dora + aka
    b.extend(pad_arr_u8(s.get("dora_indicators", []), DORA_MAX).tobytes())
    aka = pack_bits_bool([s.get("aka5m", False), s.get("aka5p", False), s.get("aka5s", False)])
    b.append(aka)

    # legal mask
    legal = s.get("legal_discards_mask")
    if legal is None:
        b.extend(b"\x00"*5)
    else:
        b.extend(pack_34bits(legal))

    legal_actions = s.get("legal_actions_mask")
    if legal_actions is None:
        b.extend(b"\x00"*32)
    else:
        b.extend(pack_253bits(legal_actions))

    # last tiles
    b.extend(pack_uint16_offset(s.get("last_draw_136", -1)).tobytes())
    b.extend(pack_uint16_offset(s.get("last_discarded_tile_136", -1)).tobytes())
    b.extend(pack_uint16_offset(s.get("last_discarder", -1)).tobytes())

    # score and rank
    b.extend(pack_uint16_offset(s.get("score", -1)).tobytes())
    b.append(np.uint8(s.get("rank", 0)))# 1


    u8arr(s.get("visible_counts"), NUM_TILES)
    u8arr(s.get("remaining_counts"), NUM_TILES)
    u8arr(s.get("shantens"), NUM_TILES)
    u8arr(s.get("ukeires"), NUM_TILES)

    # label
    b.extend(np.asarray([int(label)], dtype=np.uint8).tobytes())

    return bytes(b)

# states_iter 需要你提供一个迭代器/生成器，产生 dict 形式的 RiichiState（或直接改成你的对象字段）
# 写完后，用 webdataset 的 make_index 生成 .idx 文件（命令行或 python 调用均可）

# The second decode function
def decode_record(raw: bytes)->Tuple[RiichiState, int]:
    v = np.frombuffer(raw, dtype=np.uint8)
    off = 0

    def take(n):
        nonlocal off
        a = v[off:off+n]
        off += n
        return a

    # Ego
    hand = take(NUM_TILES).astype(np.uint8)
    meld_self = take(NUM_TILES).astype(np.uint8)
    riichi_self = bool(take(1)[0])
    turn = int(take(1)[0])
    honba = int(take(1)[0])
    sticks = int(take(1)[0])
    dealer = bool(take(1)[0])

    round_wind = int(take(1)[0])
    seat_wind = int(take(1)[0])

    river_self_raw = take(RIVER_LEN)
    river_self = [int(x) for x in river_self_raw.tolist() if x != 255]

    # Opponents
    opps = []
    for _ in range(3):
        rv_raw = take(RIVER_LEN)
        rv = [int(x) for x in rv_raw.tolist() if x != 255]
        rflag = bool(take(1)[0])
        rturn = int(take(1)[0])
        meld = take(NUM_TILES).astype(np.uint8)
        u16 = v[off:off+2].view(dtype="<u2")
        score = int(u16[0]) - 1
        off += 2
        rank = int(take(1)[0])
        opps.append((rv, rflag, rturn, meld, score, rank))

    dora_raw = take(DORA_MAX)
    dora_indicators = [int(x) for x in dora_raw.tolist() if x != 255]
    aka_flags = int(take(1)[0])

    # Legal mask: 5 bytes (little-endian 40-bit), take lower 34 bits
    legal_bytes = bytes(take(5).tolist())
    bits = int.from_bytes(legal_bytes, "little")
    legal_mask = [(bits >> i) & 1 for i in range(NUM_TILES)]

    # Legal actions mask: 32 bytes (253 little-endian bits)
    legal_actions_bytes = take(32)
    legal_actions_mask_bits = np.unpackbits(legal_actions_bytes, bitorder="little")
    legal_actions_mask = legal_actions_mask_bits[:253].astype(int).tolist()

    # Last tiles (uint16 little-endian with -1 offset)
    u16 = v[off:off+6].view(dtype="<u2")
    last_draw = int(u16[0]) - 1
    last_disc = int(u16[1]) - 1
    last_disr = int(u16[2]) - 1
    off += 6

    # score and rank
    u16 = v[off:off+2].view(dtype="<u2")
    score = int(u16[0]) - 1
    off += 2
    rank = int(take(1)[0])

    # Build RiichiState
    left = PlayerPublic(
        river=opps[0][0],
        meld_counts=opps[0][3].astype(int).tolist(),
        riichi=opps[0][1],
        riichi_turn=opps[0][2],
        score=opps[0][4],
        rank=opps[0][5],
    )
    across = PlayerPublic(
        river=opps[1][0],
        meld_counts=opps[1][3].astype(int).tolist(),
        riichi=opps[1][1],
        riichi_turn=opps[1][2],
        score=opps[1][4],
        rank=opps[1][5],
    )
    right = PlayerPublic(
        river=opps[2][0],
        meld_counts=opps[2][3].astype(int).tolist(),
        riichi=opps[2][1],
        riichi_turn=opps[2][2],
        score=opps[2][4],
        rank=opps[2][5],
    )

    visible_counts = take(NUM_TILES).tolist()
    remaining_counts = take(NUM_TILES).tolist()
    shantens = take(NUM_TILES).tolist()
    ukeires = take(NUM_TILES).tolist()

    label = int(take(1)[0])

    state = RiichiState(
        hand_counts=hand.astype(int).tolist(),
        meld_counts_self=meld_self.astype(int).tolist(),
        riichi=riichi_self,
        river_self=river_self,
        left=left,
        across=across,
        right=right,
        round_wind=round_wind,
        seat_wind_self=seat_wind,
        dealer_self=dealer,
        turn_number=turn,
        honba=honba,
        riichi_sticks=sticks,
        score=score,
        rank=rank,
        dora_indicators=dora_indicators,
        aka5m=bool(aka_flags & 0x1),
        aka5p=bool(aka_flags & 0x2),
        aka5s=bool(aka_flags & 0x4),
        legal_discards_mask=legal_mask,
        legal_actions_mask=legal_actions_mask,
        last_draw_136=last_draw,
        last_discarded_tile_136=last_disc,
        last_discarder=last_disr,
        visible_counts=visible_counts,
        remaining_counts=remaining_counts,
        shantens=shantens,
        ukeires=ukeires,
    )

    return state, label, legal_actions_mask


DEFAULT_DB_PATH = "/workspace/2018.db"
DEFAULT_OUTPUT_DIR = os.path.join("output", "webdataset")
DEFAULT_SAMPLES_PER_SHARD = 16000
DEFAULT_SQL_BATCH = 256


def _state_to_dict(state: Any) -> Dict[str, Any]:
    """Convert a :class:`RiichiState` (or similar) to a serializable mapping."""

    if isinstance(state, dict):
        return state
    if is_dataclass(state):
        data = asdict(state)
        data.pop("extra", None)
        return data
    raise TypeError(f"Unsupported state type: {type(state)!r}")


class WebDatasetShardWriter:
    def __init__(self, out_dir: str, prefix: str = "riichi", samples_per_shard: int = DEFAULT_SAMPLES_PER_SHARD):
        os.makedirs(out_dir, exist_ok=True)
        pattern = os.path.join(out_dir, f"{prefix}-%06d.tar")
        self._writer = wds.ShardWriter(pattern, maxcount=samples_per_shard)
        self._count = 0

    def write(self, state: Dict[str, Any], label: int, meta: Optional[Dict[str, Any]]=None) -> bool:
        if not (0 <= int(label) < 253):
            return False

        sample = {
            "__key__": f"{self._count:012d}",
            "bin": encode_record(state, label),
        }

        self._writer.write(sample)
        self._count += 1
        return True

    def close(self) -> None:
        self._writer.close()


def _process_single(xml: str) -> List[Tuple[Dict[str, Any], int]]:
    samples: List[Tuple[Dict[str, Any], int]] = []
    for state, who, label, meta in iter_discard_states(xml):
        who_i = int(who)
        label_i = int(label)
        samples.append((_state_to_dict(state), label_i))
    return samples
    
def _iter_action_samples(
    xmls: Sequence[str], iterator
) -> Iterable[Tuple[Dict[str, Any], int]]:
    """Yield action samples using a thread pool to parallelize XML processing."""

    cpu_count = psutil.cpu_count(logical=False) or 1
    max_workers = max(1, min(len(xmls), cpu_count - 1 if cpu_count > 1 else 1))

    if max_workers <= 1:
        for xml in xmls:
            for sample in _process_single(xml):
                yield sample
        return

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for samples in list(executor.map(_process_single, xmls)):
            for sample in samples:
                yield sample

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
    """Entry point mirroring :mod:`dataset_builder_zarr` behaviour for WebDataset output."""

    print(f"物理核心数: {psutil.cpu_count(logical=False)}, 逻辑核心数: {psutil.cpu_count(logical=True)}")

    db_path = os.environ.get("TENHOU_DB", DEFAULT_DB_PATH)
    if not os.path.exists(db_path):
        print(f"Database '{db_path}' not found; skipping dataset build")
        return

    total_logs = count_xmls_in_database(db_path)
    if total_logs <= 0:
        print("No logs available in database; nothing to do")
        return

    n_training = int(total_logs * 0.99)
    base_out = os.environ.get("RIICHI_WEB_OUT", DEFAULT_OUTPUT_DIR)
    samples_per_shard = int(os.environ.get("RIICHI_WEB_SAMPLES_PER_SHARD", DEFAULT_SAMPLES_PER_SHARD))
    sql_batch_size = int(os.environ.get("TENHOU_SQL_BATCH", DEFAULT_SQL_BATCH))

    action_configs = {
        "discard": {
            "iterator": iter_discard_states,
            "out_subdir": "discard",
        }
    }

    def run_split(split_name: str, start: int, end: int) -> None:
        if start >= end:
            print(f"Split '{split_name}' has no logs; skipping")
            return

        writers = {
            action: WebDatasetShardWriter(
                os.path.join(base_out, split_name, cfg["out_subdir"]),
                samples_per_shard=samples_per_shard,
            )
            for action, cfg in action_configs.items()
        }
        stats = {action: 0 for action in action_configs}

        try:
            for offset in tqdm(range(start, end, sql_batch_size), desc=f"{split_name} logs"):
            #for offset in range(start, end, sql_batch_size):
                num_examples = min(sql_batch_size, end - offset)
                xmls = fetch_xmls_from_database(db_path, start=offset, num_examples=num_examples)
                if not xmls:
                    break

                for action, cfg in action_configs.items():
                    iterator = cfg["iterator"]
                    for state_dict, label in _iter_action_samples(xmls, iterator):
                        if writers[action].write(state_dict, label):
                            stats[action] += 1
        finally:
            for writer in writers.values():
                writer.close()

        summary = ", ".join(f"{action}: {count}" for action, count in stats.items())
        print(f"Finished split '{split_name}' → {summary}")

    run_split("train", 0, n_training)
    run_split("test", n_training, total_logs)


if __name__ == "__main__":
    main()
