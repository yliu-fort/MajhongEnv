import os, struct, numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".", "src"))

import torch
import webdataset as wds
from typing import Dict, List, Sequence
from mahjong_features import RiichiState, PlayerPublic, NUM_TILES, RIVER_LEN, HAND_LEN, DORA_MAX

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

def pad_arr_u8(arr, length, pad=255):
    out = np.full((length,), pad, dtype=np.uint8)
    a = np.asarray(arr, dtype=np.int32)
    n = min(len(a), length)
    if n > 0:
        out[:n] = a[:n].astype(np.uint8)
    return out

def pack_uint16_offset(v):  # -1 or 0..135 -> uint16 (0..136)
    return np.uint16(0 if v < 0 else (v + 1))
    
def encode_record(s: "RiichiState-like-dict")->bytes:
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

    # last tiles
    b.extend(pack_uint16_offset(s.get("last_draw_136", -1)).tobytes())
    b.extend(pack_uint16_offset(s.get("last_discarded_tile_136", -1)).tobytes())

    u8arr(s.get("visible_counts"), NUM_TILES)
    u8arr(s.get("remaining_counts"), NUM_TILES)
    u8arr(s.get("shantens"), NUM_TILES)
    u8arr(s.get("ukeires"), NUM_TILES)

    return bytes(b)

def write_shards(states_iter, out_dir, samples_per_shard=16000):
    os.makedirs(out_dir, exist_ok=True)
    sink = wds.ShardWriter(os.path.join(out_dir, "riichi-%06d.tar"), maxcount=samples_per_shard)
    for i, s in enumerate(states_iter):
        raw = encode_record(s)
        # 用 .bin 扩展名；给每条一个唯一 key
        sample = {"__key__": f"{i:012d}", "bin": raw}
        sink.write(sample)
    sink.close()

# states_iter 需要你提供一个迭代器/生成器，产生 dict 形式的 RiichiState（或直接改成你的对象字段）
# 写完后，用 webdataset 的 make_index 生成 .idx 文件（命令行或 python 调用均可）

# The second decode function
def decode_record(raw: bytes)->RiichiState:
    import numpy as np
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
        opps.append((rv, rflag, rturn, meld))

    dora_raw = take(DORA_MAX)
    dora_indicators = [int(x) for x in dora_raw.tolist() if x != 255]
    aka_flags = int(take(1)[0])

    # Legal mask: 5 bytes (little-endian 40-bit), take lower 34 bits
    legal_bytes = bytes(take(5).tolist())
    bits = int.from_bytes(legal_bytes, "little")
    legal_mask = [(bits >> i) & 1 for i in range(NUM_TILES)]

    # Last tiles (uint16 little-endian with -1 offset)
    u16 = v[off:off+4].view(dtype="<u2")
    last_draw = int(u16[0]) - 1
    last_disc = int(u16[1]) - 1
    off += 4

    # Build RiichiState
    left = PlayerPublic(
        river=opps[0][0],
        meld_counts=opps[0][3].astype(int).tolist(),
        riichi=opps[0][1],
        riichi_turn=opps[0][2],
    )
    across = PlayerPublic(
        river=opps[1][0],
        meld_counts=opps[1][3].astype(int).tolist(),
        riichi=opps[1][1],
        riichi_turn=opps[1][2],
    )
    right = PlayerPublic(
        river=opps[2][0],
        meld_counts=opps[2][3].astype(int).tolist(),
        riichi=opps[2][1],
        riichi_turn=opps[2][2],
    )

    visible_counts = take(NUM_TILES).tolist()
    remaining_counts = take(NUM_TILES).tolist()
    shantens = take(NUM_TILES).tolist()
    ukeires = take(NUM_TILES).tolist()

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
        dora_indicators=dora_indicators,
        aka5m=bool(aka_flags & 0x1),
        aka5p=bool(aka_flags & 0x2),
        aka5s=bool(aka_flags & 0x4),
        legal_discards_mask=legal_mask,
        last_draw_136=last_draw,
        last_discarded_tile_136=last_disc,
        visible_counts=visible_counts,
        remaining_counts=remaining_counts,
        shantens=shantens,
        ukeires=ukeires,
    )

    return state

def make_loader(pattern, batch_size, num_workers=8, shard_shuffle=True, seed=1234):
    # ResampledShards 实现“分片级随机复用”，适合无限迭代式训练
    urls = wds.ResampledShards(pattern, seed=seed) if shard_shuffle else pattern
    ds = (
        wds.WebDataset(urls, resampled=shard_shuffle)
        .shuffle(2000)  # 轻度预热，先打散样本键
        .decode()       # 我们自己解码，不用自动解码器
        .to_tuple("bin")  # 只取二进制字段
        .map(lambda x: decode_record(x[0]))
        .shuffle(100000)  # 片内大缓冲区乱序（关键！）
        .batched(batch_size, partial=False)
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=None, num_workers=num_workers, pin_memory=True, prefetch_factor=4
    )
    return loader

# 用法：
# loader = make_loader("/data/riichi/riichi-{000000..004095}.tar", batch_size=1024, num_workers=12)
# for batch in loader:
#     ...
