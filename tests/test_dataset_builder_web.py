import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import unittest
import torch
from typing import Dict
from mahjong_features import RiichiState, PlayerPublic, NUM_TILES, RIVER_LEN, HAND_LEN, DORA_MAX, NUM_ACTIONS
from dataset_builder_web import encode_record, decode_record


def _decode_record(raw: bytes) -> Dict:
    # 把 bytes 解析回 numpy/torch 张量（按你网络需要返回）
    # 为了速度，尽量用 np.frombuffer + 视图切片，避免 Python 循环
    import numpy as np
    v = np.frombuffer(raw, dtype=np.uint8)  # 先全体拿到
    off = 0

    def take(n):
        nonlocal off
        a = v[off:off+n]
        off += n
        return a

    hand = take(34).astype(np.uint8)
    meld_self = take(34).astype(np.uint8)
    riichi_self = int(take(1)[0])
    turn = int(take(1)[0]); honba = int(take(1)[0]); sticks = int(take(1)[0])
    dealer = int(take(1)[0])
    round_wind = int(take(1)[0]); seat_wind = int(take(1)[0])
    river_self = take(24).astype(np.uint8)

    opp = []
    for _ in range(3):
        rv = take(24).astype(np.uint8)
        rflag = int(take(1)[0]); rturn = int(take(1)[0])
        meld = take(34).astype(np.uint8)
        u16 = v[off:off+2].view(dtype="<u2")
        score = int(u16[0]) - 1
        off += 2
        rank = int(take(1)[0])
        opp.append((rv, rflag, rturn, meld))

    dora = take(5).astype(np.uint8)
    aka_flags = int(take(1)[0])
    legal_mask = take(5).copy()  # 解 bitmask 留给后处理
    legal_actions = take((NUM_ACTIONS+7)//8).copy()  # 解 bitmask 留给后处理
    # uint16 小端：用 view
    rest = v[off:].view(np.uint8)
    last_draw = int(np.frombuffer(rest[:2].tobytes(), dtype="<u2")[0]) - 1
    last_disc = int(np.frombuffer(rest[2:4].tobytes(), dtype="<u2")[0]) - 1
    last_disr = int(np.frombuffer(rest[4:6].tobytes(), dtype="<u2")[0]) - 1
    off += 6  # consumed two uint16

    # Score and rank
    u16 = v[off:off+2].view(dtype="<u2")
    score = int(u16[0]) - 1
    off += 2
    rank = int(take(1)[0])

    # Visible counts (34), remaining counts (34), shantens (34), ukeires (34)
    visible_counts = take(NUM_TILES).astype(np.uint8)
    remaining_counts = take(NUM_TILES).astype(np.uint8)
    shantens = take(NUM_TILES).astype(np.uint8)
    ukeires = take(NUM_TILES).astype(np.uint8)

    # 返回你网络需要的张量/字典（演示返回最关键的）
    return {
        "hand_counts": torch.from_numpy(hand.astype(np.int16)),
        "river_self": torch.from_numpy(river_self.astype(np.int16)),
        "opp": opp,  # 你也可以摊平成张量
        "legal_mask_bytes": torch.from_numpy(legal_mask),
        "legal_action_bytes": torch.from_numpy(legal_actions),
        "meta": torch.tensor([riichi_self, turn, honba, sticks, dealer, round_wind, seat_wind], dtype=torch.int16),
        "last_draw": torch.tensor(last_draw, dtype=torch.int16),
        "last_disc": torch.tensor(last_disc, dtype=torch.int16),
        "last_disr": torch.tensor(last_disr, dtype=torch.int16),
        "score": torch.tensor(score, dtype=torch.int16),
        "rank": torch.tensor(rank, dtype=torch.int16),
        "dora": torch.from_numpy(dora.astype(np.int16)),
        "aka_flags": torch.tensor(aka_flags, dtype=torch.uint8),
        "meld_self": torch.from_numpy(meld_self.astype(np.int16)),
        "visible_counts": torch.from_numpy(visible_counts.astype(np.uint8)),
        "remaining_counts": torch.from_numpy(remaining_counts.astype(np.uint8)),
        "shantens": torch.from_numpy(shantens.astype(np.uint8)),
        "ukeires": torch.from_numpy(ukeires.astype(np.uint8)),
    }

def _build_sample_state_dict():
    # Minimal, deterministic sample covering all packed fields
    hand = [(i % 5) for i in range(NUM_TILES)]
    meld_self = [0] * NUM_TILES
    meld_self[0] = 1; meld_self[9] = 2; meld_self[27] = 1
    visible_counts = [(i % 5) for i in range(NUM_TILES)]
    remaining_counts = [max(0, 4 - v) for v in visible_counts]
    shantens = [(i % 6) for i in range(NUM_TILES)]
    ukeires = [(i * 3) % 17 for i in range(NUM_TILES)]

    def opp(river, riichi=False, riichi_turn=0, meld_idx=None):
        mc = [0] * NUM_TILES
        if meld_idx is not None:
            for idx in meld_idx:
                mc[idx] = 1
        return {
            "river": river,
            "riichi": riichi,
            "riichi_turn": riichi_turn,
            "meld_counts": mc,
            "score":314,
            "rank":2
        }

    state = {
        "hand_counts": hand,
        "meld_counts_self": meld_self,
        "riichi": True,
        "river_self": [1, 5, 7, 12, 18],
        "left": opp([2, 3], riichi=True, riichi_turn=3, meld_idx=[1, 2, 3]),
        "across": opp([4], riichi=False, riichi_turn=0, meld_idx=[10, 11]),
        "right": opp([6, 6, 6], riichi=False, riichi_turn=0, meld_idx=[18, 19, 20]),
        "round_wind": 1,
        "seat_wind_self": 2,
        "dealer_self": True,
        "turn_number": 9,
        "honba": 2,
        "riichi_sticks": 1,
        "score": 314,
        "rank": 1,
        "dora_indicators": [3, 28, 31],
        "aka5m": True,
        "aka5p": True,
        "aka5s": False,
        "legal_discards_mask": [1 if i in (0, 9, 18, 27, 33) else 0 for i in range(NUM_TILES)],
        "legal_actions_mask": [1 if i in (0, 9, 14, 18, 27, 33, 260) else 0 for i in range(NUM_ACTIONS)],
        "last_draw_136": 120,
        "last_discarded_tile_136": 55,
        "last_discarder": 3,
        "visible_counts": visible_counts,
        "remaining_counts": remaining_counts,
        "shantens": shantens,
        "ukeires": ukeires,
    }
    return state

class TestWebDatasetEncoderDecoder(unittest.TestCase):
    def test_encode_decode_roundtrip_basic(self):
        s = _build_sample_state_dict()
        raw = encode_record(s,260)

        # Decode variant 1 (tensor dict)
        d1 = _decode_record(raw)
        assert d1["hand_counts"].dtype == torch.int16
        assert d1["meld_self"].dtype == torch.int16
        assert d1["hand_counts"].tolist() == s["hand_counts"], "hand_counts mismatch"
        assert d1["meld_self"].tolist() == s["meld_counts_self"], "meld_self mismatch"

        # river_self is padded to length 24 with 255
        rv_self = d1["river_self"].tolist()
        n = len(s["river_self"])
        assert rv_self[:n] == s["river_self"], "river_self content mismatch"
        assert all(x == 255 for x in rv_self[n:]), "river_self padding mismatch"

        # Meta order: [riichi_self, turn, honba, sticks, dealer, round_wind, seat_wind]
        meta = d1["meta"].tolist()
        exp_meta = [1, s["turn_number"], s["honba"], s["riichi_sticks"], 1 if s["dealer_self"] else 0, s["round_wind"], s["seat_wind_self"]]
        assert meta == exp_meta, f"meta mismatch: {meta} != {exp_meta}"

        # Dora padded to DORA_MAX with 255
        dora = d1["dora"].tolist()
        m = len(s["dora_indicators"])
        assert dora[:m] == s["dora_indicators"], "dora content mismatch"
        assert all(x == 255 for x in dora[m:]), "dora padding mismatch"

        # Aka flags bitpack: 0b00000111 for m/p/s
        aka = int(d1["aka_flags"].item())
        assert (aka & 0x1) == 1 and (aka & 0x2) == 0x2 and (aka & 0x4) == 0, "aka flags mismatch"

        # Last tiles
        assert int(d1["last_draw"].item()) == s["last_draw_136"], f'last_draw mismatch: {int(d1["last_draw"].item())} True:{s["last_draw_136"]}'
        assert int(d1["last_disc"].item()) == s["last_discarded_tile_136"], "last_disc mismatch"

        assert int(d1["score"].item()) == s["score"], "score mismatch"
        assert int(d1["rank"].item()) == s["rank"], "score mismatch"

        # Counts & derived metrics
        assert d1["visible_counts"].dtype == torch.uint8
        assert d1["remaining_counts"].dtype == torch.uint8
        assert d1["shantens"].dtype == torch.uint8
        assert d1["ukeires"].dtype == torch.uint8
        assert d1["visible_counts"].tolist() == s["visible_counts"], "visible_counts mismatch"
        assert d1["remaining_counts"].tolist() == s["remaining_counts"], "remaining_counts mismatch"
        assert d1["shantens"].tolist() == s["shantens"], "shantens mismatch"
        assert d1["ukeires"].tolist() == s["ukeires"], "ukeires mismatch"

        # Decode variant 2 (RiichiState dataclass)
        d2, label, mask = decode_record(raw)
        assert d2.riichi is True and d2.dealer_self is True
        assert d2.hand_counts == s["hand_counts"], "d2 hand_counts mismatch"
        assert d2.meld_counts_self == s["meld_counts_self"], "d2 meld_self mismatch"
        assert d2.river_self == s["river_self"], "d2 river_self mismatch"
        assert d2.round_wind == s["round_wind"] and d2.seat_wind_self == s["seat_wind_self"]
        assert d2.turn_number == s["turn_number"] and d2.honba == s["honba"] and d2.riichi_sticks == s["riichi_sticks"]
        # Opponents
        assert d2.left.river == s["left"]["river"] and d2.left.riichi is True and d2.left.riichi_turn == 3
        assert d2.across.river == s["across"]["river"]
        assert d2.right.river == s["right"]["river"]
        # Dora & aka
        assert d2.dora_indicators == s["dora_indicators"], "d2 dora mismatch"
        assert d2.aka5m is True and d2.aka5p is True and d2.aka5s is False
        # Legal mask
        assert d2.legal_discards_mask == s["legal_discards_mask"], "d2 legal mask mismatch"
        # Last tiles
        assert d2.last_draw_136 == s["last_draw_136"] and d2.last_discarded_tile_136 == s["last_discarded_tile_136"]
        assert d2.score == s["score"], "score mismatch"
        assert d2.rank == s["rank"], "score mismatch"
        # Counts & derived metrics
        assert d2.visible_counts == s["visible_counts"], "d2 visible_counts mismatch"
        assert d2.remaining_counts == s["remaining_counts"], "d2 remaining_counts mismatch"
        assert d2.shantens == s["shantens"], "d2 shantens mismatch"
        assert d2.ukeires == s["ukeires"], "d2 ukeires mismatch"
        assert label == 260, f"label={label}"
        assert mask[label] == True, str(sum(mask))
