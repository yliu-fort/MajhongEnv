"""
Tenhou → RiichiState converter (reusable module)
===============================================

Given a Tenhou mjlog XML, iterate over **every discard action** and emit a
`RiichiState` snapshot **just before** that discard, along with the actor and
label (34-index of the discarded tile).

Designed to align with the data schema used by `riichi_resnet_features.py`.
If that module is importable, we reuse its `RiichiState`/`PlayerPublic`.
Otherwise we provide compatible fallback dataclasses.

Usage
-----
```python
from tenhou_to_riichi import iter_discard_states
from riichi_resnet_features import RiichiResNetFeatures  # optional

for state, who, discard_t34, meta in iter_discard_states("/path/to/mjlog.xml"):
    feats = RiichiResNetFeatures().forward(state)
    x = feats["x"]              # (C,34,34)
    legal = feats["legal_mask"] # (34,)
    target = discard_t34        # supervision label
```

CLI
---
```bash
python tenhou_to_riichi.py /path/to/file.xml --preview 5 --dump /tmp/out.pkl
```

Notes
-----
- Handles Chi/Pon/Kan/Chakan melds (4P); ignores 3P NUKI.
- Tracks dora **indicators** as 34-indices; mapping to actual dora is typically
  handled inside the feature extractor.
- `turn_number` is approx `(total_discards_so_far // 4)`.
- `aka5m/p/s` flags are set if any red-5 is ever seen in the hand/draws of the
  current round (136 IDs 16, 52, 88 for m/p/s respectively).

License: MIT
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Dict, Tuple, Iterable, Generator, Union, IO, Any
import xml.etree.ElementTree as ET
import io
import os
import sys
import pickle
import numpy as np

# ----------------------------
# Import schema from feature module if available
# ----------------------------
'''
try:
    from mahjong_features import RiichiState, PlayerPublic, NUM_TILES  # type: ignore
    _HAVE_SCHEMA = True
except Exception:
    _HAVE_SCHEMA = False
    NUM_TILES = 34

    @dataclass
    class PlayerPublic:  # fallback (compatible API)
        river: List[int] = field(default_factory=list)
        river_counts: Optional[Sequence[int]] = None
        meld_counts: Optional[Sequence[int]] = None
        riichi: bool = False

    @dataclass
    class ExtraCalcs:
        visible_count_hook: Optional[Any] = None
        remaining_count_hook: Optional[Any] = None

    @dataclass
    class RiichiState:  # fallback (compatible API)
        hand_counts: Sequence[int]
        meld_counts_self: Optional[Sequence[int]] = None
        my_river: List[int] = field(default_factory=list)
        my_river_counts: Optional[Sequence[int]] = None
        left: PlayerPublic = field(default_factory=PlayerPublic)
        across: PlayerPublic = field(default_factory=PlayerPublic)
        right: PlayerPublic = field(default_factory=PlayerPublic)
        round_wind: int = 0
        seat_wind_self: int = 0
        dealer_self: bool = False
        turn_number: int = 0
        honba: int = 0
        riichi_sticks: int = 0
        dora_indicators: Sequence[int] = field(default_factory=list)
        aka5m: bool = False
        aka5p: bool = False
        aka5s: bool = False
        legal_discards_mask: Optional[Sequence[int]] = None
'''
from mahjong_features import RiichiState, PlayerPublic, NUM_TILES, NUM_ACTIONS
from shanten_dp import compute_ukeire_advanced, compute_all_discards_ukeire_fast
from mahjong_features import get_action_index

# ----------------------------
# Tenhou basics
# ----------------------------
NUM_WALL_TILES = 136

RED_TILES = {16, 52, 88}  # Tenhou's 136 IDs for red 5m/5p/5s


def tid136_to_t34(tid: int) -> int:
    return tid // 4


# ----------------------------
# Meld decoder (Tenhou `m` integer)
# ----------------------------
class TenhouMeld:
    """Decode Tenhou meld integer.

    Sets fields:
      - who (caller index 0..3)
      - type: "chi" | "pon" | "kan" | "chakan" | "nuki"
      - from_who: 0..3 (relative to caller) or None for closed kan / nuki
      - base_t34: tile index of the base (for chi, left tile of the run)
      - called_index: position in the meld that was taken from discarder (0..2/3)
      - tiles_t34: list of involved 34-indices (duplicates included)
    """
    def __init__(self, who: int, m: int):
        self.who = who
        self.m = m
        self.type: Optional[str] = None
        self.from_who: Optional[int] = None
        self.called_index: Optional[int] = None # 0: left 1: mid 2: right
        self.base_t34: Optional[int] = None # b34 index of base tile
        self.base_t136: Optional[int] = None # b136 index of called tile
        self.tiles_t34: List[int] = []
        self.tiles_t136: List[int] = []
        self.opened: bool = True
        self._decode()

    def _get_distance(self, current_player, other_player, num_players=4):
        """计算当前玩家到其他玩家的距离（逆时针）。"""
        return (other_player - current_player) % num_players
    
    def _decode(self):
        data = self.m
        self.from_who = (self.who + (data & 0x3)) % 4 # Convert offset (data & 0x3) to from_who
        if data & 0x4:  # CHI
            self.type = "chi"
            t0 = (data >> 3) & 0x3
            t1 = (data >> 5) & 0x3
            t2 = (data >> 7) & 0x3
            base_and_called = data >> 10
            called = base_and_called % 3
            base7 = base_and_called // 3
            base9 = (base7 // 7) * 9 + (base7 % 7)
            self.called_index = called
            self.base_t34 = base9
            options = [t0, t1, t2]
            self.base_t136 = (base9 + called) * 4 + options[called]
            self.tiles_t34 = [base9 + 0, base9 + 1, base9 + 2]
            self.tiles_t136 = [base9 * 4 + t0, (base9 + 1) * 4 + t1, (base9 + 2) * 4 + t2]
        elif data & 0x18:  # PON or CHAKAN (added tile to pon)
            t4 = (data >> 5) & 0x3
            base_and_called = data >> 9
            called = base_and_called % 3
            base = base_and_called // 3
            self.called_index = called
            self.base_t34 = base
            self.base_t136 = base * 4  + called
            if data & 0x8:
                self.type = "pon"
                self.tiles_t34 = [base, base, base]
                self.tiles_t136 = [base * 4 + i for i in range(4) if i != t4]
            else:
                self.type = "chakan"
                self.tiles_t34 = [base, base, base, base]
                self.tiles_t136 = [base * 4 + i for i in range(4)]
        elif data & 0x20:  # NUKI (3P); keep for completeness
            self.type = "nuki"
            self.from_who = None
        else:  # KAN
            base_and_called = data >> 8
            if self.from_who != self.who:  # open kan
                called = base_and_called % 4
                base = base_and_called // 4
                self.called_index = called
                self.base_t34 = base
                self.base_t136 = base * 4  + called
                self.type = "kan"
                self.tiles_t34 = [base, base, base, base]
                self.tiles_t136 = [base * 4 + i for i in range(4)]
            else:  # closed kan
                base = base_and_called // 4
                self.from_who = self.who
                self.base_t34 = base
                self.base_t136 = base * 4
                self.type = "kan"
                self.tiles_t34 = [base, base, base, base]
                self.tiles_t136 = [base * 4 + i for i in range(4)]
                self.opened = False
        
    def __str__(self):
        return (
            f"Meld(who={self.who}, m={self.m}, type={self.type}, "
            f"from_who={self.from_who}, called_index={self.called_index}, "
            f"base_t34={self.base_t34}, base_t136={self.base_t136}, "
            f"tiles_t34={self.tiles_t34}, tiles_t136={self.tiles_t136}, "
            f"opened={self.opened})"
        )
    
    def to_dict(self):
        return {"type":self.type, 
                "fromwho":self.from_who, "offset":self._get_distance(self.who, self.from_who),
                "m": sorted([t for t in self.tiles_t136]), 
                "claimed_tile": self.base_t136,
                "opened": self.opened}

    def encode(self):
        base = self.base_t34
        offset = self._get_distance(self.who, self.from_who)
        match self.type:
            case "chi":
                called = self.called_index
                base_and_called = ((base // 9) * 7 + base % 9) * 3 + called
                t0 = self.tiles_t136[0] - base * 4
                t1 = self.tiles_t136[1] - (base + 1) * 4
                t2 = self.tiles_t136[2] - (base + 2) * 4
                return (base_and_called << 10) | (t2 << 7) | (t1 << 5) | (t0 << 3) | (1 << 2) | offset
            case "pon"|"chakan":
                called = self.called_index
                base_and_called = base * 3 + called
                is_kan = self.type == "chakan"
                t4 = (self.tiles_t136[-1] % 4) if is_kan else [x for x in range(4) if x not in [y % 4 for y in self.tiles_t136]][0]
                return (base_and_called << 9) | (t4 << 5) | is_kan << 4 | (not is_kan) << 3 | offset
            case "kan":
                if self.opened:
                    called = self.called_index
                    base_and_called = base * 4 + called % 4
                    return (base_and_called << 8) | offset
                else:
                    base_and_called = base * 4
                    return base_and_called << 8


# ----------------------------
# Round tracker → snapshot builder
# ----------------------------
class TenhouRoundTracker:
    def __init__(self):
        self.reset()

    # state for a single hand
    def reset(self):
        self.oya = 0
        self.round_index = 0
        self.honba = 0
        self.riichi_sticks = 0
        self.dora_inds_t34: List[int] = []

        self.hands_136: List[List[int]] = [[], [], [], []]
        self.rivers_t34: List[List[int]] = [[], [], [], []]
        self.meld_counts: List[List[int]] = [[0]*NUM_TILES for _ in range(4)]
        self.riichi_flag: List[bool] = [False, False, False, False]
        self.riichi_turn: List[int] = [0, 0, 0, 0]
        self.discard_for_riichi: List[bool] = [False, False, False, False]
        self.discards_total = 0
        self.last_discarded_tile_136 = -1
        self.last_discarder: int = -1

        self.seen_red = {"m": False, "p": False, "s": False}

        # For calculating legal masks
        self.furiten: List[bool] = [False, False, False, False] 
        self.kuikae: List[bool] = [False, False, False, False]
        self.menzen: List[bool] = [True, True, True, True]

        # Score
        self.scores: List[int] = [250, 250, 250, 250]
        self.ranks: List[int] = [-1, -1, -1, -1]

    # helpers
    def _mark_red(self, tid: int):
        if tid == 16:
            self.seen_red["m"] = True
        elif tid == 52:
            self.seen_red["p"] = True
        elif tid == 88:
            self.seen_red["s"] = True

    def _unmark_red(self, tid: int):
        if tid == 16:
            self.seen_red["m"] = False
        elif tid == 52:
            self.seen_red["p"] = False
        elif tid == 88:
            self.seen_red["s"] = False

    # INIT
    def start_init(self, seed_list: List[int], oya: int, hands: Dict[int, List[int]], scores: List[int]):
        self.reset()
        self.round_index = seed_list[0]
        self.honba = seed_list[1]
        self.riichi_sticks = seed_list[2]
        self.oya = oya
        dora_tid = seed_list[5]
        self.dora_inds_t34 = [tid136_to_t34(dora_tid)]
        for p in range(4):
            h = hands.get(p, [])
            self.hands_136[p] = list(h)
            for tid in h:
                self._mark_red(tid)
        self.scores = list(scores)
        self.ranks = list(reversed(sorted(list(range(4)), key=lambda x: self.scores[x])))

    # events
    def draw(self, who: int, tid: int):
        self.hands_136[who].append(tid)
        #print(who, "draw: ", tid)
        #assert len(self.hands_136[who]) in [2, 5, 8, 11, 14], "TenhouRoundTracker: Num tiles in hand are not any of 2, 5, 8, 11, 14!"
        self._mark_red(tid)

    def discard(self, who: int, tid: int):
        # remove exact tid if present; else remove by 34-type as fallback
        try:
            self.hands_136[who].remove(tid)
        except ValueError:
            t34 = tid136_to_t34(tid)
            for i, x in enumerate(self.hands_136[who]):
                if tid136_to_t34(x) == t34:
                    self.hands_136[who].pop(i)
                    break
        self.rivers_t34[who].append(tid136_to_t34(tid))
        self._unmark_red(tid)
        self.discards_total += 1
        self.last_discarded_tile_136 = tid
        self.last_discarder = who
        #print(who, "discard: ", tid)
        self.discard_for_riichi[who] = False

    def apply_meld(self, who: int, m_val: int):
        m = TenhouMeld(who, m_val)
        #print(who, m_val, m.to_dict())
        base = m.base_t34
        if m.type is None or base is None:
            return
        if m.type == "chi":
            used = [base, base+1, base+2]
            for idx, t34 in enumerate(used):
                if idx == m.called_index:
                    continue
                # remove one tile of this type from caller's hand
                for i in range(len(self.hands_136[who]) - 1, -1, -1):
                    if tid136_to_t34(self.hands_136[who][i]) == t34:
                        self.hands_136[who].pop(i)
                        break
                self.meld_counts[who][t34] += 1
                self.menzen[who] = False
        elif m.type == "pon":
            removed = 0
            for i in range(len(self.hands_136[who]) - 1, -1, -1):
                if tid136_to_t34(self.hands_136[who][i]) == base:
                    self.hands_136[who].pop(i)
                    removed += 1
                    if removed == 2:
                        break
            self.meld_counts[who][base] += 3
            self.menzen[who] = False
        elif m.type == "kan":
            # open or closed
            need = 4 if m.from_who == m.who else 3
            removed = 0
            for i in range(len(self.hands_136[who]) - 1, -1, -1):
                if tid136_to_t34(self.hands_136[who][i]) == base:
                    self.hands_136[who].pop(i)
                    removed += 1
                    if removed == need:
                        break
            #assert removed == need
            self.meld_counts[who][base] += 4
            if m.from_who != m.who:
                self.menzen[who] = False
        elif m.type == "chakan":
            for i in range(len(self.hands_136[who]) - 1, -1, -1):
                if tid136_to_t34(self.hands_136[who][i]) == base:
                    self.hands_136[who].pop(i)
                    break
            self.meld_counts[who][base] += 1
            self.menzen[who] = False
        # nuki ignored for 4P

    def reach(self, who: int, step: int):
        if step == 1:
            self.discard_for_riichi[who] = True
        elif step == 2:
            self.riichi_flag[who] = True
            self.riichi_sticks += 1
            self.riichi_turn[who] = self.discards_total // 4

    def add_dora(self, tid: int):
        self.dora_inds_t34.append(tid136_to_t34(tid))

    # snapshot BEFORE a specific player's discard
    def snapshot_before_action(self, who: int, legal_actions: Optional[List[int]]=None) -> RiichiState:
        counts = [0]*NUM_TILES
        for tid in self.hands_136[who]:
            counts[tid136_to_t34(tid)] += 1
        legal = [1 if c > 0 else 0 for c in counts]

        left = (who + 1) % 4
        across = (who + 2) % 4
        right = (who + 3) % 4

        seat_wind_self = (who - self.oya) % 4
        round_wind = self.round_index // 4
        dealer_self = (who == self.oya)

        pp_left = PlayerPublic(river=list(self.rivers_t34[left]),
                               meld_counts=list(self.meld_counts[left]),
                               riichi=self.riichi_flag[left],
                               riichi_turn=self.riichi_turn[left],
                               score=self.scores[left],
                               rank=self.ranks[left])
        pp_across = PlayerPublic(river=list(self.rivers_t34[across]),
                                 meld_counts=list(self.meld_counts[across]),
                                 riichi=self.riichi_flag[across],
                                 riichi_turn=self.riichi_turn[across],
                                 score=self.scores[across],
                                 rank=self.ranks[across])
        pp_right = PlayerPublic(river=list(self.rivers_t34[right]),
                                meld_counts=list(self.meld_counts[right]),
                                riichi=self.riichi_flag[right],
                                riichi_turn=self.riichi_turn[right],
                                score=self.scores[right],
                                rank=self.ranks[right])
        
        # Compute per-tile visible counts (hand + all melds + all rivers + dora indicators), clipped to 0..4
        vc = np.zeros(NUM_TILES, dtype=np.int16)
        # self hand
        hc = counts
        if hc is not None:
            vc += np.asarray(hc, dtype=np.int16)
        # self melds
        mc_self = list(self.meld_counts[who])
        if mc_self is not None:
            vc += np.asarray(mc_self, dtype=np.int16)
        # self river
        rc_self = [0]*NUM_TILES
        for tid in self.rivers_t34[who]:
            rc_self[tid136_to_t34(tid)] += 1
        if rc_self is not None:
            vc += np.asarray(rc_self, dtype=np.int16)
        # opponents rivers + melds
        for opp in [pp_left, pp_across, pp_right]:
            rc = opp.river_counts
            if rc is not None:
                vc += np.asarray(rc, dtype=np.int16)
            else:
                r = opp.river or []
                if r:
                    vc += np.bincount(np.asarray(r, dtype=np.int16), minlength=NUM_TILES)[:NUM_TILES].astype(np.int16)
            mc = opp.meld_counts
            if mc is not None:
                vc += np.asarray(mc, dtype=np.int16)
        # dora indicators are also visible
        for ind in self.dora_inds_t34 or []:
            if 0 <= int(ind) < NUM_TILES:
                vc[int(ind)] += 1
        np.clip(vc, 0, 4, out=vc)
        remaining = 4 - vc
        np.maximum(remaining, 0, out=remaining)
        visible_counts = vc.astype(np.uint8).tolist()
        remaining_counts = remaining.astype(np.uint8).tolist()

        shantens, ukeires = compute_all_discards_ukeire_fast(counts, remaining_counts)
        
        # legal actions
        if legal_actions is None:
            total_tiles = sum(counts)
            legal_actions = [False] * NUM_ACTIONS
            meld_counts_self_arr = self.meld_counts[who]
            in_riichi = self.riichi_flag[who]
            can_declare_riichi = ((not in_riichi) or (self.discard_for_riichi[who])) and (self.menzen[who])

            if total_tiles % 3 == 2:  # player's own draw phase
                for tile, cnt in enumerate(counts):
                    
                    if cnt <= 0:
                        continue

                    discard_idx = get_action_index(tile, "discard")
                    legal_actions[discard_idx] = True

                    ready_for_riichi = can_declare_riichi and shantens[tile] == 0
                    if ready_for_riichi:
                        riichi_idx = get_action_index(tile, "riichi")
                        legal_actions[riichi_idx] = True
                        legal_actions[get_action_index(None, ("pass", "riichi"))] = True

                    if cnt >= 4:
                        #open_kan_idx = get_action_index((tile, 0), "kan") # No implementation for 大明杠
                        #legal_actions[open_kan_idx] = True
                        closed_kan_idx = get_action_index((tile, None), "kan")
                        legal_actions[closed_kan_idx] = True
                        legal_actions[get_action_index(None, ("pass", "ankan"))] = True

                    if in_riichi:
                        continue

                    if meld_counts_self_arr[tile] >= 3:
                        chakan_idx = get_action_index((tile, 0), "chakan")
                        legal_actions[chakan_idx] = True
                        legal_actions[get_action_index(None, ("pass", "chakan"))] = True

            elif total_tiles % 3 == 1:  # opponent just discarded
                last_tile_136 = self.last_discarded_tile_136
                discarder = getattr(self, "last_discarder", -1)
                if last_tile_136 >= 0 and discarder != -1 and discarder != who:
                    last_tile = tid136_to_t34(last_tile_136)
                    if not (0 <= last_tile < NUM_TILES):
                        last_tile = -1
                    if last_tile == -1:
                        count_last = 0
                    else:
                        count_last = counts[last_tile]

                    if count_last >= 2 and last_tile != -1:
                        pon_idx = get_action_index((last_tile, 0), "pon")
                        legal_actions[pon_idx] = True
                        legal_actions[get_action_index(None, ("pass", "pon"))] = True

                    if count_last >= 3 and last_tile != -1:
                        kan_idx = get_action_index((last_tile, 0), "kan")
                        legal_actions[kan_idx] = True
                        legal_actions[get_action_index(None, ("pass", "kan"))] = True

                    if last_tile != -1 and ((discarder + 1) % 4) == who and last_tile < 27:
                        suit = last_tile // 9
                        suit_min = suit * 9
                        suit_max = suit_min + 8
                        base_start = max(last_tile - 2, suit_min)
                        base_end = min(last_tile, suit_max - 2)
                        for base in range(base_start, base_end + 1):
                            called = last_tile - base
                            if called < 0 or called > 2:
                                continue
                            can_chi = True
                            for seq_tile in (base, base + 1, base + 2):
                                if seq_tile == last_tile:
                                    continue
                                if counts[seq_tile] == 0:
                                    can_chi = False
                                    break
                            if not can_chi:
                                continue
                            chi_idx = get_action_index((base, called), "chi")
                            legal_actions[chi_idx] = True
                            legal_actions[get_action_index(None, ("pass", "chi"))] = True
                    #pass_idx = get_action_index(None, "pass")
                    #legal_actions[pass_idx] = True

        # end of implementation for calculation for legal actions

        state = RiichiState(
            hand_counts=counts,
            meld_counts_self=list(self.meld_counts[who]),
            riichi=self.riichi_flag[who],
            river_self=list(self.rivers_t34[who]),
            left=pp_left, across=pp_across, right=pp_right,
            round_wind=round_wind,
            seat_wind_self=seat_wind_self,
            dealer_self=dealer_self,
            turn_number=self.discards_total // 4,
            honba=self.honba,
            riichi_sticks=self.riichi_sticks,
            score=self.scores[who],
            rank=self.ranks[who],
            dora_indicators=list(self.dora_inds_t34),
            aka5m=self.seen_red["m"], aka5p=self.seen_red["p"], aka5s=self.seen_red["s"],
            legal_discards_mask=legal,
            last_draw_136=self.hands_136[who][-1],
            last_discarded_tile_136=self.last_discarded_tile_136,
            last_discarder=(self.last_discarder-who) % 4,
            visible_counts=visible_counts,
            remaining_counts=remaining_counts,
            shantens=shantens,
            ukeires=ukeires,
            legal_actions_mask=legal_actions
        )
        return state


# ----------------------------
# XML iterators
# ----------------------------
TagLike = Union[str, bytes, os.PathLike, IO[bytes], IO[str]]

def _open_xml(xml: TagLike) -> ET.Element:
    if hasattr(xml, "read"):
        content = xml.read()
        if isinstance(content, bytes):
            return ET.fromstring(content)
        else:
            return ET.fromstring(content.encode("utf-8"))
    if isinstance(xml, (bytes, bytearray)):
        return ET.fromstring(xml)
    # path-like
    return ET.parse(str(xml)).getroot()


def iter_discard_states(xml: TagLike, iter_nakis = True, iter_end_states: bool = False) -> Generator[Tuple[RiichiState, int, int, Dict[str, int]], None, None]:
    """Yield `(state, who, action, meta)` for every discard.

    - `who`: 0..3 (Tenhou seat order)
    - `action`: 0..235 (action index)
    - `meta`: {round_index, oya, honba, riichi_sticks, action_idx}
    """
    root = _open_xml(xml)
    tracker = TenhouRoundTracker()
    action_idx = 0

    for el in root:
        raw = el.tag  # e.g., 'INIT', 'D134', 'T105'
        if raw == "INIT":
            attr = el.attrib
            seed = list(map(int, attr["seed"].split(",")))
            oya = int(attr["oya"])
            hands: Dict[int, List[int]] = {}
            for p in range(4):
                key = f"hai{p}"
                if key in attr:
                    hands[p] = list(map(int, attr[key].split(",")))
            scores = list(map(int, attr["ten"].split(",")))
            tracker.start_init(seed, oya, hands, scores)
        elif raw and raw[0] in "TUVW" and raw[1:].isdigit():
            who = "TUVW".index(raw[0])
            tid = int(raw[1:])
            for p in range(4):
                if p == tracker.last_discarder or tracker.riichi_flag[p]:
                    continue
                state = tracker.snapshot_before_action(p)
                legal_mask = [i for i in state.legal_actions_mask]
                chi_mask, pon_mask, kan_mask = [False]*NUM_ACTIONS, [False]*NUM_ACTIONS, [False]*NUM_ACTIONS
                chi_mask[68:113]=legal_mask[68:113]
                pon_mask[113:147]=legal_mask[113:147]
                kan_mask[147:181]=legal_mask[147:181]
                chi_mask[get_action_index(None, ("pass","chi"))]=True
                pon_mask[get_action_index(None, ("pass","pon"))]=True
                kan_mask[get_action_index(None, ("pass","kan"))]=True
                meta = {
                    "round_index": tracker.round_index,
                    "oya": tracker.oya,
                    "honba": tracker.honba,
                    "riichi_sticks": tracker.riichi_sticks,
                    "action_idx": action_idx,
                }
                if sum(kan_mask) > 1:
                    state = tracker.snapshot_before_action(p)
                    state.legal_actions_mask = kan_mask
                    yield (state, p, get_action_index(None, ("pass","kan")), meta)
                    action_idx += 1
                if sum(pon_mask) > 1:
                    state = tracker.snapshot_before_action(p)
                    state.legal_actions_mask = pon_mask
                    yield (state, p, get_action_index(None, ("pass","pon")), meta)
                    action_idx += 1
                if sum(chi_mask) > 1:
                    state = tracker.snapshot_before_action(p)
                    state.legal_actions_mask = chi_mask
                    yield (state, p, get_action_index(None, ("pass","chi")), meta)
                    action_idx += 1
            tracker.draw(who, tid)
        elif raw and raw[0] in "DEFG" and raw[1:].isdigit():
            who = "DEFG".index(raw[0])
            tid = int(raw[1:])
            # 如果不在立直或者在打出立直宣言牌， 返回动作
            if not tracker.riichi_flag[who]:
                state = tracker.snapshot_before_action(who)
                legal_mask = [i for i in state.legal_actions_mask]
                can_riichi = sum(legal_mask[34:68]) > 0
                can_chakan = sum(legal_mask[181:215]) > 0
                can_ankan = sum(legal_mask[215:249]) > 0
                action = get_action_index(tid136_to_t34(tid), "riichi" if tracker.discard_for_riichi[who] else "discard")
                meta = {
                    "round_index": tracker.round_index,
                    "oya": tracker.oya,
                    "honba": tracker.honba,
                    "riichi_sticks": tracker.riichi_sticks,
                    "action_idx": action_idx,
                }
                if tracker.discard_for_riichi[who]:
                    state.legal_actions_mask = [False]*NUM_ACTIONS
                    state.legal_actions_mask[34:68]=legal_mask[34:68]
                    state.legal_actions_mask[get_action_index(None, ("pass","riichi"))]=True
                    assert state.legal_actions_mask[action] == True, "Riichi: Action is not valid!"
                    yield (state, who, action, meta)
                    action_idx += 1
                    if can_chakan:
                        state = tracker.snapshot_before_action(who)
                        state.legal_actions_mask = [False]*NUM_ACTIONS
                        state.legal_actions_mask[181:215]=legal_mask[181:215]
                        action = get_action_index(None, ("pass","chakan"))
                        state.legal_actions_mask[action]=True
                        assert state.legal_actions_mask[action] == True, "Chakan: Action is not valid!"
                        yield (state, who, action, meta)
                        action_idx += 1
                    if can_ankan:
                        state = tracker.snapshot_before_action(who)
                        state.legal_actions_mask = [False]*NUM_ACTIONS
                        state.legal_actions_mask[215:249]=legal_mask[215:249]
                        action = get_action_index(None, ("pass","ankan"))
                        state.legal_actions_mask[action]=True
                        assert state.legal_actions_mask[action] == True, "Ankan: Action is not valid!"
                        yield (state, who, action, meta)
                        action_idx += 1
                else:
                    state.legal_actions_mask = [False]*NUM_ACTIONS
                    state.legal_actions_mask[:34]=legal_mask[:34]
                    assert state.legal_actions_mask[action] == True, "Discard: Action is not valid!"
                    yield (state, who, action, meta)
                    action_idx += 1
                    if can_riichi:
                        state = tracker.snapshot_before_action(who)
                        state.legal_actions_mask = [False]*NUM_ACTIONS
                        state.legal_actions_mask[34:68]=legal_mask[34:68]
                        action = get_action_index(None, ("pass","riichi"))
                        state.legal_actions_mask[action]=True
                        assert state.legal_actions_mask[action] == True, "Riichi: Action is not valid!"
                        yield (state, who, action, meta)
                        action_idx += 1
                    if can_chakan:
                        state = tracker.snapshot_before_action(who)
                        state.legal_actions_mask = [False]*NUM_ACTIONS
                        state.legal_actions_mask[181:215]=legal_mask[181:215]
                        action = get_action_index(None, ("pass","chakan"))
                        state.legal_actions_mask[action]=True
                        assert state.legal_actions_mask[action] == True, "Chakan: Action is not valid!"
                        yield (state, who, action, meta)
                        action_idx += 1
                    if can_ankan:
                        state = tracker.snapshot_before_action(who)
                        state.legal_actions_mask = [False]*NUM_ACTIONS
                        state.legal_actions_mask[215:249]=legal_mask[215:249]
                        action = get_action_index(None, ("pass","ankan"))
                        state.legal_actions_mask[action]=True
                        assert state.legal_actions_mask[action] == True, "Ankan: Action is not valid!"
                        yield (state, who, action, meta)
                        action_idx += 1
            tracker.discard(who, tid)
        elif raw == "N":
            who = int(el.attrib["who"])
            m_val = int(el.attrib["m"])
            state = tracker.snapshot_before_action(who)
            tracker.apply_meld(who, m_val)
            if iter_nakis:
                meld=TenhouMeld(who, m_val)
                hand = (meld.base_t34, meld.called_index)
                action = get_action_index(hand, meld.type)
                meta = {
                    "round_index": tracker.round_index,
                    "oya": tracker.oya,
                    "honba": tracker.honba,
                    "riichi_sticks": tracker.riichi_sticks,
                    "action_idx": action_idx,
                }
                assert state.legal_actions_mask[action] == True, "Naki: Action is not valid!"
                yield (state, who, action, meta)
                action_idx += 1
        elif raw == "REACH":
            who = int(el.attrib["who"])
            step = int(el.attrib.get("step", "0"))
            tracker.reach(who, step)
        elif raw == "DORA":
            tid = int(el.attrib["hai"])
            tracker.add_dora(tid)
        elif raw in ("AGARI", "RYUUKYOKU"):
            # end of hand
            if iter_end_states:
                if raw == "RYUUKYOKU":
                    type = el.attrib.get("type", None)
                    if type:
                        action = get_action_index(None, "ryuukyoku")
                    else:
                        action = None
                elif raw == "AGARI":
                    who = int(el.attrib["who"])
                    fromwho = int(el.attrib["fromWho"])
                    action = get_action_index(None, "tsumo" if who == fromwho else "ron")
                if action:
                    meta = {
                        "round_index": tracker.round_index,
                        "oya": tracker.oya,
                        "honba": tracker.honba,
                        "riichi_sticks": tracker.riichi_sticks,
                        "action_idx": action_idx,
                    }
                    yield (state, who, action, meta)
                    action_idx += 1
        else:
            # ignore other tags
            pass

# ----------------------------
# Convenience: collect all samples
# ----------------------------
def collect_discard_samples(xml: TagLike) -> List[Dict[str, Any]]:
    """Return a list of **lightweight dicts** for easy JSON/pickle dumping.
    (If you want full dataclasses, iterate `iter_discard_states` directly.)
    """
    items: List[Dict[str, Any]] = []
    for st, who, action, meta in iter_discard_states(xml):
        items.append({
            "who": who,
            "action": action,
            "round_index": meta["round_index"],
            "oya": meta["oya"],
            "honba": meta["honba"],
            "riichi_sticks": meta["riichi_sticks"],
            "turn_number": st.turn_number,
            "dealer_self": st.dealer_self,
            "seat_wind_self": st.seat_wind_self,
            "round_wind": st.round_wind,
            "dora_indicators": list(st.dora_indicators),
            "aka5": [st.aka5m, st.aka5p, st.aka5s],
            "hand_counts": list(st.hand_counts),
            "my_river": list(getattr(st, "my_river", [])),
            "left_river": list(st.left.river),
            "across_river": list(st.across.river),
            "right_river": list(st.right.river),
            "riichi_flags": [st.riichi, st.left.riichi, st.across.riichi, st.right.riichi],
        })
    return items


# ----------------------------
# CLI
# ----------------------------
_DEF_PREVIEW = 5000

def _main(argv: Sequence[str]) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Tenhou → RiichiState converter")
    p.add_argument("--xml", type=str, default="data/debug2.xml", help="Path to Tenhou mjlog XML")
    p.add_argument("--preview", type=int, default=_DEF_PREVIEW, help="Print first N samples")
    p.add_argument("--dump", type=str, default=None, help="Path to dump pickle of lightweight dicts")
    args = p.parse_args(argv[1:])

    from mahjong_tiles_print_style import get_action_printouts
    pouts = get_action_printouts()

    items = collect_discard_samples(args.xml)
    print("Total discard states:", len(items))
    for i, it in enumerate(items[:args.preview]):
        print(f"[{i}] who={it['who']} action={pouts[it['action']]} round={it['round_index']} oya={it['oya']} turn={it['turn_number']} dora_inds={it['dora_indicators']} aka={it['aka5']}")
    if args.dump:
        with open(args.dump, "wb") as f:
            pickle.dump(items, f)
        print("Saved:", args.dump)
    
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main(sys.argv))
    ms = [52703,60615,35847]
    who = 0
    for m in ms:
        print(m, TenhouMeld(who=who, m=m).encode())
        print(TenhouMeld(who=who, m=m))
        print(TenhouMeld(who=who, m=m).to_dict())
    print(TenhouMeld(who=1, m=32768).to_dict())
