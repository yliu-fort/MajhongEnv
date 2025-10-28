"""
Reusable NumPy feature extractor for Japanese Riichi Mahjong
----------------------------------------------------------------

This module builds a (C x 34 x 34) array suitable for a ResNet-style
model that predicts the *next discard* for the current player (ego view).

It implements the **Baseline (~20 channels)** spec from the previous
message, plus a few handy utilities and masks.

You can extend it by plugging in custom calculators (e.g. shanten, ukeire)
via the hooks in `ExtraCalcs`.

Author: ChatGPT (GPT-5 Thinking)
License: MIT
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Sequence, Tuple

import numpy as np

from shanten_dp import compute_ukeire_advanced


# ----------------------------
# Tile system helpers (34-tile)
# ----------------------------
# Indexing convention:
# 0..8:  m1-m9 (characters/manzu)
# 9..17: p1-p9 (dots/pinzu)
# 18..26: s1-s9 (bamboo/souzu)
# 27..33: winds+dragons [East,South,West,North,White,Green,Red]

NUM_ACTIONS = 262
NUM_TILES = 34
WIDTH = 1
NUM_FEATURES = 136
RIVER_LEN = 24
HAND_LEN = 14
DORA_MAX = 5

# Red fives are tracked via flags, not separate indices.


def is_numbered(tile: int) -> bool:
    return 0 <= tile <= 26


def suit_of(tile: int) -> Optional[int]:
    """Return suit id for numbered tiles: 0=m,1=p,2=s ; None for honors."""
    if 0 <= tile <= 8:
        return 0
    if 9 <= tile <= 17:
        return 1
    if 18 <= tile <= 26:
        return 2
    return None


def rank_of(tile: int) -> Optional[int]:
    """Return rank 1..9 for numbered tiles; None for honors."""
    if 0 <= tile <= 8:
        return tile + 1
    if 9 <= tile <= 17:
        return tile - 9 + 1
    if 18 <= tile <= 26:
        return tile - 18 + 1
    return None


def next_rank_wrap(r: int) -> int:
    return 1 if r == 9 else r + 1


def indicator_to_dora(ind: int) -> int:
    """Map a dora indicator tile index -> actual dora tile index (34-tile space).
    Honors wrap E->S->W->N->E and W->G->R->W for dragons.
    """
    if 0 <= ind <= 26:  # numbered
        s = suit_of(ind)
        r = rank_of(ind)
        assert s is not None and r is not None
        r2 = next_rank_wrap(r)
        base = 0 if s == 0 else 9 if s == 1 else 18
        return base + (r2 - 1)
    # honors
    if 27 <= ind <= 30:  # winds E,S,W,N
        return 27 + ((ind - 27 + 1) % 4)
    if 31 <= ind <= 33:  # dragons: Wht->Grn->Red->Wht
        return 31 + ((ind - 31 + 1) % 3)
    raise ValueError("Invalid tile index")


# ----------------------------
# Data schema
# ----------------------------
@dataclass
class PlayerPublic:
    """
    Public/visible state for one opponent (or self where applicable).
    All counts are in the 34-tile space.
    """
    river: List[int] = field(default_factory=list)  # discard order as 34-idx
    river_counts: Optional[Sequence[int]] = None   # length 34 (optional; built if None)
    meld_counts: Optional[Sequence[int]] = None    # length 34 (exposed tiles from chi/pon/kan)
    riichi: bool = False
    riichi_turn: int = 0  # turn number when riichi declared (if any)
    score: int = 0
    rank: int = -1


@dataclass
class ExtraCalcs:
    """
    Optional calculators. Provide callable attributes to enrich features
    without changing this file.
    - visible_count_hook(state) -> List[int] length 34 (override default)
    - remaining_count_hook(state) -> List[int] length 34
    """
    visible_count_hook: Optional[callable] = None
    remaining_count_hook: Optional[callable] = None


@dataclass
class RiichiState:
    """Snapshot at the exact moment before the current player's discard."""
    # Ego (current player)
    hand_counts: Sequence[int]                 # length 34, 0..4
    meld_counts_self: Optional[Sequence[int]] = None  # exposed tiles only (chi/pon/kan)
    riichi: bool = False
    river_self: List[int] = field(default_factory=list)
    river_self_counts: Optional[Sequence[int]] = None

    # Opponents (relative seats: Left, Across/Center, Right)
    left: PlayerPublic = field(default_factory=PlayerPublic)
    across: PlayerPublic = field(default_factory=PlayerPublic)
    right: PlayerPublic = field(default_factory=PlayerPublic)

    # Round/seat
    round_wind: int = 0        # 0:E,1:S,2:W,3:N
    seat_wind_self: int = 0    # 0:E,1:S,2:W,3:N
    dealer_self: bool = False

    # Progress & sticks
    turn_number: int = 0       # approx 0..24 (normalize later)
    honba: int = 0
    riichi_sticks: int = 0

    # Score and rank
    score: int = 0
    rank: int = -1

    # Dora
    dora_indicators: Sequence[int] = field(default_factory=list)
    aka5m: bool = False
    aka5p: bool = False
    aka5s: bool = False

    # Legal actions (optional)
    legal_discards_mask: Optional[Sequence[int]] = None  # len 34, 0/1
    legal_actions_mask: Optional[Sequence[int]] = None  # len 253, 0/1

    # Last droped tiles (for naki)
    last_draw_136: int = -1
    last_discarded_tile_136: int = -1
    last_discarder: int = -1

    # Computed features
    visible_counts: Sequence[int] = None
    remaining_counts: Sequence[int] = None
    shantens: Sequence[int] = None
    ukeires: Sequence[int] = None

    # Hooks / extra calculators
    extra: ExtraCalcs = field(default_factory=ExtraCalcs)


# ----------------------------
# Feature Extractor
# ----------------------------
class RiichiResNetFeatures:
    """Builds a (C, 34, 34) array from :class:`RiichiState`."""

    def __init__(self,
                 max_turns: int = 24,
                 max_sticks: int = 5,
                 use_constant_width: bool = True):
        self.max_turns = max_turns
        self.max_sticks = max_sticks
        self.use_constant_width = use_constant_width  # keep 34x34 canvas

    def __call__(self, state: RiichiState) -> Dict[str, np.ndarray]:
        return self.forward(state)

    # ---------- utilities ----------
    @staticmethod
    def _to_array_1d(arr: Sequence[int], dtype: np.dtype = np.float32) -> np.ndarray:
        t = np.asarray(arr, dtype=dtype)
        if t.size != NUM_TILES:
            raise ValueError(f"Expected length {NUM_TILES}, got {t.size}")
        return t.reshape(NUM_TILES)

    @staticmethod
    def _broadcast_row(v: Sequence[float]) -> np.ndarray:
        arr = np.asarray(v, dtype=np.float32)
        if arr.size != NUM_TILES:
            raise ValueError(f"Expected length {NUM_TILES}, got {arr.size}")
        return np.broadcast_to(arr.reshape(NUM_TILES, 1), (NUM_TILES, WIDTH)).copy()

    @staticmethod
    def _const_plane(val: float) -> np.ndarray:
        return np.full((NUM_TILES, WIDTH), float(val), dtype=np.float32)

    @staticmethod
    def _one_hot_plane(index: int, num: int) -> np.ndarray:
        planes = [RiichiResNetFeatures._const_plane(1.0 if i == index else 0.0) for i in range(num)]
        return np.stack(planes, axis=0)

    @staticmethod
    def _counts_from_river(river: Sequence[int]) -> List[int]:
        counts = [0] * NUM_TILES
        for t in river:
            counts[t] += 1
        return counts

    @staticmethod
    def _one_hot_tile(t_34: int) -> List[int]:
        counts = [0] * NUM_TILES
        counts[t_34] += 1
        return counts

    @staticmethod
    def _default_visible_counts(state: RiichiState) -> List[int]:
        counts = [0] * NUM_TILES
        for i, c in enumerate(state.hand_counts):
            counts[i] += int(c)
        if state.meld_counts_self is not None:
            for i, c in enumerate(state.meld_counts_self):
                counts[i] += int(c)
        rc_self = state.river_self_counts if state.river_self_counts is not None else RiichiResNetFeatures._counts_from_river(state.river_self)
        for i, c in enumerate(rc_self):
            counts[i] += int(c)
        for opp in [state.left, state.across, state.right]:
            rc = opp.river_counts if opp.river_counts is not None else RiichiResNetFeatures._counts_from_river(opp.river)
            mc = opp.meld_counts or [0] * NUM_TILES
            for i, c in enumerate(rc):
                counts[i] += int(c)
            for i, c in enumerate(mc):
                counts[i] += int(c)
        for ind in state.dora_indicators:
            if 0 <= ind < NUM_TILES:
                counts[ind] += 1
        return [min(4, c) for c in counts]

    @staticmethod
    def _surplus_mask(counts: Sequence[int], ascending: bool) -> np.ndarray:
        c = [int(x) for x in counts]
        for i in range(NUM_TILES):
            c[i] %= 3
        rng = range(NUM_TILES - 2) if ascending else range(NUM_TILES - 3, -1, -1)
        for i in rng:
            if suit_of(i) is None or i % 9 > 6:
                continue
            m = min(c[i], c[i + 1], c[i + 2])
            if m > 0:
                c[i] -= m
                c[i + 1] -= m
                c[i + 2] -= m
        rng2 = range(NUM_TILES) if ascending else range(NUM_TILES - 1, -1, -1)
        for i in rng2:
            if suit_of(i) is None:
                continue
            for d in (1, 2):
                j = i + d if ascending else i - d
                if 0 <= j < NUM_TILES and suit_of(j) == suit_of(i):
                    m = min(c[i], c[j])
                    if m > 0:
                        c[i] -= m
                        c[j] -= m
        for i in range(NUM_TILES):
            c[i] %= 2
        return np.array([1.0 if v > 0 else 0.0 for v in c], dtype=np.float32)

    @staticmethod
    def _get_possible_moves(hand_34: Sequence[int], remaining: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        shantens = [8] * NUM_TILES
        ukeires = [0] * NUM_TILES

        for i, cnt in enumerate(hand_34):
            if cnt <= 0:
                continue
            out = compute_ukeire_advanced(hand_34, i, remaining)
            shantens[i] = out.get("shanten", 1_000_000)
            ukeires[i] = out.get("ukeire", -1)

        return (
            RiichiResNetFeatures._to_array_1d(shantens),
            RiichiResNetFeatures._to_array_1d(ukeires),
        )

    @staticmethod
    def _quantize_score(score: int) -> int:
        if score < 100:
            return 0
        if score < 200:
            return 1
        if score < 300:
            return 2
        if score < 500:
            return 3
        return 4

    # ---------- core ----------
    def forward(self, state: RiichiState) -> np.ndarray:
        planes: List[np.ndarray] = []

        # 1) Self hand
        hand = self._to_array_1d(state.hand_counts)
        hand_clamped = np.clip(hand, 0, 4)
        hand_mask = (hand_clamped > 0).astype(np.float32)
        planes.append(self._broadcast_row(hand_clamped / 4.0))  # 1 hand_count
        planes.append(self._broadcast_row(hand_mask))  # 2 hand_mask

        meld_self = self._to_array_1d(state.meld_counts_self or [0] * NUM_TILES)
        planes.append(self._broadcast_row(np.clip(meld_self, 0, 4) / 4.0))  # 3 meld_self
        planes.append(self._const_plane(0.0))  # 4 riichi_self placeholder

        # 2) Opponents public (L, C, R)
        opps = [state.left, state.across, state.right]
        for opp in opps:
            river_counts = opp.river_counts
            if river_counts is None:
                river_counts = self._counts_from_river(opp.river)
            river = self._to_array_1d(river_counts)
            planes.append(self._broadcast_row(np.clip(river, 0, 4) / 4.0))  # 5-7 river_count

            for irc in range(RIVER_LEN):
                if irc < len(opp.river):
                    planes.append(self._broadcast_row(self._to_array_1d(self._one_hot_tile(opp.river[irc]))))
                else:
                    planes.append(self._const_plane(0.0))  # 8-79 rivers

        for opp in opps:
            meld_counts = opp.meld_counts or [0] * NUM_TILES
            meld = self._to_array_1d(meld_counts)
            planes.append(self._broadcast_row(np.clip(meld, 0, 4) / 4.0))  # 80-82 meld_{L,C,R}

        for opp in opps:
            planes.append(self._const_plane(1.0 if opp.riichi else 0.0))  # 83-85 riichi flags

        # 3) Round/seat
        planes.extend(list(self._one_hot_plane(state.round_wind, 4)))  # 86-89 round wind OH
        planes.extend(list(self._one_hot_plane(state.seat_wind_self, 4)))  # 90-93 seat wind OH
        planes.append(self._const_plane(1.0 if state.dealer_self else 0.0))  # 94 dealer flag

        # 4) Progress & sticks (normalized and clipped)
        tn = min(max(int(state.turn_number), 0), self.max_turns)
        planes.append(self._const_plane(tn / float(self.max_turns)))  # 95 turn_number
        hb = min(max(int(state.honba), 0), self.max_sticks)
        rs = min(max(int(state.riichi_sticks), 0), self.max_sticks)
        planes.append(self._const_plane(hb / float(self.max_sticks)))  # 96 honba
        planes.append(self._const_plane(rs / float(self.max_sticks)))  # 97 riichi_sticks

        # 5) Dora related
        is_dora = np.zeros(NUM_TILES, dtype=np.float32)
        ind_mark = np.zeros(NUM_TILES, dtype=np.float32)
        for ind in state.dora_indicators:
            try:
                d = indicator_to_dora(ind)
            except Exception:
                continue
            is_dora[d] = 1.0
            if 0 <= ind < NUM_TILES:
                ind_mark[ind] = 1.0
        planes.append(self._broadcast_row(is_dora))  # 98 dora flag
        planes.append(self._broadcast_row(ind_mark))  # 99 indicator mark

        # 6) Aka5 flags (optional)
        aka = np.zeros(NUM_TILES, dtype=np.float32)
        if state.aka5m:
            aka[4] = 1.0
        if state.aka5p:
            aka[13] = 1.0
        if state.aka5s:
            aka[22] = 1.0
        planes.append(self._broadcast_row(aka))  # 100 aka flags

        # 7) Legal discard mask (hand>0 by default)
        if state.legal_discards_mask is not None:
            legal = self._to_array_1d(state.legal_discards_mask)
        else:
            legal = hand_mask
        planes.append(self._broadcast_row(legal))  # 101 legal mask

        # --- Intermediate features ---
        if state.visible_counts:
            visible_counts = list(state.visible_counts)
        else:
            visible_counts = self._default_visible_counts(state)
        if state.remaining_counts:
            remaining_counts = list(state.remaining_counts)
        else:
            remaining_counts = [max(0, 4 - v) for v in visible_counts]
        visible_tensor = np.asarray(visible_counts, dtype=np.float32)

        planes.append(self._broadcast_row((hand_clamped >= 2).astype(np.float32)))  # 102 tuitsu
        planes.append(self._broadcast_row((hand_clamped >= 3).astype(np.float32)))  # 103 triplet

        # Vectorised taatsu (open-ended shapes) detection per suit
        taatsu = np.zeros(NUM_TILES, dtype=np.float32)
        shuntsu = np.zeros(NUM_TILES, dtype=np.float32)
        for base in (0, 9, 18):
            h = hand_mask[base:base + 9]
            if np.sum(h) == 0:
                continue

            mask = np.zeros(9, dtype=np.float32)
            pair1 = h[:-1] * h[1:]
            pair2 = h[:-2] * h[2:]
            mask[:-1] += pair1
            mask[1:] += pair1
            mask[:-2] += pair2
            mask[2:] += pair2
            taatsu[base:base + 9] = (mask > 0).astype(np.float32)

            sh_mask = np.zeros(9, dtype=np.float32)
            triplet = h[:-2] * h[1:-1] * h[2:]
            sh_mask[:-2] += triplet
            sh_mask[1:-1] += triplet
            sh_mask[2:] += triplet
            shuntsu[base:base + 9] = (sh_mask > 0).astype(np.float32)

        planes.append(self._broadcast_row(taatsu))  # 104 taatsu
        planes.append(self._broadcast_row(shuntsu))  # 105 shuntsu

        hand_counts_list = hand_clamped.astype(int).tolist()
        planes.append(self._broadcast_row(self._surplus_mask(hand_counts_list, True)))  # 106 surplus1
        planes.append(self._broadcast_row(self._surplus_mask(hand_counts_list, False)))  # 107 surplus2

        hand_list = hand_clamped.astype(int).tolist()
        last_draw = state.last_draw_136 // 4
        sh_normal, sh_chiitoi, sh_kokushi, ukeire = 6, 8, 13, 60
        ukeire_counts = [0] * NUM_TILES
        if sum(hand_list) > 0:
            res = compute_ukeire_advanced(hand_list, last_draw, remaining_counts)
            sh_normal = res["explain"]["shanten_regular"]
            sh_chiitoi = res["explain"]["shanten_chiitoi"]
            sh_kokushi = res["explain"]["shanten_kokushi"]
            ukeire = res["ukeire"]
            for t, cnt in res["tiles"]:
                ukeire_counts[t] = cnt

        planes.append(self._const_plane(max(sh_normal, 0) / 6.0))  # 108 shanten normal
        planes.append(self._const_plane(max(sh_chiitoi, 0) / 8.0))  # 109 shanten chiitoi
        planes.append(self._const_plane(max(sh_kokushi, 0) / 13.0))  # 110 shanten kokushi

        ukeire_counts_arr = self._to_array_1d(ukeire_counts)
        planes.append(self._const_plane(min(ukeire, 60) / 60.0))  # 111 ukeire count

        for opp in opps:
            rc = opp.river_counts if opp.river_counts is not None else self._counts_from_river(opp.river)
            gen = np.array([1.0 if c > 0 else 0.0 for c in rc], dtype=np.float32)
            planes.append(self._broadcast_row(gen))  # 112-114 genbutsu

        planes.append(self._broadcast_row((visible_tensor >= 4).astype(np.float32)))  # 115 4 visible
        planes.append(self._broadcast_row((visible_tensor >= 3).astype(np.float32)))  # 116 3 visible
        planes.append(self._broadcast_row((visible_tensor >= 2).astype(np.float32)))  # 117 2 visible

        total_hand_meld = hand_clamped + meld_self
        dora_count = float(np.sum(total_hand_meld * is_dora))
        if state.aka5m and is_dora[4] == 0:
            dora_count += 1
        if state.aka5p and is_dora[13] == 0:
            dora_count += 1
        if state.aka5s and is_dora[22] == 0:
            dora_count += 1
        planes.append(self._const_plane(min(dora_count, 5) / 5.0))  # 118 dora count hand

        for opp in opps:
            meld = self._to_array_1d(opp.meld_counts or [0] * NUM_TILES)
            count = float(np.sum(meld * is_dora))
            planes.append(self._const_plane(min(count, 5) / 5.0))  # 119-121 visible dora in melds

        total_dora_visible = float(np.sum(visible_tensor * is_dora))
        if state.aka5m and is_dora[4] == 0:
            total_dora_visible += 1
        if state.aka5p and is_dora[13] == 0:
            total_dora_visible += 1
        if state.aka5s and is_dora[22] == 0:
            total_dora_visible += 1
        planes.append(self._const_plane(min(total_dora_visible, 10) / 10.0))  # 122 visible dora total

        furiten = 0
        my_river_counts = self._to_array_1d(state.river_self_counts or self._counts_from_river(state.river_self))
        if float(np.sum(ukeire_counts_arr * my_river_counts)) > 0:
            furiten = 1
        planes.append(self._const_plane(float(furiten)))  # 123 furiten self

        for opp in opps:
            rt = opp.riichi_turn if opp.riichi_turn >= 0 else 0
            rt = max(0, min(rt, self.max_turns))
            planes.append(self._const_plane(rt / float(self.max_turns)))  # 124-126 riichi turn

        if state.shantens:
            shantens = np.clip(self._to_array_1d(state.shantens), 0, 8)
            ukeires = np.clip(self._to_array_1d(state.ukeires), 0, 60)
        else:
            #shantens, ukeires = RiichiResNetFeatures._get_possible_moves(hand_list, remaining_counts)
            shantens = self._to_array_1d([8] * NUM_TILES)
            ukeires = self._to_array_1d([60] * NUM_TILES)
            shantens = np.clip(shantens, 0, 8)
            ukeires = np.clip(ukeires, 0, 60)
        planes.append(self._broadcast_row(shantens / 8.0))  # 127 shantens
        planes.append(self._broadcast_row(ukeires / 60.0))  # 128 ukeires

        planes.append(self._const_plane(RiichiResNetFeatures._quantize_score(state.score) / 4.0))
        for opp in opps:
            planes.append(self._const_plane(RiichiResNetFeatures._quantize_score(opp.score) / 4.0))

        planes.append(self._const_plane(max(0.0, state.rank) / 3.0))
        for opp in opps:
            planes.append(self._const_plane(max(0.0, opp.rank) / 3.0))

        x = np.stack(planes, axis=0).astype(np.float32)

        if state.legal_actions_mask is None:
            raise ValueError("legal_actions_mask must be provided")
        legal_actions = np.asarray(state.legal_actions_mask, dtype=np.float32)
        
        return x[None,:,:,0]

        return {
            "x": x,
            "legal_mask": legal_actions,
            "meta": {
                "num_channels": x.shape[0],
                "spec": "baseline-136ch-253ac",
            },
        }


def get_action_index(t_34, type):
    """Map an action description to the flat action index used in ``print_all_actions``."""
    """t_34 can be the tile index or the (t_34, called_index) for chi. """
    action_type = type.lower() if isinstance(type, str) else type

    if action_type == "discard":
        return int(t_34)

    if action_type == "riichi":
        return 34 + int(t_34)

    if action_type == "chi":
        base, called = t_34
        base, called = int(base), int(called)
        suit = base // 9
        rank = base % 9
        offset = 68 + suit * 15
        if called == 0:
            local_a = rank + 1
            return offset + local_a
        elif called == 1:
            local_a = rank
            return offset + 8 + local_a
        elif called == 2:
            local_a = rank
            return offset + local_a
        raise ValueError(f"Unsupported chi shape: {t_34}")

    if action_type == "pon":
        base, called = t_34
        base = int(base)
        return 113 + base

    if action_type == "kan":
        base, called = t_34
        base = int(base)
        if called != None:
            return 147 + base
        else:
            return 215 + base

    if action_type == "chakan":
        base, called = t_34
        base = int(base)
        return 181 + base

    if action_type == "ryuukyoku":
        return 249

    if action_type == "ron":
        return 250

    if action_type == "tsumo":
        return 251

    if action_type in ("cancel", "pass"):
        return 252

    if action_type == ("pass", "riichi"):
        return 253

    if action_type == ("pass", "chi"):
        return 254

    if action_type == ("pass", "pon"):
        return 255

    if action_type == ("pass", "kan"):
        return 256

    if action_type == ("pass", "ankan"):
        return 257

    if action_type == ("pass", "chakan"):
        return 258

    if action_type == ("pass", "ryuukyoku"):
        return 259

    if action_type == ("pass", "ron"):
        return 260

    if action_type == ("pass", "tsumo"):
        return 261

    raise ValueError(f"Unsupported action type: {type}")


def get_action_from_index(i):
    # discard
    if i < 34:
        return (i, False)

    # riichi
    elif i < 68:
        return (i-34, True)

    # chi
    elif i < 113:
        pouts = []
        for r in range(3):
            for j in range(8):
                s = (r*9+j, r*9+j+1)
                pouts.append((s, True))
            for j in range(7):
                s = (r*9+j, r*9+j+2)
                pouts.append((s, True))
        return pouts[i-68]

    # pon
    elif i < 147:
        k = (i-113, i-113)
        return (k, True)

    # kan
    elif i < 181:
        k = (i-147, i-147, i-147)
        return (k, True)

    # chakan
    elif i < 215:
        k = (i-181,)
        return (k, True)

    # ankan
    elif i < 249:
        k = (i-215, i-215, i-215, i-215)
        return (k, True)

    # ryuukyoku
    elif i == 249:
        return (255, True)

    # ron
    elif i == 250:
        return (255, True)

    # tsumo
    elif i == 251:
        return (255, True)

    # cancel
    elif i == 252:
        return (255, False)

    # cancel (action group specific)
    elif 252 < i < 262:
        return (255, False)

    else:
        return (-1, False)


def get_actions():
    pouts = []

    # discard
    for i in range(34):
        pouts.append((i, False))

    # riichi
    for i in range(34):
        pouts.append((i, True))

    # chi
    for r in range(3):
        for i in range(8):
            s = (r*9+i, r*9+i+1)
            pouts.append((s, True))
        for i in range(7):
            s = (r*9+i, r*9+i+2)
            pouts.append((s, True))

    # pon
    for i in range(34):
        k = (i, i)
        pouts.append((k, True))

    # kan
    for i in range(34):
        k = (i, i, i)
        pouts.append((k, True))

    # chakan
    for i in range(34):
        k = i
        pouts.append((k, True))

    # ankan
    for i in range(34):
        k = i
        pouts.append((k, True))

    # ryuukyoku
    pouts.append((255, True))

    # ron
    pouts.append((255, True))

    # tsumo
    pouts.append((255, True))

    # cancel
    pouts.append((255, False))

    # cancel (action group specific)
    for _ in range(9):
        pouts.append((255, False))

    return pouts


# ----------------------------
# Mini example / smoke test
# ----------------------------
if __name__ == "__main__":
    # Fabricate a tiny state and run the extractor
    state = RiichiState(
        hand_counts=[0]*NUM_TILES,
        meld_counts_self=[0]*NUM_TILES,
        left=PlayerPublic(river=[27, 5, 5, 31], riichi=False),
        across=PlayerPublic(river=[9, 10], riichi=True),
        right=PlayerPublic(river=[18], riichi=False),
        round_wind=0, seat_wind_self=1, dealer_self=False,
        turn_number=8, honba=1, riichi_sticks=0,
        dora_indicators=[3, 31], aka5m=True, aka5p=False, aka5s=True,
    )
    # Give the player a few tiles and legal mask
    state.hand_counts[0] = 1  # m1
    state.hand_counts[4] = 2  # m5
    state.hand_counts[27] = 1 # East
    state.legal_discards_mask = [1 if c>0 else 0 for c in state.hand_counts]
    state.legal_actions_mask = [1 if c<34 else 0 for c in range(253)]

    extractor = RiichiResNetFeatures()
    out = extractor(state)
    x = out["x"]
    legal = out["legal_mask"]
    print("Feature tensor:", x.shape, "channels=", x.shape[0])
    print("Legal mask", legal.shape, " sum:", float(np.sum(out["legal_mask"])))
