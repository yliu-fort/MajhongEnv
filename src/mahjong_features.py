"""
Reusable PyTorch feature extractor for Japanese Riichi Mahjong
----------------------------------------------------------------

This module builds a (C x 34 x 34) tensor suitable for a ResNet-style
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
import math
import torch

# ----------------------------
# Tile system helpers (34-tile)
# ----------------------------
# Indexing convention:
# 0..8:  m1-m9 (characters/manzu)
# 9..17: p1-p9 (dots/pinzu)
# 18..26: s1-s9 (bamboo/souzu)
# 27..33: winds+dragons [East,South,West,North,White,Green,Red]

NUM_TILES = 34

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

    # Dora
    dora_indicators: Sequence[int] = field(default_factory=list)
    aka5m: bool = False
    aka5p: bool = False
    aka5s: bool = False

    # Legal actions (optional)
    legal_discards_mask: Optional[Sequence[int]] = None  # len 34, 0/1

    # Hooks / extra calculators
    extra: ExtraCalcs = field(default_factory=ExtraCalcs)


# ----------------------------
# Feature Extractor
# ----------------------------
class RiichiResNetFeatures(torch.nn.Module):
    """Builds a (C, 34, 34) tensor from `RiichiState`.

    Channels implemented (Baseline ~20 + a few small extras):
        1  hand_count (/4)
        2  hand_mask
        3  meld_self (/4)
        4  riichi_self (always 0; provided for symmetry/extension)
        5-7   river_count_{L,C,R} (/4)
        8-10  meld_{L,C,R} (/4)
        11-13 riichi_{L,C,R}
        14-17 round_wind one-hot (4ch)
        18-21 seat_wind_self one-hot (4ch)
        22 dealer_flag
        23 turn_number (/24)
        24 honba (/5)
        25 riichi_sticks (/5)
        26 dora_flag (any tile that is current dora)
        27 dora_indicator_mark
        28-30 aka5 flags for m/p/s
        31 legal_discard_mask (if provided, else derived from hand_count>0)

    Total: 31 channels

    Notes:
      - Rivers can be provided as ordered lists; if `river_counts` is None
        we compute counts from the river list.
      - All per-tile features are broadcast along width to shape (34,34).
      - Global scalars are written as constant planes.
    """

    def __init__(self,
                 max_turns: int = 24,
                 max_sticks: int = 5,
                 use_constant_width: bool = True):
        super().__init__()
        self.max_turns = max_turns
        self.max_sticks = max_sticks
        self.use_constant_width = use_constant_width  # keep 34x34 canvas

    # ---------- utilities ----------
    @staticmethod
    def _to_tensor_1d(arr: Sequence[int], dtype=torch.float32) -> torch.Tensor:
        t = torch.as_tensor(arr, dtype=dtype)
        if t.numel() != NUM_TILES:
            raise ValueError(f"Expected length {NUM_TILES}, got {t.numel()}")
        return t

    @staticmethod
    def _broadcast_row(v: torch.Tensor) -> torch.Tensor:
        # v: (34,)
        return v.view(NUM_TILES, 1).expand(NUM_TILES, NUM_TILES)

    @staticmethod
    def _const_plane(val: float) -> torch.Tensor:
        return torch.full((NUM_TILES, NUM_TILES), float(val))

    @staticmethod
    def _one_hot_plane(index: int, num: int) -> torch.Tensor:
        # one-hot across 0..num-1 as constant planes stacked
        planes = []
        for i in range(num):
            planes.append(RiichiResNetFeatures._const_plane(1.0 if i == index else 0.0))
        return torch.stack(planes, dim=0)  # (num,34,34)

    @staticmethod
    def _counts_from_river(river: Sequence[int]) -> List[int]:
        counts = [0]*NUM_TILES
        for t in river:
            counts[t] += 1
        return counts

    # ---------- core ----------
    def forward(self, state: RiichiState) -> Dict[str, torch.Tensor]:
        planes: List[torch.Tensor] = []

        # 1) Self hand
        hand = self._to_tensor_1d(state.hand_counts)
        hand_clamped = hand.clamp(min=0, max=4)
        planes.append(self._broadcast_row(hand_clamped / 4.0))                  # 1 hand_count
        planes.append(self._broadcast_row((hand_clamped > 0).float()))          # 2 hand_mask

        meld_self = self._to_tensor_1d(state.meld_counts_self or [0]*NUM_TILES)
        planes.append(self._broadcast_row(meld_self.clamp(0, 4) / 4.0))         # 3 meld_self
        planes.append(self._const_plane(0.0))                                    # 4 riichi_self placeholder

        # 2) Opponents public (L, C, R)
        opps = [state.left, state.across, state.right]
        for idx, opp in enumerate(opps):
            river_counts = opp.river_counts
            if river_counts is None:
                river_counts = self._counts_from_river(opp.river)
            river = self._to_tensor_1d(river_counts)
            planes.append(self._broadcast_row(river.clamp(0, 4) / 4.0))          # 5-7 river_count

        for opp in opps:
            meld_counts = opp.meld_counts or [0]*NUM_TILES
            meld = self._to_tensor_1d(meld_counts)
            planes.append(self._broadcast_row(meld.clamp(0, 4) / 4.0))           # 8-10 meld_{L,C,R}

        for opp in opps:
            planes.append(self._const_plane(1.0 if opp.riichi else 0.0))         # 11-13 riichi flags

        # 3) Round/seat
        planes.extend(list(self._one_hot_plane(state.round_wind, 4)))            # 14-17 round wind OH
        planes.extend(list(self._one_hot_plane(state.seat_wind_self, 4)))        # 18-21 seat wind OH
        planes.append(self._const_plane(1.0 if state.dealer_self else 0.0))      # 22 dealer flag

        # 4) Progress & sticks (normalized and clipped)
        tn = min(max(int(state.turn_number), 0), self.max_turns)
        planes.append(self._const_plane(tn / float(self.max_turns)))             # 23 turn_number
        hb = min(max(int(state.honba), 0), self.max_sticks)
        rs = min(max(int(state.riichi_sticks), 0), self.max_sticks)
        planes.append(self._const_plane(hb / float(self.max_sticks)))            # 24 honba
        planes.append(self._const_plane(rs / float(self.max_sticks)))            # 25 riichi_sticks

        # 5) Dora related
        is_dora = torch.zeros(NUM_TILES, dtype=torch.float32)
        ind_mark = torch.zeros(NUM_TILES, dtype=torch.float32)
        for ind in state.dora_indicators:
            try:
                d = indicator_to_dora(ind)
                is_dora[d] = 1.0
                ind_mark[ind] = 1.0
            except Exception:
                continue
        planes.append(self._broadcast_row(is_dora))                              # 26 dora flag
        planes.append(self._broadcast_row(ind_mark))                             # 27 indicator mark

        # 6) Aka5 flags (optional)
        aka = torch.zeros(NUM_TILES, dtype=torch.float32)
        if state.aka5m:
            aka[4] = 1.0   # m5 index = 0+ (5-1) = 4
        if state.aka5p:
            aka[13] = 1.0  # p5 index = 9+ (5-1) = 13
        if state.aka5s:
            aka[22] = 1.0  # s5 index = 18+ (5-1) = 22
        planes.append(self._broadcast_row(aka))                                  # 28-30 (packed as one plane)

        # 7) Legal discard mask (hand>0 by default)
        if state.legal_discards_mask is not None:
            legal = self._to_tensor_1d(state.legal_discards_mask)
        else:
            legal = (hand_clamped > 0).float()
        planes.append(self._broadcast_row(legal))                                # 31 legal mask

        x = torch.stack(planes, dim=0)  # (C,34,34)

        return {
            "x": x,                              # model input
            "legal_mask": legal,                 # (34,)
            "meta": {
                "num_channels": x.shape[0],
                "spec": "baseline+extras-31ch",
            },
        }


# ----------------------------
# Loss utility (masked CE)
# ----------------------------
class MaskedCrossEntropy(torch.nn.Module):
    """Cross-entropy with a (34,) mask where 0-weight classes are ignored.
    Assumes logits shape (B, 34) and targets shape (B,).
    """
    def __init__(self, eps: float = 1e-9):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # logits: (B,34); mask: (B,34) or (34,)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0).expand(logits.size(0), -1)
        # set very negative logits on illegal classes so softmax prob ~0
        neg_inf = -1e9
        masked_logits = logits.clone()
        masked_logits[mask <= 0.0] = neg_inf
        return torch.nn.functional.cross_entropy(masked_logits, targets)


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

    extractor = RiichiResNetFeatures()
    out = extractor(state)
    x = out["x"]
    print("Feature tensor:", x.shape, "channels=", x.shape[0])
    print("Legal mask sum:", out["legal_mask"].sum().item())
