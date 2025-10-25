# my_types.py
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Tuple, Any

# ---------- 基础类型 ----------

class Seat(IntEnum):
    EAST = 0
    SOUTH = 1
    WEST = 2
    NORTH = 3

class ActionType(IntEnum):
    RON = 10
    TSUMO = 9
    RYUUKYOKU = 8
    RIICHI = 7
    ANKAN = 6
    CHAKAN = 5
    KAN = 4
    PON = 3
    CHI = 2
    DISCARD = 1
    PASS = 0

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:  # optional: nicer in REPL/debugging
        return f"{self.__class__.__name__}.{self.name}"
    
    
PRIORITY = {
    ActionType.RON: 140,
    ActionType.TSUMO: 140,
    ActionType.RYUUKYOKU: 100,
    ActionType.RIICHI: 10,
    ActionType.ANKAN: 10,
    ActionType.CHAKAN: 10,
    ActionType.KAN: 60,
    ActionType.PON: 60,
    ActionType.CHI: 50,
    ActionType.DISCARD: 10,
    ActionType.PASS: 0,
}

@dataclass(frozen=True)
class ActionSketch:
    action_type: ActionType
    payload: dict = field(default_factory=dict)

@dataclass(frozen=True)
class Request:
    room_id: str
    step_id: int
    request_id: str
    to_seat: Seat
    actions: List[ActionSketch]
    observation: Any
    deadline_ms: int

@dataclass(frozen=True)
class Response:
    room_id: str
    step_id: int
    request_id: str
    from_seat: Seat
    chosen: ActionSketch
