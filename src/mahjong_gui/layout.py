"""Layout helpers for the Mahjong Kivy wrapper."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Rect:
    x: float
    y: float
    width: float
    height: float

    @property
    def left(self) -> float:
        return self.x

    @left.setter
    def left(self, value: float) -> None:
        self.x = value

    @property
    def right(self) -> float:
        return self.x + self.width

    @right.setter
    def right(self, value: float) -> None:
        self.x = value - self.width

    @property
    def top(self) -> float:
        return self.y

    @top.setter
    def top(self, value: float) -> None:
        self.y = value

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @bottom.setter
    def bottom(self, value: float) -> None:
        self.y = value - self.height

    @property
    def centerx(self) -> float:
        return self.x + self.width / 2

    @centerx.setter
    def centerx(self, value: float) -> None:
        self.x = value - self.width / 2

    @property
    def centery(self) -> float:
        return self.y + self.height / 2

    @centery.setter
    def centery(self, value: float) -> None:
        self.y = value - self.height / 2

    @property
    def center(self) -> Tuple[float, float]:
        return (self.centerx, self.centery)

    @center.setter
    def center(self, value: Tuple[float, float]) -> None:
        cx, cy = value
        self.centerx = cx
        self.centery = cy


def compute_display_order(num_players: int, focus_index: int) -> List[int]:
    """Return the player order starting from ``focus_index``."""
    if num_players <= 0:
        return []
    focus_index = max(0, min(focus_index, num_players - 1))
    max_display = min(4, num_players)
    return [((focus_index + offset) % num_players) for offset in range(max_display)]


def get_relative_angle(relative_position: int) -> int:
    """Map a relative seat position to a board rotation angle."""
    angle_map = {0: 0, 1: 90, 2: 180, 3: -90}
    normalized = relative_position % 4
    return angle_map.get(normalized, 0)


def board_rotation_angle(num_players: int, focus_index: int) -> int:
    """Return the rotation angle that centers ``focus_index`` at the bottom."""
    if num_players <= 0:
        return 0
    focus_index = max(0, min(focus_index, num_players - 1))
    return -get_relative_angle(focus_index)


__all__ = ["Rect", "compute_display_order", "get_relative_angle", "board_rotation_angle"]
