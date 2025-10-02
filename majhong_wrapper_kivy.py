"""Kivy-based Mahjong table renderer following the GUI blueprint.

This module provides a high-level widget (`MahjongKivyWidget`) that mirrors the
pygame-driven layout documented in ``gui_readme.md``.  It renders the table,
players, control buttons, and score overlay while remaining agnostic of the
underlying environment.  The widget exposes event hooks so a controller can
supply Mahjong state snapshots and react to button presses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional, Sequence

from kivy.clock import Clock
from kivy.core.text import Label as CoreLabel
from kivy.graphics import (
    Color,
    Line,
    PushMatrix,
    PopMatrix,
    Rectangle,
    RoundedRectangle,
    Rotate,
    Translate,
)
from kivy.metrics import dp
from kivy.properties import BooleanProperty, NumericProperty, ObjectProperty
from kivy.uix.widget import Widget


# ---------------------------------------------------------------------------
# Data structures describing the snapshot rendered by the widget
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MeldSnapshot:
    """Represents a meld (chi/pon/kan) in the player layout."""

    tiles: Sequence[str]
    concealed: bool = False
    taken_from: Optional[int] = None  # seat index, used to rotate claimed tile


@dataclass(slots=True)
class PlayerSnapshot:
    """Data required to render a single seat."""

    name: str
    wind: str
    score: int
    hand: Sequence[str]
    discards: Sequence[str] = ()
    melds: Sequence[MeldSnapshot] = ()
    face_up_hand: bool = False
    drawn_tile: Optional[str] = None
    is_dealer: bool = False
    is_riichi: bool = False
    riichi_declaration_index: Optional[int] = None
    tenpai: bool = False


@dataclass(slots=True)
class ScoreEntry:
    """Entry rendered in the score overlay."""

    name: str
    seat_index: int
    score: int
    delta: int
    is_winner: bool = False
    is_dealer: bool = False


@dataclass(slots=True)
class ScoreSummary:
    """Score overlay payload."""

    title: str
    subtitle: str
    description: str
    entries: Sequence[ScoreEntry] = field(default_factory=tuple)
    yaku_lines: Sequence[str] = field(default_factory=tuple)


@dataclass(slots=True)
class MahjongTableState:
    """Snapshot consumed by :class:`MahjongKivyWidget`."""

    players: Sequence[PlayerSnapshot]
    round_wind: str
    hand_number: int
    honba: int
    riichi_sticks: int
    wall_remaining: int
    dora_indicators: Sequence[str]
    dead_wall_count: int
    phase: str
    active_player: int
    last_action: str
    reward: float
    episode_finished: bool
    auto_next: bool = True
    pause_on_score: bool = True
    step_enabled: bool = False
    score_summary: Optional[ScoreSummary] = None


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _rgba(rgb: Sequence[int], alpha: float = 1.0) -> Sequence[float]:
    return tuple(channel / 255.0 for channel in rgb) + (alpha,)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@lru_cache(maxsize=512)
def _measure_text(text: str, font_size: float, bold: bool = False) -> tuple[float, float]:
    label = CoreLabel(text=text, font_size=font_size, bold=bold)
    label.refresh()
    return label.texture.size


def _draw_text(
    widget: Widget,
    text: str,
    pos: tuple[float, float],
    font_size: float,
    color_rgba: Sequence[float],
    halign: str = "left",
    valign: str = "baseline",
    bold: bool = False,
) -> None:
    label = CoreLabel(text=text, font_size=font_size, bold=bold)
    label.refresh()
    texture = label.texture
    width, height = texture.size
    x, y = pos

    if halign == "center":
        x -= width / 2.0
    elif halign == "right":
        x -= width

    if valign == "middle":
        y -= height / 2.0
    elif valign == "top":
        y -= height

    with widget.canvas:
        Color(*color_rgba)
        Rectangle(texture=texture, pos=(x, y), size=texture.size)


# ---------------------------------------------------------------------------
# Mahjong widget implementation
# ---------------------------------------------------------------------------


class MahjongKivyWidget(Widget):
    """Kivy widget that renders the Mahjong table layout described in the README."""

    target_fps = NumericProperty(30)
    auto_next = BooleanProperty(True)
    pause_on_score = BooleanProperty(True)
    allow_step = BooleanProperty(False)
    table_state = ObjectProperty(allownone=True)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.register_event_type("on_toggle_auto")
        self.register_event_type("on_toggle_pause_on_score")
        self.register_event_type("on_step_requested")

        self._background = _rgba((12, 30, 60))
        self._play_area_color = _rgba((24, 60, 90))
        self._play_area_border = _rgba((40, 90, 130))
        self._panel_color = _rgba((5, 5, 5))
        self._panel_border = _rgba((90, 120, 160))
        self._accent_color = _rgba((170, 230, 255))
        self._text_color = _rgba((235, 235, 235))
        self._muted_text_color = _rgba((170, 190, 210))
        self._danger_color = _rgba((220, 120, 120))
        self._face_down_color = _rgba((18, 18, 22))
        self._face_down_border = _rgba((60, 60, 70))

        self._button_regions: dict[str, tuple[float, float, float, float, bool]] = {}
        self._refresh_trigger = Clock.create_trigger(self._refresh_canvas, -1)

        self.bind(size=self._trigger_refresh)
        self.bind(pos=self._trigger_refresh)
        self.bind(table_state=self._trigger_refresh)
        self.bind(auto_next=self._trigger_refresh)
        self.bind(pause_on_score=self._trigger_refresh)
        self.bind(allow_step=self._trigger_refresh)

    # -- event stubs -----------------------------------------------------
    def on_toggle_auto(self, value: bool) -> None:  # pragma: no cover - dispatched event
        pass

    def on_toggle_pause_on_score(self, value: bool) -> None:  # pragma: no cover
        pass

    def on_step_requested(self) -> None:  # pragma: no cover
        pass

    # -- state management ------------------------------------------------
    def set_table_state(self, state: MahjongTableState) -> None:
        """Update the rendered snapshot."""

        self.table_state = state
        if state is not None:
            self.auto_next = state.auto_next
            self.pause_on_score = state.pause_on_score
            self.allow_step = state.step_enabled
        self._trigger_refresh()

    def _trigger_refresh(self, *_args) -> None:
        self._refresh_trigger()

    # -- drawing ---------------------------------------------------------
    def _refresh_canvas(self, *_args) -> None:
        self.canvas.clear()

        with self.canvas:
            Color(*self._background)
            Rectangle(pos=self.pos, size=self.size)

        state = self.table_state
        if state is None:
            return

        play_side = min(self.height, self.width)
        play_side = _clamp(play_side, 200, float("inf"))
        play_x = self.x + (self.width - play_side) / 2.0
        play_y = self.y + self.height - play_side
        play_rect = (play_x, play_y, play_side, play_side)

        tile_width = _clamp(play_side / 18.0, 16, 72)
        tile_height = tile_width * 1.4
        tile_gap = max(6.0, tile_width * 0.25)

        self._draw_play_area(play_rect)
        self._draw_center_panel(state, play_rect, tile_height)
        self._draw_dead_wall(state, play_rect, tile_width, tile_height, tile_gap)
        self._draw_players(state, play_rect, tile_width, tile_height, tile_gap)
        self._draw_seat_labels(state, play_rect)
        self._draw_status_band(state, play_rect)
        self._draw_buttons(state)

        if state.score_summary is not None:
            self._draw_score_overlay(state)

    def _draw_play_area(self, play_rect: tuple[float, float, float, float]) -> None:
        x, y, w, h = play_rect
        with self.canvas:
            Color(*self._play_area_border)
            RoundedRectangle(pos=(x - dp(6), y - dp(6)), size=(w + dp(12), h + dp(12)), radius=[dp(24)] * 4)
            Color(*self._play_area_color)
            RoundedRectangle(pos=(x, y), size=(w, h), radius=[dp(20)] * 4)

    def _draw_center_panel(
        self, state: MahjongTableState, play_rect: tuple[float, float, float, float], tile_height: float
    ) -> None:
        x, y, w, h = play_rect
        panel_size = w * 0.24
        panel_x = x + (w - panel_size) / 2.0
        panel_y = y + (h - panel_size) / 2.0
        panel_rect = (panel_x, panel_y, panel_size, panel_size)

        with self.canvas:
            Color(*self._panel_border)
            RoundedRectangle(pos=(panel_x - dp(4), panel_y - dp(4)), size=(panel_size + dp(8), panel_size + dp(8)), radius=[dp(16)] * 4)
            Color(*self._panel_color)
            RoundedRectangle(pos=panel_rect[:2], size=panel_rect[2:], radius=[dp(12)] * 4)

        font_size = tile_height * 0.5
        info_lines = [
            f"Round {state.round_wind} {state.hand_number}",
            f"Honba: {state.honba}",
            f"Riichi sticks: {state.riichi_sticks}",
            f"Wall tiles: {state.wall_remaining}",
        ]
        line_height = font_size + dp(4)
        text_y = panel_y + panel_size - line_height - dp(8)
        for line in info_lines:
            _draw_text(self, line, (panel_x + dp(12), text_y), font_size, self._text_color, bold=False)
            text_y -= line_height

        # Player scores at cardinal points
        center_x = panel_x + panel_size / 2.0
        center_y = panel_y + panel_size / 2.0
        offsets = [
            (0, -(panel_size / 2.0 + dp(16))),  # South
            (panel_size / 2.0 + dp(16), 0),  # East
            (0, panel_size / 2.0 + dp(16)),  # North
            (-(panel_size / 2.0 + dp(16)), 0),  # West
        ]
        for idx, player in enumerate(state.players[:4]):
            px = center_x + offsets[idx][0]
            py = center_y + offsets[idx][1]
            color = self._accent_color if idx == state.active_player else self._text_color
            _draw_text(self, f"{player.name}: {player.score * 100}", (px, py), font_size, color, halign="center", valign="middle", bold=True)

    def _draw_dead_wall(
        self,
        state: MahjongTableState,
        play_rect: tuple[float, float, float, float],
        tile_width: float,
        tile_height: float,
        tile_gap: float,
    ) -> None:
        x, y, w, h = play_rect
        start_x = x + (w - (tile_width * 5 + tile_gap * 4)) / 2.0
        base_y = y + h * 0.3
        dora = state.dora_indicators
        for idx in range(state.dead_wall_count):
            tile_x = start_x + idx * (tile_width + tile_gap)
            if idx < len(dora):
                self._draw_tile(tile_x, base_y, tile_width, tile_height, dora[idx], face_up=True)
            else:
                self._draw_tile(tile_x, base_y, tile_width, tile_height, "", face_up=False)

    def _draw_players(
        self,
        state: MahjongTableState,
        play_rect: tuple[float, float, float, float],
        tile_width: float,
        tile_height: float,
        tile_gap: float,
    ) -> None:
        x, y, w, h = play_rect
        center_x = x + w / 2.0
        center_y = y + h / 2.0
        seat_angles = [0, -90, 180, 90]
        for idx, player in enumerate(state.players[:4]):
            angle = seat_angles[idx]
            with self.canvas:
                PushMatrix()
                Translate(center_x, center_y)
                Rotate(angle=angle)
                Translate(-w / 2.0, -h / 2.0)
            self._draw_seat_surface(player, w, h, tile_width, tile_height, tile_gap)
            with self.canvas:
                PopMatrix()

    def _draw_seat_surface(
        self,
        player: PlayerSnapshot,
        surface_width: float,
        surface_height: float,
        tile_width: float,
        tile_height: float,
        tile_gap: float,
    ) -> None:
        hand = list(player.hand)
        drawn_tile = player.drawn_tile
        if drawn_tile:
            hand_to_draw = hand[:-1]
        else:
            hand_to_draw = hand

        hand_count = len(hand_to_draw)
        if hand_count:
            total_width = hand_count * tile_width + (hand_count - 1) * tile_gap
            start_x = (surface_width - total_width) / 2.0
        else:
            start_x = surface_width / 2.0
        base_y = dp(32)

        for index, tile in enumerate(hand_to_draw):
            tile_x = start_x + index * (tile_width + tile_gap)
            self._draw_tile(
                tile_x,
                base_y,
                tile_width,
                tile_height,
                tile,
                face_up=player.face_up_hand,
            )

        if drawn_tile:
            offset_x = start_x + hand_count * (tile_width + tile_gap)
            self._draw_tile(
                offset_x + tile_gap,
                base_y + tile_height * 0.1,
                tile_width,
                tile_height,
                drawn_tile,
                face_up=player.face_up_hand,
            )

        if player.is_riichi:
            label_width, label_height = _measure_text("Riichi", tile_height * 0.4, bold=True)
            rect_width = label_width + dp(20)
            rect_height = label_height + dp(12)
            rect_x = (surface_width - rect_width) / 2.0
            rect_y = base_y + tile_height + dp(12)
            with self.canvas:
                Color(*self._panel_color)
                RoundedRectangle(pos=(rect_x, rect_y), size=(rect_width, rect_height), radius=[dp(12)] * 4)
                Color(*self._accent_color)
                Line(rounded_rectangle=(rect_x, rect_y, rect_width, rect_height, dp(12)), width=dp(2))
            _draw_text(self, "Riichi", (rect_x + rect_width / 2.0, rect_y + rect_height / 2.0), tile_height * 0.4, self._accent_color, halign="center", valign="middle", bold=True)

        discard_cols = 6
        discard_width = discard_cols * tile_width + (discard_cols - 1) * tile_gap
        discard_start_x = (surface_width - discard_width) / 2.0
        discard_start_y = base_y + tile_height + dp(48)

        for idx, tile in enumerate(player.discards):
            col = idx % discard_cols
            row = idx // discard_cols
            tile_x = discard_start_x + col * (tile_width + tile_gap)
            tile_y = discard_start_y + row * (tile_height + tile_gap)
            rotation = 90 if player.riichi_declaration_index == idx else 0
            self._draw_tile(tile_x, tile_y, tile_width, tile_height, tile, face_up=True, rotation=rotation)

        meld_y = discard_start_y + max(1, (len(player.discards) - 1) // discard_cols + 1) * (tile_height + tile_gap) + dp(24)
        for meld in player.melds:
            tiles = list(meld.tiles)
            for order, tile in enumerate(tiles):
                tile_x = surface_width - dp(32) - (len(tiles) - order) * (tile_width + tile_gap)
                face_up = not meld.concealed or order in (1, 2)
                rotation = 90 if meld.taken_from is not None and order == (meld.taken_from % len(tiles)) else 0
                self._draw_tile(tile_x, meld_y, tile_width, tile_height, tile, face_up=face_up, rotation=rotation)
            meld_y += tile_height + tile_gap

    def _draw_seat_labels(self, state: MahjongTableState, play_rect: tuple[float, float, float, float]) -> None:
        x, y, w, h = play_rect
        labels = [
            (x + w / 2.0, y - dp(24), "center", "top"),  # South
            (x + w + dp(24), y + h / 2.0, "left", "middle"),  # East
            (x + w / 2.0, y + h + dp(8), "center", "bottom"),  # North
            (x - dp(24), y + h / 2.0, "right", "middle"),  # West
        ]
        for idx, player in enumerate(state.players[:4]):
            px, py, halign, valign = labels[idx]
            color = self._accent_color if player.is_dealer else self._muted_text_color
            _draw_text(self, f"{player.wind} ({player.name})", (px, py), dp(18), color, halign=halign, valign=valign, bold=player.is_dealer)

    def _draw_status_band(self, state: MahjongTableState, play_rect: tuple[float, float, float, float]) -> None:
        x, y, w, h = play_rect
        band_height = dp(40)
        band_y = y + h + dp(16)
        band_padding = dp(16)

        with self.canvas:
            Color(*self._panel_color)
            RoundedRectangle(pos=(self.x + band_padding, band_y), size=(self.width - band_padding * 2, band_height), radius=[dp(12)] * 4)

        left_text = f"Phase: {state.phase} | Active: {state.players[state.active_player].name}"
        right_text = f"Last: {state.last_action} | Reward: {state.reward:.2f}"
        _draw_text(self, left_text, (self.x + band_padding * 2, band_y + band_height / 2.0), dp(16), self._text_color, valign="middle")

        right_x = self.x + self.width - band_padding * 2
        _draw_text(self, right_text, (right_x, band_y + band_height / 2.0), dp(16), self._text_color, halign="right", valign="middle")

        if state.episode_finished:
            _draw_text(
                self,
                "Episode finished",
                (self.x + self.width - band_padding * 2, band_y - dp(20)),
                dp(14),
                self._accent_color,
                halign="right",
                valign="top",
                bold=True,
            )

    def _draw_buttons(self, state: MahjongTableState) -> None:
        labels = [
            ("pause_on_score", "Pause on Score", True),
            ("auto_next", "Auto Next", True),
            ("step", "Next", self.allow_step or not self.auto_next),
        ]
        font_size = dp(18)
        padding = dp(12)
        button_height = dp(44)
        max_text_width = 0
        for _key, title, _enabled in labels:
            width, _ = _measure_text(title, font_size, bold=True)
            max_text_width = max(max_text_width, width)
        button_width = max(dp(160), max_text_width + padding * 2)

        start_x = self.x + self.width - button_width - dp(24)
        start_y = self.y + dp(32)
        spacing = dp(10)
        self._button_regions.clear()

        for index, (key, title, enabled) in enumerate(labels):
            by = start_y + index * (button_height + spacing)
            active = False
            if key == "pause_on_score":
                active = self.pause_on_score
            elif key == "auto_next":
                active = self.auto_next
            elif key == "step":
                active = False

            fill = self._panel_color
            border = self._panel_border
            text_color = self._text_color

            if not enabled:
                fill = _rgba((10, 10, 10))
                border = self._muted_text_color
                text_color = self._muted_text_color
            elif active and key != "step":
                fill = _rgba((35, 60, 90))
                border = self._accent_color
                text_color = self._accent_color

            with self.canvas:
                Color(*fill)
                RoundedRectangle(pos=(start_x, by), size=(button_width, button_height), radius=[dp(12)] * 4)
                Color(*border)
                Line(rounded_rectangle=(start_x, by, button_width, button_height, dp(12)), width=dp(2))

            _draw_text(
                self,
                title,
                (start_x + button_width / 2.0, by + button_height / 2.0),
                font_size,
                text_color,
                halign="center",
                valign="middle",
                bold=True,
            )
            self._button_regions[key] = (start_x, by, button_width, button_height, enabled)

    def _draw_score_overlay(self, state: MahjongTableState) -> None:
        summary = state.score_summary
        if summary is None:
            return

        width = min(self.width * 0.7, dp(560))
        height = min(self.height * 0.7, dp(480))
        pos_x = self.x + (self.width - width) / 2.0
        pos_y = self.y + (self.height - height) / 2.0

        with self.canvas:
            Color(0, 0, 0, 0.65)
            Rectangle(pos=self.pos, size=self.size)
            Color(*self._panel_color)
            RoundedRectangle(pos=(pos_x, pos_y), size=(width, height), radius=[dp(16)] * 4)
            Color(*self._accent_color)
            Line(rounded_rectangle=(pos_x, pos_y, width, height, dp(16)), width=dp(2))

        title_y = pos_y + height - dp(32)
        _draw_text(self, summary.title, (pos_x + dp(20), title_y), dp(24), self._accent_color, bold=True)
        _draw_text(self, summary.subtitle, (pos_x + dp(20), title_y - dp(32)), dp(18), self._text_color)
        _draw_text(self, summary.description, (pos_x + dp(20), title_y - dp(64)), dp(16), self._muted_text_color)

        row_y = title_y - dp(96)
        for entry in summary.entries:
            color = self._accent_color if entry.delta > 0 else self._danger_color if entry.delta < 0 else self._text_color
            if entry.is_winner:
                color = self._accent_color
            text = f"{entry.name} — {entry.score * 100} ({entry.delta:+})"
            _draw_text(self, text, (pos_x + dp(40), row_y), dp(18), color)
            if entry.is_dealer:
                _draw_text(self, "Dealer", (pos_x + width - dp(40), row_y), dp(16), self._accent_color, halign="right")
            row_y -= dp(28)

        if summary.yaku_lines:
            row_y -= dp(12)
            for line in summary.yaku_lines:
                _draw_text(self, line, (pos_x + dp(20), row_y), dp(16), self._text_color)
                row_y -= dp(22)

    # -- tile drawing ----------------------------------------------------
    def _draw_tile(
        self,
        pos_x: float,
        pos_y: float,
        width: float,
        height: float,
        tile_code: str,
        *,
        face_up: bool,
        rotation: float = 0.0,
    ) -> None:
        with self.canvas:
            PushMatrix()
            Translate(pos_x + width / 2.0, pos_y + height / 2.0)
            Rotate(angle=rotation)
            Translate(-width / 2.0, -height / 2.0)

            if face_up:
                Color(*_rgba((245, 245, 245)))
                RoundedRectangle(pos=(0, 0), size=(width, height), radius=[dp(6)] * 4)
                Color(*_rgba((200, 200, 200)))
                Line(rounded_rectangle=(0, 0, width, height, dp(6)), width=dp(1.5))
                if tile_code:
                    label = CoreLabel(text=tile_code, font_size=height * 0.45, bold=True)
                    label.refresh()
                    texture = label.texture
                    Color(*self._panel_color)
                    Rectangle(
                        texture=texture,
                        pos=(width / 2.0 - texture.size[0] / 2.0, height / 2.0 - texture.size[1] / 2.0),
                        size=texture.size,
                    )
            else:
                Color(*self._face_down_color)
                RoundedRectangle(pos=(0, 0), size=(width, height), radius=[dp(6)] * 4)
                Color(*self._face_down_border)
                Line(rounded_rectangle=(0, 0, width, height, dp(6)), width=dp(1.5))

            PopMatrix()

    # -- interaction -----------------------------------------------------
    def on_touch_down(self, touch) -> bool:
        if not self.collide_point(*touch.pos):
            return super().on_touch_down(touch)

        for key, (x, y, w, h, enabled) in self._button_regions.items():
            if not enabled:
                continue
            if x <= touch.x <= x + w and y <= touch.y <= y + h:
                if key == "pause_on_score":
                    self.pause_on_score = not self.pause_on_score
                    self.dispatch("on_toggle_pause_on_score", self.pause_on_score)
                elif key == "auto_next":
                    self.auto_next = not self.auto_next
                    self.dispatch("on_toggle_auto", self.auto_next)
                elif key == "step":
                    self.dispatch("on_step_requested")
                self._trigger_refresh()
                return True

        return super().on_touch_down(touch)
