"""Kivy-based Mahjong environment wrapper.

This module mirrors the :mod:`mahjong_wrapper` pygame implementation while
rendering the board using Kivy widgets.  The goal is to keep layout metrics,
colour palette, and control semantics consistent with the pygame wrapper so
that the two front-ends behave identically from the environment's
perspective.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Tuple

from kivy.clock import Clock
from kivy.core.text import Label as CoreLabel
from kivy.lang import Builder
from kivy.metrics import dp
from kivy.properties import BooleanProperty, NumericProperty, ObjectProperty
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.widget import Widget

try:  # pragma: no cover - optional dependency
    from kivy_garden.svg import Svg
except Exception:  # pragma: no cover - optional dependency
    Svg = None  # type: ignore[assignment]

from mahjong_env import MahjongEnvBase as _BaseMahjongEnv

_TILE_SYMBOLS: Tuple[str, ...] = (
    "Man1",
    "Man2",
    "Man3",
    "Man4",
    "Man5",
    "Man6",
    "Man7",
    "Man8",
    "Man9",
    "Pin1",
    "Pin2",
    "Pin3",
    "Pin4",
    "Pin5",
    "Pin6",
    "Pin7",
    "Pin8",
    "Pin9",
    "Sou1",
    "Sou2",
    "Sou3",
    "Sou4",
    "Sou5",
    "Sou6",
    "Sou7",
    "Sou8",
    "Sou9",
    "Ton",
    "Nan",
    "Shaa",
    "Pei",
    "Haku",
    "Hatsu",
    "Chun",
    "Man5-Dora",
    "Pin5-Dora",
    "Sou5-Dora",
)


@dataclass(slots=True)
class _RenderPayload:
    action: Optional[int]
    reward: float
    done: bool
    info: dict[str, Any]


class MahjongTileWidget(Widget):
    """Widget representing a single mahjong tile."""

    tile_id = NumericProperty(0)
    face_up = BooleanProperty(True)
    angle = NumericProperty(0.0)
    resources = ObjectProperty(allownone=True)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.bind(
            tile_id=self._refresh,
            face_up=self._refresh,
            angle=self._refresh,
            size=self._refresh,
            pos=self._refresh,
            resources=self._refresh,
        )
        self._svg_node: Optional[Svg] = None
        self._refresh()

    # ------------------------------------------------------------------
    def _refresh(self, *_: Any) -> None:
        resources: MahjongTileResources = self.resources  # type: ignore[assignment]
        if resources is None:
            return

        self.canvas.clear()
        tile_34 = self._normalise_tile(self.tile_id)
        width = max(1, int(self.width))
        height = max(1, int(self.height))

        if not self.face_up:
            resources.draw_face_down(self.canvas, (width, height), self.angle, self.pos)
            return

        asset_path = resources.asset_for(tile_34)
        if asset_path is not None and Svg is not None:
            try:  # pragma: no branch - runtime guard
                svg = Svg(str(asset_path))
                self._svg_node = svg
            except Exception:  # pragma: no cover - optional dependency guard
                svg = None
        else:
            svg = None

        if svg is not None:
            resources.draw_svg(self.canvas, svg, (width, height), self.angle, self.pos)
        else:
            label = _TILE_SYMBOLS[tile_34]
            resources.draw_placeholder(
                self.canvas, (width, height), self.angle, self.pos, tile_34, label
            )

    @staticmethod
    def _normalise_tile(tile_136: int) -> int:
        tile_34 = tile_136 // 4
        if tile_136 == 16:
            return 34
        if tile_136 == 52:
            return 35
        if tile_136 == 88:
            return 36
        return tile_34


class MahjongTileResources:
    """Utility for drawing tiles with consistent palette and fallbacks."""

    def __init__(self, font_name: Optional[str]) -> None:
        self._font_name = font_name
        self._asset_root = (
            Path(__file__).resolve().parent.parent / "assets" / "tiles" / "Regular"
        )
        self._placeholder_colors = {
            "m": (210 / 255.0, 90 / 255.0, 90 / 255.0, 1.0),
            "p": (90 / 255.0, 150 / 255.0, 225 / 255.0, 1.0),
            "s": (90 / 255.0, 190 / 255.0, 120 / 255.0, 1.0),
            "z": (220 / 255.0, 210 / 255.0, 150 / 255.0, 1.0),
        }
        self._face_down_color = (18 / 255.0, 18 / 255.0, 22 / 255.0, 1.0)
        self._face_down_border = (60 / 255.0, 60 / 255.0, 70 / 255.0, 1.0)
        self._tile_cache: dict[int, Optional[Path]] = {}

    # ------------------------------------------------------------------
    def asset_for(self, tile_34: int) -> Optional[Path]:
        if tile_34 in self._tile_cache:
            return self._tile_cache[tile_34]

        symbols = list(_TILE_SYMBOLS)
        asset_path: Optional[Path] = None
        if 0 <= tile_34 < len(symbols):
            candidate = self._asset_root / f"{symbols[tile_34]}.svg"
            if candidate.exists():
                asset_path = candidate

        self._tile_cache[tile_34] = asset_path
        return asset_path

    # ------------------------------------------------------------------
    def draw_svg(
        self,
        canvas,
        svg: Svg,
        size: Tuple[int, int],
        angle: float,
        pos: Tuple[float, float],
    ) -> None:
        width, height = size
        scale = min(width / max(1.0, svg.width), height / max(1.0, svg.height))
        with canvas:
            from kivy.graphics import Color, PushMatrix, PopMatrix, Rotate, Scale, Translate

            Color(1, 1, 1, 1)
            PushMatrix()
            Translate(pos[0], pos[1])
            Translate(width / 2.0, height / 2.0)
            if angle:
                Rotate(angle=angle, origin=(0, 0))
            Scale(scale, scale, 1)
            Translate(-svg.width / 2.0, -svg.height / 2.0)
            canvas.add(svg)
            PopMatrix()

    # ------------------------------------------------------------------
    def draw_placeholder(
        self,
        canvas,
        size: Tuple[int, int],
        angle: float,
        pos: Tuple[float, float],
        tile_34: int,
        label: str,
    ) -> None:
        width, height = size
        suit = "z"
        if tile_34 < 27:
            suit = "m" if tile_34 < 9 else "p" if tile_34 < 18 else "s"

        color = self._placeholder_colors.get(suit, (0.78, 0.78, 0.78, 1.0))
        outline = (245 / 255.0, 245 / 255.0, 245 / 255.0, 1.0)
        font_size = max(12, int(height * 0.45))
        core_label = CoreLabel(text=label, font_size=font_size, font_name=self._font_name)
        core_label.refresh()
        texture = core_label.texture

        from kivy.graphics import (
            Color,
            Rectangle,
            Line,
            PushMatrix,
            PopMatrix,
            Rotate,
            Translate,
        )

        with canvas:
            PushMatrix()
            Translate(pos[0], pos[1])
            Translate(width / 2.0, height / 2.0)
            if angle:
                Rotate(angle=angle, origin=(0, 0))
            Translate(-width / 2.0, -height / 2.0)
            Color(*color)
            Rectangle(pos=(0, 0), size=(width, height))
            Color(*outline)
            Line(rectangle=(0, 0, width, height), width=1.2)
            if texture is not None:
                Color(0.08, 0.08, 0.08, 1.0)
                text_pos = (
                    (width - texture.width) / 2.0,
                    (height - texture.height) / 2.0,
                )
                Rectangle(pos=text_pos, size=texture.size, texture=texture)
            PopMatrix()

    # ------------------------------------------------------------------
    def draw_face_down(
        self, canvas, size: Tuple[int, int], angle: float, pos: Tuple[float, float]
    ) -> None:
        width, height = size
        from kivy.graphics import (
            Color,
            Rectangle,
            Line,
            PushMatrix,
            PopMatrix,
            Rotate,
            Translate,
        )

        with canvas:
            PushMatrix()
            Translate(pos[0], pos[1])
            Translate(width / 2.0, height / 2.0)
            if angle:
                Rotate(angle=angle, origin=(0, 0))
            Translate(-width / 2.0, -height / 2.0)
            Color(*self._face_down_color)
            Rectangle(pos=(0, 0), size=(width, height))
            Color(*self._face_down_border)
            Line(rectangle=(0, 0, width, height), width=1.2)
            PopMatrix()


class MahjongBoardWidget(FloatLayout):
    """Widget responsible for arranging the rendered components."""

    controller = ObjectProperty(allownone=True)
    board_size = NumericProperty(0.0)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._tiles: list[MahjongTileWidget] = []
        self.bind(size=self._update_board_size, pos=self._update_board_size)
        self._update_board_size()

    def clear_tiles(self) -> None:
        for widget in self._tiles:
            self.remove_widget(widget)
        self._tiles.clear()

    def add_tile(
        self,
        tile_id: int,
        pos: Tuple[float, float],
        size: Tuple[float, float],
        face_up: bool,
        angle: float,
    ) -> MahjongTileWidget:
        widget = MahjongTileWidget(
            tile_id=tile_id,
            size=size,
            pos=pos,
            face_up=face_up,
            angle=angle,
            resources=self.controller.tile_resources if self.controller else None,
        )
        self._tiles.append(widget)
        self.add_widget(widget)
        return widget

    def _update_board_size(self, *_: Any) -> None:
        self.board_size = min(self.width, self.height)


class MahjongRootLayout(FloatLayout):
    """Root layout defined by the KV language file."""

    controller = ObjectProperty(allownone=True)
    board = ObjectProperty(allownone=True)
    status_label = ObjectProperty(allownone=True)
    auto_button = ObjectProperty(allownone=True)
    step_button = ObjectProperty(allownone=True)
    pause_button = ObjectProperty(allownone=True)


class MahjongEnvKivyWrapper:
    """Mahjong environment with a Kivy front-end."""

    def __init__(
        self,
        *args: Any,
        env: _BaseMahjongEnv = None,
        fps: int = 30,
        font_name: Optional[str] = None,
        font_size: int = 12,
        fallback_fonts: Optional[Sequence[str]] = None,
        kv_file: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        del args, kwargs
        self._env = env
        self._fps = max(1, fps)
        self._font_name = font_name
        self._font_size = font_size
        self._fallback_fonts: Tuple[str, ...] = (
            tuple(fallback_fonts)
            if fallback_fonts is not None
            else (
                "Noto Sans CJK SC",
                "Noto Sans CJK TC",
                "Noto Sans CJK JP",
                "Source Han Sans CN",
                "Source Han Sans TW",
                "Source Han Sans JP",
                "Microsoft YaHei",
                "Microsoft JhengHei",
                "Yu Gothic",
                "Meiryo",
                "MS Gothic",
                "SimHei",
                "WenQuanYi Zen Hei",
                "Arial Unicode MS",
            )
        )
        self._background_color = (12 / 255.0, 30 / 255.0, 60 / 255.0, 1.0)
        self._play_area_color = (24 / 255.0, 60 / 255.0, 90 / 255.0, 1.0)
        self._play_area_border = (40 / 255.0, 90 / 255.0, 130 / 255.0, 1.0)
        self._panel_color = (5 / 255.0, 5 / 255.0, 5 / 255.0, 0.92)
        self._panel_border = (90 / 255.0, 120 / 255.0, 160 / 255.0, 1.0)
        self._accent_color = (170 / 255.0, 230 / 255.0, 255 / 255.0, 1.0)
        self._text_color = (235 / 255.0, 235 / 255.0, 235 / 255.0, 1.0)
        self._muted_text_color = (170 / 255.0, 190 / 255.0, 210 / 255.0, 1.0)
        self._danger_color = (220 / 255.0, 120 / 255.0, 120 / 255.0, 1.0)
        self._line_height = font_size + 4
        self._tile_metrics: dict[str, Tuple[int, int]] = {}
        self._tile_resources = MahjongTileResources(font_name)
        self._clock_event = None
        self._root: Optional[MahjongRootLayout] = None
        self._board: Optional[MahjongBoardWidget] = None
        self._auto_advance = True
        self._step_once_requested = False
        self._pause_on_score = False
        self._score_pause_active = False
        self._score_pause_pending = False
        self._last_phase_is_score_last = ""
        self._last_payload = _RenderPayload(action=None, reward=0.0, done=False, info={})
        self._kv_file = kv_file
        self._riichi_states: list[bool] = []
        self._riichi_pending: list[bool] = []
        self._discard_counts: list[int] = []
        self._riichi_declarations: dict[int, int] = {}

    # ------------------------------------------------------------------
    @property
    def tile_resources(self) -> MahjongTileResources:
        return self._tile_resources

    # ------------------------------------------------------------------
    def bind_root(self, root: MahjongRootLayout) -> None:
        self._root = root
        root.controller = self
        self._board = root.board
        if self._board is not None:
            self._board.controller = self
        self._update_control_labels()

    # ------------------------------------------------------------------
    def schedule(self) -> None:
        if self._clock_event is not None:
            self._clock_event.cancel()
        interval = 1.0 / self._fps
        self._clock_event = Clock.schedule_interval(self._on_frame, interval)

    # ------------------------------------------------------------------
    def reset(self, *args: Any, **kwargs: Any):
        observation = self._env.reset(*args, **kwargs)
        self._last_payload = _RenderPayload(action=None, reward=0.0, done=self._env.done, info={})
        self._step_once_requested = False
        self._score_pause_active = False
        self._score_pause_pending = False
        self._last_phase_is_score_last = getattr(self._env, "phase", "")
        self._riichi_states = []
        self._riichi_pending = []
        self._discard_counts = []
        self._riichi_declarations = {}
        self._render()
        return observation

    # ------------------------------------------------------------------
    def step(self, action: int):
        if self._score_pause_active and not self._step_once_requested:
            return self._env.get_observation(self._env.current_player), 0.0, self._env.done, {}

        if self._step_once_requested:
            self._step_once_requested = False

        observation, reward, done, info = self._env.step(action)
        self._last_payload = _RenderPayload(action=action, reward=reward, done=done, info=info)
        self._render()
        return observation, reward, done, info

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._clock_event is not None:
            self._clock_event.cancel()
            self._clock_event = None

    # ------------------------------------------------------------------
    def toggle_auto(self) -> None:
        self._auto_advance = not self._auto_advance
        if self._auto_advance:
            self._step_once_requested = False
        self._update_control_labels()

    def request_step(self) -> None:
        self._step_once_requested = True

    def toggle_pause_on_score(self) -> None:
        self._pause_on_score = not self._pause_on_score
        if not self._pause_on_score:
            self._score_pause_pending = False
            self._score_pause_active = False
        self._update_control_labels()

    # ------------------------------------------------------------------
    def _on_frame(self, _dt: float) -> None:
        if self._env.done:
            return

        phase = getattr(self._env, "phase", "")
        is_score_phase = phase == "score"
        if is_score_phase and not self._last_phase_is_score_last == phase:
            self._score_pause_pending = True
        self._last_phase_is_score_last = phase

        if not is_score_phase and self._score_pause_active:
            self._score_pause_active = False

        if self._pause_on_score and self._score_pause_pending:
            self._score_pause_pending = False
            self._score_pause_active = True

        if self._score_pause_active:
            return

        if not self._auto_advance and not self._step_once_requested:
            return

        actions: list[int] = []
        action_space = getattr(self._env, "legal_actions", None)
        if isinstance(action_space, Iterable):
            actions = list(action_space)
        else:
            getter = getattr(self._env, "get_legal_actions", None)
            if callable(getter):
                result = getter()
                if isinstance(result, Iterable):
                    actions = list(result)
        if actions:
            chosen_action = actions[0]
            self.step(chosen_action)
            return
        self._render()

    # ------------------------------------------------------------------
    def _update_control_labels(self) -> None:
        if self._root is None:
            return

        if self._root.auto_button is not None:
            self._root.auto_button.text = "Auto" if not self._auto_advance else "Auto (On)"
        if self._root.step_button is not None:
            self._root.step_button.text = "Step"
        if self._root.pause_button is not None:
            self._root.pause_button.text = (
                "Pause on Score" if not self._pause_on_score else "Resume on Score"
            )

    # ------------------------------------------------------------------
    def _render(self) -> None:
        board = self._board
        if board is None:
            return

        board.canvas.clear()
        board.clear_tiles()
        play_area = self._compute_play_area(board)
        self._compute_tile_metrics(play_area)
        self._draw_center_panel(play_area)
        self._draw_player_areas(play_area)
        self._draw_walls(play_area)
        self._draw_seat_labels(play_area)
        self._draw_status_text()
        self._update_control_labels()

    # ------------------------------------------------------------------
    def _compute_play_area(self, board: MahjongBoardWidget) -> Tuple[float, float, float, float]:
        size = min(board.width, board.height)
        area_width = area_height = size * 0.86
        center_x, center_y = board.center
        x = center_x - area_width / 2.0
        y = center_y - area_height / 2.0
        return (x, y, area_width, area_height)

    # ------------------------------------------------------------------
    def _compute_tile_metrics(self, play_rect: Tuple[float, float, float, float]) -> None:
        width = max(1, int(play_rect[2]))
        height = max(1, int(play_rect[3]))
        base_tile_width = max(16, min(width // 27, height // 24, 72))
        base_tile_height = max(24, int(base_tile_width * 1.4))
        tile_size = (base_tile_width, base_tile_height)
        self._tile_metrics = {
            "south_hand": tile_size,
            "north_hand": tile_size,
            "side_hand": tile_size,
            "discard": tile_size,
            "wall": tile_size,
            "meld": tile_size,
        }

    # ------------------------------------------------------------------
    def _draw_center_panel(self, play_rect: Tuple[float, float, float, float]) -> None:
        if self._root is None:
            return

        x, y, width, height = play_rect
        center_width = max(100, int(width * 0.24))
        center_height = max(100, int(height * 0.24))
        center_x = x + width / 2.0
        center_y = y + height / 2.0
        panel_x = center_x - center_width / 2.0
        panel_y = center_y - center_height / 2.0

        board = self._board
        if board is None:
            return

        from kivy.graphics import Color, Rectangle, Line

        with board.canvas:
            Color(*self._play_area_color)
            Rectangle(pos=(x, y), size=(width, height))
            Color(*self._play_area_border)
            Line(rectangle=(x, y, width, height), width=dp(2))
            Color(*self._panel_color)
            Rectangle(pos=(panel_x, panel_y), size=(center_width, center_height))
            Color(*self._panel_border)
            Line(rectangle=(panel_x, panel_y, center_width, center_height), width=dp(2))

        self._update_center_text(center_x, center_y, center_width, center_height)

    # ------------------------------------------------------------------
    def _update_center_text(
        self, center_x: float, center_y: float, width: float, height: float
    ) -> None:
        if self._root is None:
            return

        round_data = getattr(self._env, "round", [0, 0])
        round_index = round_data[0] if isinstance(round_data[0], int) else 0
        wind_names = ["East", "South", "West", "North"]
        wind = wind_names[(round_index // 4) % 4]
        hand_number = round_index % 4 + 1
        round_text = f"{wind} {hand_number}"

        honba = round_data[1] if len(round_data) > 1 and isinstance(round_data[1], int) else 0
        riichi = getattr(self._env, "num_riichi", 0)
        tiles_remaining = len(getattr(self._env, "deck", []))
        counter_texts = (
            f"Honba {honba}",
            f"Riichi {riichi}",
            f"Tiles {tiles_remaining}",
        )

        top = center_y + height / 2.0 - dp(28)
        self._write_text(round_text, center_x, top, align="center", font_size=self._font_size + 6)

        next_top = top - dp(32)
        for text in counter_texts:
            self._write_text(
                text,
                center_x,
                next_top,
                align="center",
                font_size=self._font_size,
                color=self._muted_text_color,
            )
            next_top -= dp(18)

        scores = getattr(self._env, "scores", [])
        current_player = getattr(self._env, "current_player", 0)
        offsets = (
            (0, height / 2 + dp(14)),
            (width / 2 + dp(18), 0),
            (0, -height / 2 - dp(14)),
            (-width / 2 - dp(18), 0),
        )
        for idx, offset in enumerate(offsets):
            if idx >= len(scores):
                continue
            value = scores[idx] * 100
            color = self._accent_color if idx == current_player else self._text_color
            pos_x = center_x + offset[0]
            pos_y = center_y + offset[1]
            self._write_text(
                f"{value:5d}",
                pos_x,
                pos_y,
                align="center",
                font_size=self._font_size,
                color=color,
            )

    # ------------------------------------------------------------------
    def _write_text(
        self,
        text: str,
        x: float,
        y: float,
        *,
        align: str = "left",
        font_size: int,
        color: Optional[Tuple[float, float, float, float]] = None,
    ) -> None:
        board = self._board
        if board is None:
            return

        rgba = color or self._text_color
        label = CoreLabel(text=text, font_size=font_size, font_name=self._font_name)
        label.refresh()
        texture = label.texture
        if texture is None:
            return

        width, height = texture.size
        if align == "center":
            pos = (x - width / 2.0, y - height / 2.0)
        elif align == "right":
            pos = (x - width, y)
        else:
            pos = (x, y)

        from kivy.graphics import Color, Rectangle

        with board.canvas:
            Color(*rgba)
            Rectangle(pos=pos, size=texture.size, texture=texture)

    # ------------------------------------------------------------------
    def _draw_player_areas(self, play_rect: Tuple[float, float, float, float]) -> None:
        num_players = getattr(self._env, "num_players", 0)
        if num_players <= 0:
            return

        reveal_flags = self._compute_hand_reveal_flags(num_players)
        angle_map = {0: 0, 1: -90, 2: 180, 3: 90}
        board = self._board
        if board is None:
            return

        play_center = (
            play_rect[0] + play_rect[2] / 2.0,
            play_rect[1] + play_rect[3] / 2.0,
        )

        for player_idx in range(min(4, num_players)):
            face_up = (
                reveal_flags[player_idx]
                if player_idx < len(reveal_flags)
                else player_idx == 0
            )
            base_specs = self._build_player_layout(player_idx, face_up)
            rotation = angle_map.get(player_idx, 0)
            for spec in base_specs:
                tile_center = (
                    spec.pos[0] + spec.size[0] / 2.0,
                    spec.pos[1] + spec.size[1] / 2.0,
                )
                rotated_x, rotated_y = self._rotate_point(tile_center, play_center, rotation)
                pos = (
                    rotated_x - spec.size[0] / 2.0,
                    rotated_y - spec.size[1] / 2.0,
                )
                board.add_tile(
                    spec.tile_id,
                    pos,
                    spec.size,
                    spec.face_up,
                    rotation + spec.angle,
                )

    # ------------------------------------------------------------------
    def _compute_hand_reveal_flags(self, num_players: int) -> list[bool]:
        reveal = [False] * max(0, num_players)
        for idx in range(num_players):
            reveal[idx] = idx == 0

        if getattr(self._env, "phase", "") != "score":
            return reveal

        agari = getattr(self._env, "agari", None)
        if agari:
            for winner in self._extract_winner_indices(agari):
                if 0 <= winner < num_players:
                    reveal[winner] = True
        else:
            tenpai_flags = list(getattr(self._env, "tenpai", []))
            for idx in range(num_players):
                if idx < len(tenpai_flags) and tenpai_flags[idx]:
                    reveal[idx] = True

        return reveal

    # ------------------------------------------------------------------
    def _extract_winner_indices(self, agari: Any) -> Iterable[int]:
        winners = agari
        if isinstance(agari, dict):
            winners = agari.get("winner", [])
        if isinstance(winners, int):
            winners = [winners]
        return [int(idx) for idx in winners]

    # ------------------------------------------------------------------
    @dataclass(slots=True)
    class _TileSpec:
        tile_id: int
        pos: Tuple[float, float]
        size: Tuple[int, int]
        face_up: bool
        angle: float = 0.0

    def _build_player_layout(self, player_idx: int, face_up_hand: bool) -> list[_TileSpec]:
        board = self._board
        if board is None:
            return []
        tile_size = self._tile_metrics.get("south_hand", (40, 56))
        spacing = tile_size[0] + 6
        draw_gap = 0
        tiles: list[MahjongEnvKivyWrapper._TileSpec] = []

        hands = getattr(self._env, "hands", [])
        hand_tiles = list(hands[player_idx]) if player_idx < len(hands) else []
        if hand_tiles:
            if (
                getattr(self._env, "phase", "") in ["draw", "kan_draw"]
                and getattr(self._env, "current_player", 0) == player_idx
            ):
                hand_tiles = sorted(hand_tiles[:-1]) + [hand_tiles[-1]]
            else:
                hand_tiles = sorted(hand_tiles)

        x_origin = board.center_x - 7 * (tile_size[0] + 6)
        y = board.y + board.height - tile_size[1] - 14
        x = x_origin
        for idx, tile in enumerate(hand_tiles):
            if len(hand_tiles) > 1 and idx == len(hand_tiles) - 1:
                x += draw_gap
            spec = MahjongEnvKivyWrapper._TileSpec(
                tile_id=tile,
                pos=(x, y),
                size=tile_size,
                face_up=face_up_hand,
            )
            tiles.append(spec)
            x += spacing

        discard_tiles = self._get_discard_tiles(player_idx)
        discard_tile = self._tile_metrics.get("discard", tile_size)
        cols = 6
        grid_half_width = 3 * (discard_tile[0] + 4)
        grid_width = 6 * (discard_tile[0] + 4)
        grid_height = 4 * (discard_tile[1] + 4)
        centerx = board.center_x
        discard_origin_x = centerx - grid_half_width
        discard_origin_y = y - 4 * (discard_tile[1] + 4) - 24
        orientation_map: dict[int, int] | None = None
        declaration_index = self._riichi_declarations.get(player_idx)
        if declaration_index is not None:
            orientation_map = {declaration_index: 90}

        tile_specs = self._build_tile_grid(
            discard_tiles,
            (discard_origin_x, discard_origin_y, grid_width, grid_height),
            discard_tile,
            0,
            cols,
            orientation_map,
        )
        tiles.extend(tile_specs)

        meld_tile = self._tile_metrics.get("meld", tile_size)
        max_meld_width = meld_tile[0] * 4 + 12
        margin_side = 14
        area_width = board.width
        meld_origin_x = area_width - margin_side - max_meld_width
        meld_origin_x = max(margin_side, meld_origin_x)
        meld_origin_y = y
        tiles.extend(
            self._build_melds(
                player_idx,
                (meld_origin_x, meld_origin_y),
                "horizontal",
                meld_tile,
                0,
            )
        )

        return tiles

    # ------------------------------------------------------------------
    def _get_discard_tiles(self, player_idx: int) -> list[int]:
        if player_idx >= len(getattr(self._env, "discard_pile_seq", [])):
            return []
        return [t for t in self._env.discard_pile_seq[player_idx]]

    # ------------------------------------------------------------------
    def _build_tile_grid(
        self,
        tiles: list[int],
        area: Tuple[float, float, float, float],
        tile_size: Tuple[int, int],
        orientation: int,
        columns: int,
        orientation_map: Optional[dict[int, int]] = None,
    ) -> list[_TileSpec]:
        specs: list[MahjongEnvKivyWrapper._TileSpec] = []
        if not tiles:
            return specs

        columns = max(1, columns)
        spacing = 4
        x0, y0 = area[0], area[1]
        x, y = x0, y0
        for idx, tile in enumerate(tiles):
            column = idx % columns
            row_prev = -1 if idx == 0 else min(3, (idx - 1) // columns)
            row = min(3, idx // columns)
            tile_orientation = orientation
            if orientation_map and idx in orientation_map:
                tile_orientation = orientation_map[idx]
            width, height = tile_size
            if row != row_prev:
                x = x0
            elif orientation_map and (idx - 1) in orientation_map:
                x += height + spacing
            else:
                x += width + spacing
            y = area[1] + row * (height + spacing)
            specs.append(
                MahjongEnvKivyWrapper._TileSpec(
                    tile_id=tile,
                    pos=(x, y),
                    size=tile_size,
                    face_up=True,
                    angle=float(tile_orientation),
                )
            )
        return specs

    # ------------------------------------------------------------------
    def _build_melds(
        self,
        player: int,
        origin: Tuple[float, float],
        direction: str,
        tile_size: Tuple[int, int],
        orientation: int,
    ) -> list[_TileSpec]:
        specs: list[MahjongEnvKivyWrapper._TileSpec] = []
        melds = getattr(self._env, "melds", [])
        if player >= len(melds):
            return specs

        x, y = origin
        spacing = 8
        for meld in melds[player]:
            tiles = list(reversed([tile for tile in meld.get("m", [])]))
            opened = meld.get("opened", True)
            meld_type = meld.get("type", "")
            claimed_tile = meld.get("claimed_tile")
            sideways_index: Optional[int] = None
            if claimed_tile is not None:
                try:
                    sideways_index = tiles.index(claimed_tile)
                except ValueError:
                    sideways_index = None
            cur_x, cur_y = x, y
            total_tiles = len(tiles)
            for idx, tile in enumerate(tiles):
                tile_orientation = orientation
                if sideways_index is not None and idx == sideways_index:
                    tile_orientation = 90
                tile_face_up = opened
                if not opened and meld_type == "kan" and total_tiles >= 4:
                    if idx in {0, total_tiles - 1}:
                        tile_face_up = False
                    else:
                        tile_face_up = True
                specs.append(
                    MahjongEnvKivyWrapper._TileSpec(
                        tile_id=tile,
                        pos=(cur_x, cur_y),
                        size=tile_size,
                        face_up=tile_face_up,
                        angle=float(tile_orientation),
                    )
                )
                if direction == "horizontal":
                    if sideways_index is not None and (
                        (idx + 1) == sideways_index or idx == sideways_index
                    ):
                        cur_x -= tile_size[1] + 4
                    else:
                        cur_x -= tile_size[0] + 4
                else:
                    cur_y -= tile_size[1] + 4
            if direction == "horizontal":
                x = cur_x - spacing
            else:
                y = cur_y - spacing
        return specs

    # ------------------------------------------------------------------
    def _draw_walls(self, play_rect: Tuple[float, float, float, float]) -> None:
        wall_tile = self._tile_metrics.get("wall", (20, 26))
        tiles_per_side = 17
        spacing = 4
        x, y, width, height = play_rect

        top_y = y - wall_tile[1] - 12
        bottom_y = y + height + 12
        available_width = width - 40
        horizontal_spacing = max(
            wall_tile[0] + spacing,
            (available_width - wall_tile[0]) / max(1, tiles_per_side - 1),
        )
        for i in range(tiles_per_side):
            tile_x = int(x + 20 + i * horizontal_spacing)
            self._board.add_tile(0, (tile_x, top_y), wall_tile, False, 0)
            self._board.add_tile(0, (tile_x, bottom_y), wall_tile, False, 0)

        left_x = x - wall_tile[0] - 12
        right_x = x + width + 12
        available_height = height - 40
        vertical_spacing = max(
            wall_tile[1] + spacing,
            (available_height - wall_tile[1]) / max(1, tiles_per_side - 1),
        )
        for i in range(tiles_per_side):
            tile_y = int(y + 20 + i * vertical_spacing)
            self._board.add_tile(0, (left_x, tile_y), wall_tile, False, 0)
            self._board.add_tile(0, (right_x, tile_y), wall_tile, False, 0)

        self._draw_dead_wall(play_rect)

    # ------------------------------------------------------------------
    def _draw_dead_wall(self, play_rect: Tuple[float, float, float, float]) -> None:
        wall_tile = self._tile_metrics.get("wall", (20, 26))
        stack_size = 5
        gap = 6
        _, _, width, height = play_rect
        center_x = play_rect[0] + width / 2.0
        center_y = play_rect[1] + height / 2.0
        total_height = stack_size * (wall_tile[0] + gap) - gap
        start_x = center_x + total_height // 2
        y = center_y + 32
        for i in range(stack_size):
            x = start_x - i * (wall_tile[0] + gap) - wall_tile[0]
            if i < len(getattr(self._env, "dora_indicator", [])):
                tile = self._env.dora_indicator[i]
                self._board.add_tile(tile, (x, y), wall_tile, True, 0)
            else:
                self._board.add_tile(0, (x, y), wall_tile, False, 0)

    # ------------------------------------------------------------------
    def _draw_seat_labels(self, play_rect: Tuple[float, float, float, float]) -> None:
        seat_names = list(getattr(self._env, "seat_names", []))
        num_players = getattr(self._env, "num_players", 0)
        if len(seat_names) < num_players:
            seat_names.extend(["NoName"] * (num_players - len(seat_names)))

        offsets = (
            (0, play_rect[3] / 2 + dp(42)),
            (play_rect[2] / 2 + dp(42), 0),
            (0, -play_rect[3] / 2 - dp(42)),
            (-play_rect[2] / 2 - dp(42), 0),
        )
        center_x = play_rect[0] + play_rect[2] / 2.0
        center_y = play_rect[1] + play_rect[3] / 2.0
        for idx in range(min(4, num_players)):
            name = seat_names[idx]
            offset = offsets[idx]
            self._write_text(
                name,
                center_x + offset[0],
                center_y + offset[1],
                align="center",
                font_size=self._font_size,
                color=self._muted_text_color,
            )

    # ------------------------------------------------------------------
    @staticmethod
    def _rotate_point(
        point: Tuple[float, float], center: Tuple[float, float], angle_degrees: float
    ) -> Tuple[float, float]:
        from math import cos, radians, sin

        angle = radians(angle_degrees)
        cos_a = cos(angle)
        sin_a = sin(angle)
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        return (
            center[0] + dx * cos_a - dy * sin_a,
            center[1] + dx * sin_a + dy * cos_a,
        )

    # ------------------------------------------------------------------
    def _draw_status_text(self) -> None:
        if self._root is None:
            return

        lines = []
        if self._last_payload.action is not None:
            lines.append(f"Action: {self._last_payload.action}")
        if self._last_payload.reward:
            lines.append(f"Reward: {self._last_payload.reward:+.2f}")
        if self._last_payload.info:
            lines.append(f"Info: {self._last_payload.info}")
        lines.append(f"Phase: {getattr(self._env, 'phase', '')}")
        if getattr(self._env, "done", False):
            lines.append("Done")
        if self._root.status_label is not None:
            self._root.status_label.text = "\n".join(lines)


def load_kv_layout(path: str) -> Any:
    return Builder.load_file(path)

