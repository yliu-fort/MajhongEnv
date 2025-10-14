from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Tuple, Union

from kivy.base import EventLoop
from kivy.clock import Clock
from kivy.core.text import Label as CoreLabel
from kivy.graphics import (
    Color,
    Line,
    PushMatrix,
    PopMatrix,
    Rectangle,
    Rotate,
    RoundedRectangle,
    Translate,
)
from kivy.graphics.instructions import InstructionGroup
from kivy.properties import ObjectProperty
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.widget import Widget

import threading

from mahjong_env import MahjongEnvBase as _BaseMahjongEnv
from mahjong_gui.assets import TileAssetManager
from mahjong_gui.controls import apply_font, bind_controls, set_spinner_text, set_spinner_values
from mahjong_gui.layout import Rect, board_rotation_angle, compute_display_order, get_relative_angle
from mahjong_gui.localization import DEFAULT_LANGUAGE, LocalizationManager
from mahjong_gui.state import PlaybackState

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


class MahjongBoardWidget(Widget):
    """Widget that renders the Mahjong play field."""


class MahjongRoot(FloatLayout):
    board = ObjectProperty(None)
    status_label = ObjectProperty(None)
    reward_label = ObjectProperty(None)
    done_label = ObjectProperty(None)
    pause_button = ObjectProperty(None)
    auto_button = ObjectProperty(None)
    step_button = ObjectProperty(None)
    language_spinner = ObjectProperty(None)
    wrapper = ObjectProperty(None)


class MahjongEnvKivyWrapper:
    """Mahjong environment with a Kivy GUI overlay."""

    def __init__(
        self,
        *args: Any,
        env: Optional[_BaseMahjongEnv] = None,
        window_size: Tuple[int, int] = (1024, 720),
        fps: int = 30,
        font_name: Optional[str] = None,
        font_size: int = 18,
        fallback_fonts: Optional[Sequence[str]] = None,
        root_widget: Optional[MahjongRoot] = None,
        tile_texture_size: Optional[Tuple[int, int]] = None,
        tile_texture_use_tile_metrics: bool = False,
        tile_texture_background: Optional[
            Union[str, Sequence[float], Sequence[int]]
        ] = None,
        **kwargs: Any,
    ) -> None:
        self._env = env or _BaseMahjongEnv(*args, **kwargs)
        self._window_size = window_size
        self._fps = max(1, fps)
        self._user_font_name = font_name
        self._font_name = font_name or "Roboto"
        self._font_size = font_size
        self._background_color = (12 / 255.0, 30 / 255.0, 60 / 255.0, 1)
        self._play_area_color = (24 / 255.0, 60 / 255.0, 90 / 255.0, 1)
        self._play_area_border = (40 / 255.0, 90 / 255.0, 130 / 255.0, 1)
        self._panel_color = (5 / 255.0, 5 / 255.0, 5 / 255.0, 1)
        self._panel_border = (90 / 255.0, 120 / 255.0, 160 / 255.0, 1)
        self._accent_color = (170 / 255.0, 230 / 255.0, 255 / 255.0, 1)
        self._text_color = (235 / 255.0, 235 / 255.0, 235 / 255.0, 1)
        self._muted_text_color = (170 / 255.0, 190 / 255.0, 210 / 255.0, 1)
        self._danger_color = (220 / 255.0, 120 / 255.0, 120 / 255.0, 1)
        self._face_down_color = (18 / 255.0, 18 / 255.0, 22 / 255.0, 1)
        self._face_down_border = (60 / 255.0, 60 / 255.0, 70 / 255.0, 1)
        self._tile_texture_background = "#FFFFFF"
        if tile_texture_background is not None:
            self._tile_texture_background = tile_texture_background

        self._root = root_widget or MahjongRoot()
        self._root.size = window_size
        self._root.wrapper = self

        self._localization = LocalizationManager(DEFAULT_LANGUAGE)
        self._default_language = self._localization.default_language
        self._language = self._default_language
        self._updating_language_spinner = False
        self._language_spinner = None
        self._user_fallback_fonts: Tuple[str, ...] = tuple(fallback_fonts or ())
        self._fallback_fonts: Tuple[str, ...] = self._user_fallback_fonts

        self._asset_root = Path(__file__).resolve().parent.parent / "assets" / "tiles" / "Regular"
        self._asset_manager = TileAssetManager(
            asset_root=self._asset_root,
            tile_symbols=_TILE_SYMBOLS,
            background_color=self._tile_texture_background,
        )
        self._placeholder_cache: dict[tuple[int, Tuple[int, int]], CoreLabel] = {}
        self._tile_metrics: dict[str, Tuple[int, int]] = {}
        self._riichi_states: list[bool] = []
        self._riichi_pending: list[bool] = []
        self._discard_counts: list[int] = []
        self._riichi_declarations: dict[int, int] = {}
        self._focus_player = 0

        if tile_texture_size is not None:
            width, height = tile_texture_size
            self._tile_texture_explicit_size: Optional[Tuple[int, int]] = (
                max(1, int(width)),
                max(1, int(height)),
            )
        else:
            self._tile_texture_explicit_size = None
        self._tile_texture_auto_size = bool(tile_texture_use_tile_metrics)
        
        self._last_payload = _RenderPayload(action=None, reward=0.0, done=False, info={})
        self._state = PlaybackState()
        self._pending_action: Optional[int] = None
        self._step_result: Optional[Tuple[Any, float, bool, dict[str, Any]]] = None
        self._step_event = threading.Event()
        self._scheduled = False
        self._load_tile_assets(self._tile_texture_explicit_size)
        self._connect_controls()
        self._configure_language_spinner()
        self._update_language_fonts(self._language)
        self._update_language_spinner()
        self._apply_font_to_controls()

        self._clock_event = Clock.schedule_interval(self._on_frame, 1.0 / self._fps)
        self._scheduled = True

    # ------------------------------------------------------------------
    # Exposed environment API
    # ------------------------------------------------------------------
    @property
    def root(self) -> MahjongRoot:
        return self._root

    @property
    def pending_action(self) -> Optional[int]:
        return self._pending_action

    @property
    def fps(self) -> int:
        return self._fps

    def set_focus_player(self, player_index: int) -> None:
        num_players = getattr(self._env, "num_players", 0)
        if num_players > 0:
            sanitized = max(0, min(int(player_index), num_players - 1))
        else:
            sanitized = 0
        if sanitized == self._focus_player:
            return
        self._focus_player = sanitized
        self._render()

    def _get_focus_index(self, num_players: int) -> int:
        if num_players <= 0:
            return 0
        if self._focus_player < 0:
            self._focus_player = 0
        elif self._focus_player >= num_players:
            self._focus_player = num_players - 1
        return self._focus_player

    def _get_display_order(self) -> list[int]:
        num_players = getattr(self._env, "num_players", 0)
        if num_players <= 0:
            return []
        focus = self._get_focus_index(num_players)
        return compute_display_order(num_players, focus)

    def _get_relative_angle(self, relative_position: int) -> int:
        return get_relative_angle(relative_position)

    def _get_board_rotation_angle(self) -> int:
        num_players = getattr(self._env, "num_players", 0)
        if num_players <= 0:
            return 0
        focus = self._get_focus_index(num_players)
        return board_rotation_angle(num_players, focus)

    def set_language(self, language: str) -> None:
        if not language:
            return
        code = self._localization.resolve_code(language)
        if code is None:
            return
        if code == self._language:
            return
        self._language = code
        self._update_language_fonts(self._language)
        self._apply_font_to_controls()
        self._update_language_spinner()
        self._render()
        self._draw_status_labels()
        self._update_control_buttons()

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        observation = self._env.reset(*args, **kwargs)
        self._last_payload = _RenderPayload(action=None, reward=0.0, done=self._env.done, info={})
        self._state = PlaybackState()
        self._state.score_panel_was_visible = self._score_last()
        self._riichi_states = []
        self._riichi_pending = []
        self._discard_counts = []
        self._riichi_declarations = {}
        self._pending_action = None
        self._step_result = None
        self._render()
        return observation

    def step(self, action: int) -> Tuple[Any, float, bool, dict[str, Any]]:
        self.queue_action(action)
        while self._step_result is None:
            EventLoop.idle()
        result = self._step_result
        self._step_result = None
        return result

    def queue_action(self, action: int) -> None:
        if self._pending_action is not None:
            return
        self._pending_action = action
        self._step_result = None
        self._step_event.clear()

    def fetch_step_result(self) -> Optional[Tuple[Any, float, bool, dict[str, Any]]]:
        if self._step_result is None:
            return None
        result = self._step_result
        self._step_result = None
        return result

    def close(self) -> None:
        if self._scheduled:
            self._clock_event.cancel()
            self._scheduled = False
        self._env.close()

    def action_masks(self) -> Any:
        return self._env.action_masks()

    @property
    def phase(self) -> str:
        return getattr(self._env, "phase", "")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _connect_controls(self) -> None:
        if not self._root:
            return
        self._language_spinner = bind_controls(
            self._root,
            on_toggle_auto=self._toggle_auto,
            on_step_once=self._trigger_step_once,
            on_toggle_pause=self._toggle_pause,
            on_language_change=self._on_language_spinner_text,
        )

    def _get_language_dict(self, code: Optional[str] = None) -> dict[str, Any]:
        return self._localization.language_dict(code or self._language)

    def _translate(self, key: str, **kwargs: Any) -> str:
        return self._localization.translate(self._language, key, **kwargs)

    def _translate_sequence(self, key: str) -> Sequence[str]:
        return self._localization.translate_sequence(self._language, key)

    def _format_ordinal(self, value: int) -> str:
        return self._localization.format_ordinal(self._language, value)

    def _configure_language_spinner(self) -> None:
        if not self._language_spinner:
            return
        values = self._localization.spinner_values()
        set_spinner_values(self._language_spinner, values)

    def _update_language_spinner(self) -> None:
        if not self._language_spinner:
            return
        values = list(self._localization.spinner_values())
        display_name = self._localization.display_name(self._language)
        if display_name not in values:
            values.append(display_name)
        set_spinner_values(self._language_spinner, values)
        if self._language_spinner.text != display_name:
            self._updating_language_spinner = True
            set_spinner_text(self._language_spinner, display_name)
            self._updating_language_spinner = False

    def _update_language_fonts(self, language: str) -> None:
        font_name, localized_fallbacks = self._localization.resolve_fonts(
            language,
            user_font_name=self._user_font_name,
            fallback=self._font_name,
        )
        self._font_name = font_name
        combined: list[str] = []
        for candidate in list(localized_fallbacks) + list(self._user_fallback_fonts):
            if candidate and candidate not in combined:
                combined.append(candidate)
        self._fallback_fonts = tuple(combined)

    def _apply_font_to_controls(self) -> None:
        apply_font(self._root, self._font_name)

    def _on_language_spinner_text(self, value: str) -> None:
        if self._updating_language_spinner:
            return
        self.set_language(value)

    def _load_tile_assets(self, target_size: Optional[Tuple[int, int]] = None) -> None:
        if target_size is None:
            target_size = self._tile_texture_explicit_size
        self._asset_manager.background_color = self._tile_texture_background
        self._asset_manager.load(target_size)

    def _toggle_auto(self) -> None:
        self._state.toggle_auto(self._score_last())

    def _trigger_step_once(self) -> None:
        self._state.trigger_step_once()

    def _toggle_pause(self) -> None:
        self._state.toggle_pause(self._score_last())

    def _score_last(self) -> bool:
        return (
            getattr(self._env, "phase", "") == "score"
            and self._env.current_player == - 1
        )

    def _update_pause_state(self) -> None:
        self._state.update_for_score_panel(self._score_last())

    def _compute_tile_metrics(self, play_rect: Rect) -> None:
        width = max(1, int(play_rect.width))
        height = max(1, int(play_rect.height))
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
        self._update_auto_tile_texture_size()

    def _update_auto_tile_texture_size(self) -> None:
        if not self._tile_texture_auto_size:
            return
        base_size = self._tile_metrics.get("south_hand")
        if not base_size:
            return
        target_size = (max(1, int(base_size[0])), max(1, int(base_size[1])))
        if self._asset_manager.current_size == target_size:
            return
        self._load_tile_assets(target_size)

    def _ensure_riichi_tracking_capacity(self, count: int) -> None:
        count = max(0, count)
        while len(self._riichi_states) < count:
            self._riichi_states.append(False)
            self._riichi_pending.append(False)
            self._discard_counts.append(0)
        if len(self._riichi_states) > count:
            self._riichi_states = self._riichi_states[:count]
            self._riichi_pending = self._riichi_pending[:count]
            self._discard_counts = self._discard_counts[:count]
        for idx in list(self._riichi_declarations):
            if idx >= count:
                self._riichi_declarations.pop(idx, None)

    def _update_riichi_state(self) -> None:
        riichi_flags = list(getattr(self._env, "riichi", []))
        discard_seq = list(getattr(self._env, "discard_pile_seq", []))
        num_players = max(len(riichi_flags), len(discard_seq), getattr(self._env, "num_players", 0))
        if num_players <= 0:
            self._riichi_states = []
            self._riichi_pending = []
            self._discard_counts = []
            self._riichi_declarations.clear()
            return

        self._ensure_riichi_tracking_capacity(num_players)

        for idx in range(num_players):
            is_riichi = riichi_flags[idx] if idx < len(riichi_flags) else False
            was_riichi = self._riichi_states[idx]
            if is_riichi and not was_riichi:
                self._riichi_pending[idx] = True
            elif not is_riichi:
                self._riichi_pending[idx] = False
                self._riichi_declarations.pop(idx, None)
            self._riichi_states[idx] = is_riichi

        for idx in range(num_players):
            tiles = discard_seq[idx] if idx < len(discard_seq) else []
            current_count = len(tiles)
            last_count = self._discard_counts[idx] if idx < len(self._discard_counts) else 0
            if current_count > last_count:
                if self._riichi_pending[idx]:
                    self._riichi_declarations[idx] = last_count
                    self._riichi_pending[idx] = False
            self._discard_counts[idx] = current_count

    def _on_frame(self, dt: float) -> None:
        if self._pending_action is not None and self._step_result is None:
            can_step = True
            if not self._state.auto_advance and not self._state.step_once_requested:
                can_step = False
            if self._state.score_pause_active:
                can_step = False
            if self._env.done:
                can_step = True
            if can_step:
                if self._state.step_once_requested:
                    self._state.step_once_requested = False
                action = self._pending_action
                observation, reward, done, info = self._env.step(action)
                self._pending_action = None
                self._step_result = (observation, reward, done, info)
                self._last_payload = _RenderPayload(
                    action=action,
                    reward=reward,
                    done=done,
                    info=info,
                )
                self._step_event.set()
        self._update_pause_state()
        self._render()
        if self._step_result is not None:
            self._step_event.set()

    def _render(self) -> None:
        board = self._root.board
        if board is None:
            return

        width, height = board.size
        play_size = height
        play_rect = Rect(0, 0, play_size, play_size)
        play_rect.top = 0
        play_rect.centerx = width / 2

        self._compute_tile_metrics(play_rect)
        self._update_riichi_state()

        canvas = board.canvas
        canvas.clear()

        self._draw_center_panel(canvas, board, play_rect)
        self._draw_dead_wall(canvas, board, play_rect)
        self._draw_player_areas(canvas, board, play_rect)
        if self._score_last():
            self._draw_score_panel(canvas, board, play_rect)
        self._draw_status_labels()
        self._update_control_buttons()

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def _to_canvas_pos(
        self, board: MahjongBoardWidget, rect: Rect, x: float, y: float, width: float, height: float
    ) -> Tuple[float, float]:
        base_x, base_y = board.pos
        return (
            base_x + rect.left + x,
            base_y + rect.top + rect.height - y - height,
        )

    def _wrap_text(self, text: str, max_width: float, font_size: Optional[int] = None) -> list[str]:
        if not text:
            return []
        max_width = max(10.0, float(max_width))
        font_size = int(font_size or self._font_size)
        words = str(text).split()
        lines: list[str] = []
        current = ""

        def measure(candidate: str) -> float:
            label = CoreLabel(text=candidate, font_size=font_size, font_name=self._font_name)
            label.refresh()
            return float(label.texture.size[0])

        for word in words:
            candidate = word if not current else f"{current} {word}"
            if measure(candidate) <= max_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    def _draw_center_panel(
        self, canvas: InstructionGroup, board: MahjongBoardWidget, play_rect: Rect
    ) -> None:
        center_width = max(100, int(play_rect.width * 0.24))
        center_height = max(100, int(play_rect.height * 0.24))
        center_rect = Rect(0, 0, center_width, center_height)
        center_rect.center = play_rect.center

        x, y = self._to_canvas_pos(board, play_rect, center_rect.left, center_rect.top, center_rect.width, center_rect.height)
        canvas.add(Color(*self._panel_color))
        canvas.add(
            RoundedRectangle(
                pos=(x, y),
                size=(center_rect.width, center_rect.height),
                radius=[12, 12, 12, 12],
            )
        )
        canvas.add(Color(*self._panel_border))
        canvas.add(
            Line(
                rounded_rectangle=(x, y, center_rect.width, center_rect.height, 12),
                width=2,
            )
        )

        round_data = getattr(self._env, "round", [0, 0])
        round_index = round_data[0] if isinstance(round_data[0], int) else 0
        wind_names = self._translate_sequence("wind_names") or ("East", "South", "West", "North")
        wind = wind_names[(round_index // 4) % len(wind_names)]
        hand_number = round_index % 4 + 1
        round_text = self._translate("round_format", wind=wind, hand=hand_number)

        title_label = CoreLabel(text=round_text, font_size=self._font_size + 8, font_name=self._font_name)
        title_label.refresh()
        label_x = center_rect.centerx - title_label.texture.size[0] / 2
        label_y = center_rect.top + 18
        px, py = self._to_canvas_pos(board, play_rect, label_x, label_y, *title_label.texture.size)
        canvas.add(Color(*self._text_color))
        canvas.add(Rectangle(texture=title_label.texture, size=title_label.texture.size, pos=(px, py)))

        honba = round_data[1] if len(round_data) > 1 and isinstance(round_data[1], int) else 0
        riichi = getattr(self._env, "num_riichi", 0)
        tiles_remaining = len(getattr(self._env, "deck", []))
        counter_texts = (
            self._translate("counter_honba", count=honba),
            self._translate("counter_riichi", count=riichi),
            self._translate("counter_tiles", count=tiles_remaining),
        )
        next_top = label_y + title_label.texture.size[1] + 12
        for text in counter_texts:
            label = CoreLabel(text=text, font_size=self._font_size - 4, font_name=self._font_name)
            label.refresh()
            label_x = center_rect.centerx - label.texture.size[0] / 2
            px, py = self._to_canvas_pos(board, play_rect, label_x, next_top, *label.texture.size)
            canvas.add(Color(*self._muted_text_color))
            canvas.add(Rectangle(texture=label.texture, size=label.texture.size, pos=(px, py)))
            next_top += label.texture.size[1] + 4

        score_positions = (
            (center_rect.centerx, center_rect.bottom + 14),
            (center_rect.right + 18, center_rect.centery),
            (center_rect.centerx, center_rect.top - 14),
            (center_rect.left - 18, center_rect.centery),
        )
        scores = getattr(self._env, "scores", [])
        current_player = getattr(self._env, "current_player", 0)
        order = self._get_display_order()
        for relative_position, player_idx in enumerate(order):
            if relative_position >= len(score_positions):
                break
            if player_idx >= len(scores):
                continue
            position = score_positions[relative_position]
            score_value = scores[player_idx] * 100
            color = (
                self._accent_color
                if player_idx == current_player
                else self._text_color
            )
            label = CoreLabel(
                text=f"{score_value:5d}",
                font_size=self._font_size,
                font_name=self._font_name,
            )
            label.refresh()
            px, py = self._to_canvas_pos(
                board,
                play_rect,
                position[0] - label.texture.size[0] / 2,
                position[1] - label.texture.size[1] / 2,
                *label.texture.size,
            )
            canvas.add(Color(*color))
            canvas.add(
                Rectangle(
                    texture=label.texture,
                    size=label.texture.size,
                    pos=(px, py),
                )
            )

    def _draw_dead_wall(
        self, canvas: InstructionGroup, board: MahjongBoardWidget, play_rect: Rect
    ) -> None:
        wall_tile = self._tile_metrics.get("wall", (20, 26))
        stack_size = 5
        gap = 6
        margin_y = 32
        total_width = stack_size * (wall_tile[0] + gap) - gap
        start_x = play_rect.centerx + total_width // 2
        y = play_rect.centery + margin_y

        for i in range(stack_size):
            x = start_x - i * (wall_tile[0] + gap) - wall_tile[0]
            if i < len(self._env.dora_indicator):
                tile = self._env.dora_indicator[i]
                self._draw_tile(canvas, board, play_rect, tile, wall_tile, True, 0, (x, y))
            else:
                self._draw_tile(canvas, board, play_rect, 0, wall_tile, False, 0, (x, y))


    def _draw_player_areas(
        self, canvas: InstructionGroup, board: MahjongBoardWidget, play_rect: Rect
    ) -> None:
        order = self._get_display_order()
        if not order:
            return
        num_players = getattr(self._env, "num_players", 0)
        reveal_flags = self._compute_hand_reveal_flags(num_players)
        for relative_position, player_idx in enumerate(order):
            face_up = (
                reveal_flags[player_idx]
                if player_idx < len(reveal_flags)
                else player_idx == self._focus_player
            )
            angle = self._get_relative_angle(relative_position)
            self._draw_player_layout(
                canvas, board, play_rect, player_idx, face_up, angle
            )

    def _compute_hand_reveal_flags(self, num_players: int) -> list[bool]:
        reveal = [False] * max(0, num_players)
        agari = getattr(self._env, "agari", None)
        phase = getattr(self._env, "phase", "")
        winners: set[int] = set()
        if agari:
            winners = self._extract_winner_indices(agari)

        if phase == "score":
            if winners:
                for idx in winners:
                    if 0 <= idx < len(reveal):
                        reveal[idx] = True
            else:
                tenpai_flags = list(getattr(self._env, "tenpai", []))
                for idx, is_tenpai in enumerate(tenpai_flags):
                    if is_tenpai and 0 <= idx < len(reveal):
                        reveal[idx] = True
        else:
            if winners:
                for idx in winners:
                    if 0 <= idx < len(reveal):
                        reveal[idx] = True
        if reveal:
            focus_idx = self._get_focus_index(len(reveal))
            if 0 <= focus_idx < len(reveal):
                reveal[focus_idx] = True
        return reveal

    def _extract_winner_indices(self, agari: Any) -> set[int]:
        winners: set[int] = set()
        if isinstance(agari, dict):
            who = agari.get("who")
            if isinstance(who, int):
                winners.add(who)
            elif isinstance(who, Iterable) and not isinstance(who, (str, bytes)):
                for value in who:
                    if isinstance(value, int):
                        winners.add(value)
        return winners

    def _draw_player_layout(
        self,
        canvas: InstructionGroup,
        board: MahjongBoardWidget,
        play_rect: Rect,
        player_idx: int,
        face_up_hand: bool,
        angle: int,
    ) -> None:
        tile_size = self._tile_metrics.get("south_hand", (40, 56))
        meld_tile = self._tile_metrics.get("meld", tile_size)
        spacing = tile_size[0] + 6
        draw_gap = 0

        hands = getattr(self._env, "hands", [])
        hand_tiles = list(hands[player_idx]) if player_idx < len(hands) else []
        if (
            getattr(self._env, "phase", "") in ["draw", "kan_draw"]
            and self._env.current_player == player_idx
            and len(hand_tiles) > 0
        ):
            hand_tiles = sorted(hand_tiles[:-1]) + [hand_tiles[-1]]
        else:
            hand_tiles = sorted(hand_tiles)

        canvas.add(PushMatrix())
        center_px, center_py = self._to_canvas_pos(
            board, play_rect, play_rect.centerx, play_rect.centery, 0, 0
        )
        if angle:
            canvas.add(Translate(center_px, center_py))
            canvas.add(Rotate(angle=angle))
            canvas.add(Translate(-center_px, -center_py))

        x = play_rect.centerx - 7 * (tile_size[0] + 6)
        y = play_rect.bottom - tile_size[1] - 14
        for idx, tile in enumerate(hand_tiles):
            if len(hand_tiles) > 1 and idx == len(hand_tiles) - 1:
                x += draw_gap
            self._draw_tile(canvas, board, play_rect, tile, tile_size, face_up_hand, 0, (x, y))
            x += spacing

        riichi_flags = list(getattr(self._env, "riichi", []))
        in_riichi = player_idx < len(riichi_flags) and riichi_flags[player_idx]
        if in_riichi:
            label = CoreLabel(
                text=self._translate("riichi_flag"),
                font_size=self._font_size - 2,
                font_name=self._font_name,
            )
            label.refresh()
            padding = 4
            flag_width = label.texture.size[0] + padding * 2
            flag_height = label.texture.size[1] + padding * 2
            flag_left = play_rect.centerx - flag_width / 2
            flag_bottom = y - 6
            flag_top = flag_bottom - flag_height
            px, py = self._to_canvas_pos(
                board, play_rect, flag_left, flag_top, flag_width, flag_height
            )
            canvas.add(Color(*self._panel_color))
            canvas.add(
                RoundedRectangle(
                    pos=(px, py),
                    size=(flag_width, flag_height),
                    radius=[6, 6, 6, 6],
                )
            )
            canvas.add(Color(*self._accent_color))
            canvas.add(Line(rounded_rectangle=(px, py, flag_width, flag_height, 6), width=2))
            canvas.add(
                Rectangle(
                    texture=label.texture,
                    size=label.texture.size,
                    pos=(px + padding, py + padding),
                )
            )

        self._draw_discards(canvas, board, play_rect, player_idx, tile_size)
        self._draw_melds(canvas, board, play_rect, player_idx, tile_size, meld_tile)

        canvas.add(PopMatrix())

    def _draw_score_panel(
        self, canvas: InstructionGroup, board: MahjongBoardWidget, play_rect: Rect
    ) -> None:
        margin = 0
        padding = 14
        min_dimension = min(play_rect.width, play_rect.height)
        available_side = min_dimension - margin * 2
        effective_side = available_side if available_side > 0 else min_dimension
        max_text_width = max(50.0, effective_side - padding * 2)

        round_data = getattr(self._env, "round", [0, 0])
        round_index = round_data[0] if isinstance(round_data[0], int) else 0
        wind_names = self._translate_sequence("wind_names") or ("East", "South", "West", "North")
        wind = wind_names[(round_index // 4) % len(wind_names)]
        hand_number = round_index % 4 + 1
        honba = round_data[1] if len(round_data) > 1 and isinstance(round_data[1], int) else 0
        riichi_sticks = getattr(self._env, "num_riichi", 0)
        kyoutaku = getattr(self._env, "num_kyoutaku", 0)

        info_lines = [
            self._translate(
                "info_line_format",
                round=self._translate("round_format", wind=wind, hand=hand_number),
                honba=self._translate("counter_honba", count=honba),
                riichi=self._translate("counter_riichi", count=riichi_sticks),
                kyoutaku=self._translate("counter_kyoutaku", count=kyoutaku),
            )
        ]

        seat_names = list(getattr(self._env, "seat_names", []))
        num_players = getattr(self._env, "num_players", 0)
        if len(seat_names) < num_players:
            seat_names.extend(
                [
                    self._translate("seat_placeholder_format", index=idx)
                    for idx in range(len(seat_names), num_players)
                ]
            )

        agari = getattr(self._env, "agari", None)
        message_lines: list[str] = []
        if isinstance(agari, dict) and agari:
            winner = agari.get("who", -1)
            from_who = agari.get("fromwho", -1)
            if 0 <= winner < num_players:
                winner_name = seat_names[winner]
            else:
                winner_name = self._translate("player_name_format", index=winner)
            if winner == from_who:
                result_text = self._translate(
                    "result_tsumo",
                    winner=winner_name,
                    tsumo_label=self._translate("tsumo_label"),
                )
            else:
                loser_name = (
                    seat_names[from_who]
                    if 0 <= from_who < num_players
                    else self._translate("player_name_format", index=from_who)
                )
                result_text = self._translate(
                    "result_ron",
                    winner=winner_name,
                    ron_label=self._translate("ron_label"),
                    vs_label=self._translate("vs_label"),
                    loser=loser_name,
                )
            message_lines.append(result_text)
            ten = list(agari.get("ten", []))
            fu = ten[0] if len(ten) > 0 else 0
            total = ten[1] if len(ten) > 1 else 0
            han = ten[2] if len(ten) > 2 else 0
            message_lines.append(
                self._translate("result_details", han=han, fu=fu, total=total)
            )
        else:
            tenpai_flags = list(getattr(self._env, "tenpai", []))
            tenpai_players = [
                seat_names[idx]
                for idx, is_tenpai in enumerate(tenpai_flags)
                if is_tenpai and idx < num_players
            ]
            if tenpai_players:
                message_lines.append(
                    self._translate(
                        "draw_tenpai",
                        players=", ".join(tenpai_players),
                    )
                )
            else:
                message_lines.append(self._translate("draw_no_tenpai"))

        yaku_font_size = max(18, self._font_size - 4)
        yaku_label_proto = CoreLabel(
            text=self._translate("yaku_prefix"),
            font_size=yaku_font_size,
            font_name=self._font_name,
        )
        yaku_label_proto.refresh()
        yaku_label_width = yaku_label_proto.texture.size[0]

        yaku_lines: list[tuple[str, str]] = []
        if isinstance(agari, dict) and agari:
            raw_yaku = [str(item) for item in agari.get("yaku", [])]
            if raw_yaku:
                combined = ", ".join(raw_yaku)
                wrapped_yaku = self._wrap_text(
                    combined, max(10.0, max_text_width - yaku_label_width), font_size=yaku_font_size
                )
                if wrapped_yaku:
                    yaku_lines.append((self._translate("yaku_prefix"), wrapped_yaku[0]))
                    for extra in wrapped_yaku[1:]:
                        yaku_lines.append(("", extra))

        def make_label(text: str, font_size: int) -> CoreLabel:
            label = CoreLabel(text=text, font_size=font_size, font_name=self._font_name)
            label.refresh()
            return label

        title_font_size = self._font_size + 6
        info_font_size = max(10, self._font_size - 2)
        message_font_size = max(10, self._font_size - 2)
        player_font_size = self._font_size

        title_label = make_label(self._translate("round_results_title"), title_font_size)
        content_width = title_label.texture.size[0]

        info_labels = [make_label(line, info_font_size) for line in info_lines]
        if info_labels:
            info_width = max(label.texture.size[0] for label in info_labels)
            content_width = max(content_width, info_width)

        message_labels = [make_label(line, message_font_size) for line in message_lines]
        if message_labels:
            message_width = max(label.texture.size[0] for label in message_labels)
            content_width = max(content_width, message_width)

        scores = list(getattr(self._env, "scores", []))
        if len(scores) < num_players:
            scores.extend([0] * (num_players - len(scores)))
        score_deltas = list(getattr(self._env, "score_deltas", []))
        if len(score_deltas) < num_players:
            score_deltas.extend([0] * (num_players - len(score_deltas)))
        scores = [s + d for s, d in zip(scores, score_deltas)]
        dealer = getattr(self._env, "oya", -1)

        ranks: dict[int, int] = {}
        sorted_players = sorted(range(num_players), key=lambda idx: (-scores[idx], idx))
        last_score: Optional[int] = None
        current_rank = 0
        for position, player_idx in enumerate(sorted_players):
            if player_idx >= len(scores):
                continue
            score_value = scores[player_idx]
            if last_score is None or score_value != last_score:
                current_rank = position + 1
                last_score = score_value
            ranks[player_idx] = current_rank

        winner_idx = agari.get("who") if isinstance(agari, dict) else None

        player_entries = []
        player_section_height = 0.0
        for player_idx in range(num_players):
            if player_idx >= len(scores):
                continue
            display_name = seat_names[player_idx]
            if player_idx == dealer:
                display_name += self._translate("dealer_suffix")
            base_color = self._accent_color if player_idx == winner_idx else self._text_color
            rank_text = self._format_ordinal(ranks.get(player_idx, player_idx + 1))
            name_label = make_label(f"{rank_text}  {display_name}", player_font_size)
            delta_value = score_deltas[player_idx] if player_idx < len(score_deltas) else 0
            delta_points = int(round(delta_value * 100))
            delta_color = (
                self._accent_color
                if delta_points > 0
                else self._danger_color if delta_points < 0 else self._muted_text_color
            )
            delta_label = make_label(f"{delta_points:+}", player_font_size)
            score_points = int(round(scores[player_idx] * 100))
            score_label = make_label(f"{score_points:>6d}", player_font_size)

            name_width = name_label.texture.size[0]
            delta_width = delta_label.texture.size[0]
            score_width = score_label.texture.size[0]
            score_line_width = name_width + score_width + delta_width + 16
            content_width = max(content_width, score_line_width)

            row_height = max(
                name_label.texture.size[1],
                delta_label.texture.size[1],
                score_label.texture.size[1],
            ) + 4
            player_section_height += row_height

            player_entries.append(
                {
                    "name": name_label,
                    "delta": delta_label,
                    "score": score_label,
                    "name_color": base_color,
                    "delta_color": delta_color,
                    "row_height": row_height,
                }
            )

        yaku_entries = []
        yaku_height = 0.0
        for prefix, text in yaku_lines:
            prefix_label = make_label(prefix, yaku_font_size) if prefix else None
            text_label = make_label(text, yaku_font_size)
            prefix_width = prefix_label.texture.size[0] if prefix_label else 0
            text_width = text_label.texture.size[0]
            total_width = prefix_width + (0 if prefix else yaku_label_width) + text_width
            content_width = max(content_width, total_width)
            entry_height = max(
                text_label.texture.size[1],
                prefix_label.texture.size[1] if prefix_label else 0,
            ) + 2
            yaku_height += entry_height
            yaku_entries.append(
                {
                    "prefix": prefix_label,
                    "text": text_label,
                    "entry_height": entry_height,
                }
            )

        info_height = sum(label.texture.size[1] + 2 for label in info_labels)
        message_height = sum(label.texture.size[1] + 2 for label in message_labels)

        panel_height = padding * 2 + title_label.texture.size[1]
        if info_height:
            panel_height += 6 + info_height
        if message_height:
            panel_height += 6 + message_height
        if player_section_height:
            panel_height += 6 + player_section_height
        if yaku_height:
            panel_height += 6 + yaku_height

        required_width = padding * 2 + content_width
        required_height = panel_height
        panel_size = max(required_width, required_height)
        if available_side > 0:
            panel_size = min(panel_size, available_side)
        else:
            panel_size = min(panel_size, min_dimension)

        panel_rect = Rect(0, 0, panel_size, panel_size)
        panel_rect.center = play_rect.center

        panel_pos = self._to_canvas_pos(
            board, play_rect, panel_rect.left, panel_rect.top, panel_rect.width, panel_rect.height
        )
        canvas.add(Color(*self._panel_color))
        canvas.add(
            RoundedRectangle(
                pos=panel_pos,
                size=(panel_rect.width, panel_rect.height),
                radius=[12, 12, 12, 12],
            )
        )
        canvas.add(Color(*self._panel_border))
        canvas.add(
            Line(rounded_rectangle=(*panel_pos, panel_rect.width, panel_rect.height, 12), width=2)
        )

        def draw_label(label: CoreLabel, color: Tuple[float, float, float, float], left: float, top: float) -> None:
            px, py = self._to_canvas_pos(board, play_rect, left, top, *label.texture.size)
            canvas.add(Color(*color))
            canvas.add(Rectangle(texture=label.texture, size=label.texture.size, pos=(px, py)))

        current_y = panel_rect.top + padding
        title_left = panel_rect.centerx - title_label.texture.size[0] / 2
        draw_label(title_label, self._accent_color, title_left, current_y)
        current_y += title_label.texture.size[1] + 6

        for label in info_labels:
            draw_label(label, self._muted_text_color, panel_rect.left + padding, current_y)
            current_y += label.texture.size[1] + 2
        if info_labels:
            current_y += 4

        for label in message_labels:
            draw_label(label, self._text_color, panel_rect.left + padding, current_y)
            current_y += label.texture.size[1] + 2
        if message_labels:
            current_y += 4

        for entry in player_entries:
            name_label: CoreLabel = entry["name"]
            delta_label: CoreLabel = entry["delta"]
            score_label: CoreLabel = entry["score"]
            name_color = entry["name_color"]
            delta_color = entry["delta_color"]

            draw_label(name_label, name_color, panel_rect.left + padding, current_y)
            delta_left = panel_rect.right - padding - delta_label.texture.size[0]
            draw_label(delta_label, delta_color, delta_left, current_y)
            score_left = delta_left - 16 - score_label.texture.size[0]
            draw_label(score_label, name_color, score_left, current_y)
            current_y += entry["row_height"]
        if player_entries:
            current_y += 4

        for entry in yaku_entries:
            prefix_label = entry["prefix"]
            text_label = entry["text"]
            x = panel_rect.left + padding
            if prefix_label is not None:
                draw_label(prefix_label, self._accent_color, x, current_y)
                x += prefix_label.texture.size[0]
            else:
                x += yaku_label_width
            draw_label(text_label, self._text_color, x, current_y)
            current_y += entry["entry_height"]

    def _draw_discards(
        self,
        canvas: InstructionGroup,
        board: MahjongBoardWidget,
        play_rect: Rect,
        player_idx: int,
        tile_size: Tuple[int, int],
    ) -> None:
        discard_tiles = self._get_discard_tiles(player_idx)
        discard_tile = self._tile_metrics.get("discard", tile_size)
        cols = 6
        grid_half_width = 3 * (discard_tile[0] + 4)
        grid_width = 6 * (discard_tile[0] + 4)
        grid_height = 4 * (discard_tile[1] + 4)
        discard_rect = Rect(play_rect.centerx - grid_half_width, 0, grid_width, grid_height)
        discard_rect.top = play_rect.bottom - tile_size[1] - 14 - 4 * (discard_tile[1] + 4) - 24
        orientation_map: Optional[dict[int, int]] = None
        declaration_index = self._riichi_declarations.get(player_idx)
        if declaration_index is not None:
            orientation_map = {declaration_index: 90}
        self._draw_tile_grid(
            canvas,
            board,
            play_rect,
            discard_tiles,
            discard_rect,
            discard_tile,
            0,
            cols,
            orientation_map,
        )

    def _draw_melds(
        self,
        canvas: InstructionGroup,
        board: MahjongBoardWidget,
        play_rect: Rect,
        player_idx: int,
        hand_tile: Tuple[int, int],
        meld_tile: Tuple[int, int],
    ) -> None:
        melds = getattr(self._env, "melds", [])
        if player_idx >= len(melds):
            return
        margin_side = 14
        max_meld_width = meld_tile[0] * 4 + 12
        x = max(margin_side, play_rect.width - margin_side - max_meld_width)
        y = play_rect.bottom - hand_tile[1] - 14
        spacing = 8
        for meld in melds[player_idx]:
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
                tile_orientation = 0
                if sideways_index is not None and idx == sideways_index:
                    tile_orientation = 90
                tile_face_up = opened
                if not opened and meld_type == "kan" and total_tiles >= 4:
                    if idx in {0, total_tiles - 1}:
                        tile_face_up = False
                    else:
                        tile_face_up = True
                self._draw_tile(
                    canvas,
                    board,
                    play_rect,
                    tile,
                    meld_tile,
                    tile_face_up,
                    tile_orientation,
                    (cur_x, cur_y),
                )
                if sideways_index is not None and idx in {sideways_index - 1, sideways_index}:
                    cur_x -= (meld_tile[0]+meld_tile[1])/2 + 4
                else:
                    cur_x -= meld_tile[0] + 4
            x = cur_x - spacing

    def _draw_tile_grid(
        self,
        canvas: InstructionGroup,
        board: MahjongBoardWidget,
        play_rect: Rect,
        tiles: list[int],
        area: Rect,
        tile_size: Tuple[int, int],
        orientation: int,
        columns: int,
        orientation_map: Optional[dict[int, int]] = None,
    ) -> None:
        if not tiles:
            return
        columns = max(1, columns)
        spacing = 4
        x0, y0 = area.left, area.top
        x, y = x0, y0
        for idx, tile in enumerate(tiles):
            column = idx % columns
            row_prev = -1 if idx == 0 else min(3, (idx - 1) // columns)
            row = min(3, idx // columns)
            tile_orientation = orientation
            if orientation_map and idx in orientation_map:
                tile_orientation = orientation_map[idx]
            if row != row_prev:
                x = x0
            elif orientation_map and idx - 1 in orientation_map:
                x += (tile_size[0]+tile_size[1])/2 + spacing
            elif orientation_map and idx in orientation_map:
                x += (tile_size[0]+tile_size[1])/2 + spacing
            else:
                x += tile_size[0] + spacing
            y = area.top + row * (tile_size[1] + spacing)
            self._draw_tile(canvas, board, play_rect, tile, tile_size, True, tile_orientation, (x, y))

    def _draw_tile(
        self,
        canvas: InstructionGroup,
        board: MahjongBoardWidget,
        play_rect: Rect,
        tile_136: int,
        size: Tuple[int, int],
        face_up: bool,
        orientation: int,
        origin: Tuple[float, float],
    ) -> None:
        width, height = size
        x, y = origin
        pos = self._to_canvas_pos(board, play_rect, x, y, width, height)
        canvas.add(PushMatrix())
        cx = pos[0] + width / 2
        cy = pos[1] + height / 2
        canvas.add(Translate(cx, cy))
        if orientation:
            canvas.add(Rotate(angle=orientation))
        canvas.add(Translate(-width / 2, -height / 2))

        if not face_up:
            canvas.add(Color(*self._face_down_color))
            canvas.add(RoundedRectangle(pos=(0, 0), size=size, radius=[6, 6, 6, 6]))
            canvas.add(Color(*self._face_down_border))
            canvas.add(Line(rounded_rectangle=(0, 0, width, height, 6), width=2))
            canvas.add(PopMatrix())
            return

        tile_34 = tile_136 // 4
        if tile_136 == 16:
            tile_34 = 34
        elif tile_136 == 52:
            tile_34 = 35
        elif tile_136 == 88:
            tile_34 = 36

        texture = self._asset_manager.get_texture(tile_34)
        if texture is None:
            self._draw_tile_placeholder(canvas, board, play_rect, tile_34, size, origin, local=True)
            canvas.add(PopMatrix())
            return

        canvas.add(Color(1, 1, 1, 1))
        canvas.add(RoundedRectangle(texture=texture, size=size, pos=(0, 0), radius=[6, 6, 6, 6]))
        canvas.add(PopMatrix())

    def _draw_tile_placeholder(
        self,
        canvas: InstructionGroup,
        board: MahjongBoardWidget,
        play_rect: Rect,
        tile_34: int,
        size: Tuple[int, int],
        origin: Tuple[float, float],
        local: bool = False,
    ) -> None:
        width, height = size
        if local:
            pos = (0, 0)
        else:
            x, y = origin
            pos = self._to_canvas_pos(board, play_rect, x, y, width, height)
        background = self._get_tile_color(tile_34)
        bg = tuple(c / 255.0 for c in background)
        canvas.add(Color(*bg, 1))
        canvas.add(RoundedRectangle(pos=pos, size=size, radius=[6, 6, 6, 6]))
        canvas.add(Color(245 / 255.0, 245 / 255.0, 245 / 255.0, 1))
        canvas.add(Line(rounded_rectangle=(*pos, *size, 6), width=2))
        label = CoreLabel(
            text=_TILE_SYMBOLS[tile_34],
            font_size=max(12, int(height * 0.45)),
            font_name=self._font_name,
        )
        label.refresh()
        label_pos = (
            pos[0] + width / 2 - label.texture.size[0] / 2,
            pos[1] + height / 2 - label.texture.size[1] / 2,
        )
        canvas.add(Color(20 / 255.0, 20 / 255.0, 20 / 255.0, 1))
        canvas.add(Rectangle(texture=label.texture, size=label.texture.size, pos=label_pos))

    def _get_tile_color(self, tile_34: int) -> Tuple[int, int, int]:
        symbol = _TILE_SYMBOLS[tile_34]
        suit = symbol[-1] if symbol[-1] in {"m", "p", "s"} else "z"
        palette = {
            "m": (210, 90, 90),
            "p": (90, 150, 225),
            "s": (90, 190, 120),
            "z": (220, 210, 150),
        }
        return palette.get(suit, (200, 200, 200))

    def _get_discard_tiles(self, player_idx: int) -> list[int]:
        if player_idx >= len(getattr(self._env, "discard_pile_seq", [])):
            return []
        return [t for t in self._env.discard_pile_seq[player_idx]]

    def _draw_status_labels(self) -> None:
        if not self._root:
            return
        if self._env.current_player >= 0:
            player_text = self._translate(
                "seat_placeholder_format", index=self._env.current_player
            )
        else:
            player_text = self._translate("unknown_player")
        phase_text = self._translate(
            "status_format",
            phase_label=self._translate("phase_label"),
            current_player_label=self._translate("current_player_label"),
            phase=self._env.phase,
            player=player_text,
        )
        self._root.status_label.text = phase_text
        reward_color = self._danger_color if self._last_payload.reward < 0 else self._text_color
        action_value = (
            "-" if self._last_payload.action is None else str(self._last_payload.action)
        )
        reward_text = self._translate(
            "action_reward_format",
            action_label=self._translate("action_label"),
            reward_label=self._translate("reward_label"),
            action=action_value,
            reward=f"{self._last_payload.reward:.2f}",
        )
        self._root.reward_label.text = reward_text
        self._root.reward_label.color = reward_color
        if self._last_payload.done:
            self._root.done_label.text = self._translate("episode_finished")
        else:
            self._root.done_label.text = ""

    def _update_control_buttons(self) -> None:
        pause_label = (
            self._translate("pause_on_score_on")
            if self._state.pause_on_score
            else self._translate("pause_on_score_off")
        )
        self._root.pause_button.text = pause_label
        self._root.pause_button.disabled = not self._state.auto_advance
        auto_label = (
            self._translate("auto_next_on")
            if self._state.auto_advance
            else self._translate("auto_next_off")
        )
        self._root.auto_button.text = auto_label
        step_enabled = (not self._state.auto_advance) or self._state.score_pause_active
        self._root.step_button.disabled = not step_enabled
        self._root.step_button.text = self._translate("step_next")

