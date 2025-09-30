from __future__ import annotations

import importlib.util
import math
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Tuple

_spec = importlib.util.find_spec("kivy")
if _spec is None:  # pragma: no cover - dependency guard
    raise RuntimeError(
        "Kivy is required for the MahjongEnv Kivy wrapper; install kivy to continue"
    )

_cairosvg_spec = importlib.util.find_spec("cairosvg")
if _cairosvg_spec is not None:
    import cairosvg
else:  # pragma: no cover - optional dependency guard
    cairosvg = None

_pillow_spec = importlib.util.find_spec("PIL")
if _pillow_spec is None:  # pragma: no cover - dependency guard
    raise RuntimeError(
        "Pillow is required for the MahjongEnv Kivy wrapper; install pillow to continue"
    )

from PIL import Image, ImageDraw, ImageFont

from kivy.base import EventLoop
from kivy.clock import Clock
from kivy.core.text import Label as CoreLabel
from kivy.core.window import WindowBase
from kivy.graphics import (
    Color,
    Ellipse,
    InstructionGroup,
    Line,
    PopMatrix,
    PushMatrix,
    Rectangle,
    Rotate,
    RoundedRectangle,
    Translate,
)
from kivy.graphics.texture import Texture
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.widget import Widget

from mahjong_env import MahjongEnv as _BaseMahjongEnv


_ALT_TILE_SYMBOLS: Tuple[str, ...] = (
    "1m",
    "2m",
    "3m",
    "4m",
    "5m",
    "6m",
    "7m",
    "8m",
    "9m",
    "1p",
    "2p",
    "3p",
    "4p",
    "5p",
    "6p",
    "7p",
    "8p",
    "9p",
    "1s",
    "2s",
    "3s",
    "4s",
    "5s",
    "6s",
    "7s",
    "8s",
    "9s",
    "Ea",
    "So",
    "We",
    "No",
    "Wh",
    "Gr",
    "Re",
)

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


class _MahjongBoardWidget(Widget):
    """Placeholder widget for the Mahjong board rendering surface."""

    def __init__(self, env: "MahjongEnv", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._env = env

    def refresh(self) -> None:
        """Redraw the board contents."""

        if self._env is None:
            return

        width, height = self.size
        if width <= 0 or height <= 0:
            return

        self.canvas.clear()
        instructions = InstructionGroup()
        try:
            self._env._draw_board(instructions, (width, height))
        except Exception:
            return
        self.canvas.add(instructions)


class _MahjongControlPanel(BoxLayout):
    """Container widget holding control buttons and status labels."""

    def __init__(self, env: "MahjongEnv", **kwargs: Any) -> None:
        super().__init__(orientation="horizontal", spacing=dp(8), **kwargs)
        self._env = env

        self.padding = (dp(12), dp(8), dp(12), dp(8))
        self.size_hint_y = None
        self.height = dp(64)

        self._auto_button = Button(size_hint=(None, 1), width=dp(180))
        self._step_button = Button(size_hint=(None, 1), width=dp(120))
        self._pause_button = Button(size_hint=(None, 1), width=dp(220))
        for button in (self._auto_button, self._step_button, self._pause_button):
            button.background_normal = ""
            button.background_down = ""
            button.border = (0, 0, 0, 0)
        self._status_label = Label(halign="right", valign="middle")
        self._status_label.bind(size=self._update_status_text_width)

        self._auto_button.bind(on_press=self._handle_auto)
        self._step_button.bind(on_press=self._handle_step)
        self._pause_button.bind(on_press=self._handle_pause)

        self.add_widget(self._pause_button)
        self.add_widget(self._auto_button)
        self.add_widget(self._step_button)
        self.add_widget(self._status_label)

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------
    def _handle_auto(self, _instance: Button) -> None:
        self._env._toggle_auto_advance()

    def _handle_step(self, _instance: Button) -> None:
        self._env._trigger_step_once()

    def _handle_pause(self, _instance: Button) -> None:
        self._env._toggle_pause_on_score()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def update_state(
        self,
        *,
        auto_enabled: bool,
        auto_active: bool,
        pause_enabled: bool,
        pause_active: bool,
        step_enabled: bool,
        step_active: bool,
        status_text: str,
    ) -> None:
        self._auto_button.disabled = not auto_enabled
        self._auto_button.text = "Auto Next: ON" if auto_active else "Auto Next: OFF"
        self._apply_button_style(self._auto_button, enabled=auto_enabled, active=auto_active)

        self._step_button.disabled = not step_enabled
        self._step_button.text = "Next"
        self._apply_button_style(self._step_button, enabled=step_enabled, active=step_active)

        self._pause_button.disabled = not pause_enabled
        pause_label = "Pause on Score: ON" if pause_active else "Pause on Score: OFF"
        self._pause_button.text = pause_label
        self._apply_button_style(self._pause_button, enabled=pause_enabled, active=pause_active)

        self._status_label.text = status_text
        self._status_label.texture_update()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _apply_button_style(self, button: Button, *, enabled: bool, active: bool) -> None:
        base = self._env._panel_color
        accent = self._env._accent_color
        muted = self._env._muted_text_color
        text_color = self._env._text_color

        def adjust(color: Tuple[int, int, int], delta: int) -> Tuple[int, int, int]:
            return tuple(max(0, min(255, component + delta)) for component in color)

        background = base
        if active:
            background = adjust(base, 40)
        elif not enabled:
            background = adjust(base, -20)

        button.background_color = (
            background[0] / 255.0,
            background[1] / 255.0,
            background[2] / 255.0,
            1.0,
        )

        foreground = accent if active else (text_color if enabled else muted)
        button.color = (
            foreground[0] / 255.0,
            foreground[1] / 255.0,
            foreground[2] / 255.0,
            1.0,
        )

    def _update_status_text_width(self, _instance: Label, size: Tuple[float, float]) -> None:
        if size[0] == 0:
            return
        self._status_label.text_size = (size[0], None)


class _MahjongRoot(BoxLayout):
    """Top-level root widget hosting the board and control panel."""

    def __init__(self, env: "MahjongEnv", **kwargs: Any) -> None:
        super().__init__(orientation="vertical", spacing=dp(8), padding=dp(12), **kwargs)
        self.board = _MahjongBoardWidget(env, size_hint=(1, 1))
        self.control_panel = _MahjongControlPanel(env, size_hint=(1, None))
        self.add_widget(self.board)
        self.add_widget(self.control_panel)


class MahjongEnv(_BaseMahjongEnv):
    """Mahjong environment with a Kivy GUI overlay."""

    def __init__(
        self,
        *args: Any,
        window_size: Tuple[int, int] = (1024, 720),
        fps: int = 30,
        font_name: Optional[str] = None,
        font_size: int = 12,
        fallback_fonts: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> None:
        self._window_size = window_size
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
        self._background_color = (12, 30, 60)
        self._play_area_color = (24, 60, 90)
        self._play_area_border = (40, 90, 130)
        self._panel_color = (5, 5, 5)
        self._panel_border = (90, 120, 160)
        self._accent_color = (170, 230, 255)
        self._text_color = (235, 235, 235)
        self._muted_text_color = (170, 190, 210)
        self._danger_color = (220, 120, 120)
        self._face_down_color = (18, 18, 22)
        self._face_down_border = (60, 60, 70)
        self._small_font_size = max(10, font_size - 2)
        self._header_font_size = max(font_size + 10, font_size + font_size // 2)
        self._resolved_font_name = self._resolve_kivy_font_name()
        self._window: Optional[WindowBase] = None
        self._root_widget: Optional[_MahjongRoot] = None
        self._board_widget: Optional[_MahjongBoardWidget] = None
        self._control_panel: Optional[_MahjongControlPanel] = None
        self._line_height = font_size + 4
        self._quit_requested = False
        self._last_payload = _RenderPayload(action=None, reward=0.0, done=False, info={})
        self._auto_advance = True
        self._step_once_requested = False
        self._pause_on_score = False
        self._score_pause_active = False
        self._score_pause_pending = False
        self._last_phase_is_score_last = ""
        self._asset_root = Path(__file__).resolve().parent.parent / "assets" / "tiles" / "Regular"
        self._raw_tile_assets: dict[int, Image.Image] = {}
        self._tile_cache: dict[tuple[int, Tuple[int, int]], Image.Image] = {}
        self._tile_texture_cache: dict[tuple[int, Tuple[int, int]], Texture] = {}
        self._tile_orientation_cache: dict[tuple[int, Tuple[int, int], int], Image.Image] = {}
        self._tile_orientation_texture_cache: dict[
            tuple[int, Tuple[int, int], int], Texture
        ] = {}
        self._face_down_image_cache: dict[tuple[Tuple[int, int], int], Image.Image] = {}
        self._face_down_texture_cache: dict[tuple[Tuple[int, int], int], Texture] = {}
        self._placeholder_image_cache: dict[tuple[int, Tuple[int, int]], Image.Image] = {}
        self._tile_metrics: dict[str, Tuple[int, int]] = {}
        self._riichi_states: list[bool] = []
        self._riichi_pending: list[bool] = []
        self._discard_counts: list[int] = []
        self._riichi_declarations: dict[int, int] = {}
        self._frame_interval = 1.0 / float(self._fps) if self._fps > 0 else 0.0
        self._last_frame_time: Optional[float] = None
        self._pending_resize: Optional[Tuple[int, int]] = None
        self._assets_loaded = False

        self._ensure_gui()
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Overridden gym.Env interface
    # ------------------------------------------------------------------
    def reset(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        self._process_events()
        observation = super().reset(*args, **kwargs)
        self._last_payload = _RenderPayload(action=None, reward=0.0, done=self.done, info={})
        self._step_once_requested = False
        self._score_pause_active = False
        self._score_pause_pending = False
        self._last_phase_is_score_last = getattr(self, "phase", "")
        self._riichi_states = []
        self._riichi_pending = []
        self._discard_counts = []
        self._riichi_declarations = {}
        self._render()
        return observation

    def step(self, action: int):  # type: ignore[override]
        self._process_events()
        if self._quit_requested:
            observation = self.get_observation(self.current_player)
            self.done = True
            info = {"terminated_by_gui": True}
            self._last_payload = _RenderPayload(action=None, reward=0.0, done=True, info=info)
            self._render()
            return observation, 0.0, True, info

        while (
            not self._quit_requested
            and ((not self._auto_advance and not self._step_once_requested) or self._score_pause_active)
            and not getattr(self, "done", False)
        ):
            self._process_events()
            self._render()
            self._limit_fps()

        if self._step_once_requested:
            self._step_once_requested = False

        observation, reward, done, info = super().step(action)
        self._last_payload = _RenderPayload(action=action, reward=reward, done=done, info=info)
        self._render()
        return observation, reward, done, info

    def close(self) -> None:
        if self._window is not None:
            try:
                self._window.remove_widget(self._root_widget)  # type: ignore[arg-type]
            except Exception:
                pass
            try:
                self._window.close()
            except Exception:
                pass
        self._window = None
        self._root_widget = None
        self._board_widget = None
        self._control_panel = None
        self._quit_requested = True

    # ------------------------------------------------------------------
    # GUI lifecycle helpers
    # ------------------------------------------------------------------
    def _ensure_gui(self) -> None:
        if self._window is not None:
            return

        EventLoop.ensure_window()
        if not getattr(EventLoop, "_running", False):
            try:
                EventLoop.start()
            except Exception:
                # Fall back to marking the loop as running so manual draining via
                # ``Clock.tick``/``EventLoop.idle`` works even if ``start`` is
                # unavailable on the current provider implementation.
                setattr(EventLoop, "_running", True)

        window = EventLoop.window
        if window is None:
            raise RuntimeError("Failed to create Kivy window for MahjongEnv")

        # ``ensure_window`` may return a window instance that has not yet been
        # materialised on screen.  Explicitly trigger creation and showing of
        # the native window so the GUI becomes visible when the environment is
        # instantiated, matching the pygame behaviour of the original wrapper.
        if getattr(window, "canvas", None) is None and hasattr(window, "create_window"):
            window.create_window()
        if hasattr(window, "show"):
            try:
                window.show()
            except Exception:
                # Some window providers (e.g. SDL2) implicitly show during
                # ``create_window`` and raise if ``show`` is called twice.
                pass

        window.size = self._window_size
        window.clearcolor = (
            self._background_color[0] / 255.0,
            self._background_color[1] / 255.0,
            self._background_color[2] / 255.0,
            1.0,
        )
        window.bind(on_request_close=self._on_request_close)
        window.bind(on_resize=self._on_window_resize)

        root = _MahjongRoot(self)
        window.add_widget(root)

        self._window = window
        self._root_widget = root
        self._board_widget = root.board
        self._control_panel = root.control_panel
        self._update_control_panel()
        if not self._assets_loaded:
            self._load_tile_assets()

        # ``ensure_window`` returns once the native window has been created, but
        # a further event pump is required for some window backends (notably the
        # SDL2 provider on Linux) to materialise the surface on screen.  Drain
        # the event queue and request an explicit redraw so the window becomes
        # visible immediately instead of lingering as a flashing taskbar icon.
        self._drain_kivy_event_loop()
        try:
            window.canvas.ask_update()
        except Exception:
            pass

    def _process_events(self) -> None:
        if self._window is None:
            return

        self._drain_kivy_event_loop()
        if self._pending_resize is not None:
            self._window_size = self._pending_resize
            self._pending_resize = None

    def _render(self) -> None:
        if self._window is None or self._root_widget is None:
            return

        if self._score_last():
            if self._auto_advance and self._pause_on_score:
                if not self._last_phase_is_score_last:
                    self._score_pause_pending = True
                    self._score_pause_active = True
            else:
                self._score_pause_pending = False
                self._score_pause_active = False
            self._last_phase_is_score_last = self._score_last()
        elif self._score_pause_pending:
            if self._auto_advance and self._pause_on_score:
                self._score_pause_active = True
            else:
                self._score_pause_active = False
                self._score_pause_pending = False
        else:
            self._score_pause_pending = False
            self._score_pause_active = False
            self._last_phase_is_score_last = self._score_last()

        self._update_control_panel()
        if self._board_widget is not None:
            self._board_widget.refresh()

        self._drain_kivy_event_loop()
        self._limit_fps()

    def _drain_kivy_event_loop(self) -> None:
        try:
            Clock.tick()
        except Exception:
            pass
        try:
            EventLoop.idle()
        except Exception:
            pass

    def _limit_fps(self) -> None:
        if self._frame_interval <= 0:
            return
        now = time.monotonic()
        if self._last_frame_time is None:
            self._last_frame_time = now
            return
        elapsed = now - self._last_frame_time
        if elapsed < self._frame_interval:
            time.sleep(self._frame_interval - elapsed)
        self._last_frame_time = time.monotonic()

    # ------------------------------------------------------------------
    # Control button actions
    # ------------------------------------------------------------------
    def _toggle_auto_advance(self) -> None:
        self._auto_advance = not self._auto_advance
        if self._auto_advance:
            self._step_once_requested = False
            if self._score_last() and self._pause_on_score:
                self._score_pause_active = True
                self._score_pause_pending = True
            else:
                self._score_pause_active = False
                self._score_pause_pending = False
        else:
            self._step_once_requested = False
            self._score_pause_active = False
            self._score_pause_pending = False
        self._update_control_panel()

    def _trigger_step_once(self) -> None:
        if not self._auto_advance:
            self._step_once_requested = True
        elif self._score_pause_active:
            self._score_pause_active = False
            self._score_pause_pending = False
        self._update_control_panel()

    def _toggle_pause_on_score(self) -> None:
        if not self._auto_advance:
            return
        self._pause_on_score = not self._pause_on_score
        if self._pause_on_score and self._score_last():
            self._score_pause_active = True
            self._score_pause_pending = True
        else:
            self._score_pause_active = False
            self._score_pause_pending = False
        self._update_control_panel()

    def _score_last(self) -> bool:
        return getattr(self, "phase", "") == "score" and self.current_player == self.num_players - 1

    # ------------------------------------------------------------------
    # Window callbacks
    # ------------------------------------------------------------------
    def _on_request_close(self, *_args: Any) -> bool:
        self._quit_requested = True
        return True

    def _on_window_resize(self, _window: WindowBase, width: int, height: int) -> None:
        self._pending_resize = (int(width), int(height))

    # ------------------------------------------------------------------
    # Control panel updates
    # ------------------------------------------------------------------
    def _update_control_panel(self) -> None:
        if self._control_panel is None:
            return

        pause_enabled = self._auto_advance
        pause_active = self._pause_on_score and pause_enabled
        step_enabled = (not self._auto_advance) or self._score_pause_active
        status_text = self._build_status_text()
        self._control_panel.update_state(
            auto_enabled=True,
            auto_active=self._auto_advance,
            pause_enabled=pause_enabled,
            pause_active=pause_active,
            step_enabled=step_enabled,
            step_active=self._score_pause_active,
            status_text=status_text,
        )

    def _build_status_text(self) -> str:
        phase = getattr(self, "phase", "unknown")
        player = getattr(self, "current_player", 0)
        action = self._last_payload.action
        reward = self._last_payload.reward
        segments = [f"Phase: {phase}", f"Current Player: P{player}"]
        action_segment = f"Action: {action}" if action is not None else "Action: --"
        reward_segment = f"Reward: {reward:.2f}"
        segments.append(action_segment)
        segments.append(reward_segment)
        if self._last_payload.done:
            segments.append("Episode finished")
        return "  |  ".join(segments)

    # ------------------------------------------------------------------
    # Board rendering helpers
    # ------------------------------------------------------------------
    def _resolve_kivy_font_name(self) -> Optional[str]:
        if self._font_name:
            font_path = Path(self._font_name)
            if font_path.exists():
                return str(font_path)
        for candidate in self._fallback_fonts:
            candidate_path = Path(candidate)
            if candidate_path.exists():
                return str(candidate_path)
        return None

    @staticmethod
    def _color_to_float(color: Tuple[int, int, int], alpha: float = 1.0) -> Tuple[float, float, float, float]:
        r, g, b = color
        return (r / 255.0, g / 255.0, b / 255.0, alpha)

    def _draw_board(self, instructions: InstructionGroup, size: Tuple[float, float]) -> None:
        width, height = size
        if width <= 0 or height <= 0:
            return

        play_size = min(width, height)
        play_x = (width - play_size) / 2.0
        play_y = (height - play_size) / 2.0
        play_rect = (play_x, play_y, play_size, play_size)

        instructions.add(Color(*self._color_to_float(self._background_color)))
        instructions.add(Rectangle(pos=(0, 0), size=size))

        radius = 16
        instructions.add(Color(*self._color_to_float(self._play_area_color)))
        instructions.add(
            RoundedRectangle(pos=(play_x, play_y), size=(play_size, play_size), radius=[radius] * 4)
        )
        instructions.add(Color(*self._color_to_float(self._play_area_border)))
        instructions.add(
            Line(rounded_rectangle=(play_x, play_y, play_size, play_size, radius), width=3)
        )

        self._compute_tile_metrics((play_size, play_size))
        center_rect = self._draw_center_panel(instructions, play_rect)
        self._update_riichi_state()
        self._draw_dead_wall(instructions, play_rect)
        self._draw_player_areas(instructions, play_rect)
        self._draw_seat_labels(instructions, play_rect, size)

        if self._score_last():
            self._draw_score_panel(instructions, size)
        else:
            self._draw_status_text(instructions, size)

    def _compute_tile_metrics(self, play_size: Tuple[float, float]) -> None:
        width = max(1, int(play_size[0]))
        height = max(1, int(play_size[1]))

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

    def _create_label(
        self,
        text: str,
        *,
        font_size: int,
        color: Tuple[int, int, int],
        bold: bool = False,
        halign: str = "left",
        valign: str = "bottom",
        text_size: Optional[Tuple[float, float]] = None,
    ) -> CoreLabel:
        label = CoreLabel(
            text=text,
            font_size=font_size,
            font_name=self._resolved_font_name,
            bold=bold,
            halign=halign,
            valign=valign,
        )
        if text_size is not None:
            label.options["text_size"] = text_size
        label.color = (
            color[0] / 255.0,
            color[1] / 255.0,
            color[2] / 255.0,
            1.0,
        )
        label.refresh()
        return label

    def _draw_text(
        self,
        instructions: InstructionGroup,
        text: str,
        position: Tuple[float, float],
        *,
        font_size: int,
        color: Tuple[int, int, int],
        bold: bool = False,
        halign: str = "left",
        valign: str = "bottom",
        text_size: Optional[Tuple[float, float]] = None,
    ) -> Tuple[float, float]:
        if not text:
            return (0.0, 0.0)
        label = self._create_label(
            text,
            font_size=font_size,
            color=color,
            bold=bold,
            halign=halign,
            valign=valign,
            text_size=text_size,
        )
        width, height = label.texture.size
        x, y = position
        if halign == "center":
            x -= width / 2.0
        elif halign == "right":
            x -= width
        if valign == "center":
            y -= height / 2.0
        elif valign == "top":
            y -= height
        instructions.add(Color(1.0, 1.0, 1.0, 1.0))
        instructions.add(Rectangle(texture=label.texture, pos=(x, y), size=label.texture.size))
        return (width, height)

    def _measure_text(
        self,
        text: str,
        *,
        font_size: int,
        color: Tuple[int, int, int] | None = None,
        bold: bool = False,
        text_size: Optional[Tuple[float, float]] = None,
    ) -> Tuple[float, float]:
        if not text:
            return (0.0, 0.0)
        color_value = color if color is not None else self._text_color
        label = self._create_label(
            text,
            font_size=font_size,
            color=color_value,
            bold=bold,
            text_size=text_size,
        )
        return label.texture.size

    def _draw_center_panel(
        self, instructions: InstructionGroup, play_rect: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float]:
        px, py, pwidth, pheight = play_rect
        center_width = max(100.0, pwidth * 0.24)
        center_height = max(100.0, pheight * 0.24)
        cx = px + (pwidth - center_width) / 2.0
        cy = py + (pheight - center_height) / 2.0

        instructions.add(Color(*self._color_to_float(self._panel_color)))
        instructions.add(
            RoundedRectangle(pos=(cx, cy), size=(center_width, center_height), radius=[12] * 4)
        )
        instructions.add(Color(*self._color_to_float(self._panel_border)))
        instructions.add(Line(rounded_rectangle=(cx, cy, center_width, center_height, 12), width=2))

        round_data = getattr(self, "round", [0, 0])
        round_index = round_data[0] if isinstance(round_data, (list, tuple)) and round_data else 0
        if isinstance(round_data, (list, tuple)) and round_data and isinstance(round_data[0], int):
            round_index = round_data[0]
        wind_names = ["East", "South", "West", "North"]
        wind = wind_names[(round_index // 4) % 4]
        hand_number = round_index % 4 + 1
        round_text = f"{wind} {hand_number}"

        _, header_height = self._draw_text(
            instructions,
            round_text,
            (cx + center_width / 2.0, cy + center_height - 18),
            font_size=self._header_font_size,
            color=self._text_color,
            bold=True,
            halign="center",
            valign="top",
        )

        honba = 0
        if isinstance(round_data, (list, tuple)) and len(round_data) > 1 and isinstance(round_data[1], int):
            honba = round_data[1]
        riichi = int(getattr(self, "num_riichi", 0))
        tiles_remaining = len(getattr(self, "deck", []))
        counter_texts = (f"Honba {honba}", f"Riichi {riichi}", f"Tiles {tiles_remaining}")
        current_top = cy + center_height - 18 - header_height - 12
        for text in counter_texts:
            _, text_height = self._draw_text(
                instructions,
                text,
                (cx + center_width / 2.0, current_top),
                font_size=self._small_font_size,
                color=self._muted_text_color,
                halign="center",
                valign="top",
            )
            current_top -= text_height + 4

        score_positions = {
            0: (cx + center_width / 2.0, cy - 14),
            1: (cx + center_width + 18, cy + center_height / 2.0),
            2: (cx + center_width / 2.0, cy + center_height + 14),
            3: (cx - 18, cy + center_height / 2.0),
        }

        scores = list(getattr(self, "scores", []))
        current_player = getattr(self, "current_player", 0)
        for idx, position in score_positions.items():
            if idx >= len(scores):
                continue
            score_value = int(scores[idx] * 100)
            color = self._accent_color if idx == current_player else self._text_color
            self._draw_text(
                instructions,
                f"{score_value:5d}",
                position,
                font_size=self._font_size,
                color=color,
                halign="center",
                valign="center",
            )

        return (cx, cy, center_width, center_height)

    def _get_discard_tiles(self, player_idx: int) -> list[int]:
        if player_idx >= len(getattr(self, "discard_pile_seq", [])):
            return []
        return [int(tile) for tile in self.discard_pile_seq[player_idx]]

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
        riichi_flags = list(getattr(self, "riichi", []))
        discard_seq = list(getattr(self, "discard_pile_seq", []))
        num_players = max(len(riichi_flags), len(discard_seq), getattr(self, "num_players", 0))
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

    def _compute_hand_reveal_flags(self, num_players: int) -> list[bool]:
        reveal = [False] * max(0, num_players)
        for idx in range(num_players):
            reveal[idx] = idx == 0

        if getattr(self, "phase", "") != "score":
            return reveal

        agari = getattr(self, "agari", None)
        if agari:
            for winner in self._extract_winner_indices(agari):
                if 0 <= winner < num_players:
                    reveal[winner] = True
        else:
            tenpai_flags = list(getattr(self, "tenpai", []))
            for idx in range(num_players):
                if idx < len(tenpai_flags) and tenpai_flags[idx]:
                    reveal[idx] = True

        return reveal

    def _draw_tile_grid(
        self,
        instructions: InstructionGroup,
        tiles: list[int],
        area: Tuple[float, float, float, float],
        tile_size: Tuple[int, int],
        orientation: int,
        columns: int,
        orientation_map: Optional[dict[int, int]] = None,
    ) -> None:
        if not tiles:
            return

        columns = max(1, columns)
        spacing = 4
        area_x, area_y, area_width, area_height = area
        tile_width, tile_height = tile_size
        rows = min(4, max(1, math.ceil(len(tiles) / columns)))
        effective_height = rows * tile_height + max(0, rows - 1) * spacing
        top = area_y + area_height if area_height else area_y + effective_height

        for idx, tile in enumerate(tiles):
            row = min(rows - 1, idx // columns)
            column = idx % columns
            tile_orientation = orientation
            if orientation_map and idx in orientation_map:
                tile_orientation = orientation_map[idx]
            texture = self._get_tile_surface(tile, tile_size, True, tile_orientation)
            draw_width, draw_height = texture.size
            x = area_x + column * (tile_width + spacing)
            y = top - (row + 1) * tile_height - row * spacing
            if draw_width != tile_width:
                x += (tile_width - draw_width) / 2.0
            if draw_height != tile_height:
                y += (tile_height - draw_height) / 2.0
            instructions.add(Color(1.0, 1.0, 1.0, 1.0))
            instructions.add(Rectangle(texture=texture, pos=(x, y), size=texture.size))

    def _draw_melds(
        self,
        instructions: InstructionGroup,
        player: int,
        origin: Tuple[float, float],
        direction: str,
        tile_size: Tuple[int, int],
        orientation: int,
    ) -> None:
        melds = getattr(self, "melds", [])
        if player >= len(melds):
            return

        x, y = origin
        spacing = 8
        for meld in melds[player]:
            tiles = list(reversed([tile for tile in meld.get("m", [])]))
            opened = bool(meld.get("opened", True))
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
                texture = (
                    self._get_tile_surface(tile, tile_size, True, tile_orientation)
                    if tile_face_up
                    else self._get_face_down_surface(tile_size, tile_orientation)
                )
                draw_width, draw_height = texture.size
                draw_x = cur_x - draw_width
                instructions.add(Color(1.0, 1.0, 1.0, 1.0))
                instructions.add(Rectangle(texture=texture, pos=(draw_x, cur_y), size=texture.size))
                if direction == "horizontal":
                    cur_x = draw_x - spacing
                else:
                    cur_y -= draw_height + spacing
            if direction == "horizontal":
                x = cur_x - spacing
            else:
                y = cur_y - spacing

    def _draw_player_layout(
        self,
        instructions: InstructionGroup,
        play_rect: Tuple[float, float, float, float],
        player_idx: int,
        face_up_hand: bool,
        sort_hand: bool = True,
    ) -> None:
        px, py, pwidth, _ = play_rect
        margin_side = 14
        margin_bottom = 14

        tile_size = self._tile_metrics.get("south_hand", (40, 56))
        spacing = tile_size[0] + 6
        draw_gap = 0

        hands = getattr(self, "hands", [])
        hand_tiles = list(hands[player_idx]) if player_idx < len(hands) else []
        if sort_hand and hand_tiles:
            if getattr(self, "phase", "") in {"draw", "kan_draw"} and self.current_player == player_idx:
                hand_tiles = sorted(hand_tiles[:-1]) + [hand_tiles[-1]]
            else:
                hand_tiles = sorted(hand_tiles)
        riichi_flags = list(getattr(self, "riichi", []))
        in_riichi = player_idx < len(riichi_flags) and riichi_flags[player_idx]

        hand_x = px + pwidth / 2.0 - 7 * (tile_size[0] + 6)
        hand_y = py + margin_bottom
        current_x = hand_x
        for idx, tile in enumerate(hand_tiles):
            if len(hand_tiles) > 1 and idx == len(hand_tiles) - 1:
                current_x += draw_gap
            if face_up_hand:
                texture = self._get_tile_surface(tile, tile_size, True, 0)
            else:
                texture = self._get_face_down_surface(tile_size, 0)
            instructions.add(Color(1.0, 1.0, 1.0, 1.0))
            instructions.add(Rectangle(texture=texture, pos=(current_x, hand_y), size=texture.size))
            current_x += spacing

        if in_riichi:
            label_text = "Riichi"
            label_width, label_height = self._measure_text(
                label_text, font_size=self._small_font_size, color=self._accent_color
            )
            padding = 4
            flag_width = label_width + padding * 2
            flag_height = label_height + padding * 2
            flag_x = px + (pwidth - flag_width) / 2.0
            flag_y = hand_y + tile_size[1] + 6
            instructions.add(Color(*self._color_to_float(self._panel_color)))
            instructions.add(
                RoundedRectangle(pos=(flag_x, flag_y), size=(flag_width, flag_height), radius=[6] * 4)
            )
            instructions.add(Color(*self._color_to_float(self._accent_color)))
            instructions.add(Line(rounded_rectangle=(flag_x, flag_y, flag_width, flag_height, 6), width=2))
            self._draw_text(
                instructions,
                label_text,
                (flag_x + flag_width / 2.0, flag_y + flag_height / 2.0),
                font_size=self._small_font_size,
                color=self._accent_color,
                halign="center",
                valign="center",
            )

        discard_tiles = self._get_discard_tiles(player_idx)
        discard_tile = self._tile_metrics.get("discard", tile_size)
        cols = 6
        grid_width = cols * discard_tile[0] + (cols - 1) * 4
        grid_height = 4 * discard_tile[1] + 3 * 4
        grid_x = px + pwidth / 2.0 - grid_width / 2.0
        grid_y = hand_y + discard_tile[1] + 24
        orientation_map: Optional[dict[int, int]] = None
        declaration_index = self._riichi_declarations.get(player_idx)
        if declaration_index is not None:
            orientation_map = {declaration_index: 90}
        self._draw_tile_grid(
            instructions,
            discard_tiles,
            (grid_x, grid_y, grid_width, grid_height),
            discard_tile,
            0,
            cols,
            orientation_map=orientation_map,
        )

        meld_tile = self._tile_metrics.get("meld", tile_size)
        max_meld_width = meld_tile[0] * 4 + 12
        meld_origin_x = px + pwidth - margin_side - meld_tile[0]
        meld_origin_x = min(meld_origin_x, px + pwidth - margin_side)
        meld_origin_y = hand_y
        self._draw_melds(
            instructions,
            player_idx,
            (meld_origin_x, meld_origin_y),
            "horizontal",
            meld_tile,
            0,
        )

    def _draw_player_areas(
        self, instructions: InstructionGroup, play_rect: Tuple[float, float, float, float]
    ) -> None:
        num_players = getattr(self, "num_players", 0)
        if num_players <= 0:
            return

        reveal_flags = self._compute_hand_reveal_flags(num_players)
        angle_map = {0: 0, 1: -90, 2: 180, 3: 90}
        px, py, pwidth, pheight = play_rect
        center_x = px + pwidth / 2.0
        center_y = py + pheight / 2.0

        for player_idx in range(min(4, num_players)):
            face_up = (
                reveal_flags[player_idx]
                if player_idx < len(reveal_flags)
                else player_idx == 0
            )
            group = InstructionGroup()
            group.add(PushMatrix())
            group.add(Translate(center_x, center_y, 0))
            group.add(Rotate(angle=angle_map.get(player_idx, 0), axis=(0, 0, 1)))
            group.add(Translate(-center_x, -center_y, 0))
            self._draw_player_layout(group, play_rect, player_idx, face_up)
            group.add(PopMatrix())
            instructions.add(group)

    def _draw_dead_wall(
        self, instructions: InstructionGroup, play_rect: Tuple[float, float, float, float]
    ) -> None:
        wall_tile = self._tile_metrics.get("wall", (20, 26))
        stack_size = 5
        gap = 6
        margin_y = 32
        total_width = stack_size * (wall_tile[0] + gap) - gap
        px, py, pwidth, pheight = play_rect
        start_x = px + pwidth / 2.0 + total_width / 2.0
        y = py + pheight / 2.0 - margin_y

        dora_tiles = list(getattr(self, "dora_indicator", []))
        for i in range(stack_size):
            x = start_x - (i + 1) * (wall_tile[0] + gap)
            if i < len(dora_tiles):
                texture = self._get_tile_surface(dora_tiles[i], wall_tile, True, 0)
            else:
                texture = self._get_face_down_surface(wall_tile, 0)
            instructions.add(Color(1.0, 1.0, 1.0, 1.0))
            instructions.add(Rectangle(texture=texture, pos=(x, y), size=texture.size))

    def _draw_seat_labels(
        self,
        instructions: InstructionGroup,
        play_rect: Tuple[float, float, float, float],
        surface_size: Tuple[float, float],
    ) -> None:
        px, py, pwidth, pheight = play_rect
        surface_width, surface_height = surface_size
        num_players = getattr(self, "num_players", 0)
        seat_names = list(getattr(self, "seat_names", []))
        if len(seat_names) < num_players:
            seat_names.extend(["NoName"] * (num_players - len(seat_names)))

        label_positions = {
            0: (px + 40, py - 32),
            1: (px + pwidth + 40, py + pheight / 2.0),
            2: (px + pwidth - 40, py + pheight + 32),
            3: (px - 40, py + pheight / 2.0),
        }

        for idx in range(min(4, num_players)):
            name = seat_names[idx] if idx < len(seat_names) else "NoName"
            self._draw_text(
                instructions,
                name,
                label_positions[idx],
                font_size=self._small_font_size,
                color=self._muted_text_color,
                halign="center",
                valign="center",
            )

        dealer = getattr(self, "oya", 0)
        if dealer >= min(4, num_players):
            return

        offsets = {
            0: (0, -22),
            1: (-22, 0),
            2: (0, 22),
            3: (22, 0),
        }
        base_position = label_positions[dealer]
        marker_center = (
            base_position[0] + offsets[dealer][0],
            base_position[1] + offsets[dealer][1],
        )
        instructions.add(Color(*self._color_to_float(self._accent_color)))
        instructions.add(Line(circle=(*marker_center, 10), width=2))
        self._draw_text(
            instructions,
            "E",
            marker_center,
            font_size=self._small_font_size,
            color=self._accent_color,
            halign="center",
            valign="center",
        )

    def _draw_status_text(
        self, instructions: InstructionGroup, surface_size: Tuple[float, float]
    ) -> None:
        surface_width, surface_height = surface_size
        margin = 16
        phase_text = f"Phase: {getattr(self, 'phase', '')}  |  Current Player: P{getattr(self, 'current_player', 0)}"
        self._draw_text(
            instructions,
            phase_text,
            (margin, surface_height - margin),
            font_size=self._small_font_size,
            color=self._accent_color,
            valign="top",
        )

        reward_color = self._danger_color if self._last_payload.reward < 0 else self._text_color
        reward_text = f"Action: {self._last_payload.action}  Reward: {self._last_payload.reward:.2f}"
        reward_width, reward_height = self._measure_text(
            reward_text, font_size=self._small_font_size, color=reward_color
        )
        self._draw_text(
            instructions,
            reward_text,
            (surface_width - margin, surface_height - margin),
            font_size=self._small_font_size,
            color=reward_color,
            halign="right",
            valign="top",
        )

        if self._last_payload.done:
            self._draw_text(
                instructions,
                "Episode finished",
                (surface_width - margin, surface_height - margin - reward_height - 4),
                font_size=self._small_font_size,
                color=self._danger_color,
                halign="right",
                valign="top",
            )

    def _wrap_text(self, text: str, font_size: int, max_width: int) -> list[str]:
        if not text:
            return []
        words = str(text).split()
        if not words:
            return []
        max_width = max(10, max_width)
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            width, _ = self._measure_text(candidate, font_size=font_size)
            if width <= max_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines



    def _draw_score_panel(
        self, instructions: InstructionGroup, surface_size: Tuple[float, float]
    ) -> None:
        surface_width, surface_height = surface_size
        num_players = getattr(self, "num_players", 0)
        if num_players <= 0:
            return

        margin = 0
        padding = 14
        min_dimension = min(surface_width, surface_height)
        available_side = min_dimension - margin * 2
        effective_side = available_side if available_side > 0 else min_dimension
        max_text_width = max(50.0, effective_side - padding * 2)

        round_data = getattr(self, "round", [0, 0])
        round_index = 0
        if isinstance(round_data, (list, tuple)) and round_data:
            first = round_data[0]
            if isinstance(first, int):
                round_index = first
        wind_names = ["East", "South", "West", "North"]
        wind = wind_names[(round_index // 4) % 4]
        hand_number = round_index % 4 + 1
        honba = 0
        if isinstance(round_data, (list, tuple)) and len(round_data) > 1:
            second = round_data[1]
            if isinstance(second, int):
                honba = second
        riichi_sticks = int(getattr(self, "num_riichi", 0) or 0)
        kyoutaku = int(getattr(self, "num_kyoutaku", 0) or 0)

        info_lines = [
            f"{wind} {hand_number} | Honba {honba} | Riichi {riichi_sticks} | Kyoutaku {kyoutaku}"
        ]

        seat_names = list(getattr(self, "seat_names", []))
        if len(seat_names) < num_players:
            seat_names.extend([f"P{idx}" for idx in range(len(seat_names), num_players)])

        agari = getattr(self, "agari", None)
        agari_dict = agari if isinstance(agari, dict) else None
        winners = self._extract_winner_indices(agari_dict) if agari_dict else set()
        winner_idx_value = -1
        if agari_dict:
            raw_winner = agari_dict.get("who", -1)
            if isinstance(raw_winner, int):
                winner_idx_value = raw_winner
            elif isinstance(raw_winner, Iterable) and not isinstance(raw_winner, (str, bytes)):
                for value in raw_winner:
                    if isinstance(value, int):
                        winner_idx_value = value
                        break
        primary_winner: Optional[int] = None
        if winner_idx_value >= 0:
            primary_winner = winner_idx_value
        elif winners:
            primary_winner = min(winners)

        winner_highlight = set(winners)
        if isinstance(primary_winner, int):
            winner_highlight.add(primary_winner)

        message_lines: list[str] = []
        from_who = agari_dict.get("fromwho", -1) if agari_dict else -1
        if agari_dict:
            display_index = (
                primary_winner if isinstance(primary_winner, int) else winner_idx_value
            )
            winner_display = (
                seat_names[display_index]
                if isinstance(display_index, int) and 0 <= display_index < num_players
                else f"Player {display_index}"
            )
            if isinstance(winner_idx_value, int) and winner_idx_value == from_who:
                result_text = f"{winner_display} Tsumo"
            else:
                loser_display = (
                    seat_names[from_who]
                    if isinstance(from_who, int) and 0 <= from_who < num_players
                    else f"Player {from_who}"
                )
                result_text = f"{winner_display} Ron vs {loser_display}"
            message_lines.append(result_text)
            ten = list(agari_dict.get("ten", []))
            fu = ten[0] if len(ten) > 0 else 0
            total = ten[1] if len(ten) > 1 else 0
            han = ten[2] if len(ten) > 2 else 0
            message_lines.append(f"{han} Han | {fu} Fu | {total} Points")
        else:
            tenpai_flags = list(getattr(self, "tenpai", []))
            tenpai_players = [
                seat_names[idx]
                for idx, is_tenpai in enumerate(tenpai_flags)
                if is_tenpai and idx < num_players
            ]
            if tenpai_players:
                message_lines.append("Draw - Tenpai: " + ", ".join(tenpai_players))
            else:
                message_lines.append("Draw - No Tenpai")

        yaku_lines: list[tuple[str, str]] = []
        if agari_dict:
            raw_yaku = [str(item) for item in agari_dict.get("yaku", [])]
            if raw_yaku:
                combined = ", ".join(raw_yaku)
                label = "Yaku: "
                label_width = self._measure_text(label, font_size=self._small_font_size)[0]
                wrapped_yaku = self._wrap_text(
                    combined,
                    self._small_font_size,
                    int(max(10.0, max_text_width - label_width)),
                )
                if wrapped_yaku:
                    yaku_lines.append((label, wrapped_yaku[0]))
                    for extra in wrapped_yaku[1:]:
                        yaku_lines.append(("", extra))

        title_text = "Round Results"
        title_width, title_height = self._measure_text(
            title_text,
            font_size=self._header_font_size,
            color=self._accent_color,
            bold=True,
        )
        content_width = title_width

        info_metrics = [
            self._measure_text(
                line, font_size=self._small_font_size, color=self._muted_text_color
            )
            for line in info_lines
        ]
        info_heights = [height for _, height in info_metrics]
        for width, _ in info_metrics:
            content_width = max(content_width, width)

        message_metrics = [
            self._measure_text(line, font_size=self._small_font_size, color=self._text_color)
            for line in message_lines
        ]
        message_heights = [height for _, height in message_metrics]
        for width, _ in message_metrics:
            content_width = max(content_width, width)

        scores = list(getattr(self, "scores", []))
        if len(scores) < num_players:
            scores.extend([0] * (num_players - len(scores)))
        score_deltas = list(getattr(self, "score_deltas", []))
        if len(score_deltas) < num_players:
            score_deltas.extend([0] * (num_players - len(score_deltas)))
        dealer = getattr(self, "oya", -1)

        def ordinal(value: int) -> str:
            suffix = "th"
            if value % 100 not in {11, 12, 13}:
                suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
            return f"{value}{suffix}"

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

        player_entries: list[dict[str, Any]] = []
        player_heights: list[float] = []
        for player_idx in range(num_players):
            if player_idx >= len(scores):
                continue
            display_name = seat_names[player_idx]
            if player_idx == dealer:
                display_name += " (Dealer)"
            base_color = (
                self._accent_color if player_idx in winner_highlight else self._text_color
            )
            rank_text = ordinal(ranks.get(player_idx, player_idx + 1))
            name_text = f"{rank_text}  {display_name}"
            name_width, name_height = self._measure_text(
                name_text, font_size=self._font_size, color=base_color
            )
            delta_value = score_deltas[player_idx] if player_idx < len(score_deltas) else 0
            delta_points = int(round(delta_value * 100))
            delta_text = f"{delta_points:+}"
            delta_width, _ = self._measure_text(delta_text, font_size=self._font_size)
            score_points = int(round(scores[player_idx] * 100))
            score_text = f"{score_points:>6d}"
            score_width, _ = self._measure_text(score_text, font_size=self._font_size)
            score_line_width = name_width + score_width + delta_width + 16
            content_width = max(content_width, score_line_width)

            delta_color = (
                self._accent_color
                if delta_points > 0
                else self._danger_color if delta_points < 0 else self._muted_text_color
            )

            player_entries.append(
                {
                    "name_text": name_text,
                    "name_color": base_color,
                    "name_height": name_height,
                    "delta_text": delta_text,
                    "delta_color": delta_color,
                    "delta_width": delta_width,
                    "score_text": score_text,
                    "score_color": base_color,
                }
            )
            player_heights.append(name_height)

        label_width = self._measure_text("Yaku: ", font_size=self._small_font_size)[0]
        yaku_heights: list[float] = []
        for prefix, text in yaku_lines:
            prefix_width = (
                self._measure_text(prefix, font_size=self._small_font_size)[0] if prefix else 0
            )
            text_width, text_height = self._measure_text(
                text, font_size=self._small_font_size
            )
            total_width = prefix_width + (label_width if not prefix else 0) + text_width
            content_width = max(content_width, total_width)
            yaku_heights.append(text_height)

        total_height = padding + title_height + 6
        if info_heights:
            total_height += sum(height + 2 for height in info_heights) + 4
        if message_heights:
            total_height += sum(height + 2 for height in message_heights) + 4
        if player_heights:
            total_height += sum(height + 4 for height in player_heights) + 4
        if yaku_heights:
            total_height += sum(height + 2 for height in yaku_heights)
        total_height += padding

        required_width = padding * 2 + content_width
        required_height = total_height
        panel_size = max(required_width, required_height)
        if available_side > 0:
            panel_size = min(panel_size, available_side)
        else:
            panel_size = min(panel_size, min_dimension)

        panel_x = (surface_width - panel_size) / 2.0
        panel_y = (surface_height - panel_size) / 2.0
        top_edge = panel_y + panel_size

        instructions.add(Color(*self._color_to_float(self._panel_color)))
        instructions.add(
            RoundedRectangle(pos=(panel_x, panel_y), size=(panel_size, panel_size), radius=[12] * 4)
        )
        instructions.add(Color(*self._color_to_float(self._panel_border)))
        instructions.add(Line(rounded_rectangle=(panel_x, panel_y, panel_size, panel_size, 12), width=2))

        cursor = padding
        _, title_draw_height = self._draw_text(
            instructions,
            title_text,
            (panel_x + panel_size / 2.0, top_edge - cursor),
            font_size=self._header_font_size,
            color=self._accent_color,
            bold=True,
            halign="center",
            valign="top",
        )
        cursor += title_draw_height + 6

        for line in info_lines:
            _, line_height = self._draw_text(
                instructions,
                line,
                (panel_x + padding, top_edge - cursor),
                font_size=self._small_font_size,
                color=self._muted_text_color,
                valign="top",
            )
            cursor += line_height + 2
        if info_lines:
            cursor += 4

        for line in message_lines:
            _, line_height = self._draw_text(
                instructions,
                line,
                (panel_x + padding, top_edge - cursor),
                font_size=self._small_font_size,
                color=self._text_color,
                valign="top",
            )
            cursor += line_height + 2
        if message_lines:
            cursor += 4

        delta_right = panel_x + panel_size - padding
        for entry in player_entries:
            _, name_height = self._draw_text(
                instructions,
                entry["name_text"],
                (panel_x + padding, top_edge - cursor),
                font_size=self._font_size,
                color=entry["name_color"],
                valign="top",
            )
            self._draw_text(
                instructions,
                entry["delta_text"],
                (delta_right, top_edge - cursor),
                font_size=self._font_size,
                color=entry["delta_color"],
                halign="right",
                valign="top",
            )
            score_right = delta_right - entry["delta_width"] - 16
            self._draw_text(
                instructions,
                entry["score_text"],
                (score_right, top_edge - cursor),
                font_size=self._font_size,
                color=entry["score_color"],
                halign="right",
                valign="top",
            )
            cursor += name_height + 4
        if player_entries:
            cursor += 4

        if yaku_lines:
            for prefix, text in yaku_lines:
                x_offset = panel_x + padding
                prefix_height = 0.0
                if prefix:
                    prefix_width, prefix_height = self._draw_text(
                        instructions,
                        prefix,
                        (x_offset, top_edge - cursor),
                        font_size=self._small_font_size,
                        color=self._accent_color,
                        valign="top",
                    )
                    x_offset += prefix_width
                else:
                    x_offset += label_width
                _, text_height = self._draw_text(
                    instructions,
                    text,
                    (x_offset, top_edge - cursor),
                    font_size=self._small_font_size,
                    color=self._text_color,
                    valign="top",
                )
                line_height = max(prefix_height, text_height)
                cursor += line_height + 2
    # ------------------------------------------------------------------
    # Tile asset helpers
    # ------------------------------------------------------------------
    def _load_tile_assets(self) -> None:
        self._raw_tile_assets.clear()
        self._tile_cache.clear()
        self._tile_texture_cache.clear()
        self._tile_orientation_cache.clear()
        self._tile_orientation_texture_cache.clear()
        self._face_down_image_cache.clear()
        self._face_down_texture_cache.clear()
        self._placeholder_image_cache.clear()

        if not self._asset_root.exists():
            self._assets_loaded = True
            return

        if cairosvg is None:
            self._assets_loaded = True
            return

        for tile_index, symbol in enumerate(_TILE_SYMBOLS):
            path = self._asset_root / f"{symbol}.svg"
            if not path.exists():
                continue
            try:
                png_bytes = cairosvg.svg2png(url=str(path))
                foreground = Image.open(BytesIO(png_bytes)).convert("RGBA")
                foreground.load()
            except Exception:
                continue

            width, height = foreground.size
            base = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(base)
            border_radius = max(2, min(width, height) // 16)
            background_rect = (0, 0, width - 1, height - 1)
            try:
                draw.rounded_rectangle(
                    background_rect,
                    radius=border_radius,
                    fill=(245, 245, 245, 255),
                )
            except AttributeError:
                draw.rectangle(background_rect, fill=(245, 245, 245, 255))
            base.alpha_composite(foreground)
            self._raw_tile_assets[tile_index] = base

        self._assets_loaded = True

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

    def _load_placeholder_font(self, size: int) -> ImageFont.ImageFont:
        if self._font_name:
            font_path = Path(self._font_name)
            if font_path.exists():
                try:
                    return ImageFont.truetype(str(font_path), size)
                except Exception:
                    pass

        candidate_names: list[str] = []
        if self._font_name:
            candidate_names.extend(self._normalize_font_names(self._font_name))
        candidate_names.extend(name for name in self._fallback_fonts if name)

        seen: set[str] = set()
        for name in candidate_names:
            normalized = name.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            try:
                return ImageFont.truetype(normalized, size)
            except Exception:
                continue

        try:
            return ImageFont.load_default()
        except Exception as exc:  # pragma: no cover - defensive fallback
            raise RuntimeError("No usable font available for MahjongEnv placeholders") from exc

    def _create_tile_placeholder_image(
        self, tile_34: int, size: Tuple[int, int]
    ) -> Image.Image:
        key = (tile_34, size)
        if key in self._placeholder_image_cache:
            return self._placeholder_image_cache[key]

        width = max(16, int(size[0]))
        height = max(24, int(size[1]))
        background = self._get_tile_color(tile_34)
        image = Image.new("RGBA", (width, height), (*background, 255))
        draw = ImageDraw.Draw(image)
        border_radius = max(4, min(width, height) // 8)
        outline_color = (245, 245, 245, 255)
        rectangle = (0, 0, width - 1, height - 1)
        try:
            draw.rounded_rectangle(rectangle, radius=border_radius, outline=outline_color, width=2)
        except AttributeError:
            draw.rectangle(rectangle, outline=outline_color, width=2)

        label = _ALT_TILE_SYMBOLS[tile_34] if tile_34 < len(_ALT_TILE_SYMBOLS) else _TILE_SYMBOLS[tile_34]
        font_size = max(12, int(height * 0.45))
        font = self._load_placeholder_font(font_size)
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            text_width, text_height = draw.textsize(label, font=font)
        text_x = (width - text_width) / 2
        text_y = (height - text_height) / 2
        draw.text((text_x, text_y), label, font=font, fill=(20, 20, 20, 255))

        self._placeholder_image_cache[key] = image
        return image

    @staticmethod
    def _image_to_texture(image: Image.Image) -> Texture:
        rgba_image = image.convert("RGBA")
        width, height = rgba_image.size
        texture = Texture.create(size=(width, height), colorfmt="rgba")
        texture.blit_buffer(
            rgba_image.tobytes(),
            colorfmt="rgba",
            bufferfmt="ubyte",
        )
        texture.flip_vertical()
        return texture

    def _get_tile_surface(
        self,
        tile_136: int,
        size: Tuple[int, int],
        face_up: bool = True,
        orientation: int = 0,
    ) -> Texture:
        tile_34 = tile_136 // 4
        if tile_136 == 16:
            tile_34 = 34
        elif tile_136 == 52:
            tile_34 = 35
        elif tile_136 == 88:
            tile_34 = 36

        width = max(1, int(size[0]))
        height = max(1, int(size[1]))
        base_size = (width, height)

        if not face_up:
            return self._get_face_down_surface(base_size, orientation)

        cache_key = (tile_34, base_size)
        if cache_key not in self._tile_cache:
            if tile_34 in self._raw_tile_assets:
                try:
                    resized = self._raw_tile_assets[tile_34].resize(base_size, Image.LANCZOS)
                except Exception:
                    resized = self._raw_tile_assets[tile_34].resize(base_size)
            else:
                resized = self._create_tile_placeholder_image(tile_34, base_size)
            self._tile_cache[cache_key] = resized

        if cache_key not in self._tile_texture_cache:
            self._tile_texture_cache[cache_key] = self._image_to_texture(
                self._tile_cache[cache_key]
            )

        orientation = orientation % 360
        if orientation == 0:
            return self._tile_texture_cache[cache_key]

        orient_key = (tile_34, base_size, orientation)
        if orient_key not in self._tile_orientation_cache:
            rotated = self._tile_cache[cache_key].rotate(
                orientation, expand=True, resample=Image.BICUBIC
            )
            self._tile_orientation_cache[orient_key] = rotated
        if orient_key not in self._tile_orientation_texture_cache:
            self._tile_orientation_texture_cache[orient_key] = self._image_to_texture(
                self._tile_orientation_cache[orient_key]
            )
        return self._tile_orientation_texture_cache[orient_key]

    def _get_face_down_surface(
        self, size: Tuple[int, int], orientation: int = 0
    ) -> Texture:
        width = max(1, int(size[0]))
        height = max(1, int(size[1]))
        base_size = (width, height)
        orientation = orientation % 360

        base_key = (base_size, 0)
        if base_key not in self._face_down_image_cache:
            image = Image.new("RGBA", base_size, (*self._face_down_color, 255))
            draw = ImageDraw.Draw(image)
            border_radius = max(4, min(width, height) // 8)
            outline_color = (*self._face_down_border, 255)
            rectangle = (0, 0, width - 1, height - 1)
            try:
                draw.rounded_rectangle(
                    rectangle, radius=border_radius, outline=outline_color, width=2
                )
            except AttributeError:
                draw.rectangle(rectangle, outline=outline_color, width=2)
            self._face_down_image_cache[base_key] = image

        if base_key not in self._face_down_texture_cache:
            self._face_down_texture_cache[base_key] = self._image_to_texture(
                self._face_down_image_cache[base_key]
            )

        if orientation == 0:
            return self._face_down_texture_cache[base_key]

        orient_key = (base_size, orientation)
        if orient_key not in self._face_down_image_cache:
            rotated = self._face_down_image_cache[base_key].rotate(
                orientation, expand=True, resample=Image.BICUBIC
            )
            self._face_down_image_cache[orient_key] = rotated
        if orient_key not in self._face_down_texture_cache:
            self._face_down_texture_cache[orient_key] = self._image_to_texture(
                self._face_down_image_cache[orient_key]
            )
        return self._face_down_texture_cache[orient_key]

    # ------------------------------------------------------------------
    # Font helpers (placeholders for parity with pygame implementation)
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_font_names(font_name: str | Iterable[str]) -> list[str]:
        if isinstance(font_name, str):
            return [name.strip() for name in font_name.replace(";", ",").split(",") if name.strip()]
        return [name for name in font_name if isinstance(name, str) and name]
