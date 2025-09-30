from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Tuple

_spec = importlib.util.find_spec("kivy")
if _spec is None:  # pragma: no cover - dependency guard
    raise RuntimeError(
        "Kivy is required for the MahjongEnv Kivy wrapper; install kivy to continue"
    )

from kivy.base import EventLoop
from kivy.clock import Clock
from kivy.core.window import WindowBase
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.widget import Widget

from mahjong_env import MahjongEnv as _BaseMahjongEnv


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
        """Redraw the board contents.

        The actual drawing routines are implemented in later subtasks. This
        method exists to keep the rendering pipeline consistent with the
        pygame implementation.
        """


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
        status_text: str,
    ) -> None:
        self._auto_button.disabled = not auto_enabled
        self._auto_button.text = "Auto Next: ON" if auto_active else "Auto Next: OFF"

        self._step_button.disabled = not step_enabled
        self._step_button.text = "Next"

        self._pause_button.disabled = not pause_enabled
        pause_label = "Pause on Score: ON" if pause_active else "Pause on Score: OFF"
        self._pause_button.text = pause_label

        self._status_label.text = status_text
        self._status_label.texture_update()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
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
        self._raw_tile_assets: dict[int, Any] = {}
        self._tile_cache: dict[tuple[int, Tuple[int, int]], Any] = {}
        self._tile_orientation_cache: dict[tuple[int, Tuple[int, int], int], Any] = {}
        self._face_down_cache: dict[tuple[Tuple[int, int], int], Any] = {}
        self._placeholder_cache: dict[tuple[int, Tuple[int, int]], Any] = {}
        self._tile_metrics: dict[str, Tuple[int, int]] = {}
        self._riichi_states: list[bool] = []
        self._riichi_pending: list[bool] = []
        self._discard_counts: list[int] = []
        self._riichi_declarations: dict[int, int] = {}
        self._frame_interval = 1.0 / float(self._fps) if self._fps > 0 else 0.0
        self._last_frame_time: Optional[float] = None
        self._pending_resize: Optional[Tuple[int, int]] = None

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
        window = EventLoop.window
        if window is None:
            raise RuntimeError("Failed to create Kivy window for MahjongEnv")

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
    # Font helpers (placeholders for parity with pygame implementation)
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_font_names(font_name: str | Iterable[str]) -> list[str]:
        if isinstance(font_name, str):
            return [name.strip() for name in font_name.replace(";", ",").split(",") if name.strip()]
        return [name for name in font_name if isinstance(name, str) and name]
