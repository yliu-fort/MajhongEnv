"""Kivy-based GUI wrapper for the Mahjong environment.

This module mirrors the pygame based GUI helper from :mod:`mahjong_wrapper`
but renders the board using Kivy widgets.  The wrapper keeps the same public
interface so it can be swapped into existing evaluation scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import threading

try:  # pragma: no cover - the import itself is a runtime dependency guard
    from kivy.base import EventLoop
    from kivy.clock import Clock
    from kivy.core.window import Window
    from kivy.lang import Builder
    from kivy.properties import BooleanProperty, ObjectProperty, StringProperty
    from kivy.uix.boxlayout import BoxLayout
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise RuntimeError(
        "kivy is required for the MahjongEnv Kivy GUI wrapper; install kivy to continue"
    ) from exc

from mahjong_env import MahjongEnvBase as _BaseMahjongEnv

_TILE_SYMBOLS: tuple[str, ...] = (
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


class PlayerPanel(BoxLayout):
    """Simple container used by the KV layout to render a player column."""

    header_text = StringProperty("")
    status_text = StringProperty("")
    hand_text = StringProperty("")
    meld_text = StringProperty("")
    discard_text = StringProperty("")


class MahjongGUIRoot(BoxLayout):
    """Root widget used by :class:`MahjongEnvKivyWrapper`."""

    controller = ObjectProperty(None, allownone=True)
    status_text = StringProperty("Initializing GUI…")
    info_text = StringProperty("")
    auto_button_text = StringProperty("Auto: ON")
    pause_button_text = StringProperty("Pause on Score: OFF")
    step_button_disabled = BooleanProperty(True)

    def on_controller(self, *_: Any) -> None:
        if self.controller is not None:
            self.controller.set_root(self)

    # ------------------------------------------------------------------
    # Event handlers used by the KV layout
    # ------------------------------------------------------------------
    def on_toggle_auto(self) -> None:
        if self.controller is not None:
            self.controller.toggle_auto()

    def on_step_once(self) -> None:
        if self.controller is not None:
            self.controller.request_step_once()

    def on_toggle_pause(self) -> None:
        if self.controller is not None:
            self.controller.toggle_pause_on_score()

    def on_quit(self) -> None:
        if self.controller is not None:
            self.controller.request_quit()

    # ------------------------------------------------------------------
    # UI updates triggered by the controller
    # ------------------------------------------------------------------
    def update_players(self, players: Sequence[dict[str, str]]) -> None:
        panels: list[PlayerPanel] = []
        idx = 0
        while True:
            panel = self.ids.get(f"player{idx}")
            if not isinstance(panel, PlayerPanel):
                if panel is None:
                    break
                idx += 1
                continue
            panels.append(panel)
            idx += 1
        for idx, panel in enumerate(panels):
            if not isinstance(panel, PlayerPanel):
                continue
            if idx < len(players):
                data = players[idx]
                panel.header_text = data.get("header", "")
                panel.status_text = data.get("status", "")
                panel.hand_text = data.get("hand", "")
                panel.meld_text = data.get("melds", "")
                panel.discard_text = data.get("discards", "")
                panel.opacity = 1.0
            else:
                panel.header_text = f"Seat {idx + 1}"
                panel.status_text = ""
                panel.hand_text = ""
                panel.meld_text = ""
                panel.discard_text = ""
                panel.opacity = 0.25

    def update_controls(self, state: dict[str, Any]) -> None:
        self.auto_button_text = "Auto: ON" if state.get("auto", True) else "Auto: OFF"
        pause_flag = state.get("pause_on_score", False)
        self.pause_button_text = "Pause on Score: ON" if pause_flag else "Pause on Score: OFF"
        self.step_button_disabled = not state.get("step_enabled", False)
        step_button = self.ids.get("step_button")
        if step_button is not None:
            step_button.disabled = self.step_button_disabled

    def update_state(self, state: dict[str, Any]) -> None:
        self.status_text = state.get("status", "")
        info_lines = state.get("info_lines", [])
        self.info_text = "\n".join(info_lines)
        self.update_players(state.get("players", []))
        self.update_controls(state)


class MahjongEnvKivyWrapper:
    """Mahjong environment with a Kivy driven GUI overlay."""

    def __init__(
        self,
        *args: Any,
        env: _BaseMahjongEnv | None = None,
        fps: int = 30,
        font_name: Optional[str] = None,  # compatibility argument, unused in Kivy
        fallback_fonts: Optional[Sequence[str]] = None,  # compatibility placeholder
        **kwargs: Any,
    ) -> None:
        del font_name, fallback_fonts  # parameters kept for API compatibility
        if env is None:
            raise ValueError("An environment instance must be provided to MahjongEnvKivyWrapper")
        self._env = env
        self._fps = max(1, fps)
        self._quit_requested = False
        self._auto_advance = True
        self._pause_on_score = False
        self._score_pause_active = False
        self._score_pause_pending = False
        self._last_phase_is_score_last = ""
        self._step_once_requested = False
        self._last_payload = _RenderPayload(action=None, reward=0.0, done=False, info={})
        self._lock = threading.RLock()
        self._control_event = threading.Event()
        self._window_bound = False
        self._root: Optional[MahjongGUIRoot] = None
        self._ensure_app()

    # ------------------------------------------------------------------
    # Public gym.Env compatible API
    # ------------------------------------------------------------------
    def reset(self, *args: Any, **kwargs: Any) -> Any:
        self._ensure_app()
        observation = self._env.reset(*args, **kwargs)
        self._last_payload = _RenderPayload(action=None, reward=0.0, done=self._env.done, info={})
        self._step_once_requested = False
        self._score_pause_active = False
        self._score_pause_pending = False
        self._last_phase_is_score_last = getattr(self._env, "phase", "")
        self._render()
        self._pump_ui()
        return observation

    def step(self, action: int) -> tuple[Any, float, bool, dict[str, Any]]:
        self._pump_ui()
        self._process_wait_loop()
        if self._quit_requested:
            observation = self._env.get_observation(self._env.current_player)
            self._env.done = True
            info = {"terminated_by_gui": True}
            self._last_payload = _RenderPayload(action=None, reward=0.0, done=True, info=info)
            self._render()
            return observation, 0.0, True, info

        if self._step_once_requested:
            self._step_once_requested = False

        observation, reward, done, info = self._env.step(action)
        self._last_payload = _RenderPayload(action=action, reward=reward, done=done, info=info)
        self._render()
        return observation, reward, done, info

    def close(self) -> None:
        self._quit_requested = True
        self._control_event.set()
        self._pump_ui()
        if self._root is not None and self._root.parent is not None:
            Window.remove_widget(self._root)
        self._root = None
        self._pump_ui()
        self._env.close()

    def action_masks(self) -> Any:
        return self._env.action_masks()

    @property
    def phase(self) -> str:
        return getattr(self._env, "phase", "")

    # ------------------------------------------------------------------
    # UI callbacks triggered from the Kivy thread
    # ------------------------------------------------------------------
    def toggle_auto(self) -> None:
        with self._lock:
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
                self._score_pause_active = False
                self._score_pause_pending = False
        self._control_event.set()
        self._render()

    def request_step_once(self) -> None:
        with self._lock:
            if not self._auto_advance:
                self._step_once_requested = True
            elif self._score_pause_active:
                self._score_pause_active = False
                self._score_pause_pending = False
        self._control_event.set()
        self._render()

    def toggle_pause_on_score(self) -> None:
        with self._lock:
            self._pause_on_score = not self._pause_on_score
            if self._pause_on_score and self._score_last():
                self._score_pause_active = True
                self._score_pause_pending = True
            else:
                self._score_pause_active = False
                self._score_pause_pending = False
        self._control_event.set()
        self._render()

    def request_quit(self) -> None:
        self._quit_requested = True
        self._control_event.set()
        self._pump_ui()

    def set_root(self, root: MahjongGUIRoot) -> None:
        self._root = root
        self._render()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_app(self) -> None:
        if self._root is not None:
            return

        if not getattr(self.__class__, "_kv_loaded", False):
            kv_path = Path(__file__).resolve().parent / "mahjong_gui.kv"
            Builder.load_file(str(kv_path))
            setattr(self.__class__, "_kv_loaded", True)

        EventLoop.ensure_window()
        root = MahjongGUIRoot()
        root.controller = self
        if not self._window_bound:
            Window.bind(on_request_close=self._on_window_close)
            self._window_bound = True
        if root.parent is None:
            Window.add_widget(root)
        self._root = root
        self._pump_ui()

    def _process_wait_loop(self) -> None:
        while self._should_wait():
            self._render()
            wait_time = 1.0 / float(self._fps)
            self._pump_ui()
            if self._control_event.wait(timeout=wait_time):
                self._control_event.clear()

    def _should_wait(self) -> bool:
        if self._quit_requested or self._env.done:
            return False
        wait_for_manual = (not self._auto_advance) and not self._step_once_requested
        return wait_for_manual or self._score_pause_active

    def _render(self) -> None:
        if self._root is None:
            return

        state = self._build_state_payload()

        def _apply_update(_dt: float) -> None:
            if self._root is not None:
                self._root.update_state(state)

        Clock.schedule_once(_apply_update, 0)
        self._pump_ui()

    def _pump_ui(self) -> None:
        try:
            Clock.tick()
            EventLoop.idle()
        except Exception:
            pass

    def _on_window_close(self, *_: Any) -> bool:
        self._quit_requested = True
        self._control_event.set()
        return False

    # ------------------------------------------------------------------
    # State extraction helpers
    # ------------------------------------------------------------------
    def _build_state_payload(self) -> dict[str, Any]:
        with self._lock:
            self._update_score_pause_flags()

            env = self._env
            num_players = getattr(env, "num_players", len(getattr(env, "hands", [])))
            players: list[dict[str, str]] = []
            hands = list(getattr(env, "hands", []))
            melds = list(getattr(env, "melds", []))
            discards = list(getattr(env, "discard_pile_seq", []))
            scores = list(getattr(env, "scores", []))
            riichi_flags = list(getattr(env, "riichi", []))
            tenpai_flags = list(getattr(env, "tenpai", []))
            dealer = getattr(env, "dealer", 0)
            reveal_flags = self._compute_hand_reveal_flags(num_players)

            for idx in range(num_players):
                header_bits = [f"Player {idx + 1}"]
                if idx == getattr(env, "current_player", -1):
                    header_bits.append("(Active)")
                if idx == dealer:
                    header_bits.append("Dealer")
                seat_winds = list(getattr(env, "seat_wind", []))
                if idx < len(seat_winds):
                    header_bits.append(f"Seat: {self._format_wind(seat_winds[idx])}")

                status_bits: list[str] = []
                if idx < len(riichi_flags) and riichi_flags[idx]:
                    status_bits.append("Riichi")
                if idx < len(tenpai_flags) and tenpai_flags[idx]:
                    status_bits.append("Tenpai")
                if idx < len(scores):
                    status_bits.append(f"Score: {int(round(scores[idx] * 100))}")

                hand_tiles = hands[idx] if idx < len(hands) else []
                hand_repr = self._format_tiles(hand_tiles if reveal_flags[idx] else [None] * len(hand_tiles))
                meld_repr = self._format_melds(melds[idx] if idx < len(melds) else [])
                discard_repr = self._format_tiles(discards[idx] if idx < len(discards) else [])

                players.append(
                    {
                        "header": " ".join(header_bits),
                        "status": ", ".join(status_bits) if status_bits else "",
                        "hand": f"Hand: {hand_repr}" if hand_repr else "Hand: (empty)",
                        "melds": f"Melds: {meld_repr}" if meld_repr else "Melds: (none)",
                        "discards": f"Discards: {discard_repr}" if discard_repr else "Discards: (none)",
                    }
                )

            info_lines = self._build_info_lines()
            status_line = self._build_status_line(players)

            control_state = {
                "auto": self._auto_advance,
                "pause_on_score": self._pause_on_score,
                "step_enabled": (not self._auto_advance) or self._score_pause_active,
            }

            return {
                "players": players,
                "info_lines": info_lines,
                "status": status_line,
                **control_state,
            }

    def _build_status_line(self, players: Sequence[dict[str, str]]) -> str:
        env = self._env
        round_wind = self._format_wind(getattr(env, "round_wind", ""))
        phase = getattr(env, "phase", "")
        hand_number = getattr(env, "hand_number", 0)
        remaining_tiles = len(getattr(env, "wall", []))
        active_idx = getattr(env, "current_player", -1)
        active_name = players[active_idx]["header"] if 0 <= active_idx < len(players) else f"Player {active_idx + 1}"
        parts = [f"Phase: {phase}"]
        if round_wind:
            parts.append(f"Round: {round_wind}-{hand_number}")
        parts.append(f"Active: {active_name}")
        parts.append(f"Wall tiles remaining: {remaining_tiles}")
        if self._score_pause_active:
            parts.append("(Score pause active)")
        return " | ".join(parts)

    def _build_info_lines(self) -> list[str]:
        lines: list[str] = []
        payload = self._last_payload
        if payload.action is not None:
            lines.append(f"Last action: {payload.action}")
        lines.append(f"Last reward: {payload.reward:.2f}")
        lines.append(f"Episode done: {payload.done}")
        if payload.info:
            for key, value in payload.info.items():
                lines.append(f"{key}: {value}")
        if self._pause_on_score:
            lines.append("Pause on score is enabled")
        if self._score_pause_active:
            lines.append("Waiting for user confirmation (score phase)")
        if self._quit_requested:
            lines.append("Quit requested")
        return lines

    def _format_tiles(self, tiles: Iterable[Optional[int]]) -> str:
        formatted: list[str] = []
        for tile in tiles:
            if tile is None:
                formatted.append("??")
            else:
                formatted.append(self._tile_to_text(tile))
        return " ".join(formatted)

    def _format_melds(self, melds: Iterable[Any]) -> str:
        formatted: list[str] = []
        for meld in melds:
            tiles = meld.get("m", []) if isinstance(meld, dict) else []
            meld_type = meld.get("type", "") if isinstance(meld, dict) else ""
            opened = meld.get("opened", True) if isinstance(meld, dict) else True
            tile_text = self._format_tiles(tiles)
            descriptor = meld_type or ("Open" if opened else "Closed")
            formatted.append(f"[{descriptor}: {tile_text}]")
        return " ".join(formatted)

    def _tile_to_text(self, tile_136: int) -> str:
        tile_34 = tile_136 // 4
        if tile_136 == 16:
            tile_34 = 34
        elif tile_136 == 52:
            tile_34 = 35
        elif tile_136 == 88:
            tile_34 = 36
        if 0 <= tile_34 < len(_TILE_SYMBOLS):
            return _TILE_SYMBOLS[tile_34]
        return str(tile_136)

    def _format_wind(self, wind: Any) -> str:
        mapping = {0: "East", 1: "South", 2: "West", 3: "North", "E": "East", "S": "South", "W": "West", "N": "North"}
        return mapping.get(wind, str(wind) if wind is not None else "")

    def _compute_hand_reveal_flags(self, num_players: int) -> list[bool]:
        reveal = [False] * max(0, num_players)
        for idx in range(num_players):
            reveal[idx] = idx == getattr(self._env, "current_player", 0)

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

    def _score_last(self) -> bool:
        env = self._env
        return (
            getattr(env, "phase", "") == "score"
            and getattr(env, "current_player", 0) == getattr(env, "num_players", 1) - 1
        )

    def _update_score_pause_flags(self) -> None:
        if self._score_last():
            if self._auto_advance and self._pause_on_score:
                if not self._last_phase_is_score_last:
                    self._score_pause_pending = True
                    self._score_pause_active = True
            else:
                self._score_pause_pending = False
                self._score_pause_active = False
            self._last_phase_is_score_last = "score"
        elif self._score_pause_pending:
            if self._auto_advance and self._pause_on_score:
                self._score_pause_active = True
            else:
                self._score_pause_active = False
                self._score_pause_pending = False
            self._last_phase_is_score_last = ""
        else:
            self._score_pause_pending = False
            self._score_pause_active = False
            self._last_phase_is_score_last = getattr(self._env, "phase", "")

