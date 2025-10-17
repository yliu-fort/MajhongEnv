from __future__ import annotations
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".", "src"))

import threading
import time
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple

from mahjong_tiles_print_style import get_action_printouts


_TILE_NAMES: Tuple[str, ...] = (
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
    "East",
    "South",
    "West",
    "North",
    "Haku",
    "Hatsu",
    "Chun",
)

_PASS_LABELS: dict[int, str] = {
    252: "Cancel",
    253: "Pass Riichi",
    254: "Pass Chi",
    255: "Pass Pon",
    256: "Pass Kan",
    257: "Pass Ankan",
    258: "Pass Chakan",
    259: "Pass Ryuukyoku",
    260: "Pass Ron",
    261: "Pass Tsumo",
}


class HumanPlayerAgent:
    """Coordinator for human-controlled seats.

    The agent exposes a three-step workflow for each decision:

    * :meth:`begin_turn` publishes the available actions and starts a timer.
    * The UI relays the chosen action via :meth:`submit_action`.
    * :meth:`wait_for_action` blocks until an action is selected or the timer expires.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._legal_actions: set[int] = set()
        self._selected_action: Optional[int] = None
        self._deadline: float = 0.0
        self._active = False
        self._cancelled = False
        self._presenter: Optional[
            Tuple[
                Callable[[Sequence[Tuple[int, str]], Optional[float]], None],
                Callable[[], None],
            ]
        ] = None
        self._action_printouts = get_action_printouts()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def bind_presenter(
        self,
        show_callback: Callable[[Sequence[Tuple[int, str]], Optional[float]], None],
        clear_callback: Callable[[], None],
    ) -> None:
        """Attach callbacks used to present and clear UI options."""

        with self._lock:
            self._presenter = (show_callback, clear_callback)
        self._invoke_clear()

    def begin_turn(self, observation: Any, masks: Any, timeout: float) -> None:
        """Publish the available actions for the current decision."""
        if sum(masks) <= 1:
            raise TypeError
        
        _ = observation  # Reserved for future use
        actions = self._extract_actions(masks)
        labels = [(action_id, self._format_action_label(action_id)) for action_id in actions]

        with self._lock:
            if self._active:
                # Cancel any lingering turn before starting a fresh one.
                self._cancelled = True
            self._active = True
            self._cancelled = False
            self._legal_actions = set(actions)
            self._selected_action = None
            self._deadline = time.monotonic() + max(0.0, float(timeout))
            self._event.clear()

        self._invoke_show(labels, self._deadline)

    def submit_action(self, action_id: int) -> bool:
        """Record the human's selection and wake any waiting thread."""

        with self._lock:
            if not self._active or self._cancelled:
                return False
            if action_id not in self._legal_actions:
                return False
            self._selected_action = int(action_id)
            self._event.set()
            return True

    def wait_for_action(self) -> int:
        """Block until an action is chosen or the timeout elapses."""

        action: Optional[int] = None
        try:
            while True:
                with self._lock:
                    if not self._active:
                        raise TimeoutError("No active human turn is in progress")
                    if self._selected_action is not None:
                        action = self._selected_action
                        break
                    if self._cancelled:
                        raise TimeoutError("Human turn was cancelled")
                    deadline = self._deadline
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Human input timed out")
                if not self._event.wait(timeout=remaining):
                    raise TimeoutError("Human input timed out")
                self._event.clear()
        finally:
            self._invoke_clear()
            with self._lock:
                self._active = False
                self._legal_actions.clear()
                self._selected_action = None
                self._cancelled = False
                self._event.clear()
        if action is None:
            raise TimeoutError("Human input timed out")
        return action

    def cancel_turn(self) -> None:
        """Abort any pending selection and clear the UI."""

        with self._lock:
            if not self._active:
                return
            self._cancelled = True
            self._active = False
            self._legal_actions.clear()
            self._selected_action = None
            self._event.set()
        self._invoke_clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _extract_actions(self, masks: Any) -> list[int]:
        if isinstance(masks, dict):
            for key in ("action_mask", "legal_actions", "mask", "legal_mask"):
                if key in masks:
                    masks = masks[key]
                    break
        if isinstance(masks, (Sequence, Iterable)):
            actions = [idx for idx, value in enumerate(masks) if bool(value)]
            return actions
        return []

    def _format_action_label(self, action_id: int) -> str:
        return self._action_printouts[action_id]
        if action_id < 34:
            return f"Discard {_TILE_NAMES[action_id]}"
        if action_id < 68:
            tile_index = action_id - 34
            if 0 <= tile_index < len(_TILE_NAMES):
                return f"Riichi {_TILE_NAMES[tile_index]}"
            return "Riichi"
        if action_id < 113:
            return "Chi"
        if action_id < 147:
            return "Pon"
        if action_id < 181:
            return "Kan"
        if action_id < 215:
            return "Chakan"
        if action_id < 249:
            return "Ankan"
        if action_id in _PASS_LABELS:
            return "Cancel"
        if action_id == 249:
            return "Ryuukyoku"
        if action_id == 250:
            return "Ron"
        if action_id == 251:
            return "Tsumo"
        return f"Action {action_id}"

    def _invoke_show(
        self, actions: Sequence[Tuple[int, str]], deadline: Optional[float]
    ) -> None:
        presenter = None
        with self._lock:
            presenter = self._presenter
        if presenter is None:
            return
        show_callback, _ = presenter
        if show_callback is not None:
            show_callback(tuple(actions), deadline)

    def _invoke_clear(self) -> None:
        presenter = None
        with self._lock:
            presenter = self._presenter
        if presenter is None:
            return
        _, clear_callback = presenter
        if clear_callback is not None:
            clear_callback()
