"""Human-controlled Mahjong agent for interactive Kivy sessions.

This agent exposes the same ``predict`` interface as the other
environment agents but defers action selection to a human player.  The
agent is designed to work together with :class:`MahjongEnvKivyWrapper`
so that legal actions are presented inside the GUI without blocking the
render loop.  When a wrapper instance is not supplied the agent falls
back to a simple terminal prompt, which is still safe thanks to the
dedicated worker thread used by the application.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Iterable, Optional, Sequence

import numpy as np

from mahjong_tiles_print_style import get_action_printouts


@dataclass(frozen=True)
class _ActionOption:
    """Container describing a legal action option."""

    index: int
    label: str


class HumanPlayerAgent:
    """Agent that requests input from a human player.

    Parameters
    ----------
    env:
        Mahjong environment the agent interacts with.  The reference is
        only used to maintain feature parity with other agents (e.g. a
        fallback to ``env.action_masks`` when necessary).
    wrapper:
        Optional :class:`MahjongEnvKivyWrapper` used to present the
        interactive action selector.  When omitted the agent prompts via
        the terminal, which is primarily useful for testing.
    player_index:
        Seat index that should be shown in the on-screen prompts.  The
        manager updates this value for each request so that the UI stays
        in sync with the environment state.
    """

    def __init__(
        self,
        env,
        wrapper=None,
        player_index: int = 0,
    ) -> None:
        self.env = env
        self.wrapper = wrapper
        self._player_index = player_index
        self._legal_actions: list[int] = []
        self._latest_observation = None
        self._action_event = threading.Event()
        self._selected_action: Optional[int] = None
        self._action_labels = get_action_printouts()

    # ------------------------------------------------------------------
    # Manager hooks
    # ------------------------------------------------------------------
    def set_player_index(self, index: int) -> None:
        self._player_index = index

    def set_observation(self, observation) -> None:
        self._latest_observation = observation

    def set_legal_actions(self, mask: Optional[Sequence[int]]) -> None:
        """Record the legal actions for the next decision step."""

        if mask is None:
            self._legal_actions = []
            return
        if isinstance(mask, np.ndarray):
            indices = np.nonzero(mask)[0].tolist()
        else:
            indices = [index for index, allowed in enumerate(mask) if allowed]
        self._legal_actions = indices

    def cancel_pending(self) -> None:
        """Abort any outstanding prompt (e.g. after an environment reset)."""

        if self.wrapper is not None:
            self.wrapper.dismiss_human_action()
        if not self._action_event.is_set():
            self._selected_action = None
            self._action_event.set()

    # ------------------------------------------------------------------
    # Agent API
    # ------------------------------------------------------------------
    def predict(self, observation) -> int:
        """Return the human-selected action for ``observation``."""

        # Make sure we always have a fresh snapshot of the observation so
        # the GUI can display context if needed.
        self._latest_observation = observation

        if not self._legal_actions:
            # Fall back to cancel / noop when no options are available.
            return 252

        options = [
            _ActionOption(index=idx, label=self._format_action_label(idx))
            for idx in self._legal_actions
        ]

        self._action_event.clear()
        self._selected_action = None

        if self.wrapper is not None:
            self.wrapper.prompt_human_action(
                title=f"Player {self._player_index + 1}: choose action",
                options=[(opt.index, opt.label) for opt in options],
                on_select=self._on_action_selected,
            )
            # Wait for the UI callback to signal completion.
            self._action_event.wait()
        else:
            self._selected_action = self._prompt_from_terminal(options)

        if self._selected_action is None:
            return 252
        return int(self._selected_action)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _format_action_label(self, action_index: int) -> str:
        if 0 <= action_index < len(self._action_labels):
            return self._action_labels[action_index]
        return f"Action {action_index}"

    def _prompt_from_terminal(self, options: Iterable[_ActionOption]) -> int:
        print(f"\nPlayer {self._player_index + 1}: choose action")
        enumerated = list(enumerate(options))
        for idx, option in enumerated:
            print(f"  [{idx}] {option.label} (#{option.index})")
        while True:
            try:
                raw = input("Enter selection index: ").strip()
            except (EOFError, KeyboardInterrupt):  # pragma: no cover - manual input
                return 252
            if not raw:
                continue
            if raw.isdigit():
                choice = int(raw)
                if 0 <= choice < len(enumerated):
                    return enumerated[choice][1].index
            print("Invalid selection, please try again.")

    def _on_action_selected(self, action_index: int) -> None:
        self._selected_action = action_index
        self._action_event.set()

