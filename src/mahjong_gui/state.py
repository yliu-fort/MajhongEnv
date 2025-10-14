"""State containers for the Mahjong Kivy wrapper."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlaybackState:
    """Manage playback-related flags for the GUI wrapper."""

    auto_advance: bool = True
    pause_on_score: bool = False
    score_pause_active: bool = False
    score_pause_pending: bool = False
    step_once_requested: bool = False
    score_panel_was_visible: bool = False

    def toggle_auto(self, score_visible: bool) -> None:
        self.auto_advance = not self.auto_advance
        if self.auto_advance:
            self.step_once_requested = False
            if score_visible and self.pause_on_score:
                self.score_pause_active = True
                self.score_pause_pending = True
            else:
                self.score_pause_active = False
                self.score_pause_pending = False
        else:
            self.step_once_requested = False
            self.score_pause_active = False
            self.score_pause_pending = False

    def trigger_step_once(self) -> None:
        if not self.auto_advance:
            self.step_once_requested = True
        elif self.score_pause_active:
            self.score_pause_active = False
            self.score_pause_pending = False

    def toggle_pause(self, score_visible: bool) -> None:
        if not self.auto_advance:
            return
        self.pause_on_score = not self.pause_on_score
        if self.pause_on_score and score_visible:
            self.score_pause_active = True
            self.score_pause_pending = True
        else:
            self.score_pause_active = False
            self.score_pause_pending = False

    def update_for_score_panel(self, score_visible: bool) -> None:
        if score_visible:
            if self.auto_advance and self.pause_on_score:
                if not self.score_panel_was_visible:
                    self.score_pause_pending = True
                    self.score_pause_active = True
            else:
                self.score_pause_pending = False
                self.score_pause_active = False
            self.score_panel_was_visible = True
            return

        if self.score_pause_pending:
            if self.auto_advance and self.pause_on_score:
                self.score_pause_active = True
            else:
                self.score_pause_active = False
                self.score_pause_pending = False
            self.score_panel_was_visible = False
        else:
            self.score_pause_pending = False
            self.score_pause_active = False
            self.score_panel_was_visible = False


__all__ = ["PlaybackState"]
