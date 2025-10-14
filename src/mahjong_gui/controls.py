"""UI control helpers for the Mahjong Kivy wrapper."""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence


def bind_controls(
    root: Any,
    *,
    on_toggle_auto: Callable[[], None],
    on_step_once: Callable[[], None],
    on_toggle_pause: Callable[[], None],
    on_language_change: Optional[Callable[[str], None]] = None,
) -> Optional[Any]:
    """Wire the standard buttons and language spinner callbacks."""
    if root is None:
        return None

    auto_button = getattr(root, "auto_button", None)
    if auto_button is not None:
        auto_button.bind(on_release=lambda *_: on_toggle_auto())

    step_button = getattr(root, "step_button", None)
    if step_button is not None:
        step_button.bind(on_release=lambda *_: on_step_once())

    pause_button = getattr(root, "pause_button", None)
    if pause_button is not None:
        pause_button.bind(on_release=lambda *_: on_toggle_pause())

    spinner = getattr(root, "language_spinner", None)
    if spinner is not None and on_language_change is not None:
        spinner.bind(text=lambda _instance, value: on_language_change(value))
    return spinner


def apply_font(root: Any, font_name: str) -> None:
    """Apply ``font_name`` to the standard control widgets if present."""
    if root is None:
        return
    widgets = [
        getattr(root, "status_label", None),
        getattr(root, "reward_label", None),
        getattr(root, "done_label", None),
        getattr(root, "pause_button", None),
        getattr(root, "auto_button", None),
        getattr(root, "step_button", None),
        getattr(root, "language_spinner", None),
    ]
    for widget in widgets:
        if widget is None:
            continue
        try:
            widget.font_name = font_name
        except Exception:
            continue


def set_spinner_values(spinner: Any, values: Sequence[str]) -> None:
    if spinner is None:
        return
    spinner.values = tuple(values)


def set_spinner_text(spinner: Any, text: str) -> None:
    if spinner is None:
        return
    spinner.text = text


__all__ = [
    "apply_font",
    "bind_controls",
    "set_spinner_text",
    "set_spinner_values",
]
