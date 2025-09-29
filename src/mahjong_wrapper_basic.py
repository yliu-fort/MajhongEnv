from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple

try:
    import pygame
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise RuntimeError(
        "pygame is required for the MahjongEnv GUI wrapper; install pygame to continue"
    ) from exc

from mahjong_env import MahjongEnv as _BaseMahjongEnv

_TILE_SYMBOLS: Tuple[str, ...] = (
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


@dataclass(slots=True)
class _RenderPayload:
    action: Optional[int]
    reward: float
    done: bool
    info: dict[str, Any]


class MahjongEnv(_BaseMahjongEnv):
    """Mahjong environment with a lightweight pygame GUI overlay."""

    def __init__(
        self,
        *args: Any,
        window_size: Tuple[int, int] = (1024, 720),
        fps: int = 30,
        font_name: Optional[str] = None,
        font_size: int = 20,
        **kwargs: Any,
    ) -> None:
        self._window_size = window_size
        self._fps = max(1, fps)
        self._font_name = font_name
        self._font_size = font_size
        self._background_color = (20, 25, 35)
        self._accent_color = (120, 210, 255)
        self._text_color = (235, 235, 235)
        self._danger_color = (220, 120, 120)
        self._screen: Optional[pygame.Surface] = None
        self._font: Optional[pygame.font.Font] = None
        self._clock: Optional[pygame.time.Clock] = None
        self._line_height = font_size + 6
        self._quit_requested = False
        self._last_payload = _RenderPayload(action=None, reward=0.0, done=False, info={})
        self._ensure_gui()
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Overridden gym.Env interface
    # ------------------------------------------------------------------
    def reset(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        self._process_events()
        observation = super().reset(*args, **kwargs)
        self._last_payload = _RenderPayload(action=None, reward=0.0, done=self.done, info={})
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

        observation, reward, done, info = super().step(action)
        self._last_payload = _RenderPayload(action=action, reward=reward, done=done, info=info)
        self._render()
        return observation, reward, done, info

    def close(self) -> None:
        if self._screen is not None:
            pygame.display.quit()
        pygame.quit()
        self._screen = None
        self._font = None
        self._clock = None
        self._quit_requested = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_gui(self) -> None:
        if not pygame.get_init():
            pygame.init()
        if pygame.font is not None and not pygame.font.get_init():
            pygame.font.init()
        flags = pygame.RESIZABLE
        self._screen = pygame.display.set_mode(self._window_size, flags)
        pygame.display.set_caption("MahjongEnv GUI")
        self._font = pygame.font.SysFont(self._font_name, self._font_size)
        self._clock = pygame.time.Clock()
        self._line_height = self._font.get_linesize() + 4

    def _process_events(self) -> None:
        if self._screen is None:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._quit_requested = True
            elif event.type == pygame.VIDEORESIZE:
                self._window_size = (event.w, event.h)
                flags = pygame.RESIZABLE
                self._screen = pygame.display.set_mode(self._window_size, flags)

    def _hand_to_text(self, tiles: Iterable[int]) -> str:
        sorted_tiles = sorted(tiles)
        if not sorted_tiles:
            return "(empty)"
        return " ".join(_TILE_SYMBOLS[t // 4] for t in sorted_tiles)

    def _discards_to_text(self, tiles: Iterable[int]) -> str:
        discards = [idx for idx, flagged in enumerate(tiles) if flagged]
        if not discards:
            return "(none)"
        return " ".join(_TILE_SYMBOLS[t // 4] for t in discards)

    def _draw_line(self, text: str, x: int, y: int, color: Tuple[int, int, int]) -> int:
        if self._font is None or self._screen is None:
            return y
        surface = self._font.render(text, True, color)
        self._screen.blit(surface, (x, y))
        return y + self._line_height

    def _render(self) -> None:
        if self._screen is None or self._font is None or self._clock is None:
            return

        self._screen.fill(self._background_color)
        x_margin, y = 16, 16
        title = f"MahjongEnv — Phase: {self.phase} — Current Player: P{self.current_player}"
        y = self._draw_line(title, x_margin, y, self._accent_color)

        for idx, hand in enumerate(self.hands):
            prefix = "➤" if idx == self.current_player and not self.done else " "
            text = f"{prefix} P{idx} Hand: {self._hand_to_text(hand)}"
            y = self._draw_line(text, x_margin, y, self._text_color)
            if idx < len(self.discard_pile):
                discards = self._discards_to_text(self.discard_pile[idx])
                y = self._draw_line(f"   Discards: {discards}", x_margin + 12, y, self._text_color)

        if self.melds:
            y += 6
            for idx, melds in enumerate(self.melds):
                if not melds:
                    continue
                meld_text = ", ".join(meld["type"] for meld in melds)
                y = self._draw_line(f"P{idx} Melds: {meld_text}", x_margin, y, self._text_color)

        y += 8
        reward_color = self._danger_color if self._last_payload.reward < 0 else self._text_color
        y = self._draw_line(
            f"Last Action: {self._last_payload.action} — Reward: {self._last_payload.reward:.2f}",
            x_margin,
            y,
            reward_color,
        )
        msg = self._last_payload.info.get("msg") or getattr(self, "msg", "")
        if msg:
            y = self._draw_line(f"Message: {msg}", x_margin, y, self._text_color)

        if self._last_payload.done:
            y = self._draw_line("Episode finished", x_margin, y + 4, self._danger_color)

        pygame.display.flip()
        self._clock.tick(self._fps)

    # ------------------------------------------------------------------
    # Support context manager style usage
    # ------------------------------------------------------------------
    def __enter__(self) -> "MahjongEnv":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
