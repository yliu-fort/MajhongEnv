from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

import io

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
        self._background_color = (8, 30, 60)
        self._play_area_color = (24, 70, 110)
        self._panel_color = (8, 8, 8)
        self._accent_color = (200, 220, 255)
        self._text_color = (235, 235, 235)
        self._danger_color = (220, 120, 120)
        self._screen: Optional[pygame.Surface] = None
        self._font: Optional[pygame.font.Font] = None
        self._small_font: Optional[pygame.font.Font] = None
        self._clock: Optional[pygame.time.Clock] = None
        self._line_height = font_size + 6
        self._quit_requested = False
        self._last_payload = _RenderPayload(action=None, reward=0.0, done=False, info={})
        self._tile_images: dict[int, pygame.Surface] = {}
        self._tile_placeholders: dict[int, pygame.Surface] = {}
        self._tile_cache: dict[tuple[int, Tuple[int, int], bool], pygame.Surface] = {}
        self._tile_back: Optional[pygame.Surface] = None
        self._base_tile_size = (80, 112)
        self._ensure_gui()
        self._load_tile_assets()
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
        small_size = max(12, int(self._font_size * 0.8))
        self._small_font = pygame.font.SysFont(self._font_name, small_size)
        self._clock = pygame.time.Clock()
        self._line_height = self._font.get_linesize() + 4

    def _load_tile_assets(self) -> None:
        if self._font is None:
            return
        asset_root = Path(__file__).resolve().parent.parent / "data" / "assets"
        tile_font_size = max(18, int(self._base_tile_size[1] * 0.35))
        tile_font = pygame.font.SysFont(self._font_name, tile_font_size)
        back_surface = pygame.Surface(self._base_tile_size, pygame.SRCALPHA)
        back_surface.fill((15, 15, 15))
        pygame.draw.rect(back_surface, (40, 40, 40), back_surface.get_rect(), 4, border_radius=6)
        self._tile_back = back_surface

        for tile_index, symbol in enumerate(_TILE_SYMBOLS):
            surface: Optional[pygame.Surface] = None
            svg_path = asset_root / f"{symbol}.svg"
            if svg_path.exists():
                try:
                    surface = self._load_svg_surface(svg_path)
                except Exception:
                    surface = None
            if surface is not None:
                self._tile_images[tile_index] = surface.convert_alpha()
                continue

            placeholder = pygame.Surface(self._base_tile_size, pygame.SRCALPHA)
            placeholder.fill((240, 240, 240))
            pygame.draw.rect(
                placeholder,
                (30, 30, 30),
                placeholder.get_rect(),
                3,
                border_radius=6,
            )
            text_surface = tile_font.render(symbol.upper(), True, (25, 25, 25))
            text_rect = text_surface.get_rect(center=placeholder.get_rect().center)
            placeholder.blit(text_surface, text_rect)
            self._tile_placeholders[tile_index] = placeholder

    def _load_svg_surface(self, path: Path) -> Optional[pygame.Surface]:
        try:
            return pygame.image.load(str(path))
        except pygame.error:
            try:
                import cairosvg  # type: ignore
            except Exception:
                return None
            png_bytes = cairosvg.svg2png(url=str(path))
            return pygame.image.load(io.BytesIO(png_bytes))

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

    def _render(self) -> None:
        if self._screen is None or self._font is None or self._clock is None:
            return

        self._screen.fill(self._background_color)
        width, height = self._screen.get_size()
        play_size = int(min(width, height) * 0.82)
        play_rect = pygame.Rect(
            (width - play_size) // 2,
            (height - play_size) // 2,
            play_size,
            play_size,
        )

        pygame.draw.rect(self._screen, self._play_area_color, play_rect, border_radius=32)
        self._draw_walls(play_rect)
        self._draw_dead_wall(play_rect)

        self._draw_center_panel(play_rect)

        self._draw_player_area(0, play_rect)
        self._draw_player_area(1, play_rect)
        self._draw_player_area(2, play_rect)
        self._draw_player_area(3, play_rect)

        pygame.display.flip()
        self._clock.tick(self._fps)

    # Rendering helpers -------------------------------------------------
    def _draw_center_panel(self, play_rect: pygame.Rect) -> None:
        assert self._screen is not None and self._font is not None and self._small_font is not None
        panel_width = int(play_rect.width * 0.28)
        panel_height = int(play_rect.height * 0.22)
        panel_rect = pygame.Rect(0, 0, panel_width, panel_height)
        panel_rect.center = play_rect.center
        pygame.draw.rect(self._screen, self._panel_color, panel_rect, border_radius=16)
        pygame.draw.rect(self._screen, (50, 50, 50), panel_rect, 2, border_radius=16)

        round_winds = ["East", "South", "West", "North"]
        round_index = max(0, self.round[0])
        round_wind = round_winds[(round_index // 4) % 4]
        round_number = round_index % 4 + 1
        round_text = f"{round_wind} {round_number}"
        text_surface = self._font.render(round_text, True, self._accent_color)
        text_rect = text_surface.get_rect(center=(panel_rect.centerx, panel_rect.top + text_surface.get_height() + 10))
        self._screen.blit(text_surface, text_rect)

        counter_text = f"Honba: {self.round[1]}   Tiles left: {len(getattr(self, 'deck', []))}"
        counter_surface = self._small_font.render(counter_text, True, self._text_color)
        counter_rect = counter_surface.get_rect(center=(panel_rect.centerx, panel_rect.centery))
        self._screen.blit(counter_surface, counter_rect)

        timer_text = f"Phase: {self.phase}"
        timer_surface = self._small_font.render(timer_text, True, self._text_color)
        timer_rect = timer_surface.get_rect(center=(panel_rect.centerx, panel_rect.bottom - timer_surface.get_height() - 12))
        self._screen.blit(timer_surface, timer_rect)

        score_positions = {
            0: (panel_rect.centerx, panel_rect.bottom + 24),
            1: (panel_rect.right + 32, panel_rect.centery),
            2: (panel_rect.centerx, panel_rect.top - 24),
            3: (panel_rect.left - 32, panel_rect.centery),
        }
        for player, pos in score_positions.items():
            if player >= len(self.scores):
                continue
            score_text = f"{self.scores[player]:4d}"
            surface = self._font.render(score_text, True, self._text_color)
            rect = surface.get_rect()
            rect.center = pos
            self._screen.blit(surface, rect)

    def _draw_walls(self, play_rect: pygame.Rect) -> None:
        if self._screen is None:
            return
        tile_count_per_side = 17
        tile_length = max(1, int(play_rect.width / tile_count_per_side))
        tile_height = max(8, int(tile_length * 0.55))
        horizontal = self._get_tile_graphic(-1, (tile_length, tile_height), face_up=False)
        vertical = pygame.transform.rotate(horizontal, 90)

        for i in range(tile_count_per_side):
            offset = i * tile_length
            top_rect = horizontal.get_rect()
            top_rect.midbottom = (play_rect.left + offset + tile_length // 2, play_rect.top - 6)
            bottom_rect = horizontal.get_rect()
            bottom_rect.midtop = (play_rect.left + offset + tile_length // 2, play_rect.bottom + 6)
            self._screen.blit(horizontal, top_rect)
            self._screen.blit(horizontal, bottom_rect)

            left_rect = vertical.get_rect()
            left_rect.midright = (play_rect.left - 6, play_rect.top + offset + tile_length // 2)
            right_rect = vertical.get_rect()
            right_rect.midleft = (play_rect.right + 6, play_rect.top + offset + tile_length // 2)
            self._screen.blit(vertical, left_rect)
            self._screen.blit(vertical, right_rect)

    def _draw_dead_wall(self, play_rect: pygame.Rect) -> None:
        if self._screen is None:
            return
        dead_wall_tiles = 7
        tile_height = max(10, int(play_rect.height * 0.045))
        tile_width = max(12, int(play_rect.width * 0.032))
        gap = 4
        start_x = play_rect.right + tile_width
        start_y = play_rect.top
        dead_tile = self._get_tile_graphic(-1, (tile_width, tile_height), face_up=False)
        for i in range(dead_wall_tiles):
            rect = dead_tile.get_rect()
            rect.topleft = (start_x, start_y + i * (tile_height + gap))
            self._screen.blit(dead_tile, rect)

        if self.dora_indicator:
            tile_id = self.dora_indicator[-1]
            tile_surface = self._get_tile_graphic(tile_id // 4, (tile_width * 2, int(tile_height * 1.5)))
            rect = tile_surface.get_rect()
            rect.midleft = (start_x + tile_width + 12, start_y + dead_wall_tiles * (tile_height + gap) - tile_height)
            self._screen.blit(tile_surface, rect)

    def _draw_player_area(self, player: int, play_rect: pygame.Rect) -> None:
        assert self._screen is not None
        orientation = player % 4
        tile_height = int(play_rect.height * 0.12)
        tile_width = int(tile_height * 0.7)
        spacing = int(tile_width * 0.1)
        hand_tiles = self.hands[player] if player < len(self.hands) else []
        melds = self.melds[player] if player < len(self.melds) else []
        discards = self._get_discards(player)

        player_names = getattr(self, "player_names", None)
        if isinstance(player_names, (list, tuple)) and player < len(player_names):
            name = str(player_names[player]) or "NoName"
        else:
            name = "NoName"
        self._draw_seat_label(player, play_rect, name)

        if orientation == 0:
            self._draw_bottom_area(hand_tiles, discards, melds, play_rect, tile_width, tile_height, spacing)
        elif orientation == 1:
            self._draw_right_area(hand_tiles, discards, melds, play_rect, tile_width, tile_height, spacing)
        elif orientation == 2:
            self._draw_top_area(hand_tiles, discards, melds, play_rect, tile_width, tile_height, spacing)
        else:
            self._draw_left_area(hand_tiles, discards, melds, play_rect, tile_width, tile_height, spacing)

    def _get_discards(self, player: int) -> list[int]:
        if player >= len(self.discard_pile):
            return []
        flags = self.discard_pile[player]
        result = [idx for idx, flagged in enumerate(flags) if flagged]
        result.sort(key=lambda tile: getattr(self, "discard_order", {}).get((player, tile), tile))
        return result

    def _draw_seat_label(self, player: int, play_rect: pygame.Rect, name: str) -> None:
        if self._small_font is None or self._screen is None:
            return
        positions = {
            0: (play_rect.centerx, play_rect.bottom + 40),
            1: (play_rect.right + 50, play_rect.centery),
            2: (play_rect.centerx, play_rect.top - 40),
            3: (play_rect.left - 50, play_rect.centery),
        }
        marker_positions = {
            0: (play_rect.centerx, play_rect.bottom + 20),
            1: (play_rect.right + 20, play_rect.centery),
            2: (play_rect.centerx, play_rect.top - 20),
            3: (play_rect.left - 20, play_rect.centery),
        }
        label_surface = self._small_font.render(name, True, (200, 200, 220))
        label_rect = label_surface.get_rect(center=positions[player])
        self._screen.blit(label_surface, label_rect)

        if player == getattr(self, "oya", -1):
            marker_surface = self._small_font.render("East", True, self._accent_color)
            marker_rect = marker_surface.get_rect(center=marker_positions[player])
            self._screen.blit(marker_surface, marker_rect)

    def _draw_bottom_area(
        self,
        hand: Iterable[int],
        discards: Iterable[int],
        melds: Iterable[dict[str, Any]],
        play_rect: pygame.Rect,
        tile_width: int,
        tile_height: int,
        spacing: int,
    ) -> None:
        assert self._screen is not None
        sorted_hand = sorted(hand, key=lambda t: (t // 4, t % 4))
        start_x = play_rect.left + 40
        y = play_rect.bottom - tile_height - 16
        for idx, tile in enumerate(sorted_hand):
            surface = self._get_tile_graphic(tile // 4, (tile_width, tile_height))
            pos_x = start_x + idx * (tile_width + spacing)
            if idx == len(sorted_hand) - 1:
                pos_x += spacing * 2
            self._screen.blit(surface, (pos_x, y))

        self._draw_melds_horizontal(melds, (play_rect.right - 40, y), tile_width, tile_height, spacing, reverse=True)
        discard_origin = (play_rect.left + 60, y - tile_height - 16)
        self._draw_discard_grid(discards, discard_origin, tile_width, tile_height, spacing, rotation=0)

    def _draw_top_area(
        self,
        hand: Iterable[int],
        discards: Iterable[int],
        melds: Iterable[dict[str, Any]],
        play_rect: pygame.Rect,
        tile_width: int,
        tile_height: int,
        spacing: int,
    ) -> None:
        assert self._screen is not None
        count = len(hand)
        x = play_rect.left + 40
        y = play_rect.top + 16
        back_surface = self._get_tile_graphic(-1, (tile_width, tile_height), face_up=False)
        for idx in range(count):
            self._screen.blit(back_surface, (x + idx * (tile_width + spacing), y))

        discard_origin = (play_rect.left + 60, y + tile_height + 16)
        self._draw_discard_grid(discards, discard_origin, tile_width, tile_height, spacing, rotation=180)

    def _draw_right_area(
        self,
        hand: Iterable[int],
        discards: Iterable[int],
        melds: Iterable[dict[str, Any]],
        play_rect: pygame.Rect,
        tile_width: int,
        tile_height: int,
        spacing: int,
    ) -> None:
        assert self._screen is not None
        count = len(hand)
        x = play_rect.right - tile_height - 16
        y = play_rect.top + 40
        back_surface = self._get_tile_graphic(-1, (tile_width, tile_height), face_up=False)
        rotated_back = pygame.transform.rotate(back_surface, -90)
        for idx in range(count):
            pos_y = y + idx * (tile_width + spacing)
            rect = rotated_back.get_rect()
            rect.topleft = (x, pos_y)
            self._screen.blit(rotated_back, rect)

        meld_origin = (x - spacing * 4, play_rect.top + 80)
        self._draw_melds_vertical(melds, meld_origin, tile_width, tile_height, spacing, clockwise=True)

        discard_origin = (x - tile_height - 40, play_rect.top + 60)
        self._draw_discard_grid(discards, discard_origin, tile_width, tile_height, spacing, rotation=-90)

    def _draw_left_area(
        self,
        hand: Iterable[int],
        discards: Iterable[int],
        melds: Iterable[dict[str, Any]],
        play_rect: pygame.Rect,
        tile_width: int,
        tile_height: int,
        spacing: int,
    ) -> None:
        assert self._screen is not None
        count = len(hand)
        x = play_rect.left + 16
        y = play_rect.top + 40
        back_surface = self._get_tile_graphic(-1, (tile_width, tile_height), face_up=False)
        rotated_back = pygame.transform.rotate(back_surface, 90)
        for idx in range(count):
            pos_y = y + idx * (tile_width + spacing)
            rect = rotated_back.get_rect()
            rect.topleft = (x, pos_y)
            self._screen.blit(rotated_back, rect)

        meld_origin = (x + tile_height + 60, play_rect.top + 80)
        self._draw_melds_vertical(melds, meld_origin, tile_width, tile_height, spacing, clockwise=False)

        discard_origin = (meld_origin[0] + tile_height + 40, play_rect.top + 60)
        self._draw_discard_grid(discards, discard_origin, tile_width, tile_height, spacing, rotation=90)

    def _draw_melds_horizontal(
        self,
        melds: Iterable[dict[str, Any]],
        anchor: Tuple[int, int],
        tile_width: int,
        tile_height: int,
        spacing: int,
        reverse: bool = False,
    ) -> None:
        assert self._screen is not None
        meld_list = list(melds)
        if not meld_list:
            return
        x, y = anchor
        direction = -1 if reverse else 1
        offset = 0
        for meld in (reversed(meld_list) if reverse else meld_list):
            tiles = [t // 4 for t in meld.get("m", [])]
            for idx, tile_type in enumerate(tiles):
                surface = self._get_tile_graphic(tile_type, (tile_width, tile_height))
                pos_x = x + direction * (offset + idx) * (tile_width + spacing) - (tile_width if reverse else 0)
                rect = surface.get_rect()
                rect.bottomleft = (pos_x, y + tile_height)
                self._screen.blit(surface, rect)
            offset += len(tiles) + 1

    def _draw_melds_vertical(
        self,
        melds: Iterable[dict[str, Any]],
        anchor: Tuple[int, int],
        tile_width: int,
        tile_height: int,
        spacing: int,
        clockwise: bool,
    ) -> None:
        assert self._screen is not None
        meld_list = list(melds)
        if not meld_list:
            return
        x, y = anchor
        offset = 0
        for meld in meld_list:
            tiles = [t // 4 for t in meld.get("m", [])]
            for idx, tile_type in enumerate(tiles):
                surface = self._get_tile_graphic(tile_type, (tile_width, tile_height))
                rotated = pygame.transform.rotate(surface, -90 if clockwise else 90)
                rect = rotated.get_rect()
                rect.topleft = (x, y + (offset + idx) * (tile_width + spacing))
                self._screen.blit(rotated, rect)
            offset += len(tiles) + 1

    def _draw_discard_grid(
        self,
        discards: Iterable[int],
        origin: Tuple[int, int],
        tile_width: int,
        tile_height: int,
        spacing: int,
        rotation: int,
    ) -> None:
        assert self._screen is not None
        max_cols = 6
        discards_list = [tile // 4 for tile in discards]
        for idx, tile_type in enumerate(discards_list):
            col = idx % max_cols
            row = idx // max_cols
            x = origin[0] + col * (tile_width + spacing)
            y = origin[1] + row * (tile_height + spacing)
            surface = self._get_tile_graphic(tile_type, (tile_width, tile_height))
            if rotation:
                surface = pygame.transform.rotate(surface, rotation)
            rect = surface.get_rect()
            rect.topleft = (x, y)
            self._screen.blit(surface, rect)

    def _get_tile_graphic(
        self,
        tile_type: int,
        size: Tuple[int, int],
        face_up: bool = True,
    ) -> pygame.Surface:
        if tile_type < 0:
            face_up = False
        key = (tile_type, size, face_up)
        if key in self._tile_cache:
            return self._tile_cache[key]

        if not face_up and self._tile_back is not None:
            base = self._tile_back
        elif tile_type in self._tile_images:
            base = self._tile_images[tile_type]
        else:
            base = self._tile_placeholders.get(tile_type)
            if base is None:
                base = pygame.Surface(self._base_tile_size, pygame.SRCALPHA)
                base.fill((240, 240, 240))
                pygame.draw.rect(base, (40, 40, 40), base.get_rect(), 2, border_radius=6)
        scaled = pygame.transform.smoothscale(base, size)
        self._tile_cache[key] = scaled
        return scaled

    # ------------------------------------------------------------------
    # Support context manager style usage
    # ------------------------------------------------------------------
    def __enter__(self) -> "MahjongEnv":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
