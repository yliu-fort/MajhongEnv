from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

try:
    import pygame
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise RuntimeError(
        "pygame is required for the MahjongEnv GUI wrapper; install pygame to continue"
    ) from exc

from mahjong_env import MahjongEnv as _BaseMahjongEnv

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "data" / "assets"

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
    """Mahjong environment with a feature-rich pygame GUI overlay."""

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
        self._table_color = (8, 36, 64)
        self._play_area_color = (20, 70, 78)
        self._panel_color = (10, 10, 12)
        self._panel_border = (80, 120, 160)
        self._accent_color = (120, 210, 255)
        self._text_color = (235, 235, 235)
        self._muted_text_color = (180, 190, 205)
        self._danger_color = (220, 120, 120)
        self._hand_spacing = 6
        self._base_tile_size = (120, 164)
        self._screen: Optional[pygame.Surface] = None
        self._font: Optional[pygame.font.Font] = None
        self._small_font: Optional[pygame.font.Font] = None
        self._clock: Optional[pygame.time.Clock] = None
        self._line_height = font_size + 6
        self._quit_requested = False
        self._last_payload = _RenderPayload(action=None, reward=0.0, done=False, info={})
        self._tile_cache: dict[Tuple[int, str, Tuple[int, int]], pygame.Surface] = {}
        self._hidden_tile_cache: dict[Tuple[str, Tuple[int, int]], pygame.Surface] = {}
        self._base_tile_surfaces: dict[int, pygame.Surface] = {}
        self._hidden_tile_base: Optional[pygame.Surface] = None
        self._svg_assets_loaded = False
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
        self._small_font = pygame.font.SysFont(self._font_name, max(12, self._font_size - 4))
        self._clock = pygame.time.Clock()
        self._line_height = self._font.get_linesize() + 4
        self._tile_cache.clear()
        self._hidden_tile_cache.clear()

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
                self._tile_cache.clear()
                self._hidden_tile_cache.clear()

    def _load_tile_assets(self) -> None:
        self._base_tile_surfaces.clear()
        self._tile_cache.clear()
        self._hidden_tile_cache.clear()
        self._hidden_tile_base = pygame.Surface(self._base_tile_size, pygame.SRCALPHA)
        self._hidden_tile_base.fill((22, 22, 28))
        pygame.draw.rect(
            self._hidden_tile_base,
            (12, 12, 16),
            self._hidden_tile_base.get_rect(),
            width=4,
            border_radius=12,
        )

        svg_loaded = False
        if _ASSETS_DIR.exists():
            try:
                import cairosvg  # type: ignore

                svg_loaded = True
                for idx, symbol in enumerate(_TILE_SYMBOLS):
                    svg_path = _ASSETS_DIR / f"{symbol}.svg"
                    if not svg_path.exists():
                        svg_loaded = False
                        break
                    try:
                        png_bytes = cairosvg.svg2png(url=str(svg_path))
                        surface = pygame.image.load(io.BytesIO(png_bytes)).convert_alpha()
                        self._base_tile_surfaces[idx] = surface
                    except Exception:
                        svg_loaded = False
                        self._base_tile_surfaces.clear()
                        break
            except Exception:
                svg_loaded = False

        if not svg_loaded:
            fallback_font = pygame.font.SysFont(self._font_name, int(self._base_tile_size[1] * 0.4))
            palette = [
                (235, 235, 235),
                (240, 220, 200),
                (225, 230, 255),
                (240, 230, 220),
            ]
            text_color = (30, 30, 30)
            for idx, symbol in enumerate(_TILE_SYMBOLS):
                color = palette[idx % len(palette)]
                surface = pygame.Surface(self._base_tile_size, pygame.SRCALPHA)
                surface.fill(color)
                pygame.draw.rect(surface, (210, 210, 210), surface.get_rect(), width=4, border_radius=16)
                text_surface = fallback_font.render(symbol, True, text_color)
                text_rect = text_surface.get_rect(center=surface.get_rect().center)
                surface.blit(text_surface, text_rect)
                self._base_tile_surfaces[idx] = surface.convert_alpha()
        self._svg_assets_loaded = svg_loaded

    def _get_tile_surface(
        self,
        tile_index: int,
        orientation: str,
        size: Tuple[int, int],
    ) -> pygame.Surface:
        key = (tile_index, orientation, size)
        surface = self._tile_cache.get(key)
        if surface is not None:
            return surface

        base_surface = self._base_tile_surfaces.get(tile_index)
        if base_surface is None:
            base_surface = self._base_tile_surfaces.setdefault(
                tile_index,
                self._base_tile_surfaces.get(0) or pygame.Surface(self._base_tile_size, pygame.SRCALPHA),
            )

        scaled = pygame.transform.smoothscale(base_surface, size)
        angle = {"south": 0, "west": 90, "north": 180, "east": 270}.get(orientation, 0)
        if angle:
            scaled = pygame.transform.rotozoom(scaled, angle, 1.0)
        surface = scaled.convert_alpha()
        self._tile_cache[key] = surface
        return surface

    def _get_hidden_tile_surface(self, orientation: str, size: Tuple[int, int]) -> pygame.Surface:
        key = (orientation, size)
        surface = self._hidden_tile_cache.get(key)
        if surface is not None:
            return surface

        base = self._hidden_tile_base or pygame.Surface(self._base_tile_size, pygame.SRCALPHA)
        scaled = pygame.transform.smoothscale(base, size)
        angle = {"south": 0, "west": 90, "north": 180, "east": 270}.get(orientation, 0)
        if angle:
            scaled = pygame.transform.rotozoom(scaled, angle, 1.0)
        surface = scaled.convert_alpha()
        self._hidden_tile_cache[key] = surface
        return surface

    def _get_discards(self, player_idx: int) -> list[int]:
        if player_idx >= len(self.discard_pile):
            return []
        indices = self.discard_pile[player_idx].nonzero()[0]
        return [int(idx) for idx in indices]

    def _get_round_status(self) -> Tuple[str, str]:
        winds = ("East", "South", "West", "North")
        round_data = getattr(self, "round", [0, 0])
        round_index = round_data[0] if round_data else 0
        honba = round_data[1] if len(round_data) > 1 else 0
        num_players = max(1, getattr(self, "num_players", 4))
        wind = winds[(round_index // num_players) % len(winds)]
        hand_number = (round_index % num_players) + 1
        kyoutaku = getattr(self, "num_kyoutaku", 0)
        phase = getattr(self, "phase", "")
        round_text = f"{wind} {hand_number}"
        counter_text = f"Honba {honba}  |  Kyoutaku {kyoutaku}"
        if phase:
            counter_text += f"  |  Phase {phase}"
        return round_text, counter_text

    def _draw_center_panel(self, board_rect: pygame.Rect) -> pygame.Rect:
        assert self._screen is not None and self._font is not None and self._small_font is not None
        panel_width = int(board_rect.width * 0.3)
        panel_height = int(board_rect.height * 0.22)
        panel_rect = pygame.Rect(0, 0, panel_width, panel_height)
        panel_rect.center = board_rect.center
        pygame.draw.rect(self._screen, self._panel_color, panel_rect, border_radius=18)
        pygame.draw.rect(self._screen, self._panel_border, panel_rect, width=3, border_radius=18)

        round_text, counter_text = self._get_round_status()
        round_surface = self._font.render(round_text, True, self._text_color)
        round_rect = round_surface.get_rect(center=(panel_rect.centerx, panel_rect.top + panel_rect.height * 0.32))
        self._screen.blit(round_surface, round_rect)

        counter_surface = self._small_font.render(counter_text, True, self._muted_text_color)
        counter_rect = counter_surface.get_rect(center=(panel_rect.centerx, round_rect.bottom + counter_surface.get_height()))
        self._screen.blit(counter_surface, counter_rect)

        elapsed = getattr(self, "steps", 0)
        tiles_left = len(getattr(self, "deck", []))
        timer_text = f"Steps {elapsed}  |  Tiles Left {tiles_left}"
        timer_surface = self._small_font.render(timer_text, True, self._muted_text_color)
        timer_rect = timer_surface.get_rect(center=(panel_rect.centerx, panel_rect.bottom - timer_surface.get_height()))
        self._screen.blit(timer_surface, timer_rect)
        return panel_rect

    def _draw_scores(self, panel_rect: pygame.Rect, seat_orientations: dict[int, str]) -> None:
        assert self._screen is not None and self._font is not None
        scores = getattr(self, "scores", [0 for _ in range(self.num_players)])
        offsets = {
            "south": (0, panel_rect.height // 2 + 24),
            "north": (0, -(panel_rect.height // 2 + 24)),
            "west": (panel_rect.width // 2 + 70, 0),
            "east": (-(panel_rect.width // 2 + 70), 0),
        }
        for idx, orientation in seat_orientations.items():
            if idx >= len(scores):
                continue
            offset = offsets.get(orientation, (0, 0))
            position = (panel_rect.centerx + offset[0], panel_rect.centery + offset[1])
            score_text = f"{scores[idx]:,}"
            score_surface = self._font.render(score_text, True, self._text_color)
            score_rect = score_surface.get_rect(center=position)
            self._screen.blit(score_surface, score_rect)

    def _draw_seat_label(self, position: Tuple[int, int], text: str) -> None:
        if self._screen is None or self._small_font is None:
            return
        label_surface = self._small_font.render(text, True, self._muted_text_color)
        label_rect = label_surface.get_rect(center=position)
        self._screen.blit(label_surface, label_rect)

    def _draw_dealer_marker(self, position: Tuple[int, int]) -> None:
        if self._screen is None or self._small_font is None:
            return
        radius = 12
        pygame.draw.circle(self._screen, self._accent_color, position, radius, width=2)
        marker_surface = self._small_font.render("E", True, self._accent_color)
        marker_rect = marker_surface.get_rect(center=position)
        self._screen.blit(marker_surface, marker_rect)

    def _draw_player_hand(
        self,
        tiles: Iterable[int],
        board_rect: pygame.Rect,
        tile_size: Tuple[int, int],
        spacing: int,
    ) -> pygame.Rect:
        assert self._screen is not None
        tile_list = sorted(tiles)
        surfaces = [self._get_tile_surface(tile // 4, "south", tile_size) for tile in tile_list]
        draw_gap = spacing * 2
        if not surfaces:
            return pygame.Rect(
                board_rect.centerx,
                board_rect.bottom + spacing,
                0,
                tile_size[1],
            )

        total_width = sum(surface.get_width() for surface in surfaces)
        total_width += spacing * max(0, len(surfaces) - 1)
        total_width += draw_gap

        baseline = board_rect.bottom + tile_size[1] + spacing
        start_x = int(board_rect.centerx - total_width / 2)
        area_rect = pygame.Rect(start_x, baseline - tile_size[1], total_width, tile_size[1])

        x = float(start_x)
        for idx, surface in enumerate(surfaces):
            if idx == len(surfaces) - 1:
                x += draw_gap
            rect = surface.get_rect()
            rect.bottomleft = (int(round(x)), baseline)
            self._screen.blit(surface, rect)
            x += surface.get_width()
            if idx != len(surfaces) - 1:
                x += spacing

        return area_rect

    def _draw_hidden_hand(
        self,
        count: int,
        orientation: str,
        start_pos: Tuple[int, int],
        tile_size: Tuple[int, int],
        spacing: int,
    ) -> pygame.Rect:
        assert self._screen is not None
        if count <= 0:
            return pygame.Rect(start_pos[0], start_pos[1], 0, 0)

        surface = self._get_hidden_tile_surface(orientation, tile_size)
        width, height = surface.get_size()
        x, y = float(start_pos[0]), float(start_pos[1])
        max_width = width
        max_height = height
        for idx in range(count):
            rect = surface.get_rect()
            rect.topleft = (int(round(x)), int(round(y)))
            self._screen.blit(surface, rect)
            if orientation in ("south", "north"):
                x += width + spacing
                max_width = max(max_width, rect.right - start_pos[0])
            else:
                y += height + spacing
                max_height = max(max_height, rect.bottom - start_pos[1])
        return pygame.Rect(start_pos[0], start_pos[1], max_width, max_height)

    def _draw_discard_pool(
        self,
        tiles: Iterable[int],
        orientation: str,
        center: Tuple[int, int],
        tile_size: Tuple[int, int],
        max_cols: int,
    ) -> pygame.Rect:
        assert self._screen is not None
        tile_indices = [tile // 4 for tile in tiles]
        if not tile_indices:
            return pygame.Rect(center[0], center[1], 0, 0)

        surfaces = [self._get_tile_surface(idx, orientation, tile_size) for idx in tile_indices]
        sample = surfaces[0]
        tile_w, tile_h = sample.get_size()
        spacing = max(2, min(tile_w, tile_h) // 8)
        cols = max(1, min(max_cols, len(surfaces)))
        rows = math.ceil(len(surfaces) / cols)

        area_width = cols * tile_w + (cols - 1) * spacing
        area_height = rows * tile_h + (rows - 1) * spacing
        area_rect = pygame.Rect(0, 0, area_width, area_height)
        area_rect.center = (int(center[0]), int(center[1]))

        for idx, surface in enumerate(surfaces):
            col = idx % cols
            row = idx // cols
            x = area_rect.left + col * (tile_w + spacing)
            y = area_rect.top + row * (tile_h + spacing)
            rect = surface.get_rect(topleft=(int(x), int(y)))
            self._screen.blit(surface, rect)

        return area_rect

    def _draw_melds(
        self,
        melds: Iterable[dict[str, Any]],
        orientation: str,
        anchor: Tuple[int, int],
        tile_size: Tuple[int, int],
        spacing: int,
    ) -> pygame.Rect:
        assert self._screen is not None
        x, y = float(anchor[0]), float(anchor[1])
        start_x = x
        max_right = x
        max_bottom = y
        for meld in melds:
            tiles = [int(t) for t in meld.get("m", [])]
            if not tiles:
                continue
            surfaces = [self._get_tile_surface(t // 4, orientation, tile_size) for t in tiles]
            meld_x = x
            meld_height = 0
            for surface in surfaces:
                rect = surface.get_rect()
                rect.topleft = (int(round(meld_x)), int(round(y)))
                self._screen.blit(surface, rect)
                meld_x += surface.get_width() + spacing
                meld_height = max(meld_height, rect.height)
            meld_x -= spacing if surfaces else 0
            max_right = max(max_right, meld_x)
            max_bottom = max(max_bottom, y + meld_height)
            y += meld_height + spacing
        return pygame.Rect(int(start_x), int(anchor[1]), int(max_right - start_x), int(max_bottom - anchor[1]))

    def _draw_walls(self, board_rect: pygame.Rect, tile_size: Tuple[int, int]) -> None:
        assert self._screen is not None
        spacing = max(2, tile_size[0] // 10)
        count = 17

        top_surface = self._get_hidden_tile_surface("south", tile_size)
        bottom_surface = top_surface
        left_surface = self._get_hidden_tile_surface("east", tile_size)
        right_surface = self._get_hidden_tile_surface("west", tile_size)

        tile_w = top_surface.get_width()
        total_width = count * tile_w + (count - 1) * spacing
        start_x = board_rect.centerx - total_width / 2
        top_baseline = board_rect.top - spacing
        bottom_top = board_rect.bottom + spacing
        for idx in range(count):
            x = start_x + idx * (tile_w + spacing)
            rect_top = top_surface.get_rect()
            rect_top.bottomleft = (int(round(x)), int(round(top_baseline)))
            self._screen.blit(top_surface, rect_top)

            rect_bottom = bottom_surface.get_rect()
            rect_bottom.topleft = (int(round(x)), int(round(bottom_top)))
            self._screen.blit(bottom_surface, rect_bottom)

        tile_h = left_surface.get_height()
        total_height = count * tile_h + (count - 1) * spacing
        start_y = board_rect.centery - total_height / 2
        left_x = board_rect.left - spacing - left_surface.get_width()
        right_x = board_rect.right + spacing
        right_count = max(8, count - 4)
        for idx in range(count):
            y = start_y + idx * (tile_h + spacing)
            rect_left = left_surface.get_rect()
            rect_left.topright = (int(round(left_x)), int(round(y)))
            self._screen.blit(left_surface, rect_left)

        for idx in range(right_count):
            y = start_y + idx * (tile_h + spacing)
            rect_right = right_surface.get_rect()
            rect_right.topleft = (int(round(right_x)), int(round(y)))
            self._screen.blit(right_surface, rect_right)

    def _draw_dead_wall(
        self,
        board_rect: pygame.Rect,
        tile_size: Tuple[int, int],
        spacing: int,
    ) -> None:
        assert self._screen is not None
        hidden_surface = self._get_hidden_tile_surface("west", tile_size)
        tile_w, tile_h = hidden_surface.get_size()
        count = min(7, len(getattr(self, "dead_wall", [])))
        start_x = board_rect.right + tile_w + spacing * 2
        start_y = board_rect.centery - (count * (tile_h + spacing) - spacing) / 2

        for idx in range(count):
            rect = hidden_surface.get_rect()
            rect.topleft = (
                int(round(start_x)),
                int(round(start_y + idx * (tile_h + spacing))),
            )
            self._screen.blit(hidden_surface, rect)

        dora_tiles = getattr(self, "dora_indicator", [])
        if not dora_tiles:
            return
        dora_surface = self._get_tile_surface(dora_tiles[0] // 4, "west", tile_size)
        dora_rect = dora_surface.get_rect()
        dora_rect.midtop = (
            int(round(start_x + tile_w / 2)),
            int(round(start_y + count * (tile_h + spacing))),
        )
        pygame.draw.rect(
            self._screen,
            self._accent_color,
            dora_rect.inflate(8, 8),
            width=2,
            border_radius=8,
        )
        self._screen.blit(dora_surface, dora_rect)

    def _render(self) -> None:
        if self._screen is None or self._font is None or self._small_font is None or self._clock is None:
            return

        width, height = self._screen.get_size()
        self._screen.fill(self._table_color)

        board_size = max(420, int(min(width, height) * 0.62))
        board_rect = pygame.Rect(0, 0, board_size, board_size)
        board_rect.center = (width // 2, height // 2)

        pygame.draw.rect(self._screen, self._play_area_color, board_rect, border_radius=32)
        pygame.draw.rect(self._screen, (30, 100, 120), board_rect, width=4, border_radius=32)

        tile_height = max(52, board_rect.height // 7)
        tile_width = int(tile_height * 0.66)
        tile_size = (tile_width, tile_height)
        discard_tile_size = (int(tile_width * 0.9), int(tile_height * 0.9))
        wall_tile_size = (int(tile_width * 0.9), max(20, int(tile_height * 0.55)))
        spacing = max(6, tile_width // 6)

        self._draw_walls(board_rect, wall_tile_size)
        self._draw_dead_wall(board_rect, wall_tile_size, spacing // 2)

        panel_rect = self._draw_center_panel(board_rect)
        seat_orientations: dict[int, str] = {0: "south", 1: "west", 2: "north", 3: "east"}
        self._draw_scores(panel_rect, seat_orientations)

        hands = list(getattr(self, "hands", []))
        while len(hands) < 4:
            hands.append([])
        melds = list(getattr(self, "melds", []))
        while len(melds) < 4:
            melds.append([])

        seat_names = list(getattr(self, "player_names", []))
        if len(seat_names) < self.num_players:
            seat_names.extend(["NoName"] * (self.num_players - len(seat_names)))
        while len(seat_names) < 4:
            seat_names.append("NoName")

        label_positions = {
            "south": (board_rect.centerx, board_rect.bottom + int(tile_height * 1.6)),
            "north": (board_rect.centerx, board_rect.top - int(tile_height * 1.2)),
            "west": (board_rect.right + int(tile_height * 1.4), board_rect.centery),
            "east": (board_rect.left - int(tile_height * 1.4), board_rect.centery),
        }

        for idx, orientation in seat_orientations.items():
            position = label_positions.get(orientation)
            if position is None:
                continue
            name = seat_names[idx] if idx < len(seat_names) else "NoName"
            self._draw_seat_label((int(position[0]), int(position[1])), name)

        dealer_idx = getattr(self, "oya", 0)
        dealer_orientation = seat_orientations.get(dealer_idx)
        if dealer_orientation:
            dealer_pos = label_positions.get(dealer_orientation)
            if dealer_pos:
                offset_map = {
                    "south": (0, -int(tile_height * 0.45)),
                    "north": (0, int(tile_height * 0.45)),
                    "west": (-int(tile_height * 0.45), 0),
                    "east": (int(tile_height * 0.45), 0),
                }
                offset = offset_map.get(dealer_orientation, (0, 0))
                marker_position = (int(dealer_pos[0] + offset[0]), int(dealer_pos[1] + offset[1]))
                self._draw_dealer_marker(marker_position)

        south_hand_rect = self._draw_player_hand(hands[0], board_rect, tile_size, spacing)
        south_discards = self._get_discards(0)
        self._draw_discard_pool(
            south_discards,
            "south",
            (
                board_rect.centerx,
                board_rect.bottom - discard_tile_size[1] - spacing * 2,
            ),
            discard_tile_size,
            max_cols=6,
        )
        self._draw_melds(
            melds[0],
            "south",
            (
                south_hand_rect.right + spacing * 2,
                south_hand_rect.bottom - tile_size[1],
            ),
            tile_size,
            spacing,
        )

        west_count = len(hands[1])
        west_start_x = width - tile_size[1] - spacing * 4
        west_start_y = board_rect.centery - int((tile_size[0] + spacing) * west_count / 2)
        west_hand_rect = self._draw_hidden_hand(
            west_count,
            "west",
            (int(west_start_x), int(west_start_y)),
            tile_size,
            spacing,
        )
        west_meld_anchor = (
            int(west_hand_rect.left - spacing * 2 - tile_size[0] * 3),
            west_hand_rect.top,
        )
        west_discard_center = (
            board_rect.right - discard_tile_size[1] - spacing * 2,
            board_rect.centery,
        )
        self._draw_melds(
            melds[1],
            "west",
            west_meld_anchor,
            tile_size,
            spacing,
        )
        self._draw_discard_pool(
            self._get_discards(1),
            "west",
            west_discard_center,
            discard_tile_size,
            max_cols=3,
        )

        north_count = len(hands[2])
        north_start_x = board_rect.centerx - int((tile_size[0] + spacing) * north_count / 2)
        north_start_y = spacing * 3
        self._draw_hidden_hand(
            north_count,
            "north",
            (int(north_start_x), int(north_start_y)),
            tile_size,
            spacing,
        )
        self._draw_discard_pool(
            self._get_discards(2),
            "north",
            (
                board_rect.centerx,
                board_rect.top + discard_tile_size[1] + spacing * 2,
            ),
            discard_tile_size,
            max_cols=6,
        )
        self._draw_melds(
            melds[2],
            "north",
            (
                board_rect.left + spacing * 2,
                board_rect.top - tile_size[1] - spacing,
            ),
            tile_size,
            spacing,
        )

        east_count = len(hands[3])
        east_start_x = spacing * 3
        east_start_y = board_rect.centery - int((tile_size[0] + spacing) * east_count / 2)
        east_hand_rect = self._draw_hidden_hand(
            east_count,
            "east",
            (int(east_start_x), int(east_start_y)),
            tile_size,
            spacing,
        )
        east_meld_anchor = (
            east_hand_rect.right + spacing,
            east_hand_rect.top,
        )
        east_discard_center = (
            board_rect.left + discard_tile_size[1] + spacing * 2,
            board_rect.centery,
        )
        self._draw_melds(
            melds[3],
            "east",
            east_meld_anchor,
            tile_size,
            spacing,
        )
        self._draw_discard_pool(
            self._get_discards(3),
            "east",
            east_discard_center,
            discard_tile_size,
            max_cols=3,
        )

        info_line = f"Last Action: {self._last_payload.action}"
        info_line += f"  |  Reward: {self._last_payload.reward:.2f}"
        message = self._last_payload.info.get("msg") or getattr(self, "msg", "")
        if message:
            info_line += f"  |  {message}"
        footer_surface = self._small_font.render(info_line, True, self._text_color)
        footer_rect = footer_surface.get_rect()
        footer_rect.midbottom = (width // 2, height - spacing)
        self._screen.blit(footer_surface, footer_rect)

        if self._last_payload.done:
            banner_surface = self._font.render("Episode finished", True, self._danger_color)
            banner_rect = banner_surface.get_rect(center=(width // 2, height // 2))
            self._screen.blit(banner_surface, banner_rect)

        pygame.display.flip()
        self._clock.tick(self._fps)

    # ------------------------------------------------------------------
    # Support context manager style usage
    # ------------------------------------------------------------------
    def __enter__(self) -> "MahjongEnv":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
