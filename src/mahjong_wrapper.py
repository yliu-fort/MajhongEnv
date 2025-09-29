
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
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
    """Mahjong environment with a pygame GUI styled as a Riichi table."""

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
        self._table_color = (8, 24, 52)
        self._play_area_color = (18, 48, 84)
        self._panel_color = (8, 8, 12)
        self._panel_border_color = (60, 70, 90)
        self._accent_color = (140, 220, 255)
        self._text_color = (235, 235, 235)
        self._danger_color = (220, 120, 120)
        self._seat_text_color = (180, 190, 210)
        self._dealer_marker_color = (240, 200, 80)
        self._screen: Optional[pygame.Surface] = None
        self._font: Optional[pygame.font.Font] = None
        self._font_large: Optional[pygame.font.Font] = None
        self._font_small: Optional[pygame.font.Font] = None
        self._tile_font: Optional[pygame.font.Font] = None
        self._clock: Optional[pygame.time.Clock] = None
        self._line_height = font_size + 6
        self._quit_requested = False
        self._last_payload = _RenderPayload(action=None, reward=0.0, done=False, info={})
        self._tile_images: dict[str, pygame.Surface] = {}
        self._fallback_tile_cache: dict[str, pygame.Surface] = {}
        self._hidden_tile_base: Optional[pygame.Surface] = None
        self._scaled_tile_cache: dict[Tuple[str, Tuple[int, int], str, bool], pygame.Surface] = {}
        self._base_tile_size = (120, 160)
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
        self._font_large = None
        self._font_small = None
        self._tile_font = None
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
        self._font_large = pygame.font.SysFont(self._font_name, self._font_size + 12)
        self._font_small = pygame.font.SysFont(self._font_name, max(12, self._font_size - 4))
        self._tile_font = pygame.font.SysFont(self._font_name, self._font_size + 8)
        self._clock = pygame.time.Clock()
        self._line_height = self._font.get_linesize() + 4
        self._load_tile_assets()

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
                self._scaled_tile_cache.clear()

    def _load_tile_assets(self) -> None:
        if self._tile_font is None:
            return

        self._tile_images.clear()
        self._fallback_tile_cache.clear()
        self._scaled_tile_cache.clear()
        self._hidden_tile_base = None

        asset_dir = Path(__file__).resolve().parent.parent / "data" / "assets"
        if asset_dir.exists():
            try:
                import cairosvg  # type: ignore[import-not-found]
            except Exception:  # pragma: no cover - optional dependency
                cairosvg = None
            else:
                for symbol in _TILE_SYMBOLS:
                    svg_path = asset_dir / f"{symbol}.svg"
                    if not svg_path.exists():
                        continue
                    try:
                        png_bytes = cairosvg.svg2png(
                            url=str(svg_path),
                            output_width=self._base_tile_size[0],
                            output_height=self._base_tile_size[1],
                        )
                        surface = pygame.image.load(BytesIO(png_bytes)).convert_alpha()
                    except Exception:
                        continue
                    self._tile_images[symbol] = surface

        if self._hidden_tile_base is None:
            self._hidden_tile_base = self._create_hidden_tile_surface()

    def _create_hidden_tile_surface(self) -> pygame.Surface:
        width, height = self._base_tile_size
        surface = pygame.Surface((width, height), pygame.SRCALPHA)
        back_color = (26, 29, 36)
        border_color = (80, 88, 102)
        pygame.draw.rect(surface, back_color, (0, 0, width, height), border_radius=12)
        pygame.draw.rect(surface, border_color, (0, 0, width, height), width=4, border_radius=12)
        return surface

    def _create_fallback_tile_surface(self, symbol: str) -> pygame.Surface:
        if self._tile_font is None:
            raise RuntimeError("Tile font is not initialised")

        width, height = self._base_tile_size
        surface = pygame.Surface((width, height), pygame.SRCALPHA)
        body_color = (204, 204, 204)
        border_color = (70, 70, 70)
        pygame.draw.rect(surface, body_color, (0, 0, width, height), border_radius=12)
        pygame.draw.rect(surface, border_color, (0, 0, width, height), width=4, border_radius=12)

        text_surface = self._tile_font.render(symbol, True, (20, 30, 40))
        text_rect = text_surface.get_rect(center=(width // 2, height // 2))
        surface.blit(text_surface, text_rect)
        return surface

    def _tile_index_to_symbol(self, tile: int) -> str:
        base_index = tile // 4
        if base_index < 0 or base_index >= len(_TILE_SYMBOLS):
            return "??"
        return _TILE_SYMBOLS[base_index]

    def _get_tile_surface(
        self,
        symbol: Optional[str],
        size: Tuple[int, int],
        orientation: str,
        face_up: bool,
    ) -> pygame.Surface:
        if self._hidden_tile_base is None:
            self._hidden_tile_base = self._create_hidden_tile_surface()

        key_symbol = symbol if face_up and symbol is not None else "__hidden__"
        cache_key = (key_symbol, size, orientation, face_up)
        cached = self._scaled_tile_cache.get(cache_key)
        if cached is not None:
            return cached

        if face_up and symbol is not None:
            base_surface = self._tile_images.get(symbol)
            if base_surface is None:
                base_surface = self._fallback_tile_cache.get(symbol)
                if base_surface is None:
                    base_surface = self._create_fallback_tile_surface(symbol)
                    self._fallback_tile_cache[symbol] = base_surface
        else:
            base_surface = self._hidden_tile_base

        scaled = pygame.transform.smoothscale(base_surface, size)
        orientation_map = {
            "south": 0,
            "west": 90,
            "north": 180,
            "east": -90,
        }
        angle = orientation_map.get(orientation, 0)
        if angle:
            scaled = pygame.transform.rotate(scaled, angle)
        self._scaled_tile_cache[cache_key] = scaled
        return scaled

    def _prepare_hand_surfaces(
        self,
        tiles: Iterable[int],
        orientation: str,
        tile_size: Tuple[int, int],
        face_up: bool,
    ) -> list[pygame.Surface]:
        tiles_list = sorted(tiles) if face_up else list(tiles)
        surfaces: list[pygame.Surface] = []
        for tile in tiles_list:
            symbol = self._tile_index_to_symbol(tile) if face_up else None
            surfaces.append(self._get_tile_surface(symbol, tile_size, orientation, face_up))
        return surfaces

    def _measure_hand_extent(
        self,
        surfaces: list[pygame.Surface],
        orientation: str,
        gap: int,
        separate_last: bool = False,
    ) -> Tuple[int, int]:
        if not surfaces:
            return 0, 0
        if orientation in {"south", "north"}:
            width = sum(surface.get_width() for surface in surfaces)
            if len(surfaces) > 1:
                width += gap * (len(surfaces) - 1)
                if separate_last:
                    width += gap * 2
            height = max(surface.get_height() for surface in surfaces)
        else:
            height = sum(surface.get_height() for surface in surfaces)
            if len(surfaces) > 1:
                height += gap * (len(surfaces) - 1)
            width = max(surface.get_width() for surface in surfaces)
        return width, height

    def _draw_hand(
        self,
        surfaces: list[pygame.Surface],
        orientation: str,
        start_pos: Tuple[int, int],
        gap: int,
        separate_last: bool = False,
    ) -> pygame.Rect:
        if self._screen is None or not surfaces:
            return pygame.Rect(start_pos, (0, 0))

        x, y = start_pos
        min_x, min_y = x, y
        max_x, max_y = x, y
        for idx, surface in enumerate(surfaces):
            if orientation in {"south", "north"}:
                if separate_last and idx == len(surfaces) - 1 and len(surfaces) > 1:
                    x += gap * 2
                self._screen.blit(surface, (x, y))
                max_x = max(max_x, x + surface.get_width())
                max_y = max(max_y, y + surface.get_height())
                x += surface.get_width() + (gap if idx < len(surfaces) - 1 else 0)
            else:
                self._screen.blit(surface, (x, y))
                max_x = max(max_x, x + surface.get_width())
                max_y = max(max_y, y + surface.get_height())
                y += surface.get_height() + (gap if idx < len(surfaces) - 1 else 0)
        return pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)

    def _collect_discards(self, player_idx: int) -> list[int]:
        if player_idx >= len(self.discard_pile):
            return []
        return [idx for idx, flagged in enumerate(self.discard_pile[player_idx]) if flagged]

    def _draw_discard_pool(
        self,
        tiles: Iterable[int],
        orientation: str,
        area: pygame.Rect,
        tile_size: Tuple[int, int],
        gap: int,
    ) -> None:
        if self._screen is None:
            return

        tiles_list = list(tiles)
        if not tiles_list:
            return

        opposite_map = {
            "south": "north",
            "north": "south",
            "east": "west",
            "west": "east",
        }
        discard_orientation = opposite_map.get(orientation, "north")

        max_cols = 6
        for idx, tile in enumerate(tiles_list):
            symbol = self._tile_index_to_symbol(tile)
            surface = self._get_tile_surface(symbol, tile_size, discard_orientation, True)
            if discard_orientation in {"south", "north"}:
                col = idx % max_cols
                row = idx // max_cols
                x = area.left + col * (surface.get_width() + gap)
                y = area.top + row * (surface.get_height() + gap)
            else:
                row = idx % max_cols
                col = idx // max_cols
                x = area.left + col * (surface.get_width() + gap)
                y = area.top + row * (surface.get_height() + gap)
            if x + surface.get_width() <= area.right and y + surface.get_height() <= area.bottom:
                self._screen.blit(surface, (x, y))

    def _draw_melds(
        self,
        melds: Iterable[dict[str, Any]],
        orientation: str,
        start_pos: Tuple[int, int],
        tile_size: Tuple[int, int],
        gap: int,
        horizontal: bool,
    ) -> None:
        if self._screen is None:
            return

        x, y = start_pos
        for meld in melds:
            tiles = sorted(meld.get("m", []))
            if not tiles:
                continue
            if horizontal:
                for tile in tiles:
                    symbol = self._tile_index_to_symbol(tile)
                    surface = self._get_tile_surface(symbol, tile_size, orientation, True)
                    self._screen.blit(surface, (x, y))
                    x += surface.get_width() + gap
                x += gap * 2
            else:
                for tile in tiles:
                    symbol = self._tile_index_to_symbol(tile)
                    surface = self._get_tile_surface(symbol, tile_size, orientation, True)
                    self._screen.blit(surface, (x, y))
                    y += surface.get_height() + gap
                y += gap * 2

    def _format_round_wind(self) -> str:
        round_state = getattr(self, "round", [0, 0])
        round_index = max(0, round_state[0])
        winds = ("East", "South", "West", "North")
        wind = winds[(round_index // self.num_players) % len(winds)]
        hand_number = round_index % self.num_players + 1
        return f"{wind} {hand_number}"

    def _draw_center_panel(self, play_rect: pygame.Rect) -> None:
        if self._screen is None or self._font is None or self._font_large is None or self._font_small is None:
            return

        panel_width = max(260, int(play_rect.width * 0.28))
        panel_height = max(140, int(play_rect.height * 0.25))
        panel_rect = pygame.Rect(0, 0, panel_width, panel_height)
        panel_rect.center = play_rect.center

        pygame.draw.rect(self._screen, self._panel_color, panel_rect, border_radius=16)
        pygame.draw.rect(self._screen, self._panel_border_color, panel_rect, width=3, border_radius=16)

        round_surface = self._font_large.render(self._format_round_wind(), True, self._accent_color)
        round_rect = round_surface.get_rect(center=(panel_rect.centerx, panel_rect.top + round_surface.get_height()))
        self._screen.blit(round_surface, round_rect)

        counters = [
            f"Tiles Left: {len(getattr(self, 'deck', []))}",
            f"Honba: {getattr(self, 'round', [0, 0])[1]}",
            f"Riichi: {getattr(self, 'num_riichi', 0)}",
        ]
        counter_surface = self._font_small.render("  |  ".join(counters), True, self._text_color)
        counter_rect = counter_surface.get_rect(center=(panel_rect.centerx, round_rect.bottom + counter_surface.get_height()))
        self._screen.blit(counter_surface, counter_rect)

        score_positions = {
            0: (panel_rect.centerx, panel_rect.bottom + 30),
            1: (panel_rect.right + 50, panel_rect.centery),
            2: (panel_rect.centerx, panel_rect.top - 30),
            3: (panel_rect.left - 50, panel_rect.centery),
        }
        scores = getattr(self, "scores", [0, 0, 0, 0])
        for idx, position in score_positions.items():
            score_surface = self._font.render(f"{scores[idx]:>4}00", True, self._text_color)
            score_rect = score_surface.get_rect(center=position)
            self._screen.blit(score_surface, score_rect)

    def _draw_live_wall(self, play_rect: pygame.Rect, tile_size: Tuple[int, int], gap: int) -> None:
        if self._screen is None:
            return

        wall_rect = play_rect.inflate(tile_size[0] * 6, tile_size[1] * 6)
        hidden_h = self._get_tile_surface(None, tile_size, "south", False)
        hidden_v = self._get_tile_surface(None, tile_size, "west", False)
        per_side = 17

        total_top = per_side * (hidden_h.get_width() + gap) - gap
        start_x = wall_rect.centerx - total_top // 2
        top_y = wall_rect.top - hidden_h.get_height() - gap
        bottom_y = wall_rect.bottom + gap
        for idx in range(per_side):
            x = start_x + idx * (hidden_h.get_width() + gap)
            self._screen.blit(hidden_h, (x, top_y))
            self._screen.blit(hidden_h, (x, bottom_y))

        total_side = per_side * (hidden_v.get_height() + gap) - gap
        start_y = wall_rect.centery - total_side // 2
        left_x = wall_rect.left - hidden_v.get_width() - gap
        right_x = wall_rect.right + gap
        for idx in range(per_side):
            y = start_y + idx * (hidden_v.get_height() + gap)
            self._screen.blit(hidden_v, (left_x, y))
            self._screen.blit(hidden_v, (right_x, y))

    def _draw_dead_wall(
        self,
        play_rect: pygame.Rect,
        right_edge: int,
        tile_size: Tuple[int, int],
        gap: int,
    ) -> None:
        if self._screen is None:
            return

        dead_tiles = getattr(self, "dead_wall", [])
        stack_height = max(1, min(7, len(dead_tiles)))
        hidden_surface = self._get_tile_surface(None, tile_size, "west", False)
        total_height = stack_height * (hidden_surface.get_height() + gap) - gap
        x = max(play_rect.right + tile_size[0], right_edge + gap)
        y = play_rect.centery - total_height // 2
        for _ in range(stack_height):
            self._screen.blit(hidden_surface, (x, y))
            y += hidden_surface.get_height() + gap

        if getattr(self, "dora_indicator", []):
            indicator = self._tile_index_to_symbol(self.dora_indicator[0])
            face_surface = self._get_tile_surface(indicator, tile_size, "west", True)
            face_rect = face_surface.get_rect()
            face_rect.midleft = (x + hidden_surface.get_width() + gap, play_rect.centery)
            self._screen.blit(face_surface, face_rect)

    def _draw_seat_labels(
        self,
        positions: dict[int, Tuple[int, int]],
        dealer_idx: int,
    ) -> None:
        if self._screen is None or self._font_small is None:
            return

        seat_names = getattr(self, "player_names", None)
        if not seat_names:
            seat_names = ["NoName" for _ in range(self.num_players)]

        seat_orientations = ("South", "West", "North", "East")
        for idx, pos in positions.items():
            name = seat_names[idx] if idx < len(seat_names) else "NoName"
            label = f"{name} ({seat_orientations[idx]})"
            text_surface = self._font_small.render(label, True, self._seat_text_color)
            text_rect = text_surface.get_rect(center=pos)
            self._screen.blit(text_surface, text_rect)

            if idx == dealer_idx:
                radius = max(8, text_rect.height // 3)
                marker_center = (text_rect.right + radius * 2, text_rect.centery)
                pygame.draw.circle(self._screen, self._dealer_marker_color, marker_center, radius)
                east_surface = self._font_small.render("E", True, (20, 20, 20))
                east_rect = east_surface.get_rect(center=marker_center)
                self._screen.blit(east_surface, east_rect)

    def _draw_status_overlay(self, play_rect: pygame.Rect) -> None:
        if self._screen is None or self._font_small is None:
            return

        lines = [
            f"Phase: {self.phase}",
            f"Current Player: P{self.current_player}",
            f"Last Action: {self._last_payload.action}",
            f"Reward: {self._last_payload.reward:.2f}",
        ]
        msg = self._last_payload.info.get("msg") or getattr(self, "msg", "")
        if msg:
            lines.append(f"Msg: {msg}")
        if self._last_payload.done:
            lines.append("Episode finished")

        x = play_rect.left + 16
        y = play_rect.bottom - len(lines) * (self._font_small.get_linesize() + 2) - 16
        for line in lines:
            color = self._danger_color if line.startswith("Reward") and self._last_payload.reward < 0 else self._text_color
            surface = self._font_small.render(line, True, color)
            self._screen.blit(surface, (x, y))
            y += self._font_small.get_linesize() + 2

    def _render(self) -> None:
        if self._screen is None or self._font is None or self._clock is None or self._font_small is None:
            return

        width, height = self._screen.get_size()
        self._screen.fill(self._table_color)

        outer_margin = max(20, int(min(width, height) * 0.03))
        tile_height = max(40, int(min(width, height) * 0.08))
        tile_width = int(tile_height * 0.72)
        tile_gap = max(4, tile_width // 8)
        discard_tile_size = (int(tile_width * 0.85), int(tile_height * 0.85))
        rows, cols = 2, 6
        v_rows, v_cols = 6, 2
        meld_band_vertical = tile_height
        meld_band_horizontal = tile_width
        seat_line_height = self._font_small.get_linesize()

        sample_horizontal = self._get_tile_surface("1m", discard_tile_size, "north", True)
        sample_vertical = self._get_tile_surface("1m", discard_tile_size, "east", True)
        for _ in range(6):
            discard_height = sample_horizontal.get_height() * rows + tile_gap * (rows - 1)
            discard_width_h = sample_horizontal.get_width() * cols + tile_gap * (cols - 1)
            discard_height_v = sample_vertical.get_height() * v_rows + tile_gap * (v_rows - 1)
            discard_width_v = sample_vertical.get_width() * v_cols + tile_gap * (v_cols - 1)

            vertical_band = (
                tile_gap
                + discard_height
                + tile_gap
                + meld_band_vertical
                + tile_gap
                + tile_height
                + tile_gap
                + seat_line_height
            )
            horizontal_band = (
                tile_gap
                + discard_width_v
                + tile_gap
                + meld_band_horizontal
                + tile_gap
                + tile_height
                + tile_gap
                + seat_line_height
            )

            play_width_limit = width - 2 * (outer_margin + horizontal_band)
            play_height_limit = height - 2 * (outer_margin + vertical_band)
            if play_width_limit > 220 and play_height_limit > 220:
                break

            tile_height = max(32, int(tile_height * 0.9))
            tile_width = int(tile_height * 0.72)
            tile_gap = max(3, tile_width // 8)
            discard_tile_size = (int(tile_width * 0.85), int(tile_height * 0.85))
            meld_band_vertical = tile_height
            meld_band_horizontal = tile_width
            sample_horizontal = self._get_tile_surface("1m", discard_tile_size, "north", True)
            sample_vertical = self._get_tile_surface("1m", discard_tile_size, "east", True)
        else:
            discard_height = sample_horizontal.get_height() * rows + tile_gap * (rows - 1)
            discard_width_h = sample_horizontal.get_width() * cols + tile_gap * (cols - 1)
            discard_height_v = sample_vertical.get_height() * v_rows + tile_gap * (v_rows - 1)
            discard_width_v = sample_vertical.get_width() * v_cols + tile_gap * (v_cols - 1)
            play_width_limit = width - 2 * (outer_margin + horizontal_band)
            play_height_limit = height - 2 * (outer_margin + vertical_band)

        play_size = max(220, min(play_width_limit, play_height_limit))
        play_rect = pygame.Rect(0, 0, play_size, play_size)
        play_rect.center = (width // 2, height // 2)

        pygame.draw.rect(self._screen, self._play_area_color, play_rect, border_radius=40)
        self._draw_live_wall(play_rect, (tile_width, tile_height), tile_gap)
        self._draw_center_panel(play_rect)

        bottom_discard_rect = pygame.Rect(
            play_rect.centerx - discard_width_h // 2,
            play_rect.bottom + tile_gap,
            discard_width_h,
            discard_height,
        )
        top_discard_rect = pygame.Rect(
            play_rect.centerx - discard_width_h // 2,
            play_rect.top - tile_gap - discard_height,
            discard_width_h,
            discard_height,
        )
        right_discard_rect = pygame.Rect(
            play_rect.right + tile_gap,
            play_rect.centery - discard_height_v // 2,
            discard_width_v,
            discard_height_v,
        )
        left_discard_rect = pygame.Rect(
            play_rect.left - tile_gap - discard_width_v,
            play_rect.centery - discard_height_v // 2,
            discard_width_v,
            discard_height_v,
        )

        hand_tile_size = (tile_width, tile_height)

        hand_rects: dict[int, pygame.Rect] = {}
        meld_positions: dict[int, Tuple[int, int]] = {}

        bottom_surfaces = self._prepare_hand_surfaces(self.hands[0], "south", hand_tile_size, True)
        bottom_width, bottom_height = self._measure_hand_extent(bottom_surfaces, "south", tile_gap, separate_last=True)
        bottom_start_x = max(outer_margin, (width - bottom_width) // 2)
        bottom_start_y = bottom_discard_rect.bottom + tile_gap
        if bottom_start_y + bottom_height > height - outer_margin:
            bottom_start_y = height - outer_margin - bottom_height
        hand_rects[0] = self._draw_hand(bottom_surfaces, "south", (bottom_start_x, bottom_start_y), tile_gap, separate_last=True)
        meld_positions[0] = (hand_rects[0].right + tile_gap, hand_rects[0].top)

        top_surfaces = self._prepare_hand_surfaces(self.hands[2], "north", hand_tile_size, False)
        top_width, top_height = self._measure_hand_extent(top_surfaces, "north", tile_gap)
        top_meld_y = top_discard_rect.top - tile_gap - meld_band_vertical
        top_start_y = top_meld_y - tile_gap - top_height
        if top_start_y < outer_margin:
            shift = outer_margin - top_start_y
            top_start_y += shift
            top_meld_y += shift
        top_start_x = max(outer_margin, (width - top_width) // 2)
        hand_rects[2] = self._draw_hand(top_surfaces, "north", (top_start_x, top_start_y), tile_gap)
        meld_positions[2] = (hand_rects[2].left, top_meld_y)

        right_surfaces = self._prepare_hand_surfaces(self.hands[1], "west", hand_tile_size, False)
        right_width, right_height = self._measure_hand_extent(right_surfaces, "west", tile_gap)
        right_meld_x = right_discard_rect.right + tile_gap
        right_start_x = right_meld_x + meld_band_horizontal + tile_gap
        if right_start_x + right_width > width - outer_margin:
            overflow = right_start_x + right_width - (width - outer_margin)
            right_start_x -= overflow
            right_meld_x -= overflow
        if right_start_x < right_meld_x + tile_gap:
            shift = (right_meld_x + tile_gap) - right_start_x
            right_start_x += shift
            right_meld_x += shift
        right_start_y = max(outer_margin, play_rect.centery - right_height // 2)
        hand_rects[1] = self._draw_hand(right_surfaces, "west", (right_start_x, right_start_y), tile_gap)
        meld_positions[1] = (right_meld_x, right_start_y)

        left_surfaces = self._prepare_hand_surfaces(self.hands[3], "east", hand_tile_size, False)
        left_width, left_height = self._measure_hand_extent(left_surfaces, "east", tile_gap)
        left_start_x = left_discard_rect.left - tile_gap - meld_band_horizontal - left_width
        if left_start_x < outer_margin:
            shift = outer_margin - left_start_x
            left_start_x += shift
            left_discard_rect.move_ip(shift, 0)
        left_meld_x = left_start_x + left_width + tile_gap
        if left_meld_x + meld_band_horizontal > left_discard_rect.left - tile_gap:
            left_meld_x = left_discard_rect.left - tile_gap - meld_band_horizontal
        left_start_y = max(outer_margin, play_rect.centery - left_height // 2)
        hand_rects[3] = self._draw_hand(left_surfaces, "east", (left_start_x, left_start_y), tile_gap)
        meld_positions[3] = (left_meld_x, left_start_y)

        discard_sample_h = (sample_horizontal.get_width(), sample_horizontal.get_height())
        discard_sample_v = (sample_vertical.get_width(), sample_vertical.get_height())
        self._draw_discard_pool(self._collect_discards(0), "south", bottom_discard_rect, discard_sample_h, tile_gap)
        self._draw_discard_pool(self._collect_discards(1), "west", right_discard_rect, discard_sample_v, tile_gap)
        self._draw_discard_pool(self._collect_discards(2), "north", top_discard_rect, discard_sample_h, tile_gap)
        self._draw_discard_pool(self._collect_discards(3), "east", left_discard_rect, discard_sample_v, tile_gap)

        meld_tile_size = hand_tile_size
        self._draw_melds(self.melds[0], "south", meld_positions[0], meld_tile_size, tile_gap, horizontal=True)
        self._draw_melds(self.melds[1], "west", meld_positions[1], meld_tile_size, tile_gap, horizontal=False)
        self._draw_melds(self.melds[2], "north", meld_positions[2], meld_tile_size, tile_gap, horizontal=True)
        self._draw_melds(self.melds[3], "east", meld_positions[3], meld_tile_size, tile_gap, horizontal=False)

        label_positions = {
            0: (hand_rects[0].centerx, min(height - outer_margin // 2, hand_rects[0].bottom + tile_gap + seat_line_height // 2)),
            1: (min(width - outer_margin // 2, hand_rects[1].right + tile_gap + seat_line_height), hand_rects[1].centery),
            2: (hand_rects[2].centerx, max(outer_margin // 2, hand_rects[2].top - tile_gap - seat_line_height // 2)),
            3: (max(outer_margin // 2, hand_rects[3].left - tile_gap - seat_line_height), hand_rects[3].centery),
        }
        self._draw_seat_labels(label_positions, getattr(self, "oya", 0))

        right_edge = hand_rects[1].right if 1 in hand_rects else play_rect.right
        self._draw_dead_wall(play_rect, right_edge, hand_tile_size, tile_gap)

        self._draw_status_overlay(play_rect)

        pygame.display.flip()
        self._clock.tick(self._fps)

    # ------------------------------------------------------------------
    # Support context manager style usage
    # ------------------------------------------------------------------
    def __enter__(self) -> "MahjongEnv":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
