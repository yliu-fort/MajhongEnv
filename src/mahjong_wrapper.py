from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Tuple

try:
    import pygame
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise RuntimeError(
        "pygame is required for the MahjongEnv GUI wrapper; install pygame to continue"
    ) from exc

from mahjong_env import MahjongEnv as _BaseMahjongEnv

_ALT_TILE_SYMBOLS: Tuple[str, ...] = (
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

_TILE_SYMBOLS: Tuple[str, ...] = (
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


class MahjongEnv(_BaseMahjongEnv):
    """Mahjong environment with a lightweight pygame GUI overlay."""

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
        self._screen: Optional[pygame.Surface] = None
        self._font: Optional[pygame.font.Font] = None
        self._small_font: Optional[pygame.font.Font] = None
        self._header_font: Optional[pygame.font.Font] = None
        self._clock: Optional[pygame.time.Clock] = None
        self._line_height = font_size + 4
        self._quit_requested = False
        self._last_payload = _RenderPayload(action=None, reward=0.0, done=False, info={})
        self._auto_advance = True
        self._step_once_requested = False
        self._auto_button_rect = pygame.Rect(0, 0, 0, 0)
        self._step_button_rect = pygame.Rect(0, 0, 0, 0)
        self._pause_button_rect = pygame.Rect(0, 0, 0, 0)
        self._pause_on_score = False
        self._score_pause_active = False
        self._score_pause_pending = False
        self._last_phase_is_score_last = ""
        self._asset_root = Path(__file__).resolve().parent.parent / "assets" / "tiles" / "Regular"
        self._raw_tile_assets: dict[int, pygame.Surface] = {}
        self._tile_cache: dict[tuple[int, Tuple[int, int]], pygame.Surface] = {}
        self._tile_orientation_cache: dict[tuple[int, Tuple[int, int], int], pygame.Surface] = {}
        self._face_down_cache: dict[tuple[Tuple[int, int], int], pygame.Surface] = {}
        self._placeholder_cache: dict[tuple[int, Tuple[int, int]], pygame.Surface] = {}
        self._tile_metrics: dict[str, Tuple[int, int]] = {}
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
            and (
                (not self._auto_advance and not self._step_once_requested)
                or self._score_pause_active
            )
            and not getattr(self, "done", False)
        ):
            self._process_events()
            self._render()
            if self._clock is not None:
                self._clock.tick(self._fps)

        if self._step_once_requested:
            self._step_once_requested = False

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
        self._font = self._create_font(self._font_size)
        small_size = max(12, self._font_size - 4)
        self._small_font = self._create_font(small_size)
        self._header_font = self._create_font(self._font_size + 8)
        self._clock = pygame.time.Clock()
        self._line_height = self._font.get_linesize() + 4
        self._load_tile_assets()

    def _create_font(self, size: int) -> pygame.font.Font:
        if pygame.font is None:
            raise RuntimeError("pygame.font must be available to create fonts")

        if self._font_name:
            font_path = Path(self._font_name)
            if font_path.exists():
                try:
                    return pygame.font.Font(str(font_path), size)
                except Exception:
                    pass

        candidate_names: list[str] = []
        if self._font_name:
            candidate_names.extend(self._normalize_font_names(self._font_name))
        candidate_names.extend(name for name in self._fallback_fonts if name)

        seen: set[str] = set()
        for name in candidate_names:
            if name in seen:
                continue
            seen.add(name)
            try:
                matched = pygame.font.match_font(name)
            except Exception:
                matched = None
            if matched:
                try:
                    return pygame.font.Font(matched, size)
                except Exception:
                    continue

        return pygame.font.SysFont(self._font_name, size)

    @staticmethod
    def _normalize_font_names(font_name: str | Iterable[str]) -> list[str]:
        if isinstance(font_name, str):
            return [name.strip() for name in font_name.replace(";", ",").split(",") if name.strip()]
        return [name for name in font_name if isinstance(name, str) and name]

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
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._handle_mouse_click(event.pos)

    def _handle_mouse_click(self, position: Tuple[int, int]) -> None:
        if self._auto_button_rect.collidepoint(position):
            self._auto_advance = not self._auto_advance
            if self._auto_advance:
                self._step_once_requested = False
                if (
                    self._score_last()
                    and self._pause_on_score
                ):
                    self._score_pause_active = True
                    self._score_pause_pending = True
                else:
                    self._score_pause_active = False
                    self._score_pause_pending = False
            else:
                self._step_once_requested = False
                self._score_pause_active = False
                self._score_pause_pending = False
        elif self._step_button_rect.collidepoint(position):
            if not self._auto_advance:
                self._step_once_requested = True
            elif self._score_pause_active:
                self._score_pause_active = False
                self._score_pause_pending = False
        elif self._pause_button_rect.collidepoint(position) and self._auto_advance:
            self._pause_on_score = not self._pause_on_score
            if self._pause_on_score and self._score_last():
                self._score_pause_active = True
                self._score_pause_pending = True
            else:
                self._score_pause_active = False
                self._score_pause_pending = False
    
    def _score_last(self):
        return getattr(self, "phase", "") == "score" and self.current_player == self.num_players - 1

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def _load_tile_assets(self) -> None:
        """Attempt to load SVG tile assets; fall back gracefully on failure."""

        self._raw_tile_assets.clear()
        if not self._asset_root.exists():
            return

        for tile_index, symbol in enumerate(_TILE_SYMBOLS):
            path = self._asset_root / f"{symbol}.svg"
            if not path.exists():
                continue
            try:
                fg = pygame.image.load(str(path)).convert_alpha()
                surface = pygame.Surface(fg.get_size(), pygame.SRCALPHA)
                rect = surface.get_rect()
                border_radius = max(2, min(rect.width, rect.height) // 16)
                pygame.draw.rect(
                    surface,
                    (245, 245, 245),
                    rect,
                    border_radius=border_radius,
                )
                surface.blit(fg, (0, 0))
            except Exception:
                continue
            self._raw_tile_assets[tile_index] = surface

    def _get_tile_color(self, tile_34: int) -> Tuple[int, int, int]:
        symbol = _TILE_SYMBOLS[tile_34]
        suit = symbol[-1] if symbol[-1] in {"m", "p", "s"} else "z"
        palette = {
            "m": (210, 90, 90),
            "p": (90, 150, 225),
            "s": (90, 190, 120),
            "z": (220, 210, 150),
        }
        return palette.get(suit, (200, 200, 200))

    def _create_tile_placeholder(self, tile_34: int, size: Tuple[int, int]) -> pygame.Surface:
        key = (tile_34, size)
        if key in self._placeholder_cache:
            return self._placeholder_cache[key]

        width = max(16, size[0])
        height = max(24, size[1])
        surface = pygame.Surface((width, height), pygame.SRCALPHA)
        surface.fill(self._get_tile_color(tile_34))
        pygame.draw.rect(surface, (245, 245, 245), surface.get_rect(), 2, border_radius=6)

        label = _TILE_SYMBOLS[tile_34]
        font_size = max(12, int(height * 0.45))
        font = pygame.font.SysFont(self._font_name, font_size)
        text = font.render(label, True, (20, 20, 20))
        text_rect = text.get_rect(center=surface.get_rect().center)
        surface.blit(text, text_rect)

        self._placeholder_cache[key] = surface
        return surface

    def _get_tile_surface(
        self,
        tile_136: int,
        size: Tuple[int, int],
        face_up: bool = True,
        orientation: int = 0,
    ) -> pygame.Surface:
        tile_34 = tile_136 // 4
        if tile_136 == 16:
            tile_34 = 34
        elif tile_136 == 52:
            tile_34 = 35
        elif tile_136 == 88:
            tile_34 = 36
        width = max(1, size[0])
        height = max(1, size[1])
        base_size = (width, height)

        if not face_up:
            return self._get_face_down_surface(base_size, orientation)

        cache_key = (tile_34, base_size)
        if cache_key not in self._tile_cache:
            if tile_34 in self._raw_tile_assets:
                try:
                    surface = pygame.transform.smoothscale(
                        self._raw_tile_assets[tile_34], base_size
                    )
                except pygame.error:
                    surface = pygame.transform.scale(self._raw_tile_assets[tile_34], base_size)
            else:
                surface = self._create_tile_placeholder(tile_34, base_size)
            self._tile_cache[cache_key] = surface

        orientation = orientation % 360
        if orientation == 0:
            return self._tile_cache[cache_key]

        orient_key = (tile_34, base_size, orientation)
        if orient_key not in self._tile_orientation_cache:
            self._tile_orientation_cache[orient_key] = pygame.transform.rotate(
                self._tile_cache[cache_key], orientation
            )
        return self._tile_orientation_cache[orient_key]

    def _get_face_down_surface(
        self, size: Tuple[int, int], orientation: int = 0
    ) -> pygame.Surface:
        width = max(1, size[0])
        height = max(1, size[1])
        base_size = (width, height)
        orientation = orientation % 360
        cache_key = (base_size, orientation)
        if cache_key not in self._face_down_cache:
            surface = pygame.Surface(base_size, pygame.SRCALPHA)
            surface.fill(self._face_down_color)
            pygame.draw.rect(surface, self._face_down_border, surface.get_rect(), 2, border_radius=6)
            if orientation:
                surface = pygame.transform.rotate(surface, orientation)
            self._face_down_cache[cache_key] = surface
        return self._face_down_cache[cache_key]

    def _get_discard_tiles(self, player_idx: int) -> list[int]:
        if player_idx >= len(getattr(self, "discard_pile_seq", [])):
            return []
        return [t for t in self.discard_pile_seq[player_idx]]

    def _compute_tile_metrics(self, play_rect: pygame.Rect) -> None:
        width = max(1, play_rect.width)
        height = max(1, play_rect.height)
        if self._screen is not None:
            screen_width, screen_height = self._screen.get_size()
            width = min(width, screen_width)
            height = min(height, screen_height)

        base_tile_width = max(16, min(width // 27, height // 24, 72))
        base_tile_height = max(24, int(base_tile_width * 1.4))

        tile_size = (base_tile_width, base_tile_height)

        self._tile_metrics = {
            "south_hand": tile_size,
            "north_hand": tile_size,
            "side_hand": tile_size,
            "discard": tile_size,
            "wall": tile_size,
            "meld": tile_size,
        }

    def _draw_center_panel(self, play_rect: pygame.Rect) -> pygame.Rect:
        center_width = max(100, int(play_rect.width * 0.24))
        center_height = max(100, int(play_rect.height * 0.24))
        center_rect = pygame.Rect(0, 0, center_width, center_height)
        center_rect.center = play_rect.center

        pygame.draw.rect(self._screen, self._panel_color, center_rect, border_radius=12)
        pygame.draw.rect(self._screen, self._panel_border, center_rect, 2, border_radius=12)

        if self._header_font is None or self._small_font is None or self._font is None:
            return center_rect

        round_data = getattr(self, "round", [0, 0])
        round_index = round_data[0] if isinstance(round_data[0], int) else 0
        wind_names = ["East", "South", "West", "North"]
        wind = wind_names[(round_index // 4) % 4]
        hand_number = round_index % 4 + 1
        round_text = f"{wind} {hand_number}"

        title_surface = self._header_font.render(round_text, True, self._text_color)
        title_rect = title_surface.get_rect()
        title_rect.centerx = center_rect.centerx
        title_rect.top = center_rect.top + 18
        self._screen.blit(title_surface, title_rect)

        honba = round_data[1] if len(round_data) > 1 and isinstance(round_data[1], int) else 0
        riichi = getattr(self, "num_riichi", 0)
        tiles_remaining = len(getattr(self, "deck", []))
        counter_texts = (
            f"Honba {honba}",
            f"Riichi {riichi}",
            f"Tiles {tiles_remaining}",
        )
        next_top = title_rect.bottom + 12
        for text in counter_texts:
            surface = self._small_font.render(text, True, self._muted_text_color)
            rect = surface.get_rect()
            rect.centerx = center_rect.centerx
            rect.top = next_top
            self._screen.blit(surface, rect)
            next_top = rect.bottom + 4

        score_positions = {
            0: (center_rect.centerx, center_rect.bottom + 14),
            1: (center_rect.right + 18, center_rect.centery),
            2: (center_rect.centerx, center_rect.top - 14),
            3: (center_rect.left - 18, center_rect.centery),
        }

        scores = getattr(self, "scores", [])
        current_player = getattr(self, "current_player", 0)
        for idx, position in score_positions.items():
            if idx >= len(scores):
                continue
            score_value = scores[idx] * 100
            color = self._accent_color if idx == current_player else self._text_color
            score_surface = self._font.render(f"{score_value:5d}", True, color)
            score_rect = score_surface.get_rect(center=position)
            self._screen.blit(score_surface, score_rect)

        return center_rect

    def _draw_tile_grid(
        self,
        tiles: list[int],
        area: pygame.Rect,
        tile_size: Tuple[int, int],
        orientation: int,
        columns: int,
        target_surface: Optional[pygame.Surface] = None,
    ) -> None:
        surface = target_surface or self._screen
        if surface is None or not tiles:
            return

        columns = max(1, columns)
        spacing = 4
        for idx, tile in enumerate(tiles):
            column = idx % columns
            row = idx // columns
            tile_surface = self._get_tile_surface(tile, tile_size, True, orientation)
            tile_width, tile_height = tile_surface.get_size()
            x = area.left + column * (tile_width + spacing)
            y = area.top + row * (tile_height + spacing)
            surface.blit(tile_surface, (x, y))

    def _draw_melds(
        self,
        player: int,
        origin: Tuple[int, int],
        direction: str,
        tile_size: Tuple[int, int],
        orientation: int,
        target_surface: Optional[pygame.Surface] = None,
    ) -> None:
        surface = target_surface or self._screen
        if surface is None:
            return

        melds = getattr(self, "melds", [])
        if player >= len(melds):
            return

        x, y = origin
        spacing = 8
        for meld in melds[player]:
            tiles = [tile for tile in meld.get("m", [])]
            opened = meld.get("opened", True)
            cur_x, cur_y = x, y
            for tile in tiles:
                tile_surface = (
                    self._get_tile_surface(tile, tile_size, True, orientation)
                    if opened
                    else self._get_face_down_surface(tile_size, orientation)
                )
                surface.blit(tile_surface, (cur_x, cur_y))
                if direction == "horizontal":
                    cur_x -= tile_surface.get_width() + 4
                else:
                    cur_y -= tile_surface.get_height() + 4
            if direction == "horizontal":
                x = cur_x - spacing
            else:
                y = cur_y - spacing

    def _draw_player_layout(
        self,
        target_surface: pygame.Surface,
        player_idx: int,
        face_up_hand: bool,
    ) -> None:
        area = target_surface.get_rect()
        margin_side = 14
        margin_bottom = 14

        tile_size = self._tile_metrics.get("south_hand", (40, 56))
        spacing = tile_size[0] + 6
        draw_gap = 0 # or spacing // 2

        hands = getattr(self, "hands", [])
        hand_tiles = list(hands[player_idx]) if player_idx < len(hands) else []

        x = area.centerx - 7*(tile_size[0] + 6)
        y = area.bottom - tile_size[1] - margin_bottom
        for idx, tile in enumerate(hand_tiles):
            if len(hand_tiles) > 1 and idx == len(hand_tiles) - 1:
                x += draw_gap
            if face_up_hand:
                tile_surface = self._get_tile_surface(tile, tile_size, True, 0)
            else:
                tile_surface = self._get_face_down_surface(tile_size, 0)
            target_surface.blit(tile_surface, (x, y))
            x += spacing

        hand_end_x = x if hand_tiles else margin_side

        discard_tiles = self._get_discard_tiles(player_idx)
        discard_tile = self._tile_metrics.get("discard", tile_size)
        cols = 6
        rows = max(1, (len(discard_tiles) + cols - 1) // cols)
        grid_width = cols * (discard_tile[0] + 4)
        grid_height = rows * (discard_tile[1] + 4)
        discard_rect = pygame.Rect(area.centerx-grid_width/2, 0, grid_width, grid_height)
        discard_rect.top = y - 4 * (discard_tile[1] + 4) - 24 # maximum 3 rows
        self._draw_tile_grid(discard_tiles, discard_rect, discard_tile, 0, cols, target_surface)

        meld_tile = self._tile_metrics.get("meld", tile_size)
        max_meld_width = meld_tile[0] * 4 + 12
        meld_origin_x = area.width - margin_side - max_meld_width
        meld_origin_x = max(margin_side, meld_origin_x)
        meld_origin_y = y
        self._draw_melds(
            player_idx,
            (meld_origin_x, meld_origin_y),
            "horizontal",
            meld_tile,
            0,
            target_surface,
        )

    def _draw_player_areas(self, play_rect: pygame.Rect) -> None:
        if self._screen is None:
            return

        num_players = getattr(self, "num_players", 0)
        if num_players <= 0:
            return

        angle_map = {0: 0, 1: -90, 2: 180, 3: 90}

        for player_idx in range(min(4, num_players)):
            layout_surface = pygame.Surface(play_rect.size, pygame.SRCALPHA)
            layout_surface.fill((0, 0, 0, 0))
            face_up = player_idx == 0
            self._draw_player_layout(layout_surface, player_idx, face_up)
            angle = angle_map.get(player_idx, 0) % 360
            if angle:
                rendered = pygame.transform.rotate(layout_surface, angle)
            else:
                rendered = layout_surface
            rect = rendered.get_rect(center=play_rect.center)
            self._screen.blit(rendered, rect)

    def _draw_walls(self, play_rect: pygame.Rect) -> None:
        wall_tile = self._tile_metrics.get("wall", (20, 26))
        tiles_per_side = 17
        spacing = 4

        # Top and bottom walls
        top_y = play_rect.top - wall_tile[1] - 12
        bottom_y = play_rect.bottom + 12
        available_width = play_rect.width - 40
        horizontal_spacing = max(
            wall_tile[0] + spacing,
            (available_width - wall_tile[0]) / max(1, tiles_per_side - 1),
        )
        for i in range(tiles_per_side):
            x = int(play_rect.left + 20 + i * horizontal_spacing)
            surface = self._get_face_down_surface(wall_tile, 0)
            self._screen.blit(surface, (x, top_y))
            self._screen.blit(surface, (x, bottom_y))

        # Left and right walls
        left_x = play_rect.left - wall_tile[0] - 12
        right_x = play_rect.right + 12
        available_height = play_rect.height - 40
        vertical_spacing = max(
            wall_tile[1] + spacing,
            (available_height - wall_tile[1]) / max(1, tiles_per_side - 1),
        )
        for i in range(tiles_per_side):
            y = int(play_rect.top + 20 + i * vertical_spacing)
            surface = self._get_face_down_surface(wall_tile, 0)
            self._screen.blit(surface, (left_x, y))
            self._screen.blit(surface, (right_x, y))

        self._draw_dead_wall(play_rect, wall_tile)

    def _draw_dead_wall(self, play_rect: pygame.Rect) -> None:
        wall_tile = self._tile_metrics.get("wall", (20, 26))
        stack_size = 5
        gap = 6
        margin_y = 32
        total_height = stack_size * (wall_tile[0] + gap) - gap
        start_x = play_rect.centerx + total_height // 2
        y = play_rect.centery + margin_y

        tile = self.dora_indicator[-1] // 4

        for i in range(stack_size):
            x = start_x - i * (wall_tile[0] + gap) - wall_tile[0]
            if i < len(self.dora_indicator):
                surface = self._get_tile_surface(self.dora_indicator[i], wall_tile, True, 0)
            else:
                surface = self._get_face_down_surface(wall_tile, 0)
            self._screen.blit(surface, (x, y))

    def _draw_seat_labels(self, play_rect: pygame.Rect) -> None:
        if self._small_font is None:
            return

        num_players = getattr(self, "num_players", 0)
        seat_names = list(getattr(self, "seat_names", []))
        if len(seat_names) < num_players:
            seat_names.extend(["NoName"] * (num_players - len(seat_names)))

        label_positions = {
            0: (play_rect.left + 40, play_rect.bottom - 32),
            1: (play_rect.right + 40, play_rect.centery),
            2: (play_rect.right - 40, play_rect.top + 32),
            3: (play_rect.left - 40, play_rect.centery),
        }

        for idx in range(min(4, num_players)):
            name = seat_names[idx] if idx < len(seat_names) else "NoName"
            surface = self._small_font.render(name, True, self._muted_text_color)
            rect = surface.get_rect(center=label_positions[idx])
            self._screen.blit(surface, rect)

        dealer = getattr(self, "oya", 0)
        if dealer >= min(4, num_players):
            return

        offsets = {
            0: (0, -22),
            1: (-22, 0),
            2: (0, 22),
            3: (22, 0),
        }
        marker_center = (
            label_positions[dealer][0] + offsets[dealer][0],
            label_positions[dealer][1] + offsets[dealer][1],
        )
        pygame.draw.circle(self._screen, self._accent_color, marker_center, 10, width=2)
        marker_surface = self._small_font.render("E", True, self._accent_color)
        marker_rect = marker_surface.get_rect(center=marker_center)
        self._screen.blit(marker_surface, marker_rect)

    def _draw_status_text(self, surface_width: int) -> None:
        if self._small_font is None:
            return

        margin = 16
        phase_text = f"Phase: {self.phase}  |  Current Player: P{self.current_player}"
        phase_surface = self._small_font.render(phase_text, True, self._accent_color)
        self._screen.blit(phase_surface, (margin, margin))

        reward_color = self._danger_color if self._last_payload.reward < 0 else self._text_color
        reward_text = f"Action: {self._last_payload.action}  Reward: {self._last_payload.reward:.2f}"
        reward_surface = self._small_font.render(reward_text, True, reward_color)
        reward_rect = reward_surface.get_rect()
        reward_rect.topright = (surface_width - margin, margin)
        self._screen.blit(reward_surface, reward_rect)

        if self._last_payload.done:
            done_surface = self._small_font.render("Episode finished", True, self._danger_color)
            done_rect = done_surface.get_rect()
            done_rect.topright = (surface_width - margin, reward_rect.bottom + 4)
            self._screen.blit(done_surface, done_rect)

    def _draw_control_buttons(self, surface_width: int, surface_height: int) -> None:
        if self._screen is None or self._font is None or self._small_font is None:
            return

        margin = 16
        max_label_width = max(
            self._font.size("Auto Next: OFF")[0],
            self._font.size("Next")[0],
            self._font.size("Pause on Score: ON")[0],
        )
        button_width = max(160, max_label_width + 24)
        button_height = 44
        gap = 10

        base_right = surface_width - margin
        base_bottom = surface_height - margin

        step_rect = pygame.Rect(0, 0, button_width, button_height)
        step_rect.bottomright = (base_right, base_bottom)
        auto_rect = pygame.Rect(0, 0, button_width, button_height)
        auto_rect.bottomright = (base_right, step_rect.top - gap)
        pause_rect = pygame.Rect(0, 0, button_width, button_height)
        pause_rect.bottomright = (base_right, auto_rect.top - gap)

        self._step_button_rect = step_rect
        self._auto_button_rect = auto_rect
        self._pause_button_rect = pause_rect

        def draw_button(rect: pygame.Rect, label: str, enabled: bool, active: bool = False) -> None:
            base_color = self._panel_color
            border_color = self._panel_border
            if active:
                base_color = tuple(min(255, c + 40) for c in self._panel_color)
                border_color = self._accent_color
            elif not enabled:
                base_color = tuple(max(0, c - 20) for c in self._panel_color)
                border_color = self._muted_text_color

            pygame.draw.rect(self._screen, base_color, rect, border_radius=10)
            pygame.draw.rect(self._screen, border_color, rect, 2, border_radius=10)

            font = self._font
            text_color = self._text_color if enabled else self._muted_text_color
            text_surface = font.render(label, True, text_color)
            text_rect = text_surface.get_rect(center=rect.center)
            self._screen.blit(text_surface, text_rect)

        pause_label = "Pause on Score: ON" if self._pause_on_score else "Pause on Score: OFF"
        pause_enabled = self._auto_advance
        draw_button(
            pause_rect,
            pause_label,
            enabled=pause_enabled,
            active=self._pause_on_score and pause_enabled,
        )

        auto_label = "Auto Next: ON" if self._auto_advance else "Auto Next: OFF"
        draw_button(auto_rect, auto_label, enabled=True, active=self._auto_advance)

        step_label = "Next"
        step_enabled = (not self._auto_advance) or self._score_pause_active
        draw_button(step_rect, step_label, enabled=step_enabled, active=self._score_pause_active)

    def _wrap_text(
        self, font: pygame.font.Font, text: str, max_width: int
    ) -> list[str]:
        if not text:
            return []
        max_width = max(10, max_width)
        words = str(text).split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if font.size(candidate)[0] <= max_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    def _draw_score_panel(self, surface_size: Tuple[int, int]) -> None:
        if (
            self._screen is None
            or self._font is None
            or self._small_font is None
            or self._header_font is None
        ):
            return

        surface_width, surface_height = surface_size
        margin = 0
        padding = 14
        min_dimension = min(surface_width, surface_height)
        available_side = min_dimension - margin * 2
        effective_side = available_side if available_side > 0 else min_dimension
        max_text_width = max(50, effective_side - padding * 2)

        round_data = getattr(self, "round", [0, 0])
        round_index = round_data[0] if isinstance(round_data[0], int) else 0
        wind_names = ["East", "South", "West", "North"]
        wind = wind_names[(round_index // 4) % 4]
        hand_number = round_index % 4 + 1
        honba = round_data[1] if len(round_data) > 1 and isinstance(round_data[1], int) else 0
        riichi_sticks = getattr(self, "num_riichi", 0)
        kyoutaku = getattr(self, "num_kyoutaku", 0)

        info_lines: list[str] = [
            f"{wind} {hand_number} | Honba {honba} | Riichi {riichi_sticks} | Kyoutaku {kyoutaku}"
        ]

        seat_names = list(getattr(self, "seat_names", []))
        num_players = getattr(self, "num_players", 0)
        if len(seat_names) < num_players:
            seat_names.extend([f"P{idx}" for idx in range(len(seat_names), num_players)])

        agari = getattr(self, "agari", None)
        message_lines: list[str] = []
        if agari:
            winner = agari.get("who", -1)
            from_who = agari.get("fromwho", -1)
            if 0 <= winner < num_players:
                winner_name = seat_names[winner]
            else:
                winner_name = f"Player {winner}"
            if winner == from_who:
                result_text = f"{winner_name} Tsumo"
            else:
                loser_name = (
                    seat_names[from_who]
                    if 0 <= from_who < num_players
                    else f"Player {from_who}"
                )
                result_text = f"{winner_name} Ron vs {loser_name}"
            message_lines.append(result_text)
            ten = list(agari.get("ten", []))
            fu = ten[0] if len(ten) > 0 else 0
            total = ten[1] if len(ten) > 1 else 0
            han = ten[2] if len(ten) > 2 else 0
            message_lines.append(f"{han} Han | {fu} Fu | {total} Points")
        else:
            tenpai_flags = list(getattr(self, "tenpai", []))
            tenpai_players = [
                seat_names[idx]
                for idx, is_tenpai in enumerate(tenpai_flags)
                if is_tenpai and idx < num_players
            ]
            if tenpai_players:
                message_lines.append("Draw - Tenpai: " + ", ".join(tenpai_players))
            else:
                message_lines.append("Draw - No Tenpai")

        yaku_lines: list[tuple[str, str]] = []
        if agari:
            raw_yaku = [str(item) for item in agari.get("yaku", [])]
            if raw_yaku:
                combined = ", ".join(raw_yaku)
                label = "Yaku: "
                label_width = self._small_font.size(label)[0]
                wrapped_yaku = self._wrap_text(
                    self._small_font, combined, max(10, max_text_width - label_width)
                )
                if wrapped_yaku:
                    yaku_lines.append((label, wrapped_yaku[0]))
                    for extra in wrapped_yaku[1:]:
                        yaku_lines.append(("", extra))

        line_height = self._font.get_linesize() + 4
        small_height = self._small_font.get_linesize() + 2

        player_section_height = num_players * line_height if num_players else 0
        info_height = len(info_lines) * small_height
        message_height = len(message_lines) * small_height
        yaku_height = len(yaku_lines) * small_height

        title_text = "Round Results"
        title_width, title_height = self._header_font.size(title_text)
        content_width = title_width

        if info_lines:
            info_width = max(self._small_font.size(line)[0] for line in info_lines)
            content_width = max(content_width, info_width)
        if message_lines:
            message_width = max(self._small_font.size(line)[0] for line in message_lines)
            content_width = max(content_width, message_width)

        scores = list(getattr(self, "scores", []))
        if len(scores) < num_players:
            scores.extend([0] * (num_players - len(scores)))
        score_deltas = list(getattr(self, "score_deltas", []))
        if len(score_deltas) < num_players:
            score_deltas.extend([0] * (num_players - len(score_deltas)))
        dealer = getattr(self, "oya", -1)

        def ordinal(value: int) -> str:
            suffix = "th"
            if value % 100 not in {11, 12, 13}:
                suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
            return f"{value}{suffix}"

        ranks: dict[int, int] = {}
        sorted_players = sorted(
            range(num_players), key=lambda idx: (-scores[idx], idx)
        )
        last_score: Optional[int] = None
        current_rank = 0
        for position, player_idx in enumerate(sorted_players):
            if player_idx >= len(scores):
                continue
            score_value = scores[player_idx]
            if last_score is None or score_value != last_score:
                current_rank = position + 1
                last_score = score_value
            ranks[player_idx] = current_rank

        winner_idx = agari.get("who") if isinstance(agari, dict) else None

        for player_idx in range(num_players):
            if player_idx >= len(scores):
                continue
            display_name = seat_names[player_idx]
            if player_idx == dealer:
                display_name += " (Dealer)"
            rank_text = ordinal(ranks.get(player_idx, player_idx + 1))
            name_text = f"{rank_text}  {display_name}"
            name_width = self._font.size(name_text)[0]
            delta_value = score_deltas[player_idx] if player_idx < len(score_deltas) else 0
            delta_points = int(round(delta_value * 100))
            delta_width = self._font.size(f"{delta_points:+}")[0]
            score_points = int(round(scores[player_idx] * 100))
            score_width = self._font.size(f"{score_points:>6d}")[0]
            score_line_width = name_width + score_width + delta_width + 16
            content_width = max(content_width, score_line_width)

        if yaku_lines:
            label_width = self._small_font.size("Yaku: ")[0]
            for prefix, text in yaku_lines:
                prefix_width = self._small_font.size(prefix)[0] if prefix else 0
                text_width = self._small_font.size(text)[0]
                total_width = prefix_width + (label_width if not prefix else 0) + text_width
                content_width = max(content_width, total_width)

        panel_height = padding * 2 + title_height
        if info_height:
            panel_height += 6 + info_height
        if message_height:
            panel_height += 6 + message_height
        if player_section_height:
            panel_height += 6 + player_section_height
        if yaku_height:
            panel_height += 6 + yaku_height

        required_width = padding * 2 + content_width
        required_height = panel_height
        panel_size = max(required_width, required_height)
        if available_side > 0:
            panel_size = min(panel_size, available_side)
        else:
            panel_size = min(panel_size, min_dimension)
        panel_rect = pygame.Rect(0, 0, panel_size, panel_size)
        panel_rect.center = (surface_width // 2, surface_height // 2)

        pygame.draw.rect(self._screen, self._panel_color, panel_rect, border_radius=12)
        pygame.draw.rect(self._screen, self._panel_border, panel_rect, 2, border_radius=12)

        current_y = panel_rect.top + padding
        title_surface = self._header_font.render(title_text, True, self._accent_color)
        title_rect = title_surface.get_rect()
        title_rect.midtop = (panel_rect.centerx, current_y)
        self._screen.blit(title_surface, title_rect)
        current_y = title_rect.bottom + 6

        for line in info_lines:
            surface = self._small_font.render(line, True, self._muted_text_color)
            rect = surface.get_rect()
            rect.left = panel_rect.left + padding
            rect.top = current_y
            self._screen.blit(surface, rect)
            current_y = rect.bottom + 2

        if info_lines:
            current_y += 4

        for line in message_lines:
            surface = self._small_font.render(line, True, self._text_color)
            rect = surface.get_rect()
            rect.left = panel_rect.left + padding
            rect.top = current_y
            self._screen.blit(surface, rect)
            current_y = rect.bottom + 2

        if message_lines:
            current_y += 4

        for player_idx in range(num_players):
            if player_idx >= len(scores):
                continue
            display_name = seat_names[player_idx]
            if player_idx == dealer:
                display_name += " (Dealer)"
            base_color = (
                self._accent_color if player_idx == winner_idx else self._text_color
            )
            rank_text = ordinal(ranks.get(player_idx, player_idx + 1))
            name_surface = self._font.render(
                f"{rank_text}  {display_name}", True, base_color
            )
            name_rect = name_surface.get_rect()
            name_rect.left = panel_rect.left + padding
            name_rect.top = current_y
            self._screen.blit(name_surface, name_rect)

            delta_value = score_deltas[player_idx] if player_idx < len(score_deltas) else 0
            delta_points = int(round(delta_value * 100))
            delta_color = (
                self._accent_color
                if delta_points > 0
                else self._danger_color if delta_points < 0 else self._muted_text_color
            )
            delta_surface = self._font.render(f"{delta_points:+}", True, delta_color)
            delta_rect = delta_surface.get_rect()
            delta_rect.topright = (panel_rect.right - padding, current_y)
            self._screen.blit(delta_surface, delta_rect)

            score_points = int(round(scores[player_idx] * 100))
            score_surface = self._font.render(f"{score_points:>6d}", True, base_color)
            score_rect = score_surface.get_rect()
            score_rect.top = current_y
            score_rect.right = delta_rect.left - 16
            self._screen.blit(score_surface, score_rect)

            current_y += line_height

        if num_players:
            current_y += 4

        for prefix, text in yaku_lines:
            label_width = self._small_font.size("Yaku: ")[0]
            x = panel_rect.left + padding
            if prefix:
                label_surface = self._small_font.render(prefix, True, self._accent_color)
                label_rect = label_surface.get_rect()
                label_rect.left = x
                label_rect.top = current_y
                self._screen.blit(label_surface, label_rect)
                x = label_rect.right
            else:
                x += label_width
            text_surface = self._small_font.render(text, True, self._text_color)
            text_rect = text_surface.get_rect()
            text_rect.left = x
            text_rect.top = current_y
            self._screen.blit(text_surface, text_rect)
            current_y = text_rect.bottom + 2

    def _render(self) -> None:
        if self._screen is None or self._font is None or self._clock is None:
            return

        if self._score_last():
            if self._auto_advance and self._pause_on_score:
                if not self._last_phase_is_score_last:
                    self._score_pause_pending = True
                    self._score_pause_active = True
            else:
                self._score_pause_pending = False
                self._score_pause_active = False
            self._last_phase_is_score_last = self._score_last()
        elif self._score_pause_pending:
            if self._auto_advance and self._pause_on_score:
                self._score_pause_active = True
            else:
                self._score_pause_active = False
                self._score_pause_pending = False
        else:
            self._score_pause_pending = False
            self._score_pause_active = False
            self._last_phase_is_score_last = self._score_last()

        self._screen.fill(self._background_color)
        width, height = self._screen.get_size()
        play_size = height
        play_rect = pygame.Rect(0, 0, play_size, play_size)
        play_rect.top = 0
        play_rect.centerx = width // 2

        pygame.draw.rect(self._screen, self._play_area_color, play_rect, border_radius=16)
        pygame.draw.rect(self._screen, self._play_area_border, play_rect, 3, border_radius=16)

        self._compute_tile_metrics(play_rect)
        self._draw_center_panel(play_rect)
        #self._draw_walls(play_rect)
        self._draw_dead_wall(play_rect)
        self._draw_player_areas(play_rect)
        self._draw_seat_labels(play_rect)
        if self._score_last():
            self._draw_score_panel((width, height))
        else:
            self._draw_status_text(width)

        self._draw_control_buttons(width, height)

        pygame.display.flip()
        self._clock.tick(self._fps)

    # ------------------------------------------------------------------
    # Support context manager style usage
    # ------------------------------------------------------------------
    def __enter__(self) -> "MahjongEnv":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
