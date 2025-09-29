from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

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
        self._line_height = font_size + 6
        self._quit_requested = False
        self._last_payload = _RenderPayload(action=None, reward=0.0, done=False, info={})
        self._asset_root = Path(__file__).resolve().parent.parent / "data" / "assets"
        self._raw_tile_assets: dict[int, pygame.Surface] = {}
        self._tile_cache: dict[int, pygame.Surface] = {}
        self._tile_orientation_cache: dict[tuple[int, int], pygame.Surface] = {}
        self._face_down_cache: dict[int, pygame.Surface] = {}
        self._placeholder_cache: dict[int, pygame.Surface] = {}
        self._tile_metrics: dict[str, Tuple[int, int]] = {}
        self._tile_base_size: Tuple[int, int] = (64, 88)
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
        small_size = max(12, self._font_size - 4)
        self._small_font = pygame.font.SysFont(self._font_name, small_size)
        self._header_font = pygame.font.SysFont(self._font_name, self._font_size + 10)
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
                surface = pygame.image.load(str(path))
                surface = surface.convert_alpha()
            except Exception:
                continue
            self._raw_tile_assets[tile_index] = surface

        if self._raw_tile_assets:
            first_surface = next(iter(self._raw_tile_assets.values()))
            self._tile_base_size = first_surface.get_size()

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

    def _create_tile_placeholder(self, tile_34: int) -> pygame.Surface:
        if tile_34 in self._placeholder_cache:
            return self._placeholder_cache[tile_34]

        width, height = self._tile_base_size
        surface = pygame.Surface((width, height), pygame.SRCALPHA)
        surface.fill(self._get_tile_color(tile_34))
        pygame.draw.rect(surface, (245, 245, 245), surface.get_rect(), 2, border_radius=6)

        label = _TILE_SYMBOLS[tile_34]
        font_size = max(12, int(height * 0.45))
        font = pygame.font.SysFont(self._font_name, font_size)
        text = font.render(label, True, (20, 20, 20))
        text_rect = text.get_rect(center=surface.get_rect().center)
        surface.blit(text, text_rect)

        self._placeholder_cache[tile_34] = surface
        return surface

    def _get_tile_surface(
        self,
        tile_34: int,
        size: Tuple[int, int],
        face_up: bool = True,
        orientation: int = 0,
    ) -> pygame.Surface:
        del size

        if not face_up:
            return self._get_face_down_surface(orientation)

        cache_key = tile_34
        if cache_key not in self._tile_cache:
            if tile_34 in self._raw_tile_assets:
                surface = self._raw_tile_assets[tile_34].copy()
            else:
                surface = self._create_tile_placeholder(tile_34)
            self._tile_cache[cache_key] = surface

        orientation = orientation % 360
        if orientation == 0:
            return self._tile_cache[cache_key]

        orient_key = (tile_34, orientation)
        if orient_key not in self._tile_orientation_cache:
            self._tile_orientation_cache[orient_key] = pygame.transform.rotate(
                self._tile_cache[cache_key], orientation
            )
        return self._tile_orientation_cache[orient_key]

    def _get_face_down_surface(self, orientation: int = 0) -> pygame.Surface:
        orientation = orientation % 360
        if 0 not in self._face_down_cache:
            surface = pygame.Surface(self._tile_base_size, pygame.SRCALPHA)
            surface.fill(self._face_down_color)
            pygame.draw.rect(surface, self._face_down_border, surface.get_rect(), 2, border_radius=6)
            self._face_down_cache[0] = surface

        if orientation == 0:
            return self._face_down_cache[0]

        if orientation not in self._face_down_cache:
            self._face_down_cache[orientation] = pygame.transform.rotate(
                self._face_down_cache[0], orientation
            )
        return self._face_down_cache[orientation]

    def _get_discard_tiles(self, player_idx: int) -> list[int]:
        if player_idx >= len(getattr(self, "discard_pile", [])):
            return []
        tiles = [idx for idx, flagged in enumerate(self.discard_pile[player_idx]) if flagged]
        tiles.sort()
        return [tile // 4 for tile in tiles]

    def _compute_tile_metrics(self, play_rect: pygame.Rect) -> None:
        del play_rect
        tile_size = self._tile_base_size
        self._tile_metrics = {
            "south_hand": tile_size,
            "north_hand": tile_size,
            "side_hand": tile_size,
            "discard": tile_size,
            "wall": tile_size,
            "meld": tile_size,
            "tile": tile_size,
        }

    def _draw_center_panel(self, play_rect: pygame.Rect) -> pygame.Rect:
        center_width = max(100, int(play_rect.width * 0.2))
        center_height = max(100, int(play_rect.height * 0.2))
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
            tiles = [tile // 4 for tile in meld.get("m", [])]
            opened = meld.get("opened", True)
            cur_x, cur_y = x, y
            for tile in tiles:
                tile_surface = (
                    self._get_tile_surface(tile, tile_size, True, orientation)
                    if opened
                    else self._get_face_down_surface(orientation)
                )
                surface.blit(tile_surface, (cur_x, cur_y))
                if direction == "horizontal":
                    cur_x += tile_surface.get_width() + 4
                else:
                    cur_y += tile_surface.get_height() + 4
            if direction == "horizontal":
                x = cur_x + spacing
            else:
                y = cur_y + spacing

    def _draw_player_layout(
        self,
        target_surface: pygame.Surface,
        player_idx: int,
        face_up_hand: bool,
    ) -> None:
        area = target_surface.get_rect()
        margin_side = 28
        margin_top = 28
        margin_bottom = 28

        tile_size = self._tile_metrics.get("south_hand", self._tile_base_size)
        tile_width, tile_height = tile_size
        spacing = tile_width + 6
        draw_gap = spacing // 2
        section_gap = max(8, tile_height // 6)
        grid_spacing = 4

        hands = getattr(self, "hands", [])
        hand_tiles = list(hands[player_idx]) if player_idx < len(hands) else []

        discard_tiles = self._get_discard_tiles(player_idx)
        discard_count = len(discard_tiles)
        max_cols = max(1, (area.width - 2 * margin_side + grid_spacing) // (tile_width + grid_spacing))
        cols = min(6, max_cols)
        cols = max(1, cols)
        if discard_count:
            cols = min(cols, discard_count)
            rows = (discard_count + cols - 1) // cols
            while rows > 3 and cols < discard_count and cols < max_cols:
                cols += 1
                rows = (discard_count + cols - 1) // cols
            rows = max(1, min(rows, 3))
        else:
            rows = 0
        grid_width = cols * tile_width + max(0, cols - 1) * grid_spacing
        grid_height = rows * tile_height + max(0, rows - 1) * grid_spacing
        discard_left = area.centerx - grid_width // 2
        discard_left = max(margin_side, min(area.width - margin_side - grid_width, discard_left))
        discard_top = margin_top
        discard_rect = pygame.Rect(int(discard_left), int(discard_top), int(grid_width), int(grid_height))

        wall_tiles_total = len(getattr(self, "deck", []))
        wall_count = min(34, wall_tiles_total) if wall_tiles_total else 34
        wall_spacing = 4
        max_wall_cols = max(1, (area.width - 2 * margin_side + wall_spacing) // (tile_width + wall_spacing))
        wall_cols = min(17, max_wall_cols)
        wall_cols = max(1, wall_cols)
        wall_rows = max(1, (wall_count + wall_cols - 1) // wall_cols)
        wall_width = wall_cols * tile_width + max(0, wall_cols - 1) * wall_spacing
        wall_grid_height = wall_rows * tile_height + max(0, wall_rows - 1) * wall_spacing
        wall_left = area.centerx - wall_width // 2
        wall_left = max(margin_side, min(area.width - margin_side - wall_width, wall_left))

        available_height = area.height - margin_top - margin_bottom
        required_height = grid_height + wall_grid_height + tile_height + 2 * section_gap
        if required_height > available_height:
            reduction = required_height - available_height
            section_gap = max(2, section_gap - reduction // 2)
            required_height = grid_height + wall_grid_height + tile_height + 2 * section_gap
            if required_height > available_height:
                section_gap = max(2, section_gap - (required_height - available_height))

        discard_top = margin_top
        wall_top = discard_top + grid_height + section_gap
        hand_top = wall_top + wall_grid_height + section_gap
        max_hand_top = area.bottom - margin_bottom - tile_height
        if hand_top > max_hand_top:
            shift = hand_top - max_hand_top
            discard_top = max(margin_top, discard_top - shift)
            wall_top -= shift
            hand_top -= shift

        discard_rect.top = int(discard_top)
        if discard_count:
            self._draw_tile_grid(discard_tiles, discard_rect, tile_size, 0, cols, target_surface)

        wall_surface = self._get_face_down_surface(0)
        wall_tile_width, wall_tile_height = wall_surface.get_size()
        for idx in range(wall_count):
            column = idx % wall_cols
            row = idx // wall_cols
            x = wall_left + column * (wall_tile_width + wall_spacing)
            y = wall_top + row * (wall_tile_height + wall_spacing)
            target_surface.blit(wall_surface, (x, y))

        x = margin_side
        y = hand_top
        for idx, tile in enumerate(hand_tiles):
            if len(hand_tiles) > 1 and idx == len(hand_tiles) - 1:
                x += draw_gap
            if face_up_hand:
                tile_surface = self._get_tile_surface(tile // 4, tile_size, True, 0)
            else:
                tile_surface = self._get_face_down_surface(0)
            target_surface.blit(tile_surface, (x, y))
            x += spacing

        hand_end_x = x if hand_tiles else margin_side

        meld_tile = self._tile_metrics.get("meld", tile_size)
        max_meld_width = meld_tile[0] * 4 + 12
        meld_origin_x = hand_end_x + 16
        max_right = area.width - margin_side - max_meld_width
        if meld_origin_x > max_right:
            meld_origin_x = max_right
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

    def _draw_dead_wall(self, anchor_rect: pygame.Rect) -> None:
        if self._screen is None:
            return

        dead_wall_tiles = getattr(self, "dead_wall", [])
        wall_surface = self._get_face_down_surface(0)
        tile_width, tile_height = wall_surface.get_size()
        wall_tile = self._tile_metrics.get("tile", self._tile_base_size)
        stack_size = max(1, len(dead_wall_tiles) // 2) if dead_wall_tiles else 7
        gap = 6
        total_height = stack_size * (tile_height + gap) - gap
        screen_rect = self._screen.get_rect()
        x = anchor_rect.right + 24
        if x + tile_width > screen_rect.right - 24:
            x = anchor_rect.left - tile_width - 24
        x = max(screen_rect.left + 24, min(x, screen_rect.right - tile_width - 24))
        start_y = anchor_rect.centery - total_height // 2
        start_y = max(screen_rect.top + 24, min(start_y, screen_rect.bottom - total_height - 24))

        for i in range(stack_size):
            y = start_y + i * (tile_height + gap)
            self._screen.blit(wall_surface, (x, y))

        if getattr(self, "dora_indicator", []):
            tile = self.dora_indicator[-1] // 4
            surface = self._get_tile_surface(tile, wall_tile, True, 270)
            rect = surface.get_rect()
            rect.midleft = (
                x + tile_width + 12,
                start_y + (stack_size - 1) * (tile_height + gap) + tile_height // 2,
            )
            if rect.right > screen_rect.right - 16:
                rect.right = screen_rect.right - 16
            if rect.top < screen_rect.top + 16:
                rect.top = screen_rect.top + 16
            if rect.bottom > screen_rect.bottom - 16:
                rect.bottom = screen_rect.bottom - 16
            self._screen.blit(surface, rect)

    def _draw_seat_labels(self, play_rect: pygame.Rect) -> None:
        if self._small_font is None:
            return

        num_players = getattr(self, "num_players", 0)
        seat_names = list(getattr(self, "seat_names", []))
        if len(seat_names) < num_players:
            seat_names.extend(["NoName"] * (num_players - len(seat_names)))

        label_margin = 30
        label_positions = {
            0: (play_rect.centerx, play_rect.bottom - label_margin),
            1: (play_rect.right - label_margin, play_rect.centery),
            2: (play_rect.centerx, play_rect.top + label_margin),
            3: (play_rect.left + label_margin, play_rect.centery),
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
            0: (0, -18),
            1: (-18, 0),
            2: (0, 18),
            3: (18, 0),
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

        info_text = self._last_payload.info.get("msg") or getattr(self, "msg", "")
        if info_text:
            info_surface = self._small_font.render(str(info_text), True, self._text_color)
            self._screen.blit(info_surface, (margin, margin + phase_surface.get_height() + 6))

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

    def _render(self) -> None:
        if self._screen is None or self._font is None or self._clock is None:
            return

        self._screen.fill(self._background_color)
        width, height = self._screen.get_size()
        play_rect = self._screen.get_rect()

        pygame.draw.rect(self._screen, self._play_area_color, play_rect)
        pygame.draw.rect(self._screen, self._play_area_border, play_rect, 3)

        self._compute_tile_metrics(play_rect)
        center_rect = self._draw_center_panel(play_rect)
        self._draw_player_areas(play_rect)
        self._draw_dead_wall(center_rect)
        self._draw_seat_labels(play_rect)
        self._draw_status_text(width)

        pygame.display.flip()
        self._clock.tick(self._fps)

    # ------------------------------------------------------------------
    # Support context manager style usage
    # ------------------------------------------------------------------
    def __enter__(self) -> "MahjongEnv":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
