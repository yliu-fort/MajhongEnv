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
        tile_34: int,
        size: Tuple[int, int],
        face_up: bool = True,
        orientation: int = 0,
    ) -> pygame.Surface:
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
        if player_idx >= len(getattr(self, "discard_pile", [])):
            return []
        tiles = [idx for idx, flagged in enumerate(self.discard_pile[player_idx]) if flagged]
        tiles.sort()
        return [tile // 4 for tile in tiles]

    def _compute_tile_metrics(self, play_rect: pygame.Rect) -> None:
        width = max(1, play_rect.width)
        height = max(1, play_rect.height)

        base_width = max(28, min(width // 22, height // 18, 96))
        base_height = int(base_width * 1.4)
        tile_size = (base_width, base_height)

        self._tile_metrics = {
            "south_hand": tile_size,
            "north_hand": tile_size,
            "side_hand": tile_size,
            "discard": tile_size,
            "wall": tile_size,
            "meld": tile_size,
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
                    else self._get_face_down_surface(tile_size, orientation)
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
        margin_vertical = 24
        grid_spacing = 4

        tile_size = self._tile_metrics.get("south_hand", (40, 56))
        wall_tile = self._tile_metrics.get("wall", tile_size)
        discard_tile = self._tile_metrics.get("discard", tile_size)

        hands = getattr(self, "hands", [])
        hand_tiles = list(hands[player_idx]) if player_idx < len(hands) else []

        cols = 6
        discard_tiles = self._get_discard_tiles(player_idx)
        discard_rows = (
            min(3, max(1, (len(discard_tiles) + cols - 1) // cols)) if discard_tiles else 0
        )
        grid_width = cols * discard_tile[0] + max(0, cols - 1) * grid_spacing
        grid_height = (
            discard_rows * discard_tile[1] + max(0, discard_rows - 1) * grid_spacing
        )

        available_height = max(1, area.height - 2 * margin_vertical)
        total_static = grid_height + wall_tile[1] + tile_size[1]
        remaining = max(0, available_height - total_static)
        section_spacing = max(4, remaining // 3)

        hand_y = area.bottom - margin_vertical - tile_size[1]
        wall_y = hand_y - section_spacing - wall_tile[1]
        discard_bottom = wall_y - section_spacing

        discard_rect = pygame.Rect(0, 0, grid_width, grid_height)
        discard_rect.centerx = area.centerx
        discard_rect.bottom = discard_bottom

        if discard_rect.top < margin_vertical:
            shift = margin_vertical - discard_rect.top
            discard_rect.move_ip(0, shift)
            wall_y += shift
            hand_y += shift

        bottom_limit = area.bottom - margin_vertical
        hand_bottom = hand_y + tile_size[1]
        if hand_bottom > bottom_limit:
            adjust = hand_bottom - bottom_limit
            hand_y -= adjust
            wall_y -= adjust
            discard_rect.move_ip(0, -adjust)

        spacing = tile_size[0] + 6
        draw_gap = spacing // 2

        x = margin_side
        y = hand_y
        for idx, tile in enumerate(hand_tiles):
            if len(hand_tiles) > 1 and idx == len(hand_tiles) - 1:
                x += draw_gap
            if face_up_hand:
                tile_surface = self._get_tile_surface(tile // 4, tile_size, True, 0)
            else:
                tile_surface = self._get_face_down_surface(tile_size, 0)
            target_surface.blit(tile_surface, (x, y))
            x += spacing

        hand_end_x = x if hand_tiles else margin_side

        self._draw_tile_grid(discard_tiles, discard_rect, discard_tile, 0, cols, target_surface)

        wall_count = 17
        wall_spacing = wall_tile[0] + grid_spacing
        total_wall_width = (
            wall_tile[0] + (wall_count - 1) * wall_spacing if wall_count > 0 else 0
        )
        max_width = area.width - 2 * margin_side
        if wall_count > 1 and total_wall_width > max_width:
            wall_spacing = max(1.0, (max_width - wall_tile[0]) / (wall_count - 1))
            total_wall_width = wall_tile[0] + (wall_count - 1) * wall_spacing
        start_x = area.centerx - total_wall_width / 2 if wall_count else area.centerx
        wall_surface = self._get_face_down_surface(wall_tile, 0)
        for i in range(wall_count):
            wall_x = int(round(start_x + i * wall_spacing))
            target_surface.blit(wall_surface, (wall_x, int(round(wall_y))))

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

    def _draw_dead_wall(self, play_rect: pygame.Rect, wall_tile: Tuple[int, int]) -> None:
        dead_wall_tiles = getattr(self, "dead_wall", [])
        stack_size = max(1, len(dead_wall_tiles) // 2) if dead_wall_tiles else 7
        gap = 6
        total_height = stack_size * (wall_tile[1] + gap) - gap
        margin = 24
        x = play_rect.right - margin - wall_tile[0]
        start_y = play_rect.centery - total_height // 2
        start_y = max(play_rect.top + margin, start_y)
        max_y = play_rect.bottom - margin - wall_tile[1]
        if start_y + total_height > max_y:
            start_y = max(play_rect.top + margin, max_y - total_height)

        for i in range(stack_size):
            y = start_y + i * (wall_tile[1] + gap)
            surface = self._get_face_down_surface(wall_tile, 0)
            self._screen.blit(surface, (x, y))

        if getattr(self, "dora_indicator", []):
            tile = self.dora_indicator[-1] // 4
            surface = self._get_tile_surface(tile, wall_tile, True, 270)
            rect = surface.get_rect()
            stack_center = start_y + (stack_size - 1) * (wall_tile[1] + gap) / 2 + wall_tile[1] / 2
            rect.midright = (x - 16, int(round(stack_center)))
            self._screen.blit(surface, rect)

    def _draw_seat_labels(self, play_rect: pygame.Rect) -> None:
        if self._small_font is None:
            return

        num_players = getattr(self, "num_players", 0)
        seat_names = list(getattr(self, "seat_names", []))
        if len(seat_names) < num_players:
            seat_names.extend(["NoName"] * (num_players - len(seat_names)))

        margin = 40
        label_positions = {
            0: (play_rect.centerx, play_rect.bottom - margin),
            1: (play_rect.right - margin, play_rect.centery),
            2: (play_rect.centerx, play_rect.top + margin),
            3: (play_rect.left + margin, play_rect.centery),
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
        play_rect = pygame.Rect(0, 0, width, height)

        pygame.draw.rect(self._screen, self._play_area_color, play_rect)
        pygame.draw.rect(self._screen, self._play_area_border, play_rect, 3)

        self._compute_tile_metrics(play_rect)
        self._draw_center_panel(play_rect)
        self._draw_player_areas(play_rect)
        wall_tile = self._tile_metrics.get("wall", (40, 56))
        self._draw_dead_wall(play_rect, wall_tile)
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
