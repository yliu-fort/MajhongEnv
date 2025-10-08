from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Tuple

try:
    from kivy.clock import Clock
    from kivy.lang import Builder
    from kivy.properties import (
        BooleanProperty,
        ListProperty,
        NumericProperty,
        ObjectProperty,
        StringProperty,
    )
    from kivy.graphics import (
        Color,
        Line,
        PopMatrix,
        PushMatrix,
        RoundedRectangle,
        Rotate,
        Scale,
        Translate,
    )
    from kivy.graphics.svg import Svg
    from kivy.uix.boxlayout import BoxLayout
    from kivy.uix.floatlayout import FloatLayout
    from kivy.uix.label import Label
    from kivy.uix.widget import Widget
    from kivy.metrics import dp
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise RuntimeError(
        "kivy is required for the MahjongEnv Kivy GUI wrapper; install kivy to continue"
    ) from exc

from mahjong_env import MahjongEnvBase as _BaseMahjongEnv
from mahjong_features import get_action_from_index
from mahjong_tiles_print_style import tile_printout, tiles_printout


COLOR_BACKGROUND = [12 / 255.0, 30 / 255.0, 60 / 255.0, 1.0]
COLOR_PLAY_AREA = [24 / 255.0, 60 / 255.0, 90 / 255.0, 1.0]
COLOR_PLAY_AREA_BORDER = [40 / 255.0, 90 / 255.0, 130 / 255.0, 1.0]
COLOR_PANEL = [5 / 255.0, 5 / 255.0, 5 / 255.0, 0.9]
COLOR_PANEL_BORDER = [90 / 255.0, 120 / 255.0, 160 / 255.0, 1.0]
COLOR_ACCENT = [170 / 255.0, 230 / 255.0, 255 / 255.0, 1.0]
COLOR_TEXT = [235 / 255.0, 235 / 255.0, 235 / 255.0, 1.0]
COLOR_MUTED = [170 / 255.0, 190 / 255.0, 210 / 255.0, 1.0]
COLOR_DANGER = [220 / 255.0, 120 / 255.0, 120 / 255.0, 1.0]


_KV_PATH = Path(__file__).resolve().with_name("mahjong_gui.kv")
_KV_LOADED = False


_ASSET_ROOT = Path(__file__).resolve().parent.parent / "assets" / "tiles"
_DEFAULT_TILE_SET = "Regular"

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


def load_kivy_layout() -> None:
    """Ensure the Kivy .kv layout used by the Mahjong GUI is loaded."""

    global _KV_LOADED
    if not _KV_LOADED:
        if not _KV_PATH.exists():
            raise FileNotFoundError(f"Missing Kivy layout file: {_KV_PATH}")
        Builder.load_file(str(_KV_PATH))
        _KV_LOADED = True


@dataclass(slots=True)
class _RenderPayload:
    action: Optional[int]
    reward: float
    done: bool
    info: dict[str, Any]


def _tile_symbol_from_136(tile_136: int) -> str:
    tile_34 = tile_136 // 4 if tile_136 >= 0 else 0
    if tile_136 == 16:
        tile_34 = 34
    elif tile_136 == 52:
        tile_34 = 35
    elif tile_136 == 88:
        tile_34 = 36
    tile_34 = max(0, min(tile_34, len(_TILE_SYMBOLS) - 1))
    return _TILE_SYMBOLS[tile_34]


def _tile_label_from_136(tile_136: int) -> str:
    if tile_136 < 0:
        return ""
    try:
        return tile_printout(tile_136).strip()
    except Exception:
        return str(tile_136)


class TileFace(FloatLayout):
    """Widget that renders a Mahjong tile using SVG or a placeholder."""

    tile_id = NumericProperty(-1)
    face_up = BooleanProperty(True)
    orientation = NumericProperty(0)
    tile_set = StringProperty(_DEFAULT_TILE_SET)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.size_hint = kwargs.get("size_hint", (None, None))
        self._label = Label(
            text="",
            color=(0.1, 0.1, 0.1, 1.0),
            halign="center",
            valign="middle",
            size_hint=(1.0, 1.0),
        )
        self._label.bind(size=lambda inst, val: setattr(inst, "text_size", val))
        if self._label.parent is None:
            self.add_widget(self._label)
        self._current_svg: Optional[Svg] = None
        self._update_trigger = Clock.create_trigger(self._refresh_canvas)
        self.bind(
            tile_id=self._update_trigger,
            face_up=self._update_trigger,
            orientation=self._update_trigger,
            tile_set=self._update_trigger,
            pos=self._update_trigger,
            size=self._update_trigger,
        )
        Clock.schedule_once(lambda _dt: self._refresh_canvas(), 0)

    # ------------------------------------------------------------------
    def _tile_asset_path(self) -> Optional[Path]:
        if not self.face_up:
            back_path = _ASSET_ROOT / _DEFAULT_TILE_SET / "Back.svg"
            return back_path if back_path.exists() else None
        symbol = _tile_symbol_from_136(self.tile_id)
        candidate = _ASSET_ROOT / self.tile_set / f"{symbol}.svg"
        if candidate.exists():
            return candidate
        return None

    def _placeholder_color(self) -> tuple[float, float, float, float]:
        if not self.face_up:
            return (18 / 255.0, 18 / 255.0, 22 / 255.0, 1.0)
        symbol = _tile_symbol_from_136(self.tile_id)
        suit = symbol[-1] if symbol[-1] in {"m", "p", "s"} else symbol[-1:].lower()
        palette = {
            "m": (210 / 255.0, 90 / 255.0, 90 / 255.0, 1.0),
            "p": (90 / 255.0, 150 / 255.0, 225 / 255.0, 1.0),
            "s": (90 / 255.0, 190 / 255.0, 120 / 255.0, 1.0),
        }
        return palette.get(suit, (220 / 255.0, 210 / 255.0, 150 / 255.0, 1.0))

    def _draw_placeholder(self) -> None:
        label = _tile_label_from_136(self.tile_id) if self.face_up else ""
        self._label.text = label
        radius = dp(6)
        Color(*self._placeholder_color())
        RoundedRectangle(pos=(0, 0), size=self.size, radius=[radius])
        Color(245 / 255.0, 245 / 255.0, 245 / 255.0, 1.0)
        Line(rounded_rectangle=(0, 0, self.width, self.height, radius), width=dp(1.4))

    def _draw_svg(self, path: Path) -> None:
        try:
            svg = Svg(str(path))
        except Exception:
            self._draw_placeholder()
            return

        width = max(1.0, float(svg.width or 1.0))
        height = max(1.0, float(svg.height or 1.0))
        scale_x = self.width / width
        scale_y = self.height / height
        scale = min(scale_x, scale_y)
        offset_x = (self.width - width * scale) / 2.0
        offset_y = (self.height - height * scale) / 2.0
        PushMatrix()
        Translate(offset_x, offset_y)
        Scale(scale, scale, 1.0)
        self._current_svg = svg
        self.canvas.before.add(svg)
        PopMatrix()
        self._label.text = ""

    def _refresh_canvas(self, *_: Any) -> None:
        self.canvas.before.clear()
        with self.canvas.before:
            PushMatrix()
            Translate(self.x + self.width / 2.0, self.y + self.height / 2.0)
            angle = (self.orientation or 0) % 360
            if angle:
                Rotate(angle=angle)
            Translate(-self.width / 2.0, -self.height / 2.0)
            asset = self._tile_asset_path()
            if asset is not None:
                self._draw_svg(asset)
            else:
                self._draw_placeholder()
            PopMatrix()


class TileStrip(BoxLayout):
    """Layout that renders a horizontal strip of Mahjong tiles."""

    tiles = ListProperty([])
    tile_size = ListProperty([dp(42), dp(60)])
    placeholder_text = StringProperty("")
    muted_text_color = ListProperty(list(COLOR_MUTED))

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("orientation", "horizontal")
        kwargs.setdefault("spacing", dp(4))
        kwargs.setdefault("size_hint_y", None)
        super().__init__(**kwargs)
        self.height = self.tile_size[1]
        self.size_hint_x = kwargs.get("size_hint_x", None)
        self.bind(
            tiles=self._refresh_tiles,
            tile_size=self._on_tile_size,
            placeholder_text=self._refresh_tiles,
            minimum_width=self._update_width,
        )
        Clock.schedule_once(lambda _dt: self._refresh_tiles(), 0)

    def _update_width(self, *_: Any) -> None:
        if self.size_hint_x is None:
            self.width = max(self.minimum_width, 1)

    def _on_tile_size(self, *_: Any) -> None:
        self.height = self.tile_size[1]
        for child in self.children:
            if isinstance(child, TileFace):
                child.size = (self.tile_size[0], self.tile_size[1])
        self._refresh_tiles()

    def _normalize_tiles(self) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for entry in list(self.tiles or []):
            if isinstance(entry, dict):
                tile_id = int(entry.get("tile", -1))
                face_up = bool(entry.get("face_up", True))
                orientation = int(entry.get("orientation", 0))
            else:
                try:
                    tile_id = int(entry)
                except (TypeError, ValueError):
                    tile_id = -1
                face_up = True
                orientation = 0
            normalized.append(
                {
                    "tile": tile_id,
                    "face_up": face_up,
                    "orientation": orientation,
                }
            )
        return normalized

    def _refresh_tiles(self, *_: Any) -> None:
        self.clear_widgets()
        tiles = self._normalize_tiles()
        if not tiles:
            if self.placeholder_text:
                label = Label(
                    text=self.placeholder_text,
                    color=self.muted_text_color,
                    halign="left",
                    valign="middle",
                    size_hint=(None, None),
                    height=self.tile_size[1],
                )
                label.bind(
                    texture_size=lambda inst, value: setattr(
                        inst, "size", (max(self.tile_size[0], value[0]), self.tile_size[1])
                    )
                )
                self.add_widget(label)
            self._update_width()
            return

        for tile in tiles:
            widget = TileFace(
                tile_id=tile.get("tile", -1),
                face_up=tile.get("face_up", True),
                orientation=tile.get("orientation", 0),
            )
            widget.size_hint = (None, None)
            widget.size = (self.tile_size[0], self.tile_size[1])
            self.add_widget(widget)
        self._update_width()


class MeldList(BoxLayout):
    """Displays melds using horizontal tile strips."""

    melds = ListProperty([])
    placeholder_text = StringProperty("Melds: -")
    muted_text_color = ListProperty(list(COLOR_MUTED))
    tile_size = ListProperty([dp(38), dp(54)])

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("orientation", "vertical")
        kwargs.setdefault("spacing", dp(4))
        kwargs.setdefault("size_hint_y", None)
        super().__init__(**kwargs)
        self.bind(
            melds=self._refresh_melds,
            tile_size=self._refresh_melds,
            placeholder_text=self._refresh_melds,
            minimum_height=self.setter("height"),
        )
        Clock.schedule_once(lambda _dt: self._refresh_melds(), 0)

    def _refresh_melds(self, *_: Any) -> None:
        self.clear_widgets()
        melds = list(self.melds or [])
        if not melds:
            label = Label(
                text=self.placeholder_text,
                color=self.muted_text_color,
                size_hint_y=None,
                height=dp(20),
                halign="left",
                valign="middle",
            )
            label.bind(size=lambda inst, val: setattr(inst, "text_size", (val[0], None)))
            self.add_widget(label)
            return

        for meld in melds:
            row = BoxLayout(
                orientation="horizontal",
                spacing=dp(6),
                size_hint_y=None,
                height=self.tile_size[1],
            )
            label_text = meld.get("label", "")
            label_widget = Label(
                text=label_text,
                color=self.muted_text_color,
                size_hint_x=None,
                width=dp(120),
                halign="left",
                valign="middle",
            )
            label_widget.bind(size=lambda inst, val: setattr(inst, "text_size", val))
            row.add_widget(label_widget)
            strip = TileStrip(
                tiles=meld.get("tiles", []),
                tile_size=self.tile_size,
                placeholder_text="",
            )
            strip.size_hint = (1, None)
            strip.height = self.tile_size[1]
            row.add_widget(strip)
            self.add_widget(row)

class PlayerPanel(BoxLayout):
    """Panel widget used to present one player's state."""

    display_name = StringProperty("")
    seat_label = StringProperty("")
    hand_text = StringProperty("")
    hand_placeholder = StringProperty("-")
    discard_text = StringProperty("")
    discard_placeholder = StringProperty("-")
    meld_text = StringProperty("")
    meld_placeholder = StringProperty("Melds: -")
    score_text = StringProperty("")
    score_delta_text = StringProperty("")
    wind_text = StringProperty("")
    riichi = BooleanProperty(False)
    is_current = BooleanProperty(False)
    dealer = BooleanProperty(False)
    accent_color = ListProperty(list(COLOR_ACCENT))
    text_color = ListProperty(list(COLOR_TEXT))
    muted_text_color = ListProperty(list(COLOR_MUTED))
    danger_color = ListProperty(list(COLOR_DANGER))
    hand_tiles = ListProperty([])
    discard_tiles = ListProperty([])
    melds_data = ListProperty([])

    def update_from_dict(self, data: dict[str, Any]) -> None:
        self.display_name = data.get("name", "")
        self.seat_label = data.get("seat_label", "")
        self.hand_text = data.get("hand_text") or data.get("hand", "")
        self.hand_placeholder = data.get("hand_placeholder", self.hand_text or "-")
        self.discard_text = data.get("discard_text") or data.get("discards", "")
        self.discard_placeholder = data.get(
            "discard_placeholder", self.discard_text or "-"
        )
        self.meld_text = data.get("meld_text") or data.get("melds", "")
        self.meld_placeholder = data.get("meld_placeholder", self.meld_text or "Melds: -")
        self.score_text = data.get("score", "")
        self.score_delta_text = data.get("score_delta", "")
        self.wind_text = data.get("wind", "")
        self.riichi = bool(data.get("riichi", False))
        self.is_current = bool(data.get("is_current", False))
        self.dealer = bool(data.get("dealer", False))
        self.hand_tiles = list(data.get("hand_tiles", []))
        self.discard_tiles = list(data.get("discard_tiles", []))
        self.melds_data = list(data.get("melds_data", []))


class MahjongTable(Widget):
    """Central table canvas showing the play-field and round information."""

    background_color = ListProperty(list(COLOR_BACKGROUND))
    play_area_color = ListProperty(list(COLOR_PLAY_AREA))
    play_area_border = ListProperty(list(COLOR_PLAY_AREA_BORDER))
    panel_color = ListProperty(list(COLOR_PANEL))
    panel_border = ListProperty(list(COLOR_PANEL_BORDER))
    accent_color = ListProperty(list(COLOR_ACCENT))
    text_color = ListProperty(list(COLOR_TEXT))
    muted_text_color = ListProperty(list(COLOR_MUTED))
    danger_color = ListProperty(list(COLOR_DANGER))
    play_area_pos = ListProperty([0.0, 0.0])
    play_area_size = ListProperty([0.0, 0.0])
    center_panel_pos = ListProperty([0.0, 0.0])
    center_panel_size = ListProperty([0.0, 0.0])
    score_positions = ListProperty([[0.0, 0.0] for _ in range(4)])
    label_positions = ListProperty([[0.0, 0.0] for _ in range(4)])
    score_strings = ListProperty(["", "", "", ""])
    score_colors = ListProperty([])
    seat_labels = ListProperty([])
    dealer = NumericProperty(-1)
    current_player = NumericProperty(-1)
    dealer_marker_pos = ListProperty([0.0, 0.0])
    dealer_marker_visible = BooleanProperty(False)
    round_label = StringProperty("")
    honba_text = StringProperty("")
    riichi_text = StringProperty("")
    tiles_text = StringProperty("")
    dora_text = StringProperty("")
    ura_text = StringProperty("")
    dora_tiles = ListProperty([])
    ura_tiles = ListProperty([])

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.bind(size=self._update_layout, pos=self._update_layout)
        self.bind(dealer=self._update_layout)
        Clock.schedule_once(lambda _: self._update_layout(), 0)

    def _update_layout(self, *_: Any) -> None:
        width = float(max(1, int(self.width)))
        height = float(max(1, int(self.height)))
        margin = dp(24)
        min_dim = max(0.0, min(width, height))
        available = max(0.0, min_dim - margin * 2)
        play_side = max(dp(240), available)
        max_side = max(0.0, min_dim - margin)
        if max_side > 0:
            play_side = min(play_side, max_side)
        play_side = min(play_side, min_dim)
        center_x = self.x + width / 2.0
        center_y = self.y + height / 2.0
        play_x = center_x - play_side / 2.0
        play_y = center_y - play_side / 2.0
        self.play_area_pos = [play_x, play_y]
        self.play_area_size = [play_side, play_side]

        center_size = max(dp(140), play_side * 0.24)
        center_pos = [center_x - center_size / 2.0, center_y - center_size / 2.0]
        self.center_panel_pos = center_pos
        self.center_panel_size = [center_size, center_size]

        self.score_positions = [
            [center_x, center_pos[1] - dp(14)],
            [center_pos[0] + center_size + dp(18), center_y],
            [center_x, center_pos[1] + center_size + dp(14)],
            [center_pos[0] - dp(18), center_y],
        ]

        self.label_positions = [
            [play_x + dp(40), play_y - dp(28)],
            [play_x + play_side + dp(40), center_y],
            [play_x + play_side - dp(40), play_y + play_side + dp(28)],
            [play_x - dp(40), center_y],
        ]

        offsets = [
            (0.0, -dp(22)),
            (-dp(22), 0.0),
            (0.0, dp(22)),
            (dp(22), 0.0),
        ]
        dealer = int(self.dealer)
        if 0 <= dealer < len(offsets):
            base_x, base_y = self.label_positions[dealer]
            off_x, off_y = offsets[dealer]
            self.dealer_marker_pos = [base_x + off_x, base_y + off_y]
            self.dealer_marker_visible = True
        else:
            self.dealer_marker_visible = False


class MahjongBoard(BoxLayout):
    """Composite widget combining player panels and the Mahjong table."""

    def apply_state(
        self,
        players: Sequence[dict[str, Any]],
        board_info: dict[str, Any],
    ) -> None:
        order = [0, 1, 2, 3]
        ids_map = [
            self.ids.get("south_panel"),
            self.ids.get("east_panel"),
            self.ids.get("north_panel"),
            self.ids.get("west_panel"),
        ]
        for idx, panel in zip(order, ids_map):
            if panel is None:
                continue
            player_data = players[idx] if idx < len(players) else {}
            panel.update_from_dict(player_data)
            panel.opacity = 1.0 if player_data else 0.25

        table = self.ids.get("table")
        if table is None:
            return

        table.round_label = board_info.get("round_label", "")
        table.honba_text = f"Honba {board_info.get('honba', 0)}"
        table.riichi_text = f"Riichi {board_info.get('riichi_sticks', 0)}"
        table.tiles_text = f"Tiles {board_info.get('tiles_remaining', 0)}"
        table.dora_text = board_info.get("dora_text", "")
        table.ura_text = board_info.get("ura_text", "")
        table.dora_tiles = list(board_info.get("dora_tiles", []))
        table.ura_tiles = list(board_info.get("ura_tiles", []))
        table.seat_labels = list(board_info.get("seat_names", []))[:4]
        table.dealer = int(board_info.get("dealer", -1))
        table.current_player = int(board_info.get("current_player", -1))

        scores = [int(s) for s in board_info.get("scores", [])[:4]]
        score_strings = []
        score_colors = []
        for idx in range(4):
            if idx < len(scores):
                score_strings.append(f"{scores[idx]:5d}")
                if idx == table.current_player:
                    score_colors.append(list(table.accent_color))
                else:
                    score_colors.append(list(table.text_color))
            else:
                score_strings.append("")
                score_colors.append(list(table.muted_text_color))
        table.score_strings = score_strings
        table.score_colors = score_colors


class MahjongRoot(BoxLayout):
    """Root widget hosting the Mahjong GUI."""

    wrapper = ObjectProperty(None, allownone=True)
    status_text = StringProperty("")
    phase_text = StringProperty("")
    reward_text = StringProperty("")
    log_text = StringProperty("")
    auto_advance = BooleanProperty(True)
    pause_on_score = BooleanProperty(False)
    score_pause_active = BooleanProperty(False)
    episode_done = BooleanProperty(False)

    def on_toggle_auto(self) -> None:
        if self.wrapper is not None:
            self.wrapper.toggle_auto()

    def on_step(self) -> None:
        if self.wrapper is not None:
            self.wrapper.request_step()

    def on_toggle_pause(self) -> None:
        if self.wrapper is not None:
            self.wrapper.toggle_pause_on_score()

    # ------------------------------------------------------------------
    # UI update helpers
    # ------------------------------------------------------------------
    def update_state(self, state: dict[str, Any]) -> None:
        self.auto_advance = bool(state.get("auto_advance", True))
        self.pause_on_score = bool(state.get("pause_on_score", False))
        self.score_pause_active = bool(state.get("score_pause_active", False))
        self.phase_text = state.get("phase_line") or state.get("phase_text", "")
        self.reward_text = state.get("reward_line") or ""
        self.episode_done = bool(state.get("done", False))

        status_text = state.get("status_text", "")
        if self.score_pause_active:
            status_text = status_text or "Paused for score calculation"
            status_text = f"[Paused for score] {status_text}" if status_text else "[Paused for score]"
        self.status_text = status_text

        log_lines = state.get("log_lines", [])
        self.log_text = "\n".join(log_lines) if log_lines else ""

        board_state = state.get("board", {})
        players = state.get("players", [])
        board_widget = self.ids.get("board")
        if board_widget is not None:
            board_widget.apply_state(players, board_state)


class MahjongEnvKivyWrapper:
    """Mahjong environment wrapper that exposes a Kivy based GUI."""

    def __init__(
        self,
        *args: Any,
        env: Optional[_BaseMahjongEnv] = None,
        auto_advance: bool = True,
        pause_on_score: bool = False,
        **kwargs: Any,
    ) -> None:
        self._env = env if env is not None else _BaseMahjongEnv(*args, **kwargs)
        self._auto_advance = auto_advance
        self._pause_on_score = pause_on_score
        self._score_pause_active = False
        self._step_requested = False
        self._ui: Optional[MahjongRoot] = None
        self._status_text = ""
        self._log_messages: list[str] = []
        self._last_payload = _RenderPayload(action=None, reward=0.0, done=False, info={})
        self._last_observation = None

    # ------------------------------------------------------------------
    # Public environment-like interface
    # ------------------------------------------------------------------
    def reset(self, *args: Any, **kwargs: Any):
        observation = self._env.reset(*args, **kwargs)
        self._last_payload = _RenderPayload(action=None, reward=0.0, done=self._env.done, info={})
        self._last_observation = observation
        self._score_pause_active = False
        self._step_requested = False
        self._status_text = "Environment reset"
        self._log_messages.clear()
        self._append_log(self._status_text)
        self._update_ui()
        return observation

    def step(self, action: int):
        observation, reward, done, info = self._env.step(action)
        self._last_payload = _RenderPayload(action=action, reward=reward, done=done, info=info)
        self._last_observation = observation
        action_text = self._describe_action(action)
        self._status_text = f"Action: {action_text} | Reward: {reward:.2f}"
        if done:
            self._status_text += " | Hand complete"
        if info:
            info_message = info.get("msg") or info.get("message")
            if info_message:
                self._append_log(str(info_message))
        self._append_log(self._status_text)
        self._update_pause_state()
        self._update_ui()
        return observation, reward, done, info

    def close(self) -> None:
        self._env.close()

    def action_masks(self) -> Any:
        return self._env.action_masks()

    @property
    def phase(self) -> str:
        return getattr(self._env, "phase", "")

    @property
    def auto_advance(self) -> bool:
        return self._auto_advance

    @property
    def pause_on_score(self) -> bool:
        return self._pause_on_score

    @property
    def score_pause_active(self) -> bool:
        return self._score_pause_active

    @property
    def last_observation(self) -> Any:
        return self._last_observation

    @property
    def done(self) -> bool:
        return bool(getattr(self._env, "done", False))

    @property
    def base_env(self) -> _BaseMahjongEnv:
        return self._env

    # ------------------------------------------------------------------
    # GUI interaction helpers
    # ------------------------------------------------------------------
    def attach_ui(self, root: MahjongRoot) -> None:
        self._ui = root
        root.wrapper = self
        self._update_ui()

    def toggle_auto(self) -> None:
        self._auto_advance = not self._auto_advance
        if self._auto_advance:
            self._step_requested = False
        self._update_pause_state()
        self._update_ui()

    def toggle_pause_on_score(self) -> None:
        self._pause_on_score = not self._pause_on_score
        if not self._pause_on_score:
            self._score_pause_active = False
        else:
            self._update_pause_state()
        self._update_ui()

    def request_step(self) -> None:
        if not self._auto_advance:
            self._step_requested = True
        elif self._score_pause_active:
            self._score_pause_active = False
        self._update_ui()

    def wants_step(self) -> bool:
        if self._pause_on_score and self._score_pause_active:
            return False
        if self._auto_advance:
            return True
        if self._step_requested:
            self._step_requested = False
            return True
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _append_log(self, message: str) -> None:
        message = message.strip()
        if not message:
            return
        self._log_messages.append(message)
        if len(self._log_messages) > 80:
            self._log_messages[:] = self._log_messages[-80:]

    def _score_last(self) -> bool:
        if getattr(self._env, "num_players", 0) <= 0:
            return False
        return (
            getattr(self._env, "phase", "") == "score"
            and getattr(self._env, "current_player", 0)
            == getattr(self._env, "num_players", 1) - 1
        )

    def _update_pause_state(self) -> None:
        if self._pause_on_score and self._score_last():
            self._score_pause_active = True
        elif not self._score_last():
            self._score_pause_active = False

    def _update_ui(self) -> None:
        if self._ui is None:
            return
        state = self._generate_state()

        def _apply(_: float) -> None:
            if self._ui is not None:
                self._ui.update_state(state)

        try:
            Clock.schedule_once(_apply, 0)
        except Exception:  # pragma: no cover - Clock not running
            _apply(0)

    def _generate_state(self) -> dict[str, Any]:
        players: list[dict[str, Any]] = []
        num_players = getattr(self._env, "num_players", 0)
        riichi_flags = list(getattr(self._env, "riichi", []))
        scores = list(getattr(self._env, "scores", []))
        deltas = list(getattr(self._env, "score_deltas", []))
        hands = list(getattr(self._env, "hands", []))
        discards = list(getattr(self._env, "discard_pile_seq", []))
        melds = list(getattr(self._env, "melds", []))
        dealer = getattr(self._env, "oya", -1)
        current = getattr(self._env, "current_player", -1)
        winds = ["East", "South", "West", "North"]
        seat_names = list(getattr(self._env, "seat_names", []))
        min_labels = max(4, num_players)
        if len(seat_names) < min_labels:
            seat_names.extend(
                [f"Seat {idx + 1}" for idx in range(len(seat_names), min_labels)]
            )

        reveal_flags = self._compute_hand_reveal_flags(num_players)

        for idx in range(num_players):
            hand_tiles = hands[idx] if idx < len(hands) else []
            discard_tiles = discards[idx] if idx < len(discards) else []
            meld_list = melds[idx] if idx < len(melds) else []
            score_value = scores[idx] if idx < len(scores) else 0
            delta_value = deltas[idx] if idx < len(deltas) else 0
            riichi = riichi_flags[idx] if idx < len(riichi_flags) else False
            reveal_hand = reveal_flags[idx] if idx < len(reveal_flags) else (idx == 0)

            ordered_hand = self._order_hand_tiles(hand_tiles, idx, reveal_hand)
            hand_entries = self._build_tile_entries(ordered_hand, reveal_hand)
            discard_entries = self._build_tile_entries(discard_tiles, True)
            meld_entries = self._build_meld_entries(meld_list)

            hand_text = self._format_tiles(ordered_hand if reveal_hand else hand_tiles)
            discard_text = self._format_tiles(discard_tiles, sort=False)
            meld_text = self._format_melds(meld_list)

            players.append(
                {
                    "name": self._format_player_name(idx, dealer),
                    "hand": hand_text,
                    "hand_text": hand_text,
                    "hand_tiles": hand_entries,
                    "hand_placeholder": "-" if not hand_entries else "",
                    "discards": f"Discards: {discard_text}" if discard_text else "Discards: -",
                    "discard_text": discard_text,
                    "discard_tiles": discard_entries,
                    "discard_placeholder": "-" if not discard_entries else "",
                    "melds": meld_text,
                    "meld_text": meld_text,
                    "melds_data": meld_entries,
                    "meld_placeholder": meld_text if not meld_entries else "",
                    "score": f"{int(round(score_value * 100)):,}",
                    "score_delta": self._format_delta(delta_value),
                    "riichi": riichi,
                    "is_current": idx == current,
                    "dealer": idx == dealer,
                    "wind": f"Seat wind: {winds[idx % len(winds)]}",
                    "seat_label": seat_names[idx] if idx < len(seat_names) else "",
                }
            )

        dora_tiles_raw = list(getattr(self._env, "dora_indicator", []))
        ura_tiles_raw = list(getattr(self._env, "ura_indicator", []))
        dora_tiles = self._build_tile_entries(dora_tiles_raw, True)
        ura_tiles = self._build_tile_entries(ura_tiles_raw, True)
        dora_text = self._format_indicator_text("Dora", dora_tiles_raw)
        ura_text = self._format_indicator_text("Ura", ura_tiles_raw)
        status_text = self._status_text
        if self._pause_on_score and self._score_pause_active:
            status_text = status_text or "Paused for score calculation"

        round_label, honba = self._extract_round_state()
        board_state = {
            "round_label": round_label,
            "honba": honba,
            "riichi_sticks": int(getattr(self._env, "num_riichi", 0)),
            "tiles_remaining": int(len(getattr(self._env, "deck", []))),
            "scores": [int(round(value * 100)) for value in scores[:4]],
            "seat_names": seat_names[:4],
            "dealer": int(dealer),
            "current_player": int(current),
            "dora_tiles": dora_tiles,
            "ura_tiles": ura_tiles,
            "dora_text": dora_text,
            "ura_text": ura_text,
        }

        phase = getattr(self._env, "phase", "")
        phase_line = (
            f"Phase: {phase}  |  Current Player: P{current}"
            if current >= 0
            else f"Phase: {phase}"
        )
        reward_line = (
            f"Action: {self._last_payload.action}  Reward: {self._last_payload.reward:.2f}"
        )

        return {
            "players": players,
            "board": board_state,
            "phase_text": phase,
            "phase_line": phase_line,
            "reward_line": reward_line,
            "status_text": status_text,
            "auto_advance": self._auto_advance,
            "pause_on_score": self._pause_on_score,
            "score_pause_active": self._score_pause_active,
            "log_lines": list(self._log_messages),
            "done": self._last_payload.done,
        }

    def _format_player_name(self, idx: int, dealer: int) -> str:
        label = f"Player {idx + 1}"
        if idx == dealer:
            label += " (Dealer)"
        return label

    def _format_tiles(self, tiles: Sequence[int], sort: bool = True) -> str:
        if not tiles:
            return "-"
        try:
            return tiles_printout(sorted(tiles) if sort else list(tiles)).strip()
        except Exception:
            return " ".join(str(tile) for tile in (sorted(tiles) if sort else tiles))

    def _format_melds(self, meld_list: Sequence[dict[str, Any]]) -> str:
        if not meld_list:
            return "Melds: -"
        meld_texts: list[str] = []
        for meld in meld_list:
            tiles = meld.get("m") or []
            meld_type = meld.get("type", "")
            opened = meld.get("opened", True)
            try:
                tile_text = tiles_printout(list(tiles)).strip()
            except Exception:
                tile_text = " ".join(str(t) for t in tiles)
            status = "open" if opened else "closed"
            if meld_type:
                meld_texts.append(f"{meld_type.title()} ({status}): {tile_text}")
            else:
                meld_texts.append(f"Meld ({status}): {tile_text}")
        return "\n".join(meld_texts)

    def _order_hand_tiles(
        self, hand_tiles: Sequence[int], player_idx: int, reveal: bool
    ) -> list[int]:
        tiles = list(hand_tiles or [])
        if not tiles:
            return []
        if not reveal:
            return tiles
        phase = getattr(self._env, "phase", "")
        current_player = getattr(self._env, "current_player", -1)
        if phase in {"draw", "kan_draw"} and current_player == player_idx:
            ordered = sorted(tiles[:-1]) + tiles[-1:]
        else:
            ordered = sorted(tiles)
        return ordered

    def _build_tile_entries(
        self, tiles: Sequence[int], face_up: bool
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for tile in tiles or []:
            try:
                tile_id = int(tile)
            except (TypeError, ValueError):
                tile_id = -1
            entries.append({"tile": tile_id, "face_up": face_up})
        return entries

    def _build_meld_entries(
        self, meld_list: Sequence[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        meld_entries: list[dict[str, Any]] = []
        for meld in meld_list or []:
            tiles = list(reversed(list(meld.get("m", []))))
            opened = bool(meld.get("opened", True))
            meld_type = meld.get("type", "")
            claimed_tile = meld.get("claimed_tile")
            sideways_index: Optional[int] = None
            if claimed_tile is not None:
                try:
                    sideways_index = tiles.index(claimed_tile)
                except ValueError:
                    sideways_index = None
            tile_entries: list[dict[str, Any]] = []
            total_tiles = len(tiles)
            for idx, tile in enumerate(tiles):
                orientation = 0
                if sideways_index is not None and idx == sideways_index:
                    orientation = 90
                face_up = opened
                if not opened and meld_type == "kan" and total_tiles >= 4 and idx in {0, total_tiles - 1}:
                    face_up = False
                try:
                    tile_id = int(tile)
                except (TypeError, ValueError):
                    tile_id = -1
                tile_entries.append(
                    {
                        "tile": tile_id,
                        "face_up": face_up,
                        "orientation": orientation,
                    }
                )
            status = "open" if opened else "closed"
            label = f"{meld_type.title()} ({status})" if meld_type else f"Meld ({status})"
            meld_entries.append({"tiles": tile_entries, "label": label})
        return meld_entries

    def _format_delta(self, value: Any) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return ""
        points = int(round(numeric * 100))
        if points == 0:
            return "0"
        return f"{points:+,}"

    def _format_indicator_text(self, label: str, tiles: Iterable[int]) -> str:
        tiles = list(tiles or [])
        if not tiles:
            return f"{label}: -"
        try:
            return f"{label}: {tiles_printout(tiles).strip()}"
        except Exception:
            return f"{label}: {' '.join(str(t) for t in tiles)}"

    def _extract_winner_indices(self, agari: Any) -> set[int]:
        winners: set[int] = set()
        if isinstance(agari, dict):
            who = agari.get("who")
            if isinstance(who, int):
                winners.add(who)
            elif isinstance(who, Iterable) and not isinstance(who, (str, bytes)):
                for value in who:
                    if isinstance(value, int):
                        winners.add(value)
        return winners

    def _compute_hand_reveal_flags(self, num_players: int) -> list[bool]:
        reveal = [False] * max(0, num_players)
        for idx in range(num_players):
            reveal[idx] = idx == 0

        if getattr(self._env, "phase", "") != "score":
            return reveal

        agari = getattr(self._env, "agari", None)
        if agari:
            for winner in self._extract_winner_indices(agari):
                if 0 <= winner < num_players:
                    reveal[winner] = True
        else:
            tenpai_flags = list(getattr(self._env, "tenpai", []))
            for idx in range(num_players):
                if idx < len(tenpai_flags) and tenpai_flags[idx]:
                    reveal[idx] = True
        return reveal

    def _extract_round_state(self) -> Tuple[str, int]:
        round_info = getattr(self._env, "round", [-1, 0])
        if not isinstance(round_info, Sequence) or len(round_info) < 2:
            return ("East 1", 0)
        try:
            round_index = int(round_info[0])
        except (TypeError, ValueError):  # pragma: no cover - defensive
            round_index = 0
        try:
            honba = int(round_info[1])
        except (TypeError, ValueError):  # pragma: no cover - defensive
            honba = 0
        winds = ["East", "South", "West", "North"]
        num_players = max(1, getattr(self._env, "num_players", 4))
        wind_idx = (round_index // num_players) % len(winds)
        hand_number = (round_index % num_players) + 1
        return (f"{winds[wind_idx]} {hand_number}", honba)

    def _format_round_text(self) -> str:
        round_label, honba = self._extract_round_state()
        return f"{round_label} | Honba {honba}"

    def _describe_action(self, action: Optional[int]) -> str:
        if action is None:
            return "None"
        try:
            detail = get_action_from_index(action)
        except Exception:
            return str(action)

        if action < 34:
            tile_id = detail[0] * 4 + 1
            return f"Discard {tile_printout(tile_id).strip()}"
        if action < 68:
            tile_id = detail[0] * 4 + 1
            return f"Riichi discard {tile_printout(tile_id).strip()}"
        if action < 113:
            tiles = [value * 4 + 1 for value in detail[0]]
            return f"Chi {tiles_printout(tiles).strip()}"
        if action < 147:
            tiles = [detail[0][0] * 4 + 1] * 3
            return f"Pon {tiles_printout(tiles).strip()}"
        if action < 181:
            tiles = [detail[0][0] * 4 + 1] * 4
            return f"Kan {tiles_printout(tiles).strip()}"
        if action < 215:
            tile_id = detail[0][0] * 4 + 1
            return f"Added Kan {tile_printout(tile_id).strip()}"
        if action < 249:
            tiles = [detail[0][0] * 4 + 1] * 4
            return f"Closed Kan {tiles_printout(tiles).strip()}"
        mapping = {
            249: "Ryuukyoku",
            250: "Ron",
            251: "Tsumo",
            252: "Cancel",
            253: "Cancel Riichi",
            254: "Cancel Chi",
            255: "Cancel Pon",
            256: "Cancel Kan",
            257: "Cancel Ankan",
            258: "Cancel Chakan",
            259: "Cancel Ryuukyoku",
            260: "Cancel Ron",
            261: "Cancel Tsumo",
        }
        return mapping.get(action, str(action))

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------
    def __enter__(self) -> "MahjongEnvKivyWrapper":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
