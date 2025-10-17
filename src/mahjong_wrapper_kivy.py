from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, Sequence, Tuple, Union

from kivy.base import EventLoop
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from kivy.core.text import Label as CoreLabel
from kivy.graphics import (
    Color,
    Line,
    PushMatrix,
    PopMatrix,
    Rectangle,
    Rotate,
    RoundedRectangle,
    Translate,
)
from kivy.graphics.instructions import InstructionGroup
from kivy.properties import ObjectProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.widget import Widget

try:  # pragma: no cover - optional dependency
    from cairosvg import svg2png
except Exception:  # pragma: no cover - optional dependency
    svg2png = None

import threading
import time

if TYPE_CHECKING:  # pragma: no cover - typing support only
    from agent.human_player_agent import HumanPlayerAgent

from mahjong_env import MahjongEnvBase as _BaseMahjongEnv

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


_RED_FIVE_MAP: dict[int, int] = {4: 34, 13: 35, 22: 36}


_DEFAULT_LANGUAGE = "zh-Hans"
_LANGUAGE_ORDER: Tuple[str, ...] = ("zh-Hans", "en", "ja", "fr")
_ASSET_FONT_ROOT = Path(__file__).resolve().parent.parent / "assets" / "fonts"
_FONT_PATHS: dict[str, Path] = {
    "en": _ASSET_FONT_ROOT / "Noto_Sans" / "static" / "NotoSans-Regular.ttf",
    "zh-Hans": _ASSET_FONT_ROOT / "Noto_Sans_SC" / "static" / "NotoSansSC-Regular.ttf",
    "ja": _ASSET_FONT_ROOT / "Noto_Sans_JP" / "static" / "NotoSansJP-Regular.ttf",
    "fr": _ASSET_FONT_ROOT / "Noto_Sans" / "static" / "NotoSans-Regular.ttf",
}
_FALLBACK_FONT = _ASSET_FONT_ROOT / "Noto_Color_Emoji" / "NotoColorEmoji-Regular.ttf"


def _ordinal_en(value: int) -> str:
    suffix = "th"
    if value % 100 not in {11, 12, 13}:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"


def _ordinal_zh(value: int) -> str:
    return f"第{value}名"


def _ordinal_ja(value: int) -> str:
    return f"第{value}位"


def _ordinal_fr(value: int) -> str:
    return "1er" if value == 1 else f"{value}e"


_ORDINAL_FUNCTIONS: dict[str, Callable[[int], str]] = {
    "en": _ordinal_en,
    "zh-Hans": _ordinal_zh,
    "ja": _ordinal_ja,
    "fr": _ordinal_fr,
}


_LANGUAGE_STRINGS: dict[str, dict[str, Any]] = {
    "en": {
        "language_name": "EN",
        "wind_names": ["East", "South", "West", "North"],
        "round_format": "{wind} {hand}",
        "counter_honba": "Honba {count}",
        "counter_riichi": "Riichi {count}",
        "counter_tiles": "Tiles {count}",
        "counter_kyoutaku": "Kyoutaku {count}",
        "info_line_format": "{round} | {honba} | {riichi} | {kyoutaku}",
        "round_results_title": "Round Results",
        "yaku_prefix": "Yaku: ",
        "dealer_suffix": " (Dealer)",
        "player_name_format": "Player {index}",
        "seat_placeholder_format": "P{index}",
        "draw_tenpai": "Draw - Tenpai: {players}",
        "draw_no_tenpai": "Draw - No Tenpai",
        "tsumo_label": "Tsumo",
        "ron_label": "Ron",
        "vs_label": "vs",
        "result_tsumo": "{winner} {tsumo_label}",
        "result_ron": "{winner} {ron_label} {vs_label} {loser}",
        "result_details": "{han} Han | {fu} Fu | {total} Points",
        "phase_label": "Phase",
        "current_player_label": "Current Player",
        "unknown_player": "-",
        "action_label": "Action",
        "reward_label": "Reward",
        "status_format": "{phase_label}: {phase}  |  {current_player_label}: {player}",
        "action_reward_format": "{action_label}: {action}  {reward_label}: {reward}",
        "episode_finished": "Episode finished",
        "pause_on_score_on": "On-Pause on Score",
        "pause_on_score_off": "OFF-Pause on Score",
        "auto_next_on": "On-Auto Next",
        "auto_next_off": "OFF-Auto Next",
        "hints_on": "On-Hints",
        "hints_off": "OFF-Hints",
        "step_next": "Next",
        "riichi_flag": "Riichi",
        "tenpai_label": "Tenpai",
        "no_tenpai_label": "No Tenpai",
    },
    "zh-Hans": {
        "language_name": "ZH",
        "wind_names": ["东", "南", "西", "北"],
        "round_format": "{wind}{hand}局",
        "counter_honba": "本场 {count}",
        "counter_riichi": "立直棒 {count}",
        "counter_tiles": "余牌 {count}",
        "counter_kyoutaku": "供托 {count}",
        "info_line_format": "{round} | {honba} | {riichi} | {kyoutaku}",
        "round_results_title": "对局结果",
        "yaku_prefix": "役种：",
        "dealer_suffix": "（庄家）",
        "player_name_format": "玩家{index}",
        "seat_placeholder_format": "玩家{index}",
        "draw_tenpai": "流局 - 听牌: {players}",
        "draw_no_tenpai": "流局 - 无听牌",
        "tsumo_label": "自摸",
        "ron_label": "荣和",
        "vs_label": "对",
        "result_tsumo": "{winner} {tsumo_label}",
        "result_ron": "{winner} {ron_label} {vs_label} {loser}",
        "result_details": "{han} 番 | {fu} 符 | {total} 点",
        "phase_label": "阶段",
        "current_player_label": "当前玩家",
        "unknown_player": "-",
        "action_label": "动作",
        "reward_label": "奖励",
        "status_format": "{phase_label}：{phase}  |  {current_player_label}：{player}",
        "action_reward_format": "{action_label}：{action}  {reward_label}：{reward}",
        "episode_finished": "对局结束",
        "pause_on_score_on": "开启-计分暂停",
        "pause_on_score_off": "关闭-计分暂停",
        "auto_next_on": "开启-自动进行",
        "auto_next_off": "关闭-自动进行",
        "hints_on": "开启-提示",
        "hints_off": "关闭-提示",
        "step_next": "下一步",
        "riichi_flag": "立直",
        "tenpai_label": "听牌",
        "no_tenpai_label": "无听牌",
    },
    "ja": {
        "language_name": "JP",
        "wind_names": ["東", "南", "西", "北"],
        "round_format": "{wind}{hand}局",
        "counter_honba": "本場 {count}",
        "counter_riichi": "立直棒 {count}",
        "counter_tiles": "残り牌 {count}",
        "counter_kyoutaku": "供託 {count}",
        "info_line_format": "{round} | {honba} | {riichi} | {kyoutaku}",
        "round_results_title": "対局結果",
        "yaku_prefix": "役: ",
        "dealer_suffix": "（親）",
        "player_name_format": "プレイヤー{index}",
        "seat_placeholder_format": "プレイヤー{index}",
        "draw_tenpai": "流局 - 聴牌: {players}",
        "draw_no_tenpai": "流局 - ノーテン",
        "tsumo_label": "ツモ",
        "ron_label": "ロン",
        "vs_label": "対",
        "result_tsumo": "{winner} {tsumo_label}",
        "result_ron": "{winner} {ron_label} {vs_label} {loser}",
        "result_details": "{han} 翻 | {fu} 符 | {total} 点",
        "phase_label": "フェーズ",
        "current_player_label": "現在のプレイヤー",
        "unknown_player": "-",
        "action_label": "行動",
        "reward_label": "報酬",
        "status_format": "{phase_label}：{phase}  |  {current_player_label}：{player}",
        "action_reward_format": "{action_label}：{action}  {reward_label}：{reward}",
        "episode_finished": "ゲーム終了",
        "pause_on_score_on": "オン-得点で一時停止",
        "pause_on_score_off": "オフ-得点で一時停止",
        "auto_next_on": "オン-自動進行",
        "auto_next_off": "オフ-自動進行",
        "hints_on": "オン-ヒント",
        "hints_off": "オフ-ヒント",
        "step_next": "次へ",
        "riichi_flag": "立直",
        "tenpai_label": "聴牌",
        "no_tenpai_label": "ノーテン",
    },
    "fr": {
        "language_name": "FR",
        "wind_names": ["Est", "Sud", "Ouest", "Nord"],
        "round_format": "{wind} {hand}",
        "counter_honba": "Honba {count}",
        "counter_riichi": "Riichi {count}",
        "counter_tiles": "Tuiles {count}",
        "counter_kyoutaku": "Kyoutaku {count}",
        "info_line_format": "{round} | {honba} | {riichi} | {kyoutaku}",
        "round_results_title": "Résultats de la manche",
        "yaku_prefix": "Yaku : ",
        "dealer_suffix": " (Donneur)",
        "player_name_format": "Joueur {index}",
        "seat_placeholder_format": "J{index}",
        "draw_tenpai": "Égalité - Tenpai : {players}",
        "draw_no_tenpai": "Égalité - Pas de Tenpai",
        "tsumo_label": "Tsumo",
        "ron_label": "Ron",
        "vs_label": "contre",
        "result_tsumo": "{winner} {tsumo_label}",
        "result_ron": "{winner} {ron_label} {vs_label} {loser}",
        "result_details": "{han} Han | {fu} Fu | {total} Points",
        "phase_label": "Phase",
        "current_player_label": "Joueur actuel",
        "unknown_player": "-",
        "action_label": "Action",
        "reward_label": "Récompense",
        "status_format": "{phase_label} : {phase}  |  {current_player_label} : {player}",
        "action_reward_format": "{action_label} : {action}  {reward_label} : {reward}",
        "episode_finished": "Manche terminée",
        "pause_on_score_on": "Marche-Pause sur score",
        "pause_on_score_off": "Arrêt-Pause sur score",
        "auto_next_on": "Marche-Auto suivant",
        "auto_next_off": "Arrêt-Auto suivant",
        "hints_on": "Marche-Indices",
        "hints_off": "Arrêt-Indices",
        "step_next": "Suivant",
        "riichi_flag": "Riichi",
        "tenpai_label": "Tenpai",
        "no_tenpai_label": "Pas de Tenpai",
    },
}


@dataclass(slots=True)
class _RenderPayload:
    action: Optional[int]
    reward: float
    done: bool
    info: dict[str, Any]


@dataclass(frozen=True)
class _TileHighlight:
    dora: bool = False
    outline_color: Optional[Tuple[float, float, float, float]] = None


@dataclass
class _Rect:
    x: float
    y: float
    width: float
    height: float

    @property
    def left(self) -> float:
        return self.x

    @left.setter
    def left(self, value: float) -> None:
        self.x = value

    @property
    def right(self) -> float:
        return self.x + self.width

    @right.setter
    def right(self, value: float) -> None:
        self.x = value - self.width

    @property
    def top(self) -> float:
        return self.y

    @top.setter
    def top(self, value: float) -> None:
        self.y = value

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @bottom.setter
    def bottom(self, value: float) -> None:
        self.y = value - self.height

    @property
    def centerx(self) -> float:
        return self.x + self.width / 2

    @centerx.setter
    def centerx(self, value: float) -> None:
        self.x = value - self.width / 2

    @property
    def centery(self) -> float:
        return self.y + self.height / 2

    @centery.setter
    def centery(self, value: float) -> None:
        self.y = value - self.height / 2

    @property
    def center(self) -> Tuple[float, float]:
        return (self.centerx, self.centery)

    @center.setter
    def center(self, value: Tuple[float, float]) -> None:
        cx, cy = value
        self.centerx = cx
        self.centery = cy


class MahjongBoardWidget(Widget):
    """Widget that renders the Mahjong play field."""

    wrapper = ObjectProperty(None)

    def on_touch_down(self, touch: Any) -> bool:  # type: ignore[override]
        if super().on_touch_down(touch):
            return True
        pos = getattr(touch, "pos", None)
        if pos is None or not self.collide_point(*pos):
            return False
        if self.wrapper is not None and self.wrapper.handle_board_touch(self, touch):
            return True
        return False


class ActionPanel(BoxLayout):
    container = ObjectProperty(None)
    title_label = ObjectProperty(None)


class MahjongRoot(FloatLayout):
    board = ObjectProperty(None)
    status_label = ObjectProperty(None)
    reward_label = ObjectProperty(None)
    done_label = ObjectProperty(None)
    pause_button = ObjectProperty(None)
    auto_button = ObjectProperty(None)
    step_button = ObjectProperty(None)
    hints_button = ObjectProperty(None)
    language_spinner = ObjectProperty(None)
    action_panel = ObjectProperty(None)
    quick_action_bar = ObjectProperty(None)
    action_timer_label = ObjectProperty(None)
    wrapper = ObjectProperty(None)


class MahjongEnvKivyWrapper:
    """Mahjong environment with a Kivy GUI overlay."""

    def __init__(
        self,
        *args: Any,
        env: Optional[_BaseMahjongEnv] = None,
        window_size: Tuple[int, int] = (1024, 720),
        fps: int = 30,
        font_name: Optional[str] = None,
        font_size: int = 18,
        fallback_fonts: Optional[Sequence[str]] = None,
        root_widget: Optional[MahjongRoot] = None,
        tile_texture_size: Optional[Tuple[int, int]] = None,
        tile_texture_use_tile_metrics: bool = False,
        tile_texture_background: Optional[
            Union[str, Sequence[float], Sequence[int]]
        ] = None,
        **kwargs: Any,
    ) -> None:
        self._env = env or _BaseMahjongEnv(*args, **kwargs)
        self._window_size = window_size
        self._fps = max(1, fps)
        self._user_font_name = font_name
        self._font_name = font_name or "Roboto"
        self._font_size = font_size
        self._background_color = (12 / 255.0, 30 / 255.0, 60 / 255.0, 1)
        self._play_area_color = (24 / 255.0, 60 / 255.0, 90 / 255.0, 1)
        self._play_area_border = (40 / 255.0, 90 / 255.0, 130 / 255.0, 1)
        self._panel_color = (5 / 255.0, 5 / 255.0, 5 / 255.0, 1)
        self._panel_border = (90 / 255.0, 120 / 255.0, 160 / 255.0, 1)
        self._accent_color = (170 / 255.0, 230 / 255.0, 255 / 255.0, 1)
        self._text_color = (235 / 255.0, 235 / 255.0, 235 / 255.0, 1)
        self._muted_text_color = (170 / 255.0, 190 / 255.0, 210 / 255.0, 1)
        self._danger_color = (220 / 255.0, 120 / 255.0, 120 / 255.0, 1)
        self._face_down_color = (18 / 255.0, 18 / 255.0, 22 / 255.0, 1)
        self._face_down_border = (60 / 255.0, 60 / 255.0, 70 / 255.0, 1)
        self._tile_texture_background = "#FFFFFF"
        self._show_hints = True
        self._dora_overlay_color = (250 / 255.0, 210 / 255.0, 120 / 255.0, 0.45)
        self._last_draw_outline_color = (120 / 255.0, 190 / 255.0, 255 / 255.0, 1)
        self._last_discard_outline_color = (255 / 255.0, 170 / 255.0, 90 / 255.0, 1)
        self._hint_outline_width = 3
        self._dora_highlight_tiles: set[int] = set()
        self._cached_last_draw_tiles: list[int] = []
        self._cached_last_discard_tile: int = -1
        self._cached_last_discarder: int = -1

        self._root = root_widget or MahjongRoot()
        self._root.size = window_size
        self._root.wrapper = self
        self._root.bind(board=self._on_board_assigned)
        self._on_board_assigned(self._root, getattr(self._root, "board", None))
        self._human_agents: dict[int, "HumanPlayerAgent"] = {}
        self._active_action_seat: Optional[int] = None
        self._hand_hitboxes: dict[
            int, list[tuple[int, tuple[float, float, float, float]]]
        ] = {}

        self._language_code_to_name: dict[str, str] = {
            code: data.get("language_name", code) for code, data in _LANGUAGE_STRINGS.items()
        }
        self._language_name_to_code: dict[str, str] = {
            name: code for code, name in self._language_code_to_name.items()
        }
        if _LANGUAGE_STRINGS:
            self._default_language = (
                _DEFAULT_LANGUAGE if _DEFAULT_LANGUAGE in _LANGUAGE_STRINGS else next(iter(_LANGUAGE_STRINGS))
            )
        else:
            self._default_language = _DEFAULT_LANGUAGE
        self._language = self._default_language
        self._ordinal_func = _ORDINAL_FUNCTIONS.get(self._language, _ordinal_en)
        self._updating_language_spinner = False
        self._configure_language_spinner()
        self._update_language_fonts(self._language)
        self._update_language_spinner()
        self._apply_font_to_controls()
        self._clear_human_actions()

        self._asset_root = Path(__file__).resolve().parent.parent / "assets" / "tiles" / "Regular"
        self._raw_tile_textures: dict[int, Any] = {}
        self._placeholder_cache: dict[tuple[int, Tuple[int, int]], CoreLabel] = {}
        self._tile_metrics: dict[str, Tuple[int, int]] = {}
        self._riichi_states: list[bool] = []
        self._riichi_pending: list[bool] = []
        self._discard_counts: list[int] = []
        self._riichi_declarations: dict[int, int] = {}
        self._focus_player = 0

        if tile_texture_size is not None:
            width, height = tile_texture_size
            self._tile_texture_explicit_size: Optional[Tuple[int, int]] = (
                max(1, int(width)),
                max(1, int(height)),
            )
        else:
            self._tile_texture_explicit_size = None
        self._tile_texture_auto_size = bool(tile_texture_use_tile_metrics)
        
        self._current_texture_render_size: Optional[Tuple[int, int]] = None

        self._last_payload = _RenderPayload(action=None, reward=0.0, done=False, info={})
        self._auto_advance = True
        self._pause_on_score = True
        self._score_pause_active = False
        self._score_pause_pending = False
        self._step_once_requested = False
        self._score_panel_was_visible = False
        self._pending_action: Optional[int] = None
        self._step_result: Optional[Tuple[Any, float, bool, dict[str, Any]]] = None
        self._step_event = threading.Event()
        self._scheduled = False
        self._action_deadline: Optional[float] = None
        self._action_timer_event: Optional[Any] = None
        self._load_tile_assets(self._tile_texture_explicit_size)
        self._connect_controls()

        self._clock_event = Clock.schedule_interval(self._on_frame, 1.0 / self._fps)
        self._scheduled = True

    # ------------------------------------------------------------------
    # Exposed environment API
    # ------------------------------------------------------------------
    @property
    def root(self) -> MahjongRoot:
        return self._root

    @property
    def pending_action(self) -> Optional[int]:
        return self._pending_action

    @property
    def fps(self) -> int:
        return self._fps

    def set_focus_player(self, player_index: int) -> None:
        num_players = getattr(self._env, "num_players", 0)
        if num_players > 0:
            sanitized = max(0, min(int(player_index), num_players - 1))
        else:
            sanitized = 0
        if sanitized == self._focus_player:
            return
        self._focus_player = sanitized
        self._render()

    def _get_focus_index(self, num_players: int) -> int:
        if num_players <= 0:
            return 0
        if self._focus_player < 0:
            self._focus_player = 0
        elif self._focus_player >= num_players:
            self._focus_player = num_players - 1
        return self._focus_player

    def _get_display_order(self) -> list[int]:
        num_players = getattr(self._env, "num_players", 0)
        if num_players <= 0:
            return []
        focus = self._get_focus_index(num_players)
        max_display = min(4, num_players)
        return [((focus + offset) % num_players) for offset in range(max_display)]

    def _get_relative_angle(self, relative_position: int) -> int:
        angle_map = {0: 0, 1: 90, 2: 180, 3: -90}
        normalized = relative_position % 4
        return angle_map.get(normalized, 0)

    def _get_board_rotation_angle(self) -> int:
        num_players = getattr(self._env, "num_players", 0)
        if num_players <= 0:
            return 0
        focus = self._get_focus_index(num_players)
        return -self._get_relative_angle(focus)

    def set_language(self, language: str) -> None:
        if not language:
            return
        code = self._language_name_to_code.get(language, language)
        if code not in _LANGUAGE_STRINGS:
            return
        if code == self._language:
            return
        self._language = code
        self._ordinal_func = _ORDINAL_FUNCTIONS.get(self._language, _ordinal_en)
        self._update_language_fonts(self._language)
        self._apply_font_to_controls()
        self._update_language_spinner()
        self._render()
        self._draw_status_labels()
        self._update_control_buttons()

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        observation = self._env.reset(*args, **kwargs)
        self._last_payload = _RenderPayload(action=None, reward=0.0, done=self._env.done, info={})
        self._step_once_requested = False
        self._score_pause_active = False
        self._score_pause_pending = False
        self._score_panel_was_visible = self._score_last()
        self._riichi_states = []
        self._riichi_pending = []
        self._discard_counts = []
        self._riichi_declarations = {}
        self._pending_action = None
        self._step_result = None
        self._render()
        self._clear_human_actions()
        return observation

    def step(self, action: int) -> Tuple[Any, float, bool, dict[str, Any]]:
        self.queue_action(action)
        while self._step_result is None:
            EventLoop.idle()
        result = self._step_result
        self._step_result = None
        return result

    def queue_action(self, action: int) -> None:
        if self._pending_action is not None:
            return
        self._pending_action = action
        self._step_result = None
        self._step_event.clear()

    def fetch_step_result(self) -> Optional[Tuple[Any, float, bool, dict[str, Any]]]:
        if self._step_result is None:
            return None
        result = self._step_result
        self._step_result = None
        return result

    def close(self) -> None:
        if self._scheduled:
            self._clock_event.cancel()
            self._scheduled = False
        self._env.close()

    def action_masks(self) -> Any:
        return self._env.action_masks()

    def bind_human_ui(self, seat: int, agent: "HumanPlayerAgent") -> None:
        """Register callbacks that display human actions and handle input."""

        if seat < 0:
            raise ValueError("seat must be non-negative")
        self._human_agents[seat] = agent
        agent.bind_presenter(
            lambda actions, deadline, seat=seat: self._show_human_actions(
                seat, actions, deadline
            ),
            lambda seat=seat: self._clear_human_actions(seat),
        )

    @property
    def phase(self) -> str:
        return getattr(self._env, "phase", "")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _connect_controls(self) -> None:
        if not self._root:
            return
        self._root.auto_button.bind(on_release=lambda *_: self._toggle_auto())
        self._root.step_button.bind(on_release=lambda *_: self._trigger_step_once())
        self._root.pause_button.bind(on_release=lambda *_: self._toggle_pause())
        self._root.hints_button.bind(on_release=lambda *_: self._toggle_hints())
        spinner = getattr(self._root, "language_spinner", None)
        if spinner is not None:
            spinner.bind(text=self._on_language_spinner_text)

    def _on_board_assigned(
        self, _root: MahjongRoot, board: Optional[MahjongBoardWidget]
    ) -> None:
        if isinstance(board, MahjongBoardWidget):
            board.wrapper = self

    def _get_language_dict(self, code: Optional[str] = None) -> dict[str, Any]:
        lang_code = code or self._language
        if lang_code in _LANGUAGE_STRINGS:
            return _LANGUAGE_STRINGS[lang_code]
        return _LANGUAGE_STRINGS.get(self._default_language, {})

    def _translate(self, key: str, **kwargs: Any) -> str:
        language_dict = self._get_language_dict()
        template = language_dict.get(key)
        if template is None:
            template = self._get_language_dict(self._default_language).get(key, key)
        if isinstance(template, str):
            return template.format(**kwargs)
        return str(template)

    def _translate_sequence(self, key: str) -> Sequence[str]:
        value = self._get_language_dict().get(key)
        if isinstance(value, (list, tuple)):
            return tuple(value)
        default_value = self._get_language_dict(self._default_language).get(key, ())
        if isinstance(default_value, (list, tuple)):
            return tuple(default_value)
        return ()

    def _format_ordinal(self, value: int) -> str:
        try:
            return str(self._ordinal_func(int(value)))
        except Exception:
            return str(value)

    def _configure_language_spinner(self) -> None:
        if not self._root:
            return
        spinner = getattr(self._root, "language_spinner", None)
        if spinner is None:
            return
        values: list[str] = [
            self._language_code_to_name.get(code, code)
            for code in _LANGUAGE_ORDER
            if code in self._language_code_to_name
        ]
        for name in self._language_code_to_name.values():
            if name not in values:
                values.append(name)
        spinner.values = tuple(values)

    def _update_language_spinner(self) -> None:
        if not self._root:
            return
        spinner = getattr(self._root, "language_spinner", None)
        if spinner is None:
            return
        values: list[str] = [
            self._language_code_to_name.get(code, code)
            for code in _LANGUAGE_ORDER
            if code in self._language_code_to_name
        ]
        display_name = self._language_code_to_name.get(self._language, self._language)
        if display_name not in values:
            values.append(display_name)
        spinner.values = tuple(values)
        if spinner.text != display_name:
            self._updating_language_spinner = True
            spinner.text = display_name
            self._updating_language_spinner = False

    def _update_language_fonts(self, language: str) -> None:
        font_path = _FONT_PATHS.get(language)
        if font_path is not None and font_path.exists():
            self._font_name = str(font_path)
        elif self._user_font_name:
            self._font_name = str(self._user_font_name)
        # otherwise keep the existing font_name
        fallback_entries: list[str] = []
        if _FALLBACK_FONT.exists():
            fallback_entries.append(str(_FALLBACK_FONT))
        self._fallback_fonts = tuple(fallback_entries)

    def _apply_font_to_controls(self) -> None:
        if not self._root:
            return
        widgets = [
            getattr(self._root, "status_label", None),
            getattr(self._root, "reward_label", None),
            getattr(self._root, "done_label", None),
            getattr(self._root, "pause_button", None),
            getattr(self._root, "auto_button", None),
            getattr(self._root, "step_button", None),
            getattr(self._root, "hints_button", None),
            getattr(self._root, "language_spinner", None),
            getattr(getattr(self._root, "action_panel", None), "title_label", None),
        ]
        for widget in widgets:
            if widget is None:
                continue
            try:
                widget.font_name = self._font_name
            except Exception:
                continue
        panel = getattr(self._root, "action_panel", None)
        container = getattr(panel, "container", None) if panel is not None else None
        if container is not None:
            for child in container.children:
                try:
                    child.font_name = self._font_name
                except Exception:
                    continue
        quick_bar = getattr(self._root, "quick_action_bar", None)
        if quick_bar is not None:
            for child in quick_bar.children:
                try:
                    child.font_name = self._font_name
                except Exception:
                    continue

    def _on_language_spinner_text(self, _instance: Any, value: str) -> None:
        if self._updating_language_spinner:
            return
        self.set_language(value)

    def _load_tile_assets(self, target_size: Optional[Tuple[int, int]] = None) -> None:
        if target_size is None:
            target_size = self._tile_texture_explicit_size
        if target_size is not None:
            width, height = target_size
            sanitized_size: Optional[Tuple[int, int]] = (
                max(1, int(width)),
                max(1, int(height)),
            )
        else:
            sanitized_size = None
        self._current_texture_render_size = sanitized_size

        self._raw_tile_textures.clear()
        if not self._asset_root.exists():
            return

        for tile_index, symbol in enumerate(_TILE_SYMBOLS):
            path = self._asset_root / f"{symbol}.svg"
            if not path.exists():
                continue
            texture = None
            if svg2png is not None:
                svg_kwargs: dict[str, Any] = {"url": str(path)}
                if sanitized_size is not None:
                    svg_kwargs["output_width"] = sanitized_size[0]
                    svg_kwargs["output_height"] = sanitized_size[1]
                if self._tile_texture_background is not None:
                    svg_kwargs["background_color"] = self._tile_texture_background
                try:
                    png_bytes = svg2png(**svg_kwargs)
                except Exception:
                    png_bytes = None
                else:
                    try:
                        buffer = BytesIO(png_bytes)
                        image = CoreImage(buffer, ext="png")
                        texture = image.texture
                    except Exception:
                        texture = None
            if texture is None:
                try:
                    image = CoreImage(str(path))
                    texture = image.texture
                except Exception:
                    texture = None
            if texture is not None:
                self._raw_tile_textures[tile_index] = texture

    def _toggle_auto(self) -> None:
        self._auto_advance = not self._auto_advance
        if self._auto_advance:
            self._step_once_requested = False
            if self._score_last() and self._pause_on_score:
                self._score_pause_active = True
                self._score_pause_pending = True
            else:
                self._score_pause_active = False
                self._score_pause_pending = False
        else:
            self._step_once_requested = False
            self._score_pause_active = False
            self._score_pause_pending = False

    def _trigger_step_once(self) -> None:
        if not self._auto_advance:
            self._step_once_requested = True
        elif self._score_pause_active:
            self._score_pause_active = False
            self._score_pause_pending = False

    def _toggle_pause(self) -> None:
        if not self._auto_advance:
            return
        self._pause_on_score = not self._pause_on_score
        if self._pause_on_score and self._score_last():
            self._score_pause_active = True
            self._score_pause_pending = True
        else:
            self._score_pause_active = False
            self._score_pause_pending = False

    def _toggle_hints(self) -> None:
        self._show_hints = not self._show_hints

    def _score_last(self) -> bool:
        return (
            getattr(self._env, "phase", "") == "score"
            and self._env.current_player == - 1
        )

    def _update_pause_state(self) -> None:
        score_panel_visible = self._score_last()
        if score_panel_visible:
            if self._auto_advance and self._pause_on_score:
                if not self._score_panel_was_visible:
                    self._score_pause_pending = True
                    self._score_pause_active = True
            else:
                self._score_pause_pending = False
                self._score_pause_active = False
            self._score_panel_was_visible = True
        elif self._score_pause_pending:
            if self._auto_advance and self._pause_on_score:
                self._score_pause_active = True
            else:
                self._score_pause_active = False
                self._score_pause_pending = False
            self._score_panel_was_visible = False
        else:
            self._score_pause_pending = False
            self._score_pause_active = False
            self._score_panel_was_visible = False

    def _compute_tile_metrics(self, play_rect: _Rect) -> None:
        width = max(1, int(play_rect.width))
        height = max(1, int(play_rect.height))
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
        self._update_auto_tile_texture_size()

    def _update_auto_tile_texture_size(self) -> None:
        if not self._tile_texture_auto_size:
            return
        base_size = self._tile_metrics.get("south_hand")
        if not base_size:
            return
        target_size = (max(1, int(base_size[0])), max(1, int(base_size[1])))
        if self._current_texture_render_size == target_size:
            return
        self._load_tile_assets(target_size)

    def _tile_texture_index(self, tile_136: int) -> int:
        tile_34 = tile_136 // 4
        if tile_136 == 16:
            return 34
        if tile_136 == 52:
            return 35
        if tile_136 == 88:
            return 36
        return tile_34

    def _indicator_to_dora_tile(self, indicator_tile: int) -> Optional[int]:
        if indicator_tile < 0:
            return None
        if indicator_tile <= 26:
            block = indicator_tile // 9
            index = indicator_tile % 9
            next_index = (index + 1) % 9
            return block * 9 + next_index
        if 27 <= indicator_tile <= 30:
            return 27 + ((indicator_tile - 27 + 1) % 4)
        if 31 <= indicator_tile <= 33:
            return 31 + ((indicator_tile - 31 + 1) % 3)
        return None

    def _compute_dora_highlight_tiles(self) -> set[int]:
        highlight: set[int] = set()
        indicators = getattr(self._env, "dora_indicator", [])
        for indicator in indicators:
            if not isinstance(indicator, int):
                continue
            base_index = indicator // 4
            dora_tile = self._indicator_to_dora_tile(base_index)
            if dora_tile is None:
                continue
            highlight.add(dora_tile)
            red_variant = _RED_FIVE_MAP.get(dora_tile)
            if red_variant is not None:
                highlight.add(red_variant)
        return highlight

    def _is_dora_tile(self, tile_136: int) -> bool:
        if not self._dora_highlight_tiles:
            return False
        return self._tile_texture_index(tile_136) in self._dora_highlight_tiles

    def _is_last_draw_tile(self, player_idx: int, tile_136: int) -> bool:
        if not self._cached_last_draw_tiles:
            return False
        if player_idx < 0 or player_idx >= len(self._cached_last_draw_tiles):
            return False
        return self._cached_last_draw_tiles[player_idx] == tile_136

    def _ensure_riichi_tracking_capacity(self, count: int) -> None:
        count = max(0, count)
        while len(self._riichi_states) < count:
            self._riichi_states.append(False)
            self._riichi_pending.append(False)
            self._discard_counts.append(0)
        if len(self._riichi_states) > count:
            self._riichi_states = self._riichi_states[:count]
            self._riichi_pending = self._riichi_pending[:count]
            self._discard_counts = self._discard_counts[:count]
        for idx in list(self._riichi_declarations):
            if idx >= count:
                self._riichi_declarations.pop(idx, None)

    def _update_riichi_state(self) -> None:
        riichi_flags = list(getattr(self._env, "riichi", []))
        discard_seq = list(getattr(self._env, "discard_pile_seq", []))
        num_players = max(len(riichi_flags), len(discard_seq), getattr(self._env, "num_players", 0))
        if num_players <= 0:
            self._riichi_states = []
            self._riichi_pending = []
            self._discard_counts = []
            self._riichi_declarations.clear()
            return

        self._ensure_riichi_tracking_capacity(num_players)

        for idx in range(num_players):
            is_riichi = riichi_flags[idx] if idx < len(riichi_flags) else False
            was_riichi = self._riichi_states[idx]
            if is_riichi and not was_riichi:
                self._riichi_pending[idx] = True
            elif not is_riichi:
                self._riichi_pending[idx] = False
                self._riichi_declarations.pop(idx, None)
            self._riichi_states[idx] = is_riichi

        for idx in range(num_players):
            tiles = discard_seq[idx] if idx < len(discard_seq) else []
            current_count = len(tiles)
            last_count = self._discard_counts[idx] if idx < len(self._discard_counts) else 0
            if current_count > last_count:
                if self._riichi_pending[idx]:
                    self._riichi_declarations[idx] = last_count
                    self._riichi_pending[idx] = False
            self._discard_counts[idx] = current_count

    def _on_frame(self, dt: float) -> None:
        if self._pending_action is not None and self._step_result is None:
            can_step = True
            if not self._auto_advance and not self._step_once_requested:
                can_step = False
            if self._score_pause_active:
                can_step = False
            if self._env.done:
                can_step = True
            if can_step:
                if self._step_once_requested:
                    self._step_once_requested = False
                action = self._pending_action
                observation, reward, done, info = self._env.step(action)
                self._pending_action = None
                self._step_result = (observation, reward, done, info)
                self._last_payload = _RenderPayload(
                    action=action,
                    reward=reward,
                    done=done,
                    info=info,
                )
                self._step_event.set()
        self._update_pause_state()
        self._render()
        if self._step_result is not None:
            self._step_event.set()

    def _render(self) -> None:
        board = self._root.board
        if board is None:
            return

        width, height = board.size
        play_size = height
        play_rect = _Rect(0, 0, play_size, play_size)
        play_rect.top = 0
        play_rect.centerx = width / 2

        if self._show_hints:
            self._dora_highlight_tiles = self._compute_dora_highlight_tiles()
            self._cached_last_draw_tiles = list(
                getattr(self._env, "last_draw_tiles", [])
            )
            self._cached_last_discard_tile = getattr(
                self._env, "last_discarded_tile", -1
            )
            self._cached_last_discarder = getattr(self._env, "last_discarder", -1)
        else:
            self._dora_highlight_tiles = set()
            self._cached_last_draw_tiles = []
            self._cached_last_discard_tile = -1
            self._cached_last_discarder = -1

        self._compute_tile_metrics(play_rect)
        self._update_riichi_state()

        self._hand_hitboxes.clear()

        canvas = board.canvas
        canvas.clear()

        self._draw_center_panel(canvas, board, play_rect)
        self._draw_dead_wall(canvas, board, play_rect)
        self._draw_player_areas(canvas, board, play_rect)
        if self._score_last():
            self._draw_score_panel(canvas, board, play_rect)
        self._draw_status_labels()
        self._update_control_buttons()

    def _show_human_actions(
        self,
        seat: int,
        actions: Sequence[Tuple[int, str]],
        deadline: Optional[float],
    ) -> None:
        actions_list = list(actions)

        def apply(_dt: float) -> None:
            quick_bar = getattr(self._root, "quick_action_bar", None)

            if not actions_list:
                if quick_bar is not None:
                    quick_bar.clear_widgets()
                    quick_bar.opacity = 0.0
                    quick_bar.disabled = True
                self._active_action_seat = None
                self._stop_action_countdown(clear=True)
                return

            quick_bar_used = False
            if quick_bar is not None and seat == self._focus_player:
                quick_entries: list[Tuple[int, str]] = []
                for action_id, label in actions_list:
                    label_text = str(label)
                    display_text: Optional[str] = None
                    if action_id > 33:
                        display_text: Optional[str] = label_text
                    if display_text is not None:
                        quick_entries.append((action_id, display_text))
                if quick_entries:
                    quick_bar.clear_widgets()
                    for action_id, label_text in quick_entries:
                        button = Button(
                            text=label_text,
                            size_hint=(None, None),
                            height=48,
                            width=max(144, int(len(label_text) * 18)),
                            background_normal="",
                            background_color=self._panel_border,
                            color=self._text_color,
                        )
                        button.background_down = ""
                        try:
                            button.font_name = self._font_name
                        except Exception:
                            pass
                        button.bind(
                            on_release=lambda _instance, act=action_id, seat_idx=seat: self._on_human_action_selected(seat_idx, act)
                        )
                        quick_bar.add_widget(button)
                    quick_bar.opacity = 1.0
                    quick_bar.disabled = False
                    quick_bar_used = True
                else:
                    quick_bar.clear_widgets()
                    quick_bar.opacity = 0.0
                    quick_bar.disabled = True
            elif quick_bar is not None:
                quick_bar.clear_widgets()
                quick_bar.opacity = 0.0
                quick_bar.disabled = True

            if quick_bar_used:
                self._active_action_seat = seat
                self._start_action_countdown(deadline)
                return

            self._active_action_seat = seat
            self._start_action_countdown(deadline)

        Clock.schedule_once(apply, 0)

    def _clear_human_actions(self, seat: Optional[int] = None) -> None:
        def apply(_dt: float) -> None:
            if seat is not None and self._active_action_seat != seat:
                return
            quick_bar = getattr(self._root, "quick_action_bar", None)
            if quick_bar is not None:
                quick_bar.clear_widgets()
                quick_bar.opacity = 0.0
                quick_bar.disabled = True
            self._active_action_seat = None
            
            self._stop_action_countdown(clear=True)

        Clock.schedule_once(apply, 0)

    def _on_human_action_selected(self, seat: int, action: int) -> None:
        agent = self._human_agents.get(seat)
        if agent is None:
            return
        self._stop_action_countdown(clear=True)
        try:
            agent.submit_action(action)
        except Exception:
            return

    def _start_action_countdown(self, deadline: Optional[float]) -> None:
        if deadline is None:
            self._stop_action_countdown(clear=True)
            return
        self._action_deadline = float(deadline)
        self._update_action_timer_text()
        if self._action_timer_event is None:
            self._action_timer_event = Clock.schedule_interval(
                self._on_action_timer_tick, 0.1
            )

    def _stop_action_countdown(self, clear: bool = False) -> None:
        if self._action_timer_event is not None:
            self._action_timer_event.cancel()
            self._action_timer_event = None
        self._action_deadline = None
        if clear:
            self._set_action_timer_text("")

    def _on_action_timer_tick(self, _dt: float) -> None:
        self._update_action_timer_text()

    def _update_action_timer_text(self) -> None:
        if self._action_deadline is None:
            self._set_action_timer_text("")
            return
        remaining = self._action_deadline - time.monotonic()
        if remaining <= 0:
            self._set_action_timer_text("")
            self._stop_action_countdown(clear=False)
            return
        self._set_action_timer_text(f"{remaining:0.1f}s")

    def _set_action_timer_text(self, text: str) -> None:
        label = getattr(self._root, "action_timer_label", None)
        if label is None:
            return
        label.text = text
        label.opacity = 1.0 if text else 0.0

    def handle_board_touch(self, board: MahjongBoardWidget, touch: Any) -> bool:
        pos = getattr(touch, "pos", None)
        if pos is None:
            return False
        canvas_x, canvas_y = float(pos[0]), float(pos[1])
        base_x, base_y = board.pos
        play_x = canvas_x - base_x
        play_y = canvas_y - base_y
        canvas_x = base_x + play_x
        canvas_y = base_y + play_y

        seat = self._active_action_seat
        if seat is None:
            return False
        hitboxes = self._hand_hitboxes.get(seat)
        if not hitboxes:
            return False
        agent = self._human_agents.get(seat)
        if agent is None:
            return False

        for tile_index, rect in reversed(hitboxes):
            left, bottom, right, top = rect
            if left <= canvas_x <= right and bottom <= canvas_y <= top:
                return agent.submit_action(tile_index)
        return False

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def _to_canvas_pos(
        self, board: MahjongBoardWidget, rect: _Rect, x: float, y: float, width: float, height: float
    ) -> Tuple[float, float]:
        base_x, base_y = board.pos
        return (
            base_x + rect.left + x,
            base_y + rect.top + rect.height - y - height,
        )

    def _tile_to_discard_action(self, tile_136: int) -> Optional[int]:
        if not isinstance(tile_136, int):
            return None
        tile_index = tile_136 // 4
        if 0 <= tile_index < 34:
            return tile_index
        return None

    def _wrap_text(self, text: str, max_width: float, font_size: Optional[int] = None) -> list[str]:
        if not text:
            return []
        max_width = max(10.0, float(max_width))
        font_size = int(font_size or self._font_size)
        words = str(text).split()
        lines: list[str] = []
        current = ""

        def measure(candidate: str) -> float:
            label = CoreLabel(text=candidate, font_size=font_size, font_name=self._font_name)
            label.refresh()
            return float(label.texture.size[0])

        for word in words:
            candidate = word if not current else f"{current} {word}"
            if measure(candidate) <= max_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    def _draw_center_panel(
        self, canvas: InstructionGroup, board: MahjongBoardWidget, play_rect: _Rect
    ) -> None:
        center_width = max(100, int(play_rect.width * 0.24))
        center_height = max(100, int(play_rect.height * 0.24))
        center_rect = _Rect(0, 0, center_width, center_height)
        center_rect.center = play_rect.center

        x, y = self._to_canvas_pos(board, play_rect, center_rect.left, center_rect.top, center_rect.width, center_rect.height)
        canvas.add(Color(*self._panel_color))
        canvas.add(
            RoundedRectangle(
                pos=(x, y),
                size=(center_rect.width, center_rect.height),
                radius=[12, 12, 12, 12],
            )
        )
        canvas.add(Color(*self._panel_border))
        canvas.add(
            Line(
                rounded_rectangle=(x, y, center_rect.width, center_rect.height, 12),
                width=2,
            )
        )

        round_data = getattr(self._env, "round", [0, 0])
        round_index = round_data[0] if isinstance(round_data[0], int) else 0
        wind_names = self._translate_sequence("wind_names") or ("East", "South", "West", "North")
        wind = wind_names[(round_index // 4) % len(wind_names)]
        hand_number = round_index % 4 + 1
        round_text = self._translate("round_format", wind=wind, hand=hand_number)

        title_label = CoreLabel(text=round_text, font_size=self._font_size + 8, font_name=self._font_name)
        title_label.refresh()
        label_x = center_rect.centerx - title_label.texture.size[0] / 2
        label_y = center_rect.top + 18
        px, py = self._to_canvas_pos(board, play_rect, label_x, label_y, *title_label.texture.size)
        canvas.add(Color(*self._text_color))
        canvas.add(Rectangle(texture=title_label.texture, size=title_label.texture.size, pos=(px, py)))

        honba = round_data[1] if len(round_data) > 1 and isinstance(round_data[1], int) else 0
        riichi = getattr(self._env, "num_riichi", 0)
        tiles_remaining = len(getattr(self._env, "deck", []))
        counter_texts = (
            self._translate("counter_honba", count=honba),
            self._translate("counter_riichi", count=riichi),
            self._translate("counter_tiles", count=tiles_remaining),
        )
        next_top = label_y + title_label.texture.size[1] + 12
        for text in counter_texts:
            label = CoreLabel(text=text, font_size=self._font_size - 4, font_name=self._font_name)
            label.refresh()
            label_x = center_rect.centerx - label.texture.size[0] / 2
            px, py = self._to_canvas_pos(board, play_rect, label_x, next_top, *label.texture.size)
            canvas.add(Color(*self._muted_text_color))
            canvas.add(Rectangle(texture=label.texture, size=label.texture.size, pos=(px, py)))
            next_top += label.texture.size[1] + 4

        score_positions = (
            (center_rect.centerx, center_rect.bottom + 14),
            (center_rect.right + 18, center_rect.centery),
            (center_rect.centerx, center_rect.top - 14),
            (center_rect.left - 18, center_rect.centery),
        )
        scores = getattr(self._env, "scores", [])
        current_player = getattr(self._env, "current_player", 0)
        order = self._get_display_order()
        for relative_position, player_idx in enumerate(order):
            if relative_position >= len(score_positions):
                break
            if player_idx >= len(scores):
                continue
            position = score_positions[relative_position]
            score_value = scores[player_idx] * 100
            color = (
                self._accent_color
                if player_idx == current_player
                else self._text_color
            )
            label = CoreLabel(
                text=f"{score_value:5d}",
                font_size=self._font_size,
                font_name=self._font_name,
            )
            label.refresh()
            px, py = self._to_canvas_pos(
                board,
                play_rect,
                position[0] - label.texture.size[0] / 2,
                position[1] - label.texture.size[1] / 2,
                *label.texture.size,
            )
            canvas.add(Color(*color))
            canvas.add(
                Rectangle(
                    texture=label.texture,
                    size=label.texture.size,
                    pos=(px, py),
                )
            )

    def _draw_dead_wall(
        self, canvas: InstructionGroup, board: MahjongBoardWidget, play_rect: _Rect
    ) -> None:
        wall_tile = self._tile_metrics.get("wall", (20, 26))
        stack_size = 5
        gap = 6
        margin_y = 32
        total_width = stack_size * (wall_tile[0] + gap) - gap
        start_x = play_rect.centerx + total_width // 2
        y = play_rect.centery + margin_y

        for i in range(stack_size):
            x = start_x - i * (wall_tile[0] + gap) - wall_tile[0]
            if i < len(self._env.dora_indicator):
                tile = self._env.dora_indicator[i]
                self._draw_tile(canvas, board, play_rect, tile, wall_tile, True, 0, (x, y))
            else:
                self._draw_tile(canvas, board, play_rect, 0, wall_tile, False, 0, (x, y))


    def _draw_player_areas(
        self, canvas: InstructionGroup, board: MahjongBoardWidget, play_rect: _Rect
    ) -> None:
        order = self._get_display_order()
        if not order:
            return
        num_players = getattr(self._env, "num_players", 0)
        reveal_flags = self._compute_hand_reveal_flags(num_players)
        for relative_position, player_idx in enumerate(order):
            face_up = (
                reveal_flags[player_idx]
                if player_idx < len(reveal_flags)
                else player_idx == self._focus_player
            )
            angle = self._get_relative_angle(relative_position)
            self._draw_player_layout(
                canvas, board, play_rect, player_idx, face_up, angle
            )

    def _compute_hand_reveal_flags(self, num_players: int) -> list[bool]:
        reveal = [False] * max(0, num_players)
        agari = getattr(self._env, "agari", None)
        phase = getattr(self._env, "phase", "")
        winners: set[int] = set()
        if agari:
            winners = self._extract_winner_indices(agari)

        if phase == "score":
            if winners:
                for idx in winners:
                    if 0 <= idx < len(reveal):
                        reveal[idx] = True
            else:
                tenpai_flags = list(getattr(self._env, "tenpai", []))
                for idx, is_tenpai in enumerate(tenpai_flags):
                    if is_tenpai and 0 <= idx < len(reveal):
                        reveal[idx] = True
        else:
            if winners:
                for idx in winners:
                    if 0 <= idx < len(reveal):
                        reveal[idx] = True
        if reveal:
            focus_idx = self._get_focus_index(len(reveal))
            if 0 <= focus_idx < len(reveal):
                reveal[focus_idx] = True
        return reveal

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

    def _draw_player_layout(
        self,
        canvas: InstructionGroup,
        board: MahjongBoardWidget,
        play_rect: _Rect,
        player_idx: int,
        face_up_hand: bool,
        angle: int,
    ) -> None:
        tile_size = self._tile_metrics.get("south_hand", (40, 56))
        meld_tile = self._tile_metrics.get("meld", tile_size)
        spacing = tile_size[0] + 6
        draw_gap = 0

        hands = getattr(self._env, "hands", [])
        hand_tiles = list(hands[player_idx]) if player_idx < len(hands) else []

        canvas.add(PushMatrix())
        center_px, center_py = self._to_canvas_pos(
            board, play_rect, play_rect.centerx, play_rect.centery, 0, 0
        )
        if angle:
            canvas.add(Translate(center_px, center_py))
            canvas.add(Rotate(angle=angle))
            canvas.add(Translate(-center_px, -center_py))

        x = play_rect.centerx - 7 * (tile_size[0] + 6)
        y = play_rect.bottom - tile_size[1] - 14
        is_focus_player = player_idx == self._focus_player
        if is_focus_player:
            self._hand_hitboxes[player_idx] = []
        for idx, tile in enumerate(hand_tiles):
            if len(hand_tiles) > 1 and idx == len(hand_tiles) - 1:
                x += draw_gap
            if is_focus_player:
                discard_action = self._tile_to_discard_action(tile)
                if discard_action is not None:
                    px, py = self._to_canvas_pos(
                        board, play_rect, x, y, tile_size[0], tile_size[1]
                    )
                    self._hand_hitboxes[player_idx].append(
                        (
                            discard_action,
                            (px, py, px + tile_size[0], py + tile_size[1]),
                        )
                    )
            highlight: Optional[_TileHighlight] = None
            if face_up_hand and self._show_hints:
                highlight_dora = self._is_dora_tile(tile)
                outline_color = None
                if self._is_last_draw_tile(player_idx, tile):
                    outline_color = self._last_draw_outline_color
                if highlight_dora or outline_color:
                    highlight = _TileHighlight(
                        dora=highlight_dora,
                        outline_color=outline_color,
                    )
            self._draw_tile(
                canvas,
                board,
                play_rect,
                tile,
                tile_size,
                face_up_hand,
                0,
                (x, y),
                highlight,
            )
            x += spacing

        riichi_flags = list(getattr(self._env, "riichi", []))
        in_riichi = player_idx < len(riichi_flags) and riichi_flags[player_idx]
        if in_riichi:
            label = CoreLabel(
                text=self._translate("riichi_flag"),
                font_size=self._font_size - 2,
                font_name=self._font_name,
            )
            label.refresh()
            padding = 4
            flag_width = label.texture.size[0] + padding * 2
            flag_height = label.texture.size[1] + padding * 2
            flag_left = play_rect.centerx - flag_width / 2
            flag_bottom = y - 6
            flag_top = flag_bottom - flag_height
            px, py = self._to_canvas_pos(
                board, play_rect, flag_left, flag_top, flag_width, flag_height
            )
            canvas.add(Color(*self._panel_color))
            canvas.add(
                RoundedRectangle(
                    pos=(px, py),
                    size=(flag_width, flag_height),
                    radius=[6, 6, 6, 6],
                )
            )
            canvas.add(Color(*self._accent_color))
            canvas.add(Line(rounded_rectangle=(px, py, flag_width, flag_height, 6), width=2))
            canvas.add(
                Rectangle(
                    texture=label.texture,
                    size=label.texture.size,
                    pos=(px + padding, py + padding),
                )
            )

        self._draw_discards(canvas, board, play_rect, player_idx, tile_size)
        self._draw_melds(canvas, board, play_rect, player_idx, tile_size, meld_tile)

        canvas.add(PopMatrix())

    def _draw_score_panel(
        self, canvas: InstructionGroup, board: MahjongBoardWidget, play_rect: _Rect
    ) -> None:
        margin = 0
        padding = 14
        min_dimension = min(play_rect.width, play_rect.height)
        available_side = min_dimension - margin * 2
        effective_side = available_side if available_side > 0 else min_dimension
        max_text_width = max(50.0, effective_side - padding * 2)

        round_data = getattr(self._env, "round", [0, 0])
        round_index = round_data[0] if isinstance(round_data[0], int) else 0
        wind_names = self._translate_sequence("wind_names") or ("East", "South", "West", "North")
        wind = wind_names[(round_index // 4) % len(wind_names)]
        hand_number = round_index % 4 + 1
        honba = round_data[1] if len(round_data) > 1 and isinstance(round_data[1], int) else 0
        riichi_sticks = getattr(self._env, "num_riichi", 0)
        kyoutaku = getattr(self._env, "num_kyoutaku", 0)

        info_lines = [
            self._translate(
                "info_line_format",
                round=self._translate("round_format", wind=wind, hand=hand_number),
                honba=self._translate("counter_honba", count=honba),
                riichi=self._translate("counter_riichi", count=riichi_sticks),
                kyoutaku=self._translate("counter_kyoutaku", count=kyoutaku),
            )
        ]

        seat_names = list(getattr(self._env, "seat_names", []))
        num_players = getattr(self._env, "num_players", 0)
        if len(seat_names) < num_players:
            seat_names.extend(
                [
                    self._translate("seat_placeholder_format", index=idx)
                    for idx in range(len(seat_names), num_players)
                ]
            )

        agari = getattr(self._env, "agari", None)
        message_lines: list[str] = []
        if isinstance(agari, dict) and agari:
            winner = agari.get("who", -1)
            from_who = agari.get("fromwho", -1)
            if 0 <= winner < num_players:
                winner_name = seat_names[winner]
            else:
                winner_name = self._translate("player_name_format", index=winner)
            if winner == from_who:
                result_text = self._translate(
                    "result_tsumo",
                    winner=winner_name,
                    tsumo_label=self._translate("tsumo_label"),
                )
            else:
                loser_name = (
                    seat_names[from_who]
                    if 0 <= from_who < num_players
                    else self._translate("player_name_format", index=from_who)
                )
                result_text = self._translate(
                    "result_ron",
                    winner=winner_name,
                    ron_label=self._translate("ron_label"),
                    vs_label=self._translate("vs_label"),
                    loser=loser_name,
                )
            message_lines.append(result_text)
            ten = list(agari.get("ten", []))
            fu = ten[0] if len(ten) > 0 else 0
            total = ten[1] if len(ten) > 1 else 0
            han = ten[2] if len(ten) > 2 else 0
            message_lines.append(
                self._translate("result_details", han=han, fu=fu, total=total)
            )
        else:
            tenpai_flags = list(getattr(self._env, "tenpai", []))
            tenpai_players = [
                seat_names[idx]
                for idx, is_tenpai in enumerate(tenpai_flags)
                if is_tenpai and idx < num_players
            ]
            if tenpai_players:
                message_lines.append(
                    self._translate(
                        "draw_tenpai",
                        players=", ".join(tenpai_players),
                    )
                )
            else:
                message_lines.append(self._translate("draw_no_tenpai"))

        yaku_font_size = max(18, self._font_size - 4)
        yaku_label_proto = CoreLabel(
            text=self._translate("yaku_prefix"),
            font_size=yaku_font_size,
            font_name=self._font_name,
        )
        yaku_label_proto.refresh()
        yaku_label_width = yaku_label_proto.texture.size[0]

        yaku_lines: list[tuple[str, str]] = []
        if isinstance(agari, dict) and agari:
            raw_yaku = [str(item) for item in agari.get("yaku", [])]
            if raw_yaku:
                combined = ", ".join(raw_yaku)
                wrapped_yaku = self._wrap_text(
                    combined, max(10.0, max_text_width - yaku_label_width), font_size=yaku_font_size
                )
                if wrapped_yaku:
                    yaku_lines.append((self._translate("yaku_prefix"), wrapped_yaku[0]))
                    for extra in wrapped_yaku[1:]:
                        yaku_lines.append(("", extra))

        def make_label(text: str, font_size: int) -> CoreLabel:
            label = CoreLabel(text=text, font_size=font_size, font_name=self._font_name)
            label.refresh()
            return label

        title_font_size = self._font_size + 6
        info_font_size = max(10, self._font_size - 2)
        message_font_size = max(10, self._font_size - 2)
        player_font_size = self._font_size

        title_label = make_label(self._translate("round_results_title"), title_font_size)
        content_width = title_label.texture.size[0]

        info_labels = [make_label(line, info_font_size) for line in info_lines]
        if info_labels:
            info_width = max(label.texture.size[0] for label in info_labels)
            content_width = max(content_width, info_width)

        message_labels = [make_label(line, message_font_size) for line in message_lines]
        if message_labels:
            message_width = max(label.texture.size[0] for label in message_labels)
            content_width = max(content_width, message_width)

        scores = list(getattr(self._env, "scores", []))
        if len(scores) < num_players:
            scores.extend([0] * (num_players - len(scores)))
        score_deltas = list(getattr(self._env, "score_deltas", []))
        if len(score_deltas) < num_players:
            score_deltas.extend([0] * (num_players - len(score_deltas)))
        scores = [s + d for s, d in zip(scores, score_deltas)]
        dealer = getattr(self._env, "oya", -1)

        ranks: dict[int, int] = {}
        sorted_players = sorted(range(num_players), key=lambda idx: (-scores[idx], idx))
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

        player_entries = []
        player_section_height = 0.0
        for player_idx in range(num_players):
            if player_idx >= len(scores):
                continue
            display_name = seat_names[player_idx]
            if player_idx == dealer:
                display_name += self._translate("dealer_suffix")
            base_color = self._accent_color if player_idx == winner_idx else self._text_color
            rank_text = self._format_ordinal(ranks.get(player_idx, player_idx + 1))
            name_label = make_label(f"{rank_text}  {display_name}", player_font_size)
            delta_value = score_deltas[player_idx] if player_idx < len(score_deltas) else 0
            delta_points = int(round(delta_value * 100))
            delta_color = (
                self._accent_color
                if delta_points > 0
                else self._danger_color if delta_points < 0 else self._muted_text_color
            )
            delta_label = make_label(f"{delta_points:+}", player_font_size)
            score_points = int(round(scores[player_idx] * 100))
            score_label = make_label(f"{score_points:>6d}", player_font_size)

            name_width = name_label.texture.size[0]
            delta_width = delta_label.texture.size[0]
            score_width = score_label.texture.size[0]
            score_line_width = name_width + score_width + delta_width + 16
            content_width = max(content_width, score_line_width)

            row_height = max(
                name_label.texture.size[1],
                delta_label.texture.size[1],
                score_label.texture.size[1],
            ) + 4
            player_section_height += row_height

            player_entries.append(
                {
                    "name": name_label,
                    "delta": delta_label,
                    "score": score_label,
                    "name_color": base_color,
                    "delta_color": delta_color,
                    "row_height": row_height,
                }
            )

        yaku_entries = []
        yaku_height = 0.0
        for prefix, text in yaku_lines:
            prefix_label = make_label(prefix, yaku_font_size) if prefix else None
            text_label = make_label(text, yaku_font_size)
            prefix_width = prefix_label.texture.size[0] if prefix_label else 0
            text_width = text_label.texture.size[0]
            total_width = prefix_width + (0 if prefix else yaku_label_width) + text_width
            content_width = max(content_width, total_width)
            entry_height = max(
                text_label.texture.size[1],
                prefix_label.texture.size[1] if prefix_label else 0,
            ) + 2
            yaku_height += entry_height
            yaku_entries.append(
                {
                    "prefix": prefix_label,
                    "text": text_label,
                    "entry_height": entry_height,
                }
            )

        info_height = sum(label.texture.size[1] + 2 for label in info_labels)
        message_height = sum(label.texture.size[1] + 2 for label in message_labels)

        panel_height = padding * 2 + title_label.texture.size[1]
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

        panel_rect = _Rect(0, 0, panel_size, panel_size)
        panel_rect.center = play_rect.center

        panel_pos = self._to_canvas_pos(
            board, play_rect, panel_rect.left, panel_rect.top, panel_rect.width, panel_rect.height
        )
        canvas.add(Color(*self._panel_color))
        canvas.add(
            RoundedRectangle(
                pos=panel_pos,
                size=(panel_rect.width, panel_rect.height),
                radius=[12, 12, 12, 12],
            )
        )
        canvas.add(Color(*self._panel_border))
        canvas.add(
            Line(rounded_rectangle=(*panel_pos, panel_rect.width, panel_rect.height, 12), width=2)
        )

        def draw_label(label: CoreLabel, color: Tuple[float, float, float, float], left: float, top: float) -> None:
            px, py = self._to_canvas_pos(board, play_rect, left, top, *label.texture.size)
            canvas.add(Color(*color))
            canvas.add(Rectangle(texture=label.texture, size=label.texture.size, pos=(px, py)))

        current_y = panel_rect.top + padding
        title_left = panel_rect.centerx - title_label.texture.size[0] / 2
        draw_label(title_label, self._accent_color, title_left, current_y)
        current_y += title_label.texture.size[1] + 6

        for label in info_labels:
            draw_label(label, self._muted_text_color, panel_rect.left + padding, current_y)
            current_y += label.texture.size[1] + 2
        if info_labels:
            current_y += 4

        for label in message_labels:
            draw_label(label, self._text_color, panel_rect.left + padding, current_y)
            current_y += label.texture.size[1] + 2
        if message_labels:
            current_y += 4

        for entry in player_entries:
            name_label: CoreLabel = entry["name"]
            delta_label: CoreLabel = entry["delta"]
            score_label: CoreLabel = entry["score"]
            name_color = entry["name_color"]
            delta_color = entry["delta_color"]

            draw_label(name_label, name_color, panel_rect.left + padding, current_y)
            delta_left = panel_rect.right - padding - delta_label.texture.size[0]
            draw_label(delta_label, delta_color, delta_left, current_y)
            score_left = delta_left - 16 - score_label.texture.size[0]
            draw_label(score_label, name_color, score_left, current_y)
            current_y += entry["row_height"]
        if player_entries:
            current_y += 4

        for entry in yaku_entries:
            prefix_label = entry["prefix"]
            text_label = entry["text"]
            x = panel_rect.left + padding
            if prefix_label is not None:
                draw_label(prefix_label, self._accent_color, x, current_y)
                x += prefix_label.texture.size[0]
            else:
                x += yaku_label_width
            draw_label(text_label, self._text_color, x, current_y)
            current_y += entry["entry_height"]

    def _draw_discards(
        self,
        canvas: InstructionGroup,
        board: MahjongBoardWidget,
        play_rect: _Rect,
        player_idx: int,
        tile_size: Tuple[int, int],
    ) -> None:
        discard_tiles = self._get_discard_tiles(player_idx)
        discard_tile = self._tile_metrics.get("discard", tile_size)
        cols = 6
        grid_half_width = 3 * (discard_tile[0] + 4)
        grid_width = 6 * (discard_tile[0] + 4)
        grid_height = 4 * (discard_tile[1] + 4)
        discard_rect = _Rect(play_rect.centerx - grid_half_width, 0, grid_width, grid_height)
        discard_rect.top = play_rect.bottom - tile_size[1] - 14 - 4 * (discard_tile[1] + 4) - 24
        orientation_map: Optional[dict[int, int]] = None
        declaration_index = self._riichi_declarations.get(player_idx)
        if declaration_index is not None:
            orientation_map = {declaration_index: 90}
        highlight_map: dict[int, _TileHighlight] = {}
        if self._show_hints:
            last_index = len(discard_tiles) - 1
            for idx, tile in enumerate(discard_tiles):
                highlight_dora = self._is_dora_tile(tile)
                outline_color = None
                if (
                    player_idx == self._cached_last_discarder
                    and idx == last_index
                    and tile == self._cached_last_discard_tile
                ):
                    outline_color = self._last_discard_outline_color
                if highlight_dora or outline_color:
                    highlight_map[idx] = _TileHighlight(
                        dora=highlight_dora,
                        outline_color=outline_color,
                    )
        self._draw_tile_grid(
            canvas,
            board,
            play_rect,
            discard_tiles,
            discard_rect,
            discard_tile,
            0,
            cols,
            orientation_map,
            highlight_map if highlight_map else None,
        )

    def _draw_melds(
        self,
        canvas: InstructionGroup,
        board: MahjongBoardWidget,
        play_rect: _Rect,
        player_idx: int,
        hand_tile: Tuple[int, int],
        meld_tile: Tuple[int, int],
    ) -> None:
        melds = getattr(self._env, "melds", [])
        if player_idx >= len(melds):
            return
        margin_side = 14
        max_meld_width = meld_tile[0] * 4 + 12
        x = max(margin_side, play_rect.width - margin_side - max_meld_width)
        y = play_rect.bottom - hand_tile[1] - 14
        spacing = 8
        for meld in melds[player_idx]:
            tiles = list(reversed([tile for tile in meld.get("m", [])]))
            opened = meld.get("opened", True)
            meld_type = meld.get("type", "")
            claimed_tile = meld.get("claimed_tile")
            sideways_index: Optional[int] = None
            if claimed_tile is not None:
                try:
                    sideways_index = tiles.index(claimed_tile)
                except ValueError:
                    sideways_index = None
            cur_x, cur_y = x, y
            total_tiles = len(tiles)
            for idx, tile in enumerate(tiles):
                tile_orientation = 0
                if sideways_index is not None and idx == sideways_index:
                    tile_orientation = 90
                tile_face_up = opened
                if not opened and meld_type == "kan" and total_tiles >= 4:
                    if idx in {0, total_tiles - 1}:
                        tile_face_up = False
                    else:
                        tile_face_up = True
                highlight: Optional[_TileHighlight] = None
                if self._show_hints and tile_face_up and self._is_dora_tile(tile):
                    highlight = _TileHighlight(dora=True)
                self._draw_tile(
                    canvas,
                    board,
                    play_rect,
                    tile,
                    meld_tile,
                    tile_face_up,
                    tile_orientation,
                    (cur_x, cur_y),
                    highlight,
                )
                if sideways_index is not None and idx in {sideways_index - 1, sideways_index}:
                    cur_x -= (meld_tile[0]+meld_tile[1])/2 + 4
                else:
                    cur_x -= meld_tile[0] + 4
            x = cur_x - spacing

    def _draw_tile_grid(
        self,
        canvas: InstructionGroup,
        board: MahjongBoardWidget,
        play_rect: _Rect,
        tiles: list[int],
        area: _Rect,
        tile_size: Tuple[int, int],
        orientation: int,
        columns: int,
        orientation_map: Optional[dict[int, int]] = None,
        highlight_map: Optional[dict[int, _TileHighlight]] = None,
    ) -> None:
        if not tiles:
            return
        columns = max(1, columns)
        spacing = 4
        x0, y0 = area.left, area.top
        x, y = x0, y0
        for idx, tile in enumerate(tiles):
            column = idx % columns
            row_prev = -1 if idx == 0 else min(3, (idx - 1) // columns)
            row = min(3, idx // columns)
            tile_orientation = orientation
            if orientation_map and idx in orientation_map:
                tile_orientation = orientation_map[idx]
            if row != row_prev:
                x = x0
            elif orientation_map and idx - 1 in orientation_map:
                x += (tile_size[0]+tile_size[1])/2 + spacing
            elif orientation_map and idx in orientation_map:
                x += (tile_size[0]+tile_size[1])/2 + spacing
            else:
                x += tile_size[0] + spacing
            y = area.top + row * (tile_size[1] + spacing)
            highlight = highlight_map.get(idx) if highlight_map else None
            self._draw_tile(
                canvas,
                board,
                play_rect,
                tile,
                tile_size,
                True,
                tile_orientation,
                (x, y),
                highlight,
            )

    def _draw_tile(
        self,
        canvas: InstructionGroup,
        board: MahjongBoardWidget,
        play_rect: _Rect,
        tile_136: int,
        size: Tuple[int, int],
        face_up: bool,
        orientation: int,
        origin: Tuple[float, float],
        highlight: Optional[_TileHighlight] = None,
    ) -> None:
        width, height = size
        x, y = origin
        pos = self._to_canvas_pos(board, play_rect, x, y, width, height)
        canvas.add(PushMatrix())
        cx = pos[0] + width / 2
        cy = pos[1] + height / 2
        canvas.add(Translate(cx, cy))
        if orientation:
            canvas.add(Rotate(angle=orientation))
        canvas.add(Translate(-width / 2, -height / 2))

        if not face_up:
            canvas.add(Color(*self._face_down_color))
            canvas.add(RoundedRectangle(pos=(0, 0), size=size, radius=[6, 6, 6, 6]))
            canvas.add(Color(*self._face_down_border))
            canvas.add(Line(rounded_rectangle=(0, 0, width, height, 6), width=2))
            canvas.add(PopMatrix())
            return

        tile_34 = self._tile_texture_index(tile_136)
        highlight_data = highlight if highlight else None

        texture = self._raw_tile_textures.get(tile_34)
        if texture is None:
            self._draw_tile_placeholder(canvas, board, play_rect, tile_34, size, origin, local=True)
            canvas.add(PopMatrix())
            return

        canvas.add(Color(1, 1, 1, 1))
        canvas.add(RoundedRectangle(texture=texture, size=size, pos=(0, 0), radius=[6, 6, 6, 6]))
        if highlight_data and highlight_data.dora:
            canvas.add(Color(*self._dora_overlay_color))
            canvas.add(RoundedRectangle(pos=(0, 0), size=size, radius=[6, 6, 6, 6]))
        if highlight_data and highlight_data.outline_color is not None:
            canvas.add(Color(*highlight_data.outline_color))
            canvas.add(
                Line(
                    rounded_rectangle=(0, 0, width, height, 6),
                    width=self._hint_outline_width,
                )
            )
        canvas.add(PopMatrix())

    def _draw_tile_placeholder(
        self,
        canvas: InstructionGroup,
        board: MahjongBoardWidget,
        play_rect: _Rect,
        tile_34: int,
        size: Tuple[int, int],
        origin: Tuple[float, float],
        local: bool = False,
    ) -> None:
        width, height = size
        if local:
            pos = (0, 0)
        else:
            x, y = origin
            pos = self._to_canvas_pos(board, play_rect, x, y, width, height)
        background = self._get_tile_color(tile_34)
        bg = tuple(c / 255.0 for c in background)
        canvas.add(Color(*bg, 1))
        canvas.add(RoundedRectangle(pos=pos, size=size, radius=[6, 6, 6, 6]))
        canvas.add(Color(245 / 255.0, 245 / 255.0, 245 / 255.0, 1))
        canvas.add(Line(rounded_rectangle=(*pos, *size, 6), width=2))
        label = CoreLabel(
            text=_TILE_SYMBOLS[tile_34],
            font_size=max(12, int(height * 0.45)),
            font_name=self._font_name,
        )
        label.refresh()
        label_pos = (
            pos[0] + width / 2 - label.texture.size[0] / 2,
            pos[1] + height / 2 - label.texture.size[1] / 2,
        )
        canvas.add(Color(20 / 255.0, 20 / 255.0, 20 / 255.0, 1))
        canvas.add(Rectangle(texture=label.texture, size=label.texture.size, pos=label_pos))

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

    def _get_discard_tiles(self, player_idx: int) -> list[int]:
        if player_idx >= len(getattr(self._env, "discard_pile_seq", [])):
            return []
        return [t for t in self._env.discard_pile_seq[player_idx]]

    def _draw_status_labels(self) -> None:
        if not self._root:
            return
        if self._env.current_player >= 0:
            player_text = self._translate(
                "seat_placeholder_format", index=self._env.current_player
            )
        else:
            player_text = self._translate("unknown_player")
        phase_text = self._translate(
            "status_format",
            phase_label=self._translate("phase_label"),
            current_player_label=self._translate("current_player_label"),
            phase=self._env.phase,
            player=player_text,
        )
        self._root.status_label.text = phase_text
        reward_color = self._danger_color if self._last_payload.reward < 0 else self._text_color
        action_value = (
            "-" if self._last_payload.action is None else str(self._last_payload.action)
        )
        reward_text = self._translate(
            "action_reward_format",
            action_label=self._translate("action_label"),
            reward_label=self._translate("reward_label"),
            action=action_value,
            reward=f"{self._last_payload.reward:.2f}",
        )
        self._root.reward_label.text = reward_text
        self._root.reward_label.color = reward_color
        if self._last_payload.done:
            self._root.done_label.text = self._translate("episode_finished")
        else:
            self._root.done_label.text = ""

    def _update_control_buttons(self) -> None:
        pause_label = (
            self._translate("pause_on_score_on")
            if self._pause_on_score
            else self._translate("pause_on_score_off")
        )
        self._root.pause_button.text = pause_label
        self._root.pause_button.disabled = not self._auto_advance
        auto_label = (
            self._translate("auto_next_on")
            if self._auto_advance
            else self._translate("auto_next_off")
        )
        self._root.auto_button.text = auto_label
        hints_label = (
            self._translate("hints_on")
            if self._show_hints
            else self._translate("hints_off")
        )
        self._root.hints_button.text = hints_label
        step_enabled = (not self._auto_advance) or self._score_pause_active
        self._root.step_button.disabled = not step_enabled
        self._root.step_button.text = self._translate("step_next")

