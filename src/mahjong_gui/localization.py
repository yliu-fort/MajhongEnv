"""Localization helpers for the Mahjong Kivy wrapper."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

DEFAULT_LANGUAGE = "en"
LANGUAGE_ORDER: Tuple[str, ...] = ("en", "zh-Hans", "ja", "fr")
_ASSET_FONT_ROOT = Path(__file__).resolve().parent.parent / "assets" / "fonts"
_FONT_PATHS: Dict[str, Path] = {
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


_ORDINAL_FUNCTIONS: Dict[str, Callable[[int], str]] = {
    "en": _ordinal_en,
    "zh-Hans": _ordinal_zh,
    "ja": _ordinal_ja,
    "fr": _ordinal_fr,
}

_LANGUAGE_STRINGS: Dict[str, Dict[str, Any]] = {
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
        "step_next": "Suivant",
        "riichi_flag": "Riichi",
        "tenpai_label": "Tenpai",
        "no_tenpai_label": "Pas de Tenpai",
    },
}


class LocalizationManager:
    """Provide access to localized strings and formatting helpers."""

    def __init__(self, default_language: str = DEFAULT_LANGUAGE) -> None:
        self._strings = _LANGUAGE_STRINGS
        if self._strings:
            if default_language in self._strings:
                self._default_language = default_language
            else:
                self._default_language = next(iter(self._strings))
        else:
            self._default_language = default_language
        self._code_to_name = {
            code: data.get("language_name", code) for code, data in self._strings.items()
        }
        self._name_to_code = {name: code for code, name in self._code_to_name.items()}

    @property
    def default_language(self) -> str:
        return self._default_language

    @property
    def available_codes(self) -> Tuple[str, ...]:
        return tuple(self._strings.keys())

    def display_name(self, code: str) -> str:
        return self._code_to_name.get(code, code)

    def resolve_code(self, identifier: str) -> Optional[str]:
        if not identifier:
            return None
        if identifier in self._strings:
            return identifier
        return self._name_to_code.get(identifier)

    def spinner_values(self) -> Tuple[str, ...]:
        values: list[str] = []
        for code in LANGUAGE_ORDER:
            name = self._code_to_name.get(code)
            if name and name not in values:
                values.append(name)
        for name in self._code_to_name.values():
            if name not in values:
                values.append(name)
        return tuple(values)

    def language_dict(self, code: Optional[str]) -> Dict[str, Any]:
        language = code or self._default_language
        if language in self._strings:
            return self._strings[language]
        return self._strings.get(self._default_language, {})

    def translate(self, code: str, key: str, **kwargs: Any) -> str:
        language_dict = self.language_dict(code)
        template = language_dict.get(key)
        if template is None:
            template = self.language_dict(self._default_language).get(key, key)
        if isinstance(template, str):
            return template.format(**kwargs)
        return str(template)

    def translate_sequence(self, code: str, key: str) -> Tuple[str, ...]:
        value = self.language_dict(code).get(key)
        if isinstance(value, (list, tuple)):
            return tuple(value)
        default_value = self.language_dict(self._default_language).get(key, ())
        if isinstance(default_value, (list, tuple)):
            return tuple(default_value)
        return ()

    def ordinal_formatter(self, code: str) -> Callable[[int], str]:
        default_func = _ORDINAL_FUNCTIONS.get(
            self._default_language, lambda value: str(value)
        )
        return _ORDINAL_FUNCTIONS.get(code, default_func)

    def format_ordinal(self, code: str, value: int) -> str:
        try:
            formatter = self.ordinal_formatter(code)
            return str(formatter(int(value)))
        except Exception:
            return str(value)

    def resolve_fonts(
        self, code: str, *, user_font_name: Optional[str], fallback: str
    ) -> Tuple[str, Tuple[str, ...]]:
        font_path = _FONT_PATHS.get(code)
        if font_path is not None and font_path.exists():
            font_name = str(font_path)
        elif user_font_name:
            font_name = user_font_name
        else:
            font_name = fallback
        fallback_entries: list[str] = []
        if _FALLBACK_FONT.exists():
            fallback_entries.append(str(_FALLBACK_FONT))
        return font_name, tuple(fallback_entries)


__all__ = [
    "LocalizationManager",
    "DEFAULT_LANGUAGE",
    "LANGUAGE_ORDER",
]
