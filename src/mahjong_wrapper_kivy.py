from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

try:
    from kivy.clock import Clock
    from kivy.lang import Builder
    from kivy.properties import (
        BooleanProperty,
        ObjectProperty,
        StringProperty,
    )
    from kivy.uix.boxlayout import BoxLayout
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise RuntimeError(
        "kivy is required for the MahjongEnv Kivy GUI wrapper; install kivy to continue"
    ) from exc

from mahjong_env import MahjongEnvBase as _BaseMahjongEnv
from mahjong_features import get_action_from_index
from mahjong_tiles_print_style import tile_printout, tiles_printout


_KV_PATH = Path(__file__).resolve().with_name("mahjong_gui.kv")
_KV_LOADED = False


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


class PlayerPanel(BoxLayout):
    """Panel widget used to present one player's state."""

    display_name = StringProperty("")
    hand_text = StringProperty("")
    discard_text = StringProperty("")
    meld_text = StringProperty("")
    score_text = StringProperty("")
    score_delta_text = StringProperty("")
    wind_text = StringProperty("")
    riichi = BooleanProperty(False)
    is_current = BooleanProperty(False)
    dealer = BooleanProperty(False)

    def update_from_dict(self, data: dict[str, Any]) -> None:
        self.display_name = data.get("name", "")
        self.hand_text = data.get("hand", "")
        self.discard_text = data.get("discards", "")
        self.meld_text = data.get("melds", "")
        self.score_text = data.get("score", "")
        self.score_delta_text = data.get("score_delta", "")
        self.wind_text = data.get("wind", "")
        self.riichi = bool(data.get("riichi", False))
        self.is_current = bool(data.get("is_current", False))
        self.dealer = bool(data.get("dealer", False))


class MahjongRoot(BoxLayout):
    """Root widget hosting the Mahjong GUI."""

    wrapper = ObjectProperty(None, allownone=True)
    status_text = StringProperty("")
    phase_text = StringProperty("")
    round_text = StringProperty("")
    dora_text = StringProperty("")
    ura_text = StringProperty("")
    log_text = StringProperty("")
    auto_advance = BooleanProperty(True)
    pause_on_score = BooleanProperty(False)
    score_pause_active = BooleanProperty(False)

    def on_toggle_auto(self) -> None:
        if self.wrapper is not None:
            self.wrapper.toggle_auto()

    def on_step(self) -> None:
        if self.wrapper is not None:
            self.wrapper.request_step()

    def on_toggle_pause(self) -> None:
        if self.wrapper is not None:
            self.wrapper.toggle_pause_on_score()

    def on_reset(self) -> None:
        if self.wrapper is not None:
            self.wrapper.reset()

    # ------------------------------------------------------------------
    # UI update helpers
    # ------------------------------------------------------------------
    def update_state(self, state: dict[str, Any]) -> None:
        self.status_text = state.get("status_text", "")
        self.phase_text = state.get("phase_text", "")
        self.round_text = state.get("round_text", "")
        self.dora_text = state.get("dora_text", "")
        self.ura_text = state.get("ura_text", "")
        self.auto_advance = bool(state.get("auto_advance", True))
        self.pause_on_score = bool(state.get("pause_on_score", False))
        self.score_pause_active = bool(state.get("score_pause_active", False))

        log_lines = state.get("log_lines", [])
        if log_lines:
            self.log_text = "\n".join(log_lines)
        else:
            self.log_text = ""

        players = state.get("players", [])
        for idx in range(4):
            panel = self.ids.get(f"player{idx}")
            if panel is None:
                continue
            panel_state = players[idx] if idx < len(players) else {}
            panel.update_from_dict(panel_state)


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

        for idx in range(num_players):
            hand_tiles = hands[idx] if idx < len(hands) else []
            discard_tiles = discards[idx] if idx < len(discards) else []
            meld_list = melds[idx] if idx < len(melds) else []
            score_value = scores[idx] if idx < len(scores) else 0
            delta_value = deltas[idx] if idx < len(deltas) else 0
            riichi = riichi_flags[idx] if idx < len(riichi_flags) else False

            players.append(
                {
                    "name": self._format_player_name(idx, dealer),
                    "hand": self._format_tiles(hand_tiles),
                    "discards": f"Discards: {self._format_tiles(discard_tiles, sort=False)}"
                    if discard_tiles
                    else "Discards: -",
                    "melds": self._format_melds(meld_list),
                    "score": f"{int(round(score_value * 100)):,}",
                    "score_delta": self._format_delta(delta_value),
                    "riichi": riichi,
                    "is_current": idx == current,
                    "dealer": idx == dealer,
                    "wind": f"Seat wind: {winds[idx % len(winds)]}",
                }
            )

        round_text = self._format_round_text()
        dora_text = self._format_indicator_text("Dora", getattr(self._env, "dora_indicator", []))
        ura_text = self._format_indicator_text("Ura", getattr(self._env, "ura_indicator", []))
        status_text = self._status_text
        if self._pause_on_score and self._score_pause_active:
            status_text = status_text or "Paused for score calculation"

        return {
            "players": players,
            "phase_text": getattr(self._env, "phase", ""),
            "round_text": round_text,
            "dora_text": dora_text,
            "ura_text": ura_text,
            "status_text": status_text,
            "auto_advance": self._auto_advance,
            "pause_on_score": self._pause_on_score,
            "score_pause_active": self._score_pause_active,
            "log_lines": list(self._log_messages),
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

    def _format_round_text(self) -> str:
        round_info = getattr(self._env, "round", [-1, 0])
        if not isinstance(round_info, Sequence) or len(round_info) < 2:
            return ""
        round_index = int(round_info[0])
        honba = int(round_info[1])
        if round_index < 0:
            return "East 1 | Honba 0"
        winds = ["East", "South", "West", "North"]
        num_players = max(1, getattr(self._env, "num_players", 4))
        wind_idx = (round_index // num_players) % len(winds)
        hand_number = (round_index % num_players) + 1
        return f"{winds[wind_idx]} {hand_number} | Honba {honba}"

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
