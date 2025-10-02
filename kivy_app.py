"""Entry point for the Kivy Mahjong GUI demo."""
from __future__ import annotations

import random
from itertools import cycle

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window

from majhong_wrapper_kivy import (
    MahjongKivyWidget,
    MahjongTableState,
    MeldSnapshot,
    PlayerSnapshot,
    ScoreEntry,
    ScoreSummary,
)


_TILE_CODES = [
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
    "E",
    "S",
    "W",
    "N",
    "Wh",
    "Gr",
    "Rd",
]


class DemoStateController:
    """Generates illustrative Mahjong table states for the demo app."""

    def __init__(self, widget: MahjongKivyWidget) -> None:
        self.widget = widget
        self.step_index = 0
        self.auto_next = True
        self.pause_on_score = True
        self.pending_step = False
        self.current_state: MahjongTableState | None = None

        widget.bind(on_toggle_auto=self._toggle_auto)
        widget.bind(on_toggle_pause_on_score=self._toggle_pause)
        widget.bind(on_step_requested=self._request_step)

        Clock.schedule_interval(self._tick, 1.0 / widget.target_fps)
        self._update_state()

    def _toggle_auto(self, _widget: MahjongKivyWidget, value: bool) -> None:
        self.auto_next = value
        self._update_state()

    def _toggle_pause(self, _widget: MahjongKivyWidget, value: bool) -> None:
        self.pause_on_score = value
        self._update_state()

    def _request_step(self, _widget: MahjongKivyWidget) -> None:
        if (
            self.current_state
            and self.current_state.score_summary
            and self.pause_on_score
        ) or not self.auto_next:
            self.pending_step = True

    def _tick(self, _dt: float) -> None:
        if self.current_state and self.current_state.score_summary and self.pause_on_score:
            # Locked on score summary until user resumes
            if self.pending_step or not self.auto_next:
                if self.pending_step:
                    self.pending_step = False
                    self.step_index += 1
                    self._update_state()
            return

        if self.auto_next or self.pending_step:
            if self.pending_step:
                self.pending_step = False
            self.step_index += 1
            self._update_state()

    def _update_state(self) -> None:
        state = self._build_state()
        self.current_state = state
        self.widget.set_table_state(state)

    def _build_state(self) -> MahjongTableState:
        random.seed(self.step_index)
        winds = ["South", "East", "North", "West"]
        names = ["You", "Bot A", "Bot B", "Bot C"]

        phase_cycle = cycle(["draw", "discard", "meld", "score"])
        for _ in range(self.step_index % 4):
            next(phase_cycle)
        phase = next(phase_cycle)

        discards = [
            [random.choice(_TILE_CODES) for _ in range(random.randint(5, 14))],
            [random.choice(_TILE_CODES) for _ in range(random.randint(4, 12))],
            [random.choice(_TILE_CODES) for _ in range(random.randint(4, 13))],
            [random.choice(_TILE_CODES) for _ in range(random.randint(3, 11))],
        ]

        players: list[PlayerSnapshot] = []
        for index in range(4):
            hand_tiles = random.sample(_TILE_CODES, 13)
            drawn_tile = random.choice(_TILE_CODES) if index == 0 else None
            melds = []
            if random.random() < 0.3:
                meld = MeldSnapshot(tiles=random.sample(_TILE_CODES, 3), concealed=False, taken_from=random.randint(0, 3))
                melds.append(meld)
            players.append(
                PlayerSnapshot(
                    name=names[index],
                    wind=winds[index],
                    score=250 + random.randint(-30, 30),
                    hand=hand_tiles,
                    discards=discards[index],
                    melds=melds,
                    face_up_hand=index == 0 or phase == "score",
                    drawn_tile=drawn_tile,
                    is_dealer=index == 1,
                    is_riichi=random.random() < 0.25,
                    riichi_declaration_index=random.randint(0, len(discards[index]) - 1) if discards[index] else None,
                    tenpai=random.random() < 0.3,
                )
            )

        dora_count = 5
        dora = [random.choice(_TILE_CODES) for _ in range(random.randint(2, dora_count))]

        score_summary = None
        if self.step_index % 12 == 8:
            entries = [
                ScoreEntry(name=p.name, seat_index=i, score=p.score + random.randint(-5, 15), delta=random.randint(-8, 18), is_winner=(i == 0), is_dealer=p.is_dealer)
                for i, p in enumerate(players)
            ]
            yaku_lines = ["Riichi", "Pinfu", "Dora x2"]
            score_summary = ScoreSummary(
                title="South 2 — Tsumo",
                subtitle="Mangan",
                description="You won the hand with riichi tsumo.",
                entries=entries,
                yaku_lines=yaku_lines,
            )

        last_action = random.choice(["Draw 5m", "Discard 9s", "Pon", "Chi", "Riichi"])
        reward = random.uniform(-1.0, 1.0)
        wall_remaining = 70 - (self.step_index % 60)

        step_enabled = (not self.auto_next) or (
            score_summary is not None and self.pause_on_score
        )

        return MahjongTableState(
            players=players,
            round_wind="South",
            hand_number=1 + (self.step_index % 4),
            honba=self.step_index % 3,
            riichi_sticks=self.step_index % 4,
            wall_remaining=max(0, wall_remaining),
            dora_indicators=dora,
            dead_wall_count=dora_count,
            phase=phase,
            active_player=self.step_index % 4,
            last_action=last_action,
            reward=reward,
            episode_finished=self.step_index % 18 == 0,
            auto_next=self.auto_next,
            pause_on_score=self.pause_on_score,
            step_enabled=step_enabled,
            score_summary=score_summary,
        )


class MahjongKivyApp(App):
    """Standalone Kivy application showcasing the Mahjong GUI widget."""

    def build(self) -> MahjongKivyWidget:
        Window.size = (1024, 720)
        widget = MahjongKivyWidget()
        DemoStateController(widget)
        return widget


if __name__ == "__main__":
    MahjongKivyApp().run()
