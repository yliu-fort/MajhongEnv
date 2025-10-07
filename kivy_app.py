import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".", "agent"))

from kivy.app import App
from kivy.clock import Clock

from mahjong_env import MahjongEnv
from mahjong_wrapper_kivy import (
    MahjongEnvKivyWrapper,
    MahjongRoot,
    load_kivy_layout,
)
from agent.random_discard_agent import RandomDiscardAgent


class MahjongKivyApp(App):
    """Standalone Kivy application for observing MahjongEnv games."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.title = "MahjongEnv Kivy GUI"
        self.wrapper = MahjongEnvKivyWrapper(env=MahjongEnv(num_players=4))
        self.agent = RandomDiscardAgent(self.wrapper)
        self._observation = None
        self._reset_event = None

    def build(self):
        load_kivy_layout()
        root = MahjongRoot()
        self.wrapper.attach_ui(root)
        self._observation = self.wrapper.reset()
        Clock.schedule_interval(self._auto_step, 0.6)
        return root

    def _auto_step(self, _: float) -> None:
        if self.wrapper.done:
            if self.wrapper.auto_advance:
                self._schedule_reset()
            return

        if not self.wrapper.wants_step():
            return

        self._observation = self.wrapper.last_observation
        action = self.agent.predict(self._observation)
        observation, _reward, done, _info = self.wrapper.step(action)
        self._observation = observation

        if done and self.wrapper.auto_advance:
            self._schedule_reset()

    def _schedule_reset(self, delay: float = 1.0) -> None:
        if self._reset_event is not None:
            return

        def _do_reset(_dt):
            self._reset_event = None
            self._observation = self.wrapper.reset()

        self._reset_event = Clock.schedule_once(_do_reset, delay)

    def on_stop(self) -> None:
        self.wrapper.close()


def main() -> None:
    MahjongKivyApp().run()


if __name__ == "__main__":
    main()
