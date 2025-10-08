import os
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".", "agent"))

from kivy.app import App
from kivy.clock import Clock
from kivy.lang import Builder

from agent.visual_agent import VisualAgent
from mahjong_env import MahjongEnv
from mahjong_wrapper_kivy import MahjongEnvKivyWrapper


class MahjongKivyApp(App):
    """Entry point for the Kivy Mahjong visualiser."""

    def build(self):
        base_path = Path(__file__).resolve().parent
        layout_path = base_path / "assets" / "layout" / "mahjong_gui.kv"
        if layout_path.exists():
            Builder.load_file(str(layout_path))

        self.env = MahjongEnv(num_players=4)
        self.wrapper = MahjongEnvKivyWrapper(env=self.env)
        self.agent = VisualAgent(self.env, backbone="resnet50")
        self.agent.load_model("model_weights/latest.pt")
        self._observation = self.wrapper.reset()

        interval = 1.0 / max(1, self.wrapper.fps)
        Clock.schedule_interval(self._drive_environment, interval)
        return self.wrapper.root

    def _drive_environment(self, _dt: float) -> None:
        result = self.wrapper.fetch_step_result()
        if result is not None:
            self._observation, _, done, _ = result
            if done:
                return

        if self.env.done and self.wrapper.pending_action is None:
            self._observation = self.wrapper.reset()
            return

        if self.wrapper.pending_action is not None:
            return

        action = self.agent.predict(self._observation)
        self.wrapper.queue_action(action)


if __name__ == "__main__":
    MahjongKivyApp().run()
