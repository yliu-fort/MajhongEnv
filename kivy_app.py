import os
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".", "src"))

from kivy.app import App
from kivy.lang import Builder

from mahjong_env import MahjongEnv
from mahjong_wrapper_kivy import MahjongEnvKivyWrapper, MahjongRootLayout


class MahjongKivyApp(App):
    """Entry point for the Kivy Mahjong application."""

    def build(self):
        env = MahjongEnv(num_players=4)
        kv_path = Path(__file__).resolve().parent / "assets" / "layout" / "mahjong_gui.kv"
        root = Builder.load_file(str(kv_path))
        if not isinstance(root, MahjongRootLayout):
            raise TypeError("KV layout did not produce MahjongRootLayout instance")

        self.wrapper = MahjongEnvKivyWrapper(env=env, kv_file=str(kv_path))
        self.wrapper.bind_root(root)
        self.wrapper.reset()
        self.wrapper.schedule()
        return root

    def on_stop(self):
        if hasattr(self, "wrapper"):
            self.wrapper.close()


if __name__ == "__main__":
    MahjongKivyApp().run()
