"""A small Kivy demo that switches between multiple languages at runtime.

Run with ``python kivy_multi_lang_demo.py``. The app shows a simple
interface where changing the language updates all visible texts without
restarting the application.
"""

import os
import sys
from pathlib import Path
from typing import Iterable

from kivy.app import App
from kivy.core.text import LabelBase
from kivy.lang import Builder
from kivy.logger import Logger
from kivy.metrics import dp
from kivy.properties import DictProperty, ListProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout


KV = """
<LocalizedSpinnerOption@SpinnerOption>:
    font_name: app.ui_font_name if app else 'Roboto'

<LanguageSwitcher>:
    orientation: "vertical"
    padding: dp(20)
    spacing: dp(16)

    Label:
        font_name: root.ui_font_name
        font_size: "26sp"
        markup: True
        text: "[b]" + root.translated_title + "[/b]"
        text_size: self.width, None
        halign: "center"
        size_hint_y: None
        height: self.texture_size[1]

    Label:
        font_name: root.ui_font_name
        text: root.translated_body
        halign: "center"
        valign: "middle"
        text_size: self.width, None
        size_hint_y: None
        height: self.texture_size[1]

    BoxLayout:
        size_hint_y: None
        height: dp(52)
        spacing: dp(10)

        Label:
            font_name: root.ui_font_name
            text: root.translated_select_language
            size_hint_x: None
            width: dp(160)
            text_size: self.size
            halign: "left"
            valign: "middle"

        Spinner:
            id: language_spinner
            text: root.current_language_display
            values: root.language_display_names
            font_name: root.ui_font_name
            option_cls: 'LocalizedSpinnerOption'
            on_text: root.on_language_selected(self.text)

    Button:
        font_name: root.ui_font_name
        text: root.translated_action_button
        size_hint_y: None
        height: dp(48)
        on_release: root.on_action_pressed()

    Label:
        font_name: root.ui_font_name
        text: root.last_action_message
        color: 0.2, 0.6, 0.2, 1
        text_size: self.width, None
        size_hint_y: None
        height: self.texture_size[1]
"""


def _iter_font_search_paths() -> Iterable[Path]:
    """Yield platform-specific directories that may contain fonts."""

    # Common font directories for Linux, macOS, and Windows.
    unix_paths = [
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        Path.home() / ".fonts",
    ]
    mac_paths = [
        Path("/System/Library/Fonts"),
        Path("/Library/Fonts"),
        Path.home() / "Library" / "Fonts",
    ]
    windows_fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"

    if sys.platform.startswith("darwin"):
        yield from mac_paths
    elif sys.platform.startswith("win"):
        yield windows_fonts
    else:
        yield from unix_paths

    # Also search the repository's assets directory if it exists.
    project_assets = Path(__file__).resolve().parent / "assets"
    if project_assets.exists():
        for maybe_fonts in (project_assets, project_assets / "fonts"):
            yield maybe_fonts


def register_multilingual_font() -> str:
    """Register and return the font name that supports CJK scripts.

    The function tries to locate a font file that covers Simplified Chinese
    and Japanese glyphs. If no candidate is found, the default Kivy font is
    returned so the UI remains functional even though glyph coverage may be
    limited.
    """

    font_name = "Roboto"
    candidates = (
        "NotoSansCJK-Regular.ttc",
        "NotoSansCJKsc-Regular.otf",
        "NotoSansCJKjp-Regular.otf",
        "NotoSansSC-Regular.otf",
        "NotoSansJP-Regular.otf",
        "SourceHanSansSC-Regular.otf",
        "SourceHanSansCN-Regular.otf",
        "SourceHanSansJP-Regular.otf",
        "SourceHanSans-Regular.otf",
        "WenQuanYiMicroHei.ttf",
        "DroidSansFallback.ttf",
        "SimHei.ttf",
        "MSYH.TTC",
        "msgothic.ttc",
    )

    search_tokens = ("noto", "sourcehan", "wenquanyi", "droid", "msyh", "msmincho", "msgothic", "arialuni", "ume", "ipaex")

    for directory in _iter_font_search_paths():
        if not directory.exists():
            continue

        # First, check direct matches.
        for candidate in candidates:
            candidate_path = directory / candidate
            if candidate_path.exists():
                LabelBase.register(
                    name="LocalizedFallback",
                    fn_regular=str(candidate_path),
                )
                Logger.info("LanguageDemo: Using multilingual font %s", candidate_path)
                return "LocalizedFallback"

        # If not found, perform a broader scan for partial matches.
        try:
            for entry in directory.iterdir():
                potential_files: Iterable[Path]
                if entry.is_dir():
                    potential_files = (
                        sub_entry
                        for ext in ("*.ttf", "*.otf", "*.ttc")
                        for sub_entry in entry.rglob(ext)
                    )
                elif entry.is_file():
                    potential_files = (entry,)
                else:
                    continue

                for font_path in potential_files:
                    if font_path.suffix.lower() not in {".ttf", ".otf", ".ttc"}:
                        continue
                    lowered = font_path.name.lower()
                    if any(token in lowered for token in search_tokens):
                        LabelBase.register(
                            name="LocalizedFallback",
                            fn_regular=str(font_path),
                        )
                        Logger.info("LanguageDemo: Using multilingual font %s", font_path)
                        return "LocalizedFallback"
        except (PermissionError, OSError):
            # Some system directories may be protected or unreadable; skip them quietly.
            continue

    Logger.warning(
        "LanguageDemo: No CJK-capable font found; falling back to Roboto."
    )
    return font_name


class LanguageSwitcher(BoxLayout):
    """Root widget responsible for handling translations."""

    translations = DictProperty({
        "en": {
            "title": "Welcome!",
            "body": "This demo shows how Kivy text can update when you switch the language.",
            "select_language": "Select language",
            "action_button": "Press me",
            "action_response": "You pressed the button!",
        },
        "zh": {
            "title": "欢迎！",
            "body": "此示例展示了在切换语言时如何即时更新 Kivy 文本。",
            "select_language": "选择语言",
            "action_button": "点我",
            "action_response": "你点击了按钮！",
        },
        "ja": {
            "title": "ようこそ！",
            "body": "このデモでは、言語を切り替えると Kivy のテキストが即座に更新されます。",
            "select_language": "言語を選択",
            "action_button": "押してね",
            "action_response": "ボタンが押されました！",
        },
        "fr": {
            "title": "Bienvenue !",
            "body": "Cette démo montre comment les textes Kivy se mettent à jour lorsque vous changez de langue.",
            "select_language": "Choisissez la langue",
            "action_button": "Appuyez sur moi",
            "action_response": "Vous avez appuyé sur le bouton !",
        },
    })

    language_display_map = DictProperty({
        "English": "en",
        "简体中文": "zh",
        "日本語": "ja",
        "Français": "fr",
    })

    language_display_names = ListProperty()
    current_language = StringProperty("en")
    current_language_display = StringProperty("English")

    ui_font_name = StringProperty("Roboto")

    translated_title = StringProperty()
    translated_body = StringProperty()
    translated_select_language = StringProperty()
    translated_action_button = StringProperty()
    last_action_message = StringProperty("")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.language_display_names = list(self.language_display_map.keys())
        self.update_translated_texts()

    def get_translation(self, key: str) -> str:
        lang_data = self.translations.get(self.current_language, {})
        return lang_data.get(key, self.translations["en"].get(key, key))

    def update_translated_texts(self) -> None:
        self.translated_title = self.get_translation("title")
        self.translated_body = self.get_translation("body")
        self.translated_select_language = self.get_translation("select_language")
        self.translated_action_button = self.get_translation("action_button")

    def on_language_selected(self, display_name: str) -> None:
        language_code = self.language_display_map.get(display_name)
        if not language_code or language_code == self.current_language:
            return

        self.current_language = language_code
        self.current_language_display = display_name
        self.last_action_message = ""
        self.update_translated_texts()

    def on_action_pressed(self) -> None:
        self.last_action_message = self.get_translation("action_response")


class MultiLanguageApp(App):
    ui_font_name = StringProperty("Roboto")

    def build(self):
        Builder.load_string(KV)
        self.ui_font_name = register_multilingual_font()
        return LanguageSwitcher(ui_font_name=self.ui_font_name)


if __name__ == "__main__":
    MultiLanguageApp().run()
