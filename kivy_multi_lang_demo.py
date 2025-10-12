"""A small Kivy demo that switches between multiple languages at runtime.

Run with ``python kivy_multi_lang_demo.py``. The app shows a simple
interface where changing the language updates all visible texts without
restarting the application.
"""

import logging
import os
from pathlib import Path
from typing import Iterable

from kivy.app import App
from kivy.core.text import Label as CoreTextLabel
from kivy.lang import Builder
from kivy.metrics import dp
from kivy.properties import DictProperty, ListProperty, StringProperty
from kivy.resources import resource_find
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.spinner import SpinnerOption


logger = logging.getLogger(__name__)


FONT_SEARCH_DIRECTORIES = (
    Path(__file__).parent / "assets" / "fonts",
    Path(__file__).parent / "assets",
    Path(__file__).parent,
    Path.home() / ".fonts",
    Path.home() / "Library/Fonts",
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
)

FONT_NAME_CANDIDATES = {
    "notosanscjk-regular.ttc",
    "notosanscjksc-regular.otf",
    "notosanscjksc-regular.ttc",
    "notosanssc-regular.otf",
    "notosanssc-regular.ttf",
    "sourcehansanssc-regular.otf",
    "sourcehansans-regular.otf",
    "droidsansfallback.ttf",
    "wenquanyimicrohei.ttf",
    "wenquanyimicroheimonospaced.ttf",
    "wenquanyimicrohei_mono.ttf",
    "simhei.ttf",
    "simhei.ttc",
    "msyh.ttf",
    "msyh.ttc",
    "msgothic.ttc",
    "msmincho.ttc",
    "yugothm.ttc",
    "notosansjp-regular.otf",
    "notosansjp-regular.ttf",
    "notosansjp-regular.ttc",
    "notosans-regular.ttf",
    "arialuni.ttf",
    "arialunicodems.ttf",
    "ipagp.ttf",
    "ipag.ttf",
    "ipam.ttf",
    "ipagui.ttf",
}

TEST_STRINGS = ("欢迎！", "ようこそ！")


KV = """
<TranslatedSpinnerOption@SpinnerOption>:
    font_name: app.ui_font_name or self.font_name

<LanguageSwitcher>:
    orientation: "vertical"
    padding: dp(20)
    spacing: dp(16)

    Label:
        font_size: "26sp"
        markup: True
        text: "[b]" + root.translated_title + "[/b]"
        text_size: self.width, None
        halign: "center"
        size_hint_y: None
        height: self.texture_size[1]
        font_name: root.ui_font_name

    Label:
        text: root.translated_body
        halign: "center"
        valign: "middle"
        text_size: self.width, None
        size_hint_y: None
        height: self.texture_size[1]
        font_name: root.ui_font_name

    BoxLayout:
        size_hint_y: None
        height: dp(52)
        spacing: dp(10)

        Label:
            text: root.translated_select_language
            size_hint_x: None
            width: dp(160)
            text_size: self.size
            halign: "left"
            valign: "middle"
            font_name: root.ui_font_name

        Spinner:
            text: root.current_language_display
            values: root.language_display_names
            on_text: root.on_language_selected(self.text)
            font_name: root.ui_font_name
            option_cls: "TranslatedSpinnerOption"

    Button:
        text: root.translated_action_button
        size_hint_y: None
        height: dp(48)
        on_release: root.on_action_pressed()
        font_name: root.ui_font_name

    Label:
        text: root.last_action_message
        color: 0.2, 0.6, 0.2, 1
        text_size: self.width, None
        size_hint_y: None
        height: self.texture_size[1]
        font_name: root.ui_font_name
"""


def _iter_candidate_fonts() -> Iterable[Path]:
    """Yield candidate font paths that might support CJK glyphs."""

    lower_candidates = {name.lower() for name in FONT_NAME_CANDIDATES}
    keyword_hints = (
        "noto",
        "sourcehan",
        "wenquanyi",
        "droid",
        "unicode",
        "goth",
        "hei",
        "song",
        "ipa",
    )
    visited = set()

    for directory in FONT_SEARCH_DIRECTORIES:
        if not directory:
            continue
        try:
            if not directory.exists():
                continue
        except OSError:
            continue

        try:
            for path in directory.rglob("*"):
                if not path.is_file():
                    continue
                suffix = path.suffix.lower()
                if suffix not in {".ttf", ".ttc", ".otf"}:
                    continue

                name_lower = path.name.lower()
                if name_lower not in lower_candidates and not any(
                    hint in name_lower for hint in keyword_hints
                ):
                    continue

                resolved = path.resolve()
                if resolved in visited:
                    continue
                visited.add(resolved)
                yield resolved
        except (FileNotFoundError, PermissionError, OSError):
            continue


def _font_supports_text(font_path: str) -> bool:
    """Return ``True`` if the font renders sample multilingual strings."""

    try:
        for sample in TEST_STRINGS:
            label = CoreTextLabel(text=sample, font_name=font_path)
            label.refresh()
            texture = label.texture
            if not texture or texture.size == (0, 0):
                return False
    except Exception:  # pragma: no cover - Kivy specific exceptions
        logger.debug("Font %s failed to render sample text", font_path, exc_info=True)
        return False

    return True


def find_multilingual_font() -> str:
    """Locate a font file that contains Chinese and Japanese glyphs."""

    env_font = os.environ.get("KIVY_DEMO_FONT")
    if env_font:
        env_path = Path(env_font).expanduser()
        if env_path.is_file() and _font_supports_text(str(env_path)):
            return str(env_path)

    for font_path in _iter_candidate_fonts():
        font_str = str(font_path)
        if _font_supports_text(font_str):
            logger.info("Using multilingual font: %s", font_str)
            return font_str

    for candidate in FONT_NAME_CANDIDATES:
        resolved = resource_find(candidate)
        if resolved and _font_supports_text(resolved):
            logger.info("Using packaged multilingual font: %s", resolved)
            return resolved

    fallback = resource_find("data/fonts/Roboto-Regular.ttf")
    if fallback:
        if not _font_supports_text(fallback):
            logger.warning(
                "No multilingual font with CJK glyphs was found. Using %s as a fallback; "
                "some characters may not render correctly.",
                fallback,
            )
        return fallback

    logger.warning(
        "Unable to locate any font file for multilingual rendering. Text may appear blank."
    )
    return ""


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

    translated_title = StringProperty()
    translated_body = StringProperty()
    translated_select_language = StringProperty()
    translated_action_button = StringProperty()
    last_action_message = StringProperty("")
    ui_font_name = StringProperty("")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.language_display_names = list(self.language_display_map.keys())
        if not self.ui_font_name:
            self.ui_font_name = find_multilingual_font()
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
    def build(self):
        # Preload a font capable of displaying multilingual text before
        # kv rules access ``app.ui_font_name``.
        self.ui_font_name = find_multilingual_font()
        Builder.load_string(KV)
        return LanguageSwitcher(ui_font_name=self.ui_font_name)


if __name__ == "__main__":
    MultiLanguageApp().run()
