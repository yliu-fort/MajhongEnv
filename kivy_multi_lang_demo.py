"""Demo Kivy application with runtime language switching.

This module defines a simple user interface that allows the user to
switch between multiple languages (English, Chinese (Simplified), Japanese,
and French) at runtime.  The UI is defined entirely in Python for the sake
of portability in this repository and keeps the demo self-contained.

Run the module directly to see the demo:

    python kivy_multi_lang_demo.py

The demo attempts to register a fallback font bundled with Kivy so that
Chinese and Japanese glyphs render correctly.  If those characters do not
appear as expected, provide a custom font by editing the `_register_default_font`
method to point at a font file installed on your system.
"""

from __future__ import annotations

from dataclasses import dataclass

from kivy.app import App
from kivy.lang import Builder
from kivy.properties import DictProperty, ListProperty, StringProperty
from kivy.resources import resource_find
from kivy.uix.boxlayout import BoxLayout
from kivy.core.text import LabelBase


@dataclass(frozen=True)
class Translation:
    """Container mapping message identifiers to translated strings."""

    greeting: str
    instructions: str


class LanguagePanel(BoxLayout):
    """Widget responsible for rendering the language switching UI."""

    translations = DictProperty({})
    language_labels = DictProperty({})
    language_order = ListProperty([])
    display_languages = ListProperty([])
    current_language = StringProperty("en")
    current_display_language = StringProperty("")
    current_greeting = StringProperty("")
    current_instructions = StringProperty("")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._initialize_translations()
        self._register_default_font()
        self.set_language(self.current_language)

    def _initialize_translations(self) -> None:
        """Populate the translations used throughout the demo."""

        self.translations = {
            "en": Translation(
                greeting="Hello!",
                instructions="Use the menu below to change the language.",
            ),
            "zh": Translation(
                greeting="你好！",
                instructions="使用下面的菜单切换显示语言。",
            ),
            "ja": Translation(
                greeting="こんにちは！",
                instructions="下のメニューで表示言語を切り替えてください。",
            ),
            "fr": Translation(
                greeting="Bonjour !",
                instructions="Utilisez le menu ci-dessous pour changer la langue.",
            ),
        }
        self.language_labels = {
            "en": "English",
            "zh": "中文 (简体)",
            "ja": "日本語",
            "fr": "Français",
        }
        self.language_order = ["en", "zh", "ja", "fr"]
        self.display_languages = [
            self.language_labels[code] for code in self.language_order
        ]

    def _register_default_font(self) -> None:
        """Register a fallback font that supports multiple writing systems."""

        fallback_font = (
            resource_find("data/fonts/DejaVuSans.ttf")
            or resource_find("data/fonts/Roboto-Regular.ttf")
        )
        if fallback_font:
            LabelBase.register(name="MultilingualFallback", fn_regular=fallback_font)

    def set_language(self, language_code: str) -> None:
        """Set the language currently displayed in the UI."""

        if language_code not in self.translations:
            return
        self.current_language = language_code
        translation: Translation = self.translations[language_code]
        self.current_greeting = translation.greeting
        self.current_instructions = translation.instructions
        self.current_display_language = self.language_labels.get(
            language_code, language_code
        )

    def on_display_language_selected(self, display_name: str) -> None:
        """Handle the Spinner selection and map it back to a language code."""

        code = self._code_from_display(display_name)
        if code and code != self.current_language:
            self.set_language(code)

    def _code_from_display(self, display_name: str) -> str | None:
        for code, label in self.language_labels.items():
            if label == display_name:
                return code
        return None


KV = """
<LanguagePanel>:
    orientation: "vertical"
    padding: "20dp"
    spacing: "15dp"

    canvas.before:
        Color:
            rgba: 0.15, 0.15, 0.18, 1
        Rectangle:
            pos: self.pos
            size: self.size

    Label:
        text: root.current_greeting
        font_size: "42sp"
        bold: True
        color: 1, 1, 1, 1
        font_name: "MultilingualFallback"

    Label:
        text: root.current_instructions
        font_size: "20sp"
        color: 0.85, 0.85, 0.9, 1
        text_size: self.width, None
        halign: "center"
        valign: "middle"
        font_name: "MultilingualFallback"

    Widget:
        size_hint_y: None
        height: "20dp"

    Label:
        text: "Language"
        color: 0.9, 0.9, 0.95, 1
        font_size: "18sp"
        size_hint_y: None
        height: "30dp"

    Spinner:
        id: language_spinner
        text: root.current_display_language
        values: root.display_languages
        size_hint: (1, None)
        height: "48dp"
        background_color: 0.2, 0.2, 0.25, 1
        color: 1, 1, 1, 1
        on_text: root.on_display_language_selected(self.text)
"""


class MultiLangApp(App):
    """Entry point for the multilingual demo application."""

    def build(self):  # type: ignore[override]
        Builder.load_string(KV)
        return LanguagePanel()


if __name__ == "__main__":
    MultiLangApp().run()
