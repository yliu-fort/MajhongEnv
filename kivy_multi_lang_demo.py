"""Kivy demo app that switches between multiple languages at runtime.

Run with::

    python kivy_multi_lang_demo.py

This example demonstrates a simple runtime language switcher that updates
interface text immediately when a different language is selected.
"""

from kivy.app import App
from kivy.lang import Builder
from kivy.metrics import dp
from kivy.properties import DictProperty, ListProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout

LANGUAGE_DATA = [
    ("English", "en"),
    ("中文 (简体)", "zh"),
    ("日本語", "ja"),
    ("Français", "fr"),
]

KV = """
<MultiLanguageRoot>:
    orientation: "vertical"
    padding: dp(24)
    spacing: dp(18)

    Label:
        text: root.get_translation("title")
        font_size: "28sp"
        bold: True
        halign: "center"
        valign: "middle"
        text_size: self.size

    Label:
        text: root.get_translation("instructions")
        font_size: "18sp"
        halign: "center"
        valign: "middle"
        text_size: self.width, None
        size_hint_y: None
        height: self.texture_size[1]

    Spinner:
        id: language_spinner
        size_hint_y: None
        height: dp(48)
        text: root.spinner_text
        values: root.language_names
        on_text: root.on_language_selected(self.text)

    Button:
        size_hint_y: None
        height: dp(48)
        text: root.get_translation("next_button")
        on_release: root.cycle_language()

    Label:
        size_hint_y: None
        height: self.texture_size[1]
        font_size: "16sp"
        halign: "center"
        valign: "middle"
        text_size: self.width, None
        text: root.get_translation("current_language", language_name=root.spinner_text)
"""


class MultiLanguageRoot(BoxLayout):
    """Root widget that handles language selection and translation lookup."""

    translations = DictProperty(
        {
            "en": {
                "title": "Welcome",
                "instructions": (
                    "Pick a language from the dropdown or press the button to cycle through "
                    "available translations."
                ),
                "next_button": "Next language",
                "current_language": "Current language: {language_name}",
            },
            "zh": {
                "title": "欢迎",
                "instructions": "从下拉列表中选择一种语言，或按按钮循环切换可用翻译。",
                "next_button": "下一种语言",
                "current_language": "当前语言：{language_name}",
            },
            "ja": {
                "title": "ようこそ",
                "instructions": "ドロップダウンから言語を選ぶか、ボタンで利用可能な翻訳を切り替えます。",
                "next_button": "次の言語",
                "current_language": "現在の言語：{language_name}",
            },
            "fr": {
                "title": "Bienvenue",
                "instructions": "Liste déroulante ou bouton : choisissez la langue à afficher instantanément.",
                "next_button": "Langue suivante",
                "current_language": "Langue actuelle : {language_name}",
            },
        }
    )

    language_names = ListProperty([name for name, _ in LANGUAGE_DATA])
    language_codes = DictProperty({name: code for name, code in LANGUAGE_DATA})
    spinner_text = StringProperty(LANGUAGE_DATA[0][0])
    current_language = StringProperty(LANGUAGE_DATA[0][1])

    def on_kv_post(self, base_widget):
        """Ensure the translations match the initial spinner text."""
        self.on_language_selected(self.spinner_text)

    def on_language_selected(self, display_name: str) -> None:
        """Update the current language when the user picks a new option."""
        if display_name not in self.language_codes:
            return

        self.spinner_text = display_name
        self.current_language = self.language_codes[display_name]

    def cycle_language(self) -> None:
        """Advance to the next available language, looping back when needed."""
        try:
            index = self.language_names.index(self.spinner_text)
        except ValueError:
            index = -1

        next_index = (index + 1) % len(self.language_names)
        next_language = self.language_names[next_index]
        self.on_language_selected(next_language)

    def get_translation(self, key: str, **kwargs) -> str:
        """Return the translated string for the active language.

        ``kwargs`` are forwarded to ``str.format`` for dynamic messages.
        """

        language_strings = self.translations.get(self.current_language, {})
        template = language_strings.get(key, key)
        if kwargs:
            try:
                return template.format(**kwargs)
            except (KeyError, ValueError):
                return template
        return template


class MultiLanguageDemoApp(App):
    """Application entry point."""

    def build(self):
        Builder.load_string(KV)
        return MultiLanguageRoot()


if __name__ == "__main__":
    MultiLanguageDemoApp().run()
