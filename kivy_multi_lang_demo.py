"""Kivy multi-language switching demo.

This script creates a small Kivy application that showcases
how to switch the user interface language at runtime without
restarting the app. It supports English, Simplified Chinese,
Japanese, and French, and can serve as a starting point for
more advanced internationalisation work.
"""

from kivy.app import App
from kivy.lang import Builder
from kivy.metrics import dp
from kivy.properties import DictProperty, ListProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout


KV = """
<LanguageDemo>:
    orientation: "vertical"
    padding: dp(20)
    spacing: dp(12)

    Spinner:
        id: language_spinner
        size_hint_y: None
        height: dp(44)
        values: root.language_names
        text: root.current_language_name
        on_text: root.on_language_selected(self.text)

    Label:
        id: heading_label
        text: root.heading
        font_size: "24sp"
        halign: "center"
        valign: "middle"
        text_size: self.width, None
        size_hint_y: None
        height: self.texture_size[1]

    Label:
        id: instructions_label
        text: root.instructions
        halign: "center"
        valign: "middle"
        text_size: self.width, None
        size_hint_y: None
        height: self.texture_size[1]

    Button:
        id: action_button
        text: root.button_text
        size_hint_y: None
        height: dp(48)
        on_release: root.on_button_press()

    Label:
        id: status_label
        text: root.status
        halign: "center"
        valign: "middle"
        text_size: self.width, None
        size_hint_y: None
        height: max(self.texture_size[1], dp(24))
"""


class LanguageDemo(BoxLayout):
    """Root widget that manages translation state and UI strings."""

    heading = StringProperty()
    instructions = StringProperty()
    button_text = StringProperty()
    status = StringProperty()

    language_names = ListProperty()
    current_language_name = StringProperty()
    translations = DictProperty()

    _LANGUAGES = (
        ("en", "English"),
        ("zh-Hans", "简体中文"),
        ("ja", "日本語"),
        ("fr", "Français"),
    )

    _TRANSLATIONS = {
        "en": {
            "heading": "Hello!",
            "instructions": "Use the dropdown above to switch between languages.",
            "button_text": "Press me",
            "status_idle": "Waiting for you to press the button…",
            "status_pressed": "You pressed the button!",
        },
        "zh-Hans": {
            "heading": "你好！",
            "instructions": "使用上方的下拉菜单切换显示语言。",
            "button_text": "点我",
            "status_idle": "等待你按下按钮……",
            "status_pressed": "你按下了按钮！",
        },
        "ja": {
            "heading": "こんにちは！",
            "instructions": "上のドロップダウンで表示言語を切り替えます。",
            "button_text": "押してね",
            "status_idle": "ボタンが押されるのを待っています…",
            "status_pressed": "ボタンが押されました！",
        },
        "fr": {
            "heading": "Bonjour !",
            "instructions": "Utilisez la liste déroulante ci-dessus pour changer la langue.",
            "button_text": "Appuie sur moi",
            "status_idle": "En attente que vous appuyiez sur le bouton…",
            "status_pressed": "Vous avez appuyé sur le bouton !",
        },
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.language_names = [language_name for _, language_name in self._LANGUAGES]

        # Default language is English.
        default_language_code, default_language_name = self._LANGUAGES[0]
        self.current_language_name = default_language_name
        self.apply_language(default_language_code)

    def on_language_selected(self, language_name: str) -> None:
        """Switch to the language matching *language_name*."""

        language_code = self._language_code_from_name(language_name)
        self.apply_language(language_code)

    def on_button_press(self) -> None:
        """Update the status text to show an acknowledgement message."""

        translation = self.translations.get(self.current_language_name, {})
        status_pressed = translation.get("status_pressed")

        if not status_pressed:
            # Fallback: if the mapping fails, reapply the language to refresh texts.
            language_code = self._language_code_from_name(self.current_language_name)
            self.apply_language(language_code)
            translation = self.translations.get(self.current_language_name, {})
            status_pressed = translation.get("status_pressed", "")

        self.status = status_pressed

    def apply_language(self, language_code: str) -> None:
        """Apply translations for the given *language_code*."""

        translation = self._TRANSLATIONS.get(language_code)
        if translation is None:
            return

        language_name = self._language_name_from_code(language_code)
        self.current_language_name = language_name

        self.heading = translation["heading"]
        self.instructions = translation["instructions"]
        self.button_text = translation["button_text"]
        self.status = translation["status_idle"]

        # Store translations keyed by the language display name for quick access later.
        if language_name not in self.translations:
            self.translations[language_name] = translation
        else:
            self.translations[language_name].update(translation)

    def _language_code_from_name(self, language_name: str) -> str:
        for code, name in self._LANGUAGES:
            if name == language_name:
                return code
        return self._LANGUAGES[0][0]

    def _language_name_from_code(self, language_code: str) -> str:
        for code, name in self._LANGUAGES:
            if code == language_code:
                return name
        return self._LANGUAGES[0][1]


class MultiLangDemoApp(App):
    """Application entry point."""

    def build(self):
        Builder.load_string(KV, filename="kivy_multi_lang_demo.kv")
        return LanguageDemo()


def main() -> None:
    MultiLangDemoApp().run()


if __name__ == "__main__":
    main()
