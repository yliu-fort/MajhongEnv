"""A small Kivy demo that switches between multiple languages at runtime.

Run with ``python kivy_multi_lang_demo.py``. The app shows a simple
interface where changing the language updates all visible texts without
restarting the application.
"""

from kivy.app import App
from kivy.lang import Builder
from kivy.metrics import dp
from kivy.properties import DictProperty, ListProperty, StringProperty
from kivy.resources import resource_find
from kivy.uix.boxlayout import BoxLayout


KV = """
<LocalizedSpinnerOption@SpinnerOption>:
    font_name: app.font_path or self.font_name

<LanguageSwitcher>:
    orientation: "vertical"
    padding: dp(20)
    spacing: dp(16)

    Label:
        font_name: root.font_path or self.font_name
        font_size: "26sp"
        markup: True
        text: "[b]" + root.translated_title + "[/b]"
        text_size: self.width, None
        halign: "center"
        size_hint_y: None
        height: self.texture_size[1]

    Label:
        font_name: root.font_path or self.font_name
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
            font_name: root.font_path or self.font_name
            text: root.translated_select_language
            size_hint_x: None
            width: dp(160)
            text_size: self.size
            halign: "left"
            valign: "middle"

        Spinner:
            font_name: root.font_path or self.font_name
            text: root.current_language_display
            values: root.language_display_names
            option_cls: "LocalizedSpinnerOption"
            on_text: root.on_language_selected(self.text)

    Button:
        font_name: root.font_path or self.font_name
        text: root.translated_action_button
        size_hint_y: None
        height: dp(48)
        on_release: root.on_action_pressed()

    Label:
        font_name: root.font_path or self.font_name
        text: root.last_action_message
        color: 0.2, 0.6, 0.2, 1
        text_size: self.width, None
        size_hint_y: None
        height: self.texture_size[1]
"""


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

    font_path = StringProperty(resource_find("data/fonts/DejaVuSans.ttf") or "")
    translated_title = StringProperty()
    translated_body = StringProperty()
    translated_select_language = StringProperty()
    translated_action_button = StringProperty()
    last_action_message = StringProperty("")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.font_path:
            app = App.get_running_app()
            if app and getattr(app, "font_path", ""):
                self.font_path = app.font_path
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
    font_path = StringProperty(resource_find("data/fonts/DejaVuSans.ttf") or "")

    def build(self):
        Builder.load_string(KV)
        return LanguageSwitcher(font_path=self.font_path)


if __name__ == "__main__":
    MultiLanguageApp().run()
